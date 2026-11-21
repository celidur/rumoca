//! The projection arena holds every frame chain the kernel opens: a refresh,
//! the initialization frames around it, and a chart transfer, which keeps its
//! row outputs while it refreshes the target chart.
//!
//! The harness fills the arena with a marker before each call and reads back
//! the highest double the call wrote; every call must succeed without an
//! exhausted workspace, its high-water mark inside the arena.

use super::*;

/// The checks: settle the start point, refresh, and (for a switching
/// component) decide a completed step and transfer to the chart it latches,
/// each from a marked arena.
const CHECKS: &str = r#"
#include <stdio.h>
static int failures, exhausted;
#define CHECK(condition, what) do { if (!(condition)) { printf("FAILED: %s\n", what); failures += 1; } } while (0)
static void capture(fmi3InstanceEnvironment environment, fmi3Status status, fmi3String category, fmi3String message) {
    (void)environment; (void)status; (void)category;
    if (strstr(message, "workspace exhausted")) exhausted += 1;
}
static const uint64_t MARK = 0x7ff4deadbeefcafeull;
static void mark(ModelInstance* m) {
    for (size_t i = 0; i < RMC_DWORK; ++i) memcpy(&m->rmc_d[i], &MARK, sizeof(double));
}
static size_t high_water(const ModelInstance* m) {
    size_t top = 0;
    for (size_t i = 0; i < RMC_DWORK; ++i) if (memcmp(&m->rmc_d[i], &MARK, sizeof(double)) != 0) top = i + 1;
    return top;
}
int main(void) {
    ModelInstance* m = calloc(1, sizeof *m);
    CHECK(m != NULL, "allocate the instance");
    initialize_values(m);
    m->logger = capture;
    size_t peak = 0;
    mark(m);
    CHECK(settle_values(m) == fmi3OK, "settle the start point");
    peak = high_water(m) > peak ? high_water(m) : peak;
    mark(m);
    CHECK(refresh_algebraics(m) == fmi3OK, "refresh");
    peak = high_water(m) > peak ? high_water(m) : peak;
#ifdef RMC_NCHARTS
    /* Round the circle, an unreachable keep reference makes a completed step
       latch the other chart where it is regular, and the transfer runs to
       it. */
    bool request = false, changed = false, nominals = false;
    for (int step = 1; step < 100 && !request; ++step) {
        m->y[0] = (double)step / 100.0;
        CHECK(refresh_algebraics(m) == fmi3OK, "refresh round the circle");
        m->rmc_chart_reference = 1e300;
        mark(m);
        CHECK(rmc_chart_completed(m, &request) == fmi3OK, "decide a completed step");
        peak = high_water(m) > peak ? high_water(m) : peak;
    }
    CHECK(request, "a completed step requests a switch");
    mark(m);
    CHECK(rmc_chart_transfer(m, &changed, &nominals) == fmi3OK, "transfer to the latched chart");
    peak = high_water(m) > peak ? high_water(m) : peak;
#endif
    CHECK(exhausted == 0, "no call exhausts the workspace");
    CHECK(peak <= RMC_DWORK, "the high-water mark stays inside the arena");
    printf("peak %zu of %d\n", peak, RMC_DWORK);
    if (failures == 0) printf("ok\n");
    free(m);
    return failures != 0;
}
"#;

fn headers() -> Option<PathBuf> {
    if !conformance_prerequisites_are_available() {
        return None;
    }
    let headers = standard_roots().1.root.join("headers");
    headers.join("fmi3Functions.h").is_file().then_some(headers)
}

fn run(compiled: &rumoca::CompilationResult, model: &str) -> String {
    let headers = headers().expect("checked by the caller");
    let work = tempdir().expect("arena harness work directory");
    let fmu = build_named_fmu(work.path(), compiled, "fmi3", model);
    let sources = fmu.root.join("sources");
    let harness = work.path().join("arena.c");
    fs::write(&harness, format!("#include \"model.c\"\n{CHECKS}")).expect("write the harness");
    let binary = work.path().join("arena");
    let units = [
        "rmc_rows.c",
        "rmc_jacobian.c",
        "rmc_isolators.c",
        "rmc_assign.c",
        "rmc_functions.c",
    ];
    let mut command = Command::new("cc");
    command
        .args([
            "-std=c11",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(format!("-I{}", sources.display()))
        .arg(format!("-I{}", headers.display()))
        .arg(&harness)
        .args(units.iter().map(|unit| sources.join(unit)))
        .arg("-lm")
        .arg("-o")
        .arg(&binary);
    checked_output(&mut command, "compile the arena harness");
    let output = Command::new(&binary)
        .output()
        .expect("run the arena harness");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success() && stdout.contains("\nok"),
        "{model}: {stdout}"
    );
    stdout
}

/// Every frame chain fits the arena, a chart transfer's included.
#[test]
fn every_frame_chain_fits_the_projection_arena() {
    if headers().is_none() {
        return;
    }
    let chart = include_str!("../../fixtures/index_reduction/SplitCircleChart.mo");
    let models: [(rumoca::CompilationResult, &str); 3] = [
        (
            super::affine_promotion::compile(),
            super::affine_promotion::MODEL,
        ),
        (
            rumoca::Compiler::new()
                .model("SplitCircleChart")
                .compile_str(chart, "SplitCircleChart.mo")
                .expect("compile the split circle"),
            "SplitCircleChart",
        ),
        (
            rumoca::Compiler::new()
                .model(super::torn_tangent_harness::MODEL)
                .compile_str(&super::torn_tangent_harness::source(), "TornTangent.mo")
                .expect("compile the torn loop"),
            super::torn_tangent_harness::MODEL,
        ),
    ];
    for (compiled, model) in &models {
        let output = run(compiled, model);
        eprintln!(
            "{model}: {}",
            output
                .lines()
                .find(|line| line.starts_with("peak"))
                .unwrap_or("")
        );
    }
}
