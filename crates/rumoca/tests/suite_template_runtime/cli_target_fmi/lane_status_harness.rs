//! A lane build of the shared projection kernel (`RMC_LANE_STATUS`, one
//! instance per lane of a device integrator, with no logger) reports what an
//! FMI component logs (SPEC_0044 ME-PROJ-003).
//!
//! One C harness is compiled twice from the same generated sources, as the
//! FMI component and as a lane build. Each scenario drives the kernel into
//! fallbacks or faults on a fresh instance: the FMI build prints what it
//! logs, the lane build prints what it logs (nothing from the kernel) and
//! what `rmc_lane_report` reads from the instance afterwards. The lane's
//! fallback reports are the FMI component's `projection` messages, its
//! first fault is the first `error` message the kernel logged, and its fault
//! count is the number of them.

use super::*;

/// Renames every dispatcher and the assignment entry of the program units
/// before they are included, so the wrappers below take their names.
const RENAMES: &str = "#define rmc_row_call rmc_row_call_real\n#define rmc_inv_call rmc_inv_call_real\n#define rmc_dep_call rmc_dep_call_real\n#define rmc_jvp_call rmc_jvp_call_real\n#define rmc_lane_call rmc_lane_call_real\n#define rmc_iso_call rmc_iso_call_real\n#define rmc_causal_call rmc_causal_call_real\n#define rmc_assign rmc_assign_real\n";

const UNDEFS: &str = "#undef rmc_row_call\n#undef rmc_inv_call\n#undef rmc_dep_call\n#undef rmc_jvp_call\n#undef rmc_lane_call\n#undef rmc_iso_call\n#undef rmc_causal_call\n#undef rmc_assign\n";

/// The residual rows can answer as built, forward an index past their
/// family (a program fault), or poison their first output (a projection
/// that cannot converge); the next assignment can report a non-finite
/// store, which sends a staged refresh to its complete-plan rescue.
const WRAPPERS: &str = r#"
enum { ROWS_AS_BUILT, ROWS_BAD_INDEX, ROWS_POISONED };
static int row_mode;
RMC_API int rmc_row_call(size_t p, ModelInstance* m, double* o) {
    if (row_mode == ROWS_BAD_INDEX) return rmc_row_call_real((size_t)-1, m, o);
    int result = rmc_row_call_real(p, m, o);
    if (row_mode == ROWS_POISONED && result == 0) o[0] = NAN;
    return result;
}
RMC_API int rmc_inv_call(size_t p, ModelInstance* m, double* o) { return rmc_inv_call_real(p, m, o); }
RMC_API int rmc_dep_call(size_t p, ModelInstance* m, const double* s, double* o) { return rmc_dep_call_real(p, m, s, o); }
RMC_API int rmc_jvp_call(size_t p, ModelInstance* m, const double* s, double* o) { return rmc_jvp_call_real(p, m, s, o); }
RMC_API int rmc_lane_call(size_t p, ModelInstance* m, const double* s, double* o) { return rmc_lane_call_real(p, m, s, o); }
RMC_API int rmc_iso_call(size_t p, ModelInstance* m, double* o) { return rmc_iso_call_real(p, m, o); }
RMC_API int rmc_causal_call(size_t p, ModelInstance* m, const size_t* t) { return rmc_causal_call_real(p, m, t); }
RMC_API int rmc_assign(ModelInstance* m, size_t first, size_t count);
#if HAS_ASSIGN
static int assign_nonfinite_once;
RMC_API int rmc_assign(ModelInstance* m, size_t first, size_t count) {
    if (assign_nonfinite_once) { assign_nonfinite_once = 0; return 1; }
    return rmc_assign_real(m, first, count);
}
#endif
"#;

const SCENARIOS: &str = r#"
#include <stdio.h>
static void capture(fmi3InstanceEnvironment environment, fmi3Status status, fmi3String category, fmi3String message) {
    (void)environment; (void)status;
    printf("log %s | %s\n", category, message);
}
#if defined(RMC_LANE_STATUS)
static void sink(void* environment, const char* category, const char* message) {
    (void)environment;
    printf("report %s | %s\n", category, message);
}
#endif
static ModelInstance* begin(const char* scenario, bool settle) {
    printf("scenario %s\n", scenario);
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m) { printf("FAILED: allocate the instance\n"); exit(1); }
    initialize_values(m);
    m->logger = capture;
    if (settle && settle_values(m) != fmi3OK) printf("settle failed\n");
    return m;
}
static void finish(ModelInstance* m) {
#if defined(RMC_LANE_STATUS)
    printf("faults %zu\n", m->rmc_lane_faults);
    size_t reports = rmc_lane_report(m, sink, NULL);
    printf("reports %zu\n", reports);
#endif
    printf("end\n");
    free(m);
}
int main(void) {
    ModelInstance* m;
#if HAS_ASSIGN && defined(RMC_FB_PLAN)
    /* Every staged refresh rescues a non-finite assignment: the plan's
       fallback rate crosses the policy rate after the minimum of calls. */
    m = begin("rescue", true);
    for (int call = 0; call < 2 * RMC_FALLBACK_MIN_CALLS; ++call) { assign_nonfinite_once = 1; (void)refresh_algebraics(m); }
    assign_nonfinite_once = 0;
    finish(m);
#endif
#if defined(RMC_FB_PLAN)
    /* One fallback in the first forty calls of block 0 stays below the rate;
       three calls that each fall back twice cross it. */
    m = begin("block", true);
    for (int call = 0; call < 40; ++call) { m->rmc_fb_calls[0] += 1; m->rmc_fb_active[0] = true; if (call == 0) rmc_fb_note(m, 0); m->rmc_fb_active[0] = false; }
    for (int call = 0; call < 3; ++call) { m->rmc_fb_calls[0] += 1; m->rmc_fb_active[0] = true; rmc_fb_note(m, 0); rmc_fb_note(m, 0); m->rmc_fb_active[0] = false; }
    finish(m);
#endif
    m = begin("poisoned", true);
    row_mode = ROWS_POISONED; (void)refresh_algebraics(m); (void)refresh_algebraics(m); row_mode = ROWS_AS_BUILT;
    finish(m);
    m = begin("bad-index", true);
    row_mode = ROWS_BAD_INDEX; (void)refresh_algebraics(m); (void)refresh_derivative_values(m); row_mode = ROWS_AS_BUILT;
    finish(m);
#if defined(RMC_DWORK)
    m = begin("workspace", true);
    size_t top = m->rmc_dtop; m->rmc_dtop = RMC_DWORK; (void)refresh_algebraics(m); m->rmc_dtop = top;
    finish(m);
#endif
#if defined(RMC_INIT_ROWS)
    /* The MLS initialization, whose settled view runs the poisoned or faulted
       refresh, or whose frames find the arena full. */
    m = begin("initial-poisoned", false);
    row_mode = ROWS_POISONED; (void)rmc_initialize(m, 0.0); row_mode = ROWS_AS_BUILT;
    finish(m);
    m = begin("initial-bad-index", false);
    row_mode = ROWS_BAD_INDEX; (void)rmc_initialize(m, 0.0); row_mode = ROWS_AS_BUILT;
    finish(m);
    m = begin("initial-workspace", false);
    m->rmc_dtop = RMC_DWORK; (void)rmc_initialize(m, 0.0); m->rmc_dtop = 0;
    finish(m);
#endif
    return 0;
}
"#;

/// A limiter loop whose strict `noEvent` guard reads its own unknown: its
/// block reports its ES016 fold when its projection fails.
const FOLD_LIMITER: &str = "model FoldLimiter\n  parameter Real V0 = 15000;\n  parameter Real k = 0.5;\n  Real v_in;\n  Real v_out;\n  Real x(start = 1, fixed = true);\nequation\n  v_in = k * v_out - 10 * sin(20 * time) + 0.01 * x;\n  v_out = smooth(0, noEvent(if V0 * v_in > 15 then 15 else if V0 * v_in < -15 then -15 else V0 * v_in));\n  der(x) = -x + v_out;\nend FoldLimiter;\n";

/// A cubic whose start the initialization settles from an initial equation
/// on its algebraic image.
const INITIAL_CUBIC: &str = "model InitialCubic\n  Real x(start = 0.3);\n  Real y;\ninitial equation\n  y = 0.5;\nequation\n  y = x * x * x + x;\n  der(x) = -x;\nend InitialCubic;\n";

/// What one scenario printed.
#[derive(Debug, Default)]
struct Scenario {
    name: String,
    logs: Vec<String>,
    reports: Vec<String>,
    faults: Option<usize>,
}

fn parse(stdout: &str) -> Vec<Scenario> {
    let mut scenarios = Vec::new();
    let mut current: Option<Scenario> = None;
    for line in stdout.lines() {
        if let Some(name) = line.strip_prefix("scenario ") {
            current = Some(Scenario {
                name: name.to_owned(),
                ..Scenario::default()
            });
        } else if line == "end" {
            scenarios.extend(current.take());
        } else if let Some(scenario) = current.as_mut() {
            if let Some(log) = line.strip_prefix("log ") {
                scenario.logs.push(log.to_owned());
            } else if let Some(report) = line.strip_prefix("report ") {
                scenario.reports.push(report.to_owned());
            } else if let Some(faults) = line.strip_prefix("faults ") {
                scenario.faults = Some(faults.parse().expect("a fault count"));
            }
        }
    }
    scenarios
}

/// Builds the harness of `model` as the FMI component or as a lane build,
/// and runs it.
fn run(sources: &Path, headers: &Path, work: &Path, (model, lane): (&str, bool)) -> String {
    let assign =
        fs::read_to_string(sources.join("rmc_assign.c")).expect("read the assignment unit");
    let has_assign = u8::from(assign.contains("rmc_assign(ModelInstance"));
    let includes: String = [
        "rmc_rows.c",
        "rmc_jacobian.c",
        "rmc_isolators.c",
        "rmc_assign.c",
    ]
    .iter()
    .map(|unit| format!("#include \"{unit}\"\n"))
    .collect();
    let text = format!(
        "#define HAS_ASSIGN {has_assign}\n{RENAMES}{includes}{UNDEFS}{WRAPPERS}#include \"model.c\"\n{SCENARIOS}"
    );
    let name = format!("{model}_{}", if lane { "lane" } else { "fmi" });
    let harness = work.join(format!("{name}.c"));
    fs::write(&harness, text).expect("write the lane status harness");
    let binary = work.join(&name);
    let mut command = Command::new("cc");
    command.args([
        "-std=c11",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]);
    if lane {
        command.arg("-DRMC_LANE_STATUS");
    }
    command
        .arg(format!("-I{}", sources.display()))
        .arg(format!("-I{}", headers.display()))
        .arg(&harness)
        .arg(sources.join("rmc_functions.c"))
        .arg("-lm")
        .arg("-o")
        .arg(&binary);
    checked_output(&mut command, "compile the lane status harness");
    let output = checked_output(&mut Command::new(&binary), "run the lane status harness");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The kernel's part of one FMI scenario's log (what the lane build, which
/// logs only outside the kernel, did not log), split into its `projection`
/// messages and its `error` messages.
fn kernel_log<'a>(fmi: &'a Scenario, lane: &Scenario) -> (Vec<&'a str>, Vec<&'a str>) {
    let mut outside = lane.logs.clone();
    let mut kernel = Vec::new();
    for log in &fmi.logs {
        match outside.iter().position(|other| other == log) {
            Some(index) => {
                outside.remove(index);
            }
            None => kernel.push(log.as_str()),
        }
    }
    assert!(
        outside.is_empty(),
        "{}: the lane build logged what the FMI component did not: {outside:?}",
        fmi.name
    );
    let projection = kernel
        .iter()
        .copied()
        .filter(|log| log.starts_with("projection | "))
        .collect();
    let errors = kernel
        .iter()
        .copied()
        .filter(|log| log.starts_with("error | "))
        .collect();
    (projection, errors)
}

/// Compares the two builds of `model` scenario by scenario; the messages the
/// kernel logged.
fn compare(model: &str, fmi: &str, lane: &str) -> Vec<String> {
    let (fmi, lane) = (parse(fmi), parse(lane));
    assert_eq!(
        fmi.iter().map(|s| &s.name).collect::<Vec<_>>(),
        lane.iter().map(|s| &s.name).collect::<Vec<_>>(),
        "{model}: both builds run the same scenarios"
    );
    let mut covered = Vec::new();
    for (fmi, lane) in fmi.iter().zip(&lane) {
        let context = format!("{model} {}", fmi.name);
        let (mut projection, errors) = kernel_log(fmi, lane);
        let mut fallbacks: Vec<&str> = lane
            .reports
            .iter()
            .map(String::as_str)
            .filter(|report| report.starts_with("projection | "))
            .collect();
        projection.sort_unstable();
        fallbacks.sort_unstable();
        assert_eq!(
            fallbacks, projection,
            "{context}: the lane reports the FMI fallback messages"
        );
        let faults: Vec<&str> = lane
            .reports
            .iter()
            .map(String::as_str)
            .filter(|report| report.starts_with("error | "))
            .collect();
        assert_eq!(
            faults,
            errors.first().copied().into_iter().collect::<Vec<_>>(),
            "{context}: the lane reports the first fault the FMI component logged ({errors:?})"
        );
        assert_eq!(
            lane.faults,
            Some(errors.len()),
            "{context}: the lane counts every fault the FMI component logged"
        );
        assert_eq!(
            lane.reports.len(),
            fallbacks.len() + faults.len(),
            "{context}: every lane report is a fallback or a fault"
        );
        covered.extend(
            projection
                .iter()
                .chain(&errors)
                .map(|log| (*log).to_owned()),
        );
    }
    covered
}

#[test]
fn a_lane_build_reports_what_the_fmi_component_logs() {
    if !conformance_prerequisites_are_available() {
        return;
    }
    let headers = standard_roots().1.root.join("headers");
    let torn = super::torn_tangent_harness::MODEL;
    let compile = |source: &str, model: &str| {
        rumoca::Compiler::new()
            .model(model)
            .compile_str(source, &format!("{model}.mo"))
            .unwrap_or_else(|error| panic!("compile {model}: {error:?}"))
    };
    let models = [
        (
            super::affine_promotion::compile(),
            super::affine_promotion::MODEL,
        ),
        (
            compile(super::residual_split_harness::LOOP, "BadIndexLoop"),
            "BadIndexLoop",
        ),
        (compile(&super::torn_tangent_harness::source(), torn), torn),
        (compile(FOLD_LIMITER, "FoldLimiter"), "FoldLimiter"),
        (compile(INITIAL_CUBIC, "InitialCubic"), "InitialCubic"),
    ];
    let mut covered = Vec::new();
    for (compiled, model) in &models {
        let work = tempdir().expect("lane status work directory");
        let fmu = build_named_fmu(work.path(), compiled, "fmi3", model);
        let sources = fmu.root.join("sources");
        let fmi = run(&sources, &headers, work.path(), (model, false));
        let lane = run(&sources, &headers, work.path(), (model, true));
        covered.extend(compare(model, &fmi, &lane));
    }
    // The scenarios reach a fallback report of a block and of the staged
    // plan, a program fault, an exhausted workspace, a failed projection,
    // an ES016 fold, and an initialization fault.
    let expected = [
        "projection | projection block ",
        "projection | the staged algebraic refresh fell back on ",
        "error | projection program index outside its dispatcher",
        "error | algebraic projection workspace exhausted",
        "algebraic projection did not establish coordinate convergence",
        "ES016",
        "error | initial",
    ];
    let missing: Vec<_> = expected
        .iter()
        .filter(|needle| !covered.iter().any(|log| log.contains(*needle)))
        .collect();
    assert!(
        missing.is_empty(),
        "reports no scenario reached: {missing:?}\n{covered:#?}"
    );
}

/// The scenarios a device unit runs without wrapped dispatchers, written once
/// for the FMI component (`FMI_BUILD`) and for the device unit compiled as
/// host C++: a block's fallback rate crossing the policy rate, an exhausted
/// arena in a refresh, and one in the initialization.
const DEVICE_SCENARIOS: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#if FMI_BUILD
static void capture(fmi3InstanceEnvironment environment, fmi3Status status, fmi3String category, fmi3String message) {
    (void)environment; (void)status;
    printf("log %s | %s\n", category, message);
}
#else
static void sink(void* environment, const char* category, const char* message) {
    (void)environment;
    printf("report %s | %s\n", category, message);
}
#endif
static ModelInstance* begin(const char* scenario) {
    printf("scenario %s\n", scenario);
    ModelInstance* m = (ModelInstance*)calloc(1, sizeof *m);
    if (!m) { printf("FAILED: allocate the instance\n"); exit(1); }
#if FMI_BUILD
    initialize_values(m);
    m->logger = capture;
    if (refresh_parameters(m) != fmi3OK) printf("parameters failed\n");
#else
    rmc_lane_reset(m); m->time = 0.0; m->parameters_dirty = false;
    for (size_t i = 0; i < Y_LEN; ++i) m->y[i] = rmc_y0[i];
    for (size_t i = 0; i < P_LEN; ++i) m->p[i] = rmc_p0[i];
#endif
    return m;
}
static void finish(ModelInstance* m) {
#if !FMI_BUILD
    printf("faults %zu\n", m->rmc_lane_faults);
    printf("reports %zu\n", rmc_lane_report(m, sink, NULL));
#endif
    printf("end\n");
    free(m);
}
int main(void) {
    ModelInstance* m;
#if FMI_BUILD
    m = begin("start");
    printf("Y"); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
    printf("P"); for (size_t i = 0; i < P_LEN; ++i) printf(" %a", m->p[i]); printf("\n");
    finish(m);
#endif
#if defined(RMC_FB_PLAN)
    m = begin("block");
    for (int call = 0; call < 40; ++call) { m->rmc_fb_calls[0] += 1; m->rmc_fb_active[0] = true; if (call == 0) rmc_fb_note(m, 0); m->rmc_fb_active[0] = false; }
    for (int call = 0; call < 3; ++call) { m->rmc_fb_calls[0] += 1; m->rmc_fb_active[0] = true; rmc_fb_note(m, 0); rmc_fb_note(m, 0); m->rmc_fb_active[0] = false; }
    finish(m);
#endif
#if defined(RMC_DWORK)
    m = begin("workspace");
    m->rmc_dtop = RMC_DWORK; (void)refresh_algebraics(m); m->rmc_dtop = 0;
    finish(m);
#endif
#if defined(RMC_INIT_ROWS)
    m = begin("initial-workspace");
    m->rmc_dtop = RMC_DWORK; (void)rmc_initialize(m, 0.0); m->rmc_dtop = 0;
    finish(m);
#endif
    return 0;
}
"#;

/// The `Y` or `P` values the FMI build printed, as C initializers.
fn start_values(fmi: &str, tag: &str) -> String {
    fmi.lines()
        .find_map(|line| line.strip_prefix(tag))
        .unwrap_or_else(|| panic!("the FMI build prints its `{tag}` line: {fmi}"))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(", ")
}

/// The device unit of the ME projection kernel (`device_instance`,
/// `device_kernel`, every device build being a lane build), compiled by the
/// host C++ compiler with the CUDA qualifiers defined away, reports what the
/// FMI component of the same model logs: its lanes log nothing, and the
/// host's `rmc_lane_report` of each lane gives the FMI messages.
#[test]
fn a_device_unit_reports_what_the_fmi_component_logs() {
    if !conformance_prerequisites_are_available() {
        return;
    }
    let headers = standard_roots().1.root.join("headers");
    let mut covered = Vec::new();
    for (compiled, model) in crate::me_projection_device::models() {
        let work = tempdir().expect("device lane status work directory");
        let fmu = build_named_fmu(work.path(), &compiled, "fmi3", model);
        let sources = fmu.root.join("sources");
        let driver = work.path().join("fmi_reports.c");
        fs::write(
            &driver,
            format!("#define FMI_BUILD 1\n#include \"model.c\"\n{DEVICE_SCENARIOS}"),
        )
        .expect("write the FMI driver");
        let binary = work.path().join("fmi_reports");
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
            .arg(&driver)
            .args(
                [
                    "rmc_rows.c",
                    "rmc_jacobian.c",
                    "rmc_isolators.c",
                    "rmc_assign.c",
                    "rmc_functions.c",
                ]
                .iter()
                .map(|unit| sources.join(unit)),
            )
            .arg("-lm")
            .arg("-o")
            .arg(&binary);
        checked_output(&mut command, "compile the FMI report driver");
        let fmi = String::from_utf8_lossy(
            &checked_output(&mut Command::new(&binary), "run the FMI report driver").stdout,
        )
        .into_owned();
        let unit = crate::me_projection_device::device_unit(&compiled);
        let device_source = format!(
            "#define FMI_BUILD 0\n#include <stdio.h>\n{unit}\nstatic const double rmc_y0[Y_LEN] = {{ {} }};\nstatic const double rmc_p0[P_LEN] = {{ {} }};\n{DEVICE_SCENARIOS}",
            start_values(&fmi, "Y"),
            start_values(&fmi, "P"),
        );
        let harness = work.path().join("device_reports.cpp");
        fs::write(&harness, device_source).expect("write the device report harness");
        let binary = work.path().join("device_reports");
        let mut command = Command::new("c++");
        // The unit holds every refresh entry; the scenarios call only some.
        command
            .args([
                "-std=c++17",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-unused-function",
            ])
            .args(["-D__device__=", "-D__host__="])
            .arg(&harness)
            .arg("-lm")
            .arg("-o")
            .arg(&binary);
        checked_output(
            &mut command,
            "compile the device report harness as host C++",
        );
        let device = String::from_utf8_lossy(
            &checked_output(&mut Command::new(&binary), "run the device report harness").stdout,
        )
        .into_owned();
        let fmi_scenarios = fmi
            .split_once("end\n")
            .map_or(fmi.as_str(), |(_, rest)| rest);
        covered.extend(compare(model, fmi_scenarios, &device));
    }
    for needle in [
        "projection | projection block ",
        "error | algebraic projection workspace exhausted",
    ] {
        assert!(
            covered.iter().any(|log| log.contains(needle)),
            "no device scenario reached `{needle}`: {covered:#?}"
        );
    }
}
