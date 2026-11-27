//! The kernel built for a device refreshes as the FMI component does.
//!
//! For each test model, the FMI 3 component settles its parameters at the
//! start point and prints its coordinate and parameters; at five perturbed
//! state points, a new instance then refreshes the algebraic coordinates and
//! the derivative values, and a switching chart set decides its chart and
//! transfers when the decision asks (its points sweep the first state towards
//! the edge of the first chart, so the last ones switch). The device unit of the same model
//! (`crate::me_projection_device`), compiled by a host C++ compiler with the
//! CUDA qualifiers defined away, starts one lane per point from that
//! coordinate and those parameters and runs the same entries. Every status,
//! chart decision, failed-assertion flag and coordinate must be the same, bit
//! for bit. The initialization entry is compiled and not run: the unit
//! settles no parameter binding.

use super::*;
use crate::me_projection_device::{POINTS, device_unit, lane_harness, models, solve_device_unit};

const FMI_DRIVER: &str = r#"
#include <stdio.h>
static ModelInstance* fresh(void) {
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m) exit(1);
    initialize_values(m);
    if (refresh_parameters(m) != fmi3OK) { printf("parameters failed\n"); exit(1); }
    return m;
}
int main(void) {
    ModelInstance* m = fresh();
    printf("Y"); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
    printf("P"); for (size_t i = 0; i < P_LEN; ++i) printf(" %a", m->p[i]); printf("\n");
    double start[Y_LEN]; memcpy(start, m->y, sizeof start);
    free(m);
    for (int k = 1; k <= RMC_POINTS; ++k) {
        m = fresh();
        for (size_t i = 0; i < RMC_STATES; ++i) m->y[i] = start[i] * (1.0 + 0.05 * k) + 0.01 * (double)(i + 1) + (RMC_CHARTS && i == 0 ? 0.196 * k : 0.0);
        int code = refresh_algebraics(m) == fmi3OK;
        code |= (refresh_derivative_values(m) == fmi3OK) << 1;
#if RMC_CHARTS
        bool request = false, values = false, nominals = false;
        code |= (rmc_chart_completed(m, &request) == fmi3OK) << 2;
        code |= (int)request << 3;
        if (request) code |= (rmc_chart_transfer(m, &values, &nominals) == fmi3OK) << 4;
        code |= (int)values << 5;
        printf("%zu ", m->rmc_chart);
#endif
        code |= (int)m->assertion_failed << 6;
        printf("%d", code); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
        free(m);
    }
    return 0;
}
"#;

/// The prerequisites of the comparison: the FMI conformance set with its FMI
/// 3 headers and a host C compiler, for the component, and a host C++
/// compiler, for the device unit. A run that requires external tools fails
/// on a missing one instead of skipping.
fn headers() -> Option<PathBuf> {
    if !conformance_prerequisites_are_available() {
        return None;
    }
    let runs = |tool: &str| Command::new(tool).arg("--version").output().is_ok();
    let headers = standard_roots().1.root.join("headers");
    crate::template_runtime_policy::prerequisites_are_available(
        "host build of the device projection kernel and its FMI 3 driver",
        &[
            ("cc", runs("cc")),
            ("c++", runs("c++")),
            (
                "headers/fmi3Functions.h",
                headers.join("fmi3Functions.h").is_file(),
            ),
        ],
    )
    .then_some(headers)
}

/// The FMI 3 component's start point and refreshed points.
fn fmi_points(
    work: &Path,
    compiled: &rumoca::CompilationResult,
    model: &str,
    states: usize,
    charts: bool,
) -> String {
    let headers = headers().expect("checked by the caller");
    let fmu = build_named_fmu(work, compiled, "fmi3", model);
    let sources = fmu.root.join("sources");
    let driver = work.join("fmi_points.c");
    fs::write(
        &driver,
        format!(
            "#include \"model.c\"\n#define RMC_POINTS {POINTS}\n#define RMC_STATES {states}\n#define RMC_CHARTS {}\n{FMI_DRIVER}",
            u8::from(charts)
        ),
    )
    .expect("write the FMI driver");
    let binary = work.join("fmi_points");
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
        .arg(&driver)
        .args(units.iter().map(|unit| sources.join(unit)))
        .arg("-lm")
        .arg("-o")
        .arg(&binary);
    checked_output(&mut command, "compile the FMI driver");
    let output = checked_output(&mut Command::new(&binary), "run the FMI driver");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The device unit's lanes, compiled by the host C++ compiler, from the
/// FMI component's start point.
fn device_points(work: &Path, unit: &str, start: &str, states: usize) -> String {
    let harness = work.join("device_points.cpp");
    fs::write(&harness, lane_harness(unit, start, states)).expect("write the device harness");
    let binary = work.join("device_points");
    let mut command = Command::new("c++");
    command
        .args([
            "-std=c++17",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .args(["-D__device__=", "-D__host__="])
        .arg(&harness)
        .arg("-lm")
        .arg("-o")
        .arg(&binary);
    checked_output(&mut command, "compile the device unit as host C++");
    let output = checked_output(&mut Command::new(&binary), "run the device lanes");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn a_device_build_refreshes_every_lane_as_the_fmi_component() {
    if headers().is_none() {
        return;
    }
    for (compiled, model) in models() {
        let work = tempdir().expect("device projection work directory");
        let states = crate::me_projection_device::state_count(&compiled);
        let unit = device_unit(&compiled);
        let charts = unit.contains(" rmc_chart_completed(");
        let fmi = fmi_points(work.path(), &compiled, model, states, charts);
        // The driver prints the start point on two lines (`Y`, `P`), then one
        // line per point, whatever its status.
        let (start, points) = fmi.split_at(
            fmi.match_indices('\n')
                .nth(1)
                .expect("the FMI driver prints its start point")
                .0
                + 1,
        );
        assert_eq!(points.lines().count(), POINTS, "{model}: {fmi}");
        let device = device_points(work.path(), &unit, start, states);
        assert_eq!(
            device, points,
            "{model}: the device lanes differ from the FMI component"
        );
        // A device target passes the simulation lowering of the model.
        let solve = device_points(work.path(), &solve_device_unit(&compiled), start, states);
        assert_eq!(
            solve, points,
            "{model}: the lanes of the simulation lowering differ from the FMI component"
        );
    }
}

/// A device unit unrolls the elimination solve of an eliminating affine
/// block with no promoted step into straight-line code over its pattern,
/// with the pooled loops kept for every other call: through it, the lanes of
/// `a_device_build_refreshes_every_lane_as_the_fmi_component` refresh as the
/// FMI component, whose C keeps the pooled loops, bit for bit.
#[test]
fn a_device_unit_unrolls_the_elimination_of_an_affine_block() {
    let mut unrolled = Vec::new();
    for (compiled, model) in models() {
        let unit = device_unit(&compiled);
        if unit.contains(": return rmc_eliminate_solve_") {
            assert!(
                unit.contains("if (e->k == RMC_BS(b, nelim_tear))"),
                "{model}: an unrolled solve serves only a call with no promoted step"
            );
            unrolled.push(model);
        }
    }
    assert!(
        unrolled.contains(&"DeviceAffineSplit"),
        "the eliminating affine block unrolls its solve: {unrolled:?}"
    );
}

/// A device unit unrolls the origin convergence test of a coupled affine
/// block over its coordinates and pattern: through it, the lanes of
/// `a_device_build_refreshes_every_lane_as_the_fmi_component` refresh as the
/// FMI component, whose C keeps the pooled loops, bit for bit.
#[test]
fn a_device_unit_unrolls_the_origin_test_of_an_affine_block() {
    let mut unrolled = Vec::new();
    for (compiled, model) in models() {
        if device_unit(&compiled).contains(": return rmc_origin_converged_") {
            unrolled.push(model);
        }
    }
    assert!(
        unrolled.contains(&"DeviceAffineSplit") && unrolled.contains(&"DeviceAffineChain"),
        "the coupled affine blocks unroll their origin test: {unrolled:?}"
    );
}

/// A lowered model reaches the device kernel through the admission of the C
/// component: one whose initial vector disagrees with its layout is refused
/// before any unit is rendered, with the component's reason.
#[test]
fn a_lowered_model_the_c_kernel_refuses_renders_no_device_unit() {
    let (compiled, _) = models().remove(0);
    let component =
        rumoca_sim::lower_fmi_c_export_component(&compiled.dae).expect("lower the FMI C component");
    let mut model = component.runtime_view().model().clone();
    model.initial_y = rumoca_ir_solve::SolveInitialValues::repeat(0.0, model.initial_y.len() - 1)
        .expect("a shorter initial vector");
    let refused = rumoca_phase_codegen::SolveTemplateRenderer::new_owned_model_with_me_projection(
        std::sync::Arc::new(model),
        &compiled.dae,
    );
    let Err(error) = refused else {
        panic!("the short initial vector is refused");
    };
    assert!(
        error
            .to_string()
            .contains("initial-value owners disagree with checked storage capacity"),
        "{error}"
    );
}

/// A lane reset leaves the retained factors of a lane as they are and clears
/// only their key flags: from a lane of garbage, the reset writes none of the
/// retained storage but each block's key flag, and the lanes still refresh
/// as the FMI component (`a_device_build_refreshes_every_lane_as_the_fmi_component`).
#[test]
fn a_lane_reset_leaves_the_retained_factors_alone() {
    let runs = Command::new("c++").arg("--version").output().is_ok();
    if !crate::template_runtime_policy::prerequisites_are_available(
        "host build of the device lane reset",
        &[("c++", runs)],
    ) {
        return;
    }
    let (compiled, _) = models()
        .into_iter()
        .find(|(_, model)| *model == "DeviceAffineChain")
        .expect("the affine chain test model");
    let driver = r#"
int main(void) {
    ModelInstance* m = (ModelInstance*)malloc(sizeof *m);
    if (!m) return 2;
    memset((void*)m, 0xA5, sizeof *m);
    rmc_lane_reset(m);
    size_t written = 0, keys = 0;
    const unsigned char* bytes = (const unsigned char*)m->rmc_keep_d;
    for (size_t i = 0; i < sizeof m->rmc_keep_d; ++i) written += bytes[i] != 0xA5;
    for (size_t b = 0; b < sizeof rmc_block_desc / sizeof rmc_block_desc[0]; ++b) {
        if (!rmc_block_desc[b].keep) continue;
        keys += 1;
        if (m->rmc_keep_s[rmc_block_desc[b].keep_s] != 0) written += 1000000;
        m->rmc_keep_s[rmc_block_desc[b].keep_s] = (size_t)0xA5A5A5A5A5A5A5A5ull;
    }
    bytes = (const unsigned char*)m->rmc_keep_s;
    for (size_t i = 0; i < sizeof m->rmc_keep_s; ++i) written += bytes[i] != 0xA5;
    printf("%zu %zu\n", keys, written);
    free(m);
    return 0;
}
"#;
    let work = tempdir().expect("lane reset work directory");
    let source = work.path().join("lane_reset.cpp");
    let binary = work.path().join("lane_reset");
    fs::write(
        &source,
        format!(
            "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n{}\n{driver}",
            device_unit(&compiled)
        ),
    )
    .expect("write the lane reset driver");
    checked_output(
        Command::new("c++")
            .args([
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffp-contract=off",
            ])
            .args(["-D__device__=", "-D__host__="])
            .arg(&source)
            .arg("-o")
            .arg(&binary),
        "compile the lane reset driver",
    );
    let output = checked_output(&mut Command::new(&binary), "run the lane reset driver");
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "1 0");
}

/// A device unit defines no strong external symbol, so the units of several
/// models link into one host program.
#[test]
fn a_device_unit_defines_no_external_symbol() {
    let runs = |tool: &str| Command::new(tool).arg("--version").output().is_ok();
    if !crate::template_runtime_policy::prerequisites_are_available(
        "symbol check of the device projection unit",
        &[("c++", runs("c++")), ("nm", runs("nm"))],
    ) {
        return;
    }
    let work = tempdir().expect("device unit work directory");
    for (compiled, model) in models() {
        let source = work.path().join(format!("{model}.cpp"));
        let object = work.path().join(format!("{model}.o"));
        fs::write(&source, device_unit(&compiled)).expect("write the device unit");
        checked_output(
            Command::new("c++")
                .args(["-std=c++17", "-Wall", "-Wextra", "-Werror", "-c"])
                .args(["-D__device__=", "-D__host__="])
                .arg(&source)
                .arg("-o")
                .arg(&object),
            "compile the device unit as host C++",
        );
        let symbols = checked_output(
            Command::new("nm")
                .args(["-g", "--defined-only"])
                .arg(&object),
            "list the device unit's external symbols",
        );
        // Weak symbols (`W`, `V`: inline functions of the C++ library) merge
        // at link time; any other external symbol would collide.
        let listing = String::from_utf8_lossy(&symbols.stdout);
        let strong: Vec<&str> = listing
            .lines()
            .filter(|line| !matches!(line.split_whitespace().nth(1), Some("W" | "V")))
            .collect();
        assert!(strong.is_empty(), "{model}: {strong:?}");
    }
}

/// The device macros include no header after their shims (`device_instance`
/// includes `<stdio.h>` before it redefines `snprintf`): a unit that includes
/// the documented headers before `device_instance` compiles whatever the
/// model, a switching chart set included.
#[test]
fn the_device_unit_includes_no_header_after_its_shims() {
    for (compiled, model) in models() {
        let unit = device_unit(&compiled);
        let shims = unit.find("#define snprintf").expect("the device shims");
        assert!(!unit[shims..].contains("#include"), "{model}");
        // Nor a compound literal, which C++ does not have.
        assert!(!unit.contains("[]){"), "{model}");
    }
}

/// A device reserves a thread's stack for every resident thread, so the
/// unit's parameter bindings keep their previous values and outputs in the
/// lane's instance: `rmc_apply_updates` holds no array sized to its binding
/// targets (one of 9133 doubles twice set the stack of an integrator kernel
/// over 100 KB per thread on the IEEE 39 bus model).
#[test]
fn the_device_bindings_keep_no_array_on_the_thread_stack() {
    let bound = rumoca::Compiler::new()
        .model("BoundParameters")
        .compile_str(
            "model BoundParameters
  parameter Real a = 2;
  parameter Real b = 3*a;
  parameter Real c = a + b;
  Real x(start=1, fixed=true);
  Real u;
equation
  u + 0.1*u*u*u = c*x;
  der(x) = -u;
end BoundParameters;",
            "BoundParameters.mo",
        )
        .unwrap_or_else(|error| panic!("compile BoundParameters: {error:?}"));
    let mut checked = 0;
    for (compiled, model) in models().into_iter().chain([(bound, "BoundParameters")]) {
        let unit = device_unit(&compiled);
        let Some(start) = unit.find(" int rmc_apply_updates(") else {
            continue;
        };
        if !unit.contains(" rmc_binding_runs(") {
            continue;
        }
        let body = &unit[start..start + unit[start..].find("\n}\n").expect("the function ends")];
        assert!(
            !body.contains("double before[") && !body.contains("double binding_output["),
            "{model}: {body}"
        );
        assert!(
            body.contains("m->rmc_bind_before") && body.contains("m->rmc_bind_output"),
            "{model}: {body}"
        );
        checked += 1;
    }
    assert!(checked > 0, "BoundParameters has parameter bindings");
}

/// The device unit's `memcmp` (the device library has none) compares aligned
/// words, in runs of eight, before bytes, which the kernel's bit-for-bit
/// comparisons of doubles read: it orders every pair of buffers as the C
/// library does, aligned or not, whichever byte differs.
#[test]
fn the_device_memcmp_orders_buffers_as_the_c_library() {
    let runs = |tool: &str| Command::new(tool).arg("--version").output().is_ok();
    if !crate::template_runtime_policy::prerequisites_are_available(
        "host build of the device memcmp",
        &[("c++", runs("c++"))],
    ) {
        return;
    }
    let (compiled, model) = models().into_iter().next().expect("a device test model");
    let work = tempdir().expect("device memcmp work directory");
    let harness = work.path().join("memcmp.cpp");
    let driver = r#"
#include <cstdio>
int main() {
    alignas(8) unsigned char a[160], b[160];
    int bad = 0, cases = 0;
    for (size_t offset = 0; offset < 9; ++offset)
        for (size_t n = 0; n + offset <= 152; ++n)
            for (int at = -1; at < (int)n; ++at)
                for (int sign = -1; sign <= 1; sign += 2) {
                    for (size_t i = 0; i < 160; ++i) a[i] = b[i] = (unsigned char)(7 * i + 3);
                    if (at >= 0) b[offset + at] = (unsigned char)(a[offset + at] + sign);
                    int got = rmc_device_memcmp(a + offset, b + offset, n), want = (memcmp)(a + offset, b + offset, n);
                    bad += (got > 0) != (want > 0) || (got < 0) != (want < 0);
                    ++cases;
                }
    printf("%d %d\n", bad, cases);
    return 0;
}
"#;
    fs::write(&harness, format!("{}\n{driver}", device_unit(&compiled)))
        .expect("write the memcmp harness");
    let binary = work.path().join("memcmp");
    checked_output(
        Command::new("c++")
            .args(["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"])
            .args(["-D__device__=", "-D__host__="])
            .arg(&harness)
            .arg("-lm")
            .arg("-o")
            .arg(&binary),
        "compile the device memcmp harness",
    );
    let output = checked_output(&mut Command::new(&binary), "run the device memcmp harness");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut counts = stdout.split_whitespace();
    assert_eq!(counts.next(), Some("0"), "{model}: {stdout}");
    assert!(
        counts.next().is_some_and(|cases| cases != "0"),
        "{model}: {stdout}"
    );
}
