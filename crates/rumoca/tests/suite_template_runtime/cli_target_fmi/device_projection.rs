//! The kernel built for a device refreshes as the FMI component does.
//!
//! For each test model, the FMI 3 component settles its parameters at the
//! start point and prints its coordinate and parameters; it then refreshes
//! the algebraic coordinates at five perturbed state points. The device unit
//! of the same model (`crate::me_projection_device`), compiled by a host C++
//! compiler with the CUDA qualifiers defined away, starts one lane per point
//! from that coordinate and those parameters and refreshes it. Every status
//! failed-assertion flag and every coordinate must be the same, bit for bit.

use super::*;
use crate::me_projection_device::{POINTS, device_unit, lane_harness, models};

const FMI_DRIVER: &str = r#"
#include <stdio.h>
int main(void) {
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m) return 1;
    initialize_values(m);
    if (refresh_parameters(m) != fmi3OK) { printf("parameters failed\n"); return 1; }
    printf("Y"); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
    printf("P"); for (size_t i = 0; i < P_LEN; ++i) printf(" %a", m->p[i]); printf("\n");
    double start[Y_LEN]; memcpy(start, m->y, sizeof start);
    for (int k = 1; k <= RMC_POINTS; ++k) {
        memcpy(m->y, start, sizeof start);
        for (size_t i = 0; i < RMC_STATES; ++i) m->y[i] = start[i] * (1.0 + 0.05 * k) + 0.01 * (double)(i + 1);
        int ok = refresh_algebraics(m) == fmi3OK;
        printf("%d %d", ok, (int)m->assertion_failed); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
        m->assertion_failed = false;
    }
    free(m);
    return 0;
}
"#;

fn headers() -> Option<PathBuf> {
    if !conformance_prerequisites_are_available() {
        return None;
    }
    let headers = standard_roots().1.root.join("headers");
    headers.join("fmi3Functions.h").is_file().then_some(headers)
}

/// The FMI 3 component's start point and refreshed points.
fn fmi_points(
    work: &Path,
    compiled: &rumoca::CompilationResult,
    model: &str,
    states: usize,
) -> String {
    let headers = headers().expect("checked by the caller");
    let fmu = build_named_fmu(work, compiled, "fmi3", model);
    let sources = fmu.root.join("sources");
    let driver = work.join("fmi_points.c");
    fs::write(
        &driver,
        format!("#include \"model.c\"\n#define RMC_POINTS {POINTS}\n#define RMC_STATES {states}\n{FMI_DRIVER}"),
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
fn device_points(
    work: &Path,
    compiled: &rumoca::CompilationResult,
    start: &str,
    states: usize,
) -> String {
    let harness = work.join("device_points.cpp");
    fs::write(
        &harness,
        lane_harness(&device_unit(compiled), start, states),
    )
    .expect("write the device harness");
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
        let fmi = fmi_points(work.path(), &compiled, model, states);
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
        let device = device_points(work.path(), &compiled, start, states);
        assert_eq!(
            device, points,
            "{model}: the device lanes differ from the FMI component"
        );
    }
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

/// The device macros include no header: a unit that includes the documented
/// headers before `device_instance`, which redefines `snprintf`, compiles
/// whatever the model, a switching chart set included.
#[test]
fn the_device_unit_includes_no_header_after_its_shims() {
    for (compiled, model) in models() {
        let unit = device_unit(&compiled);
        let shims = unit.find("#define snprintf").expect("the device shims");
        assert!(!unit[shims..].contains("#include"), "{model}");
    }
}
