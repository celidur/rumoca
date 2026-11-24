//! A lane target's initialization certifies the settled states on the
//! retained state manifold, as the linked kernel does (MLS §8.6, SPEC_0044
//! ME-PROJ-001).
//!
//! The retained manifold rows close the initialization residual as ordinary
//! rows, which the runtime's `settle_initialization_system` settles with the
//! initial equations. The linked kernel then certifies the settled states on
//! the manifold (`certify_state_manifold`, its `CertifyInitial`). The shared
//! kernel's `rmc_initialize` does the same through `rmc_certify_state_manifold`
//! for a target that emits `manifold_kernel`. The test renders the device unit
//! of `PinnedStart` (rows on the settled refresh) and `PinnedSeed` (rows over
//! state seeds) from the lowered model (`device_instance`,
//! `binding_updates`, `device_kernel`, `manifold_kernel`). It builds that unit
//! as host C++ with the CUDA qualifiers defined away and initializes one lane
//! per start guess. Each lane is compared with the runtime on the same
//! guesses.

use std::sync::Arc;

use rumoca::Compiler;
use rumoca_eval_solve::projection_policy::ALGEBRAIC_REFRESH_MAX_ITERS;
use rumoca_phase_codegen::SolveTemplateRenderer;
use rumoca_sim::{SimOptions, lower_correlated_for_simulation_with_overrides};
use rumoca_solver::SolveRuntime;

/// The tolerance both initializations and certifications run at.
const TOLERANCE: f64 = 1.0e-10;

/// `x^3 = 1` is retained over the state `x`; the initial equation constrains
/// the algebraic `w`, which the refresh computes from both states, so the
/// C initialization evaluates its rows on the settled refresh.
const PINNED_START: &str = r#"
model PinnedStart
  Real x(start=1.2);
  Real z(start=0.5);
  Real lambda;
  Real w;
initial equation
  w = 2;
equation
  der(x) = lambda*x;
  der(z) = 1 - z;
  w = z + x*x;
  x*x*x = 1;
end PinnedStart;
"#;

/// The same retained row with no initial equation: the residual reads no
/// continuous algebraic, so the C initialization evaluates it on the instance
/// coordinates and solves `x` from its state seed.
const PINNED_SEED: &str = r#"
model PinnedSeed
  Real x(start=1.2);
  Real z(start=0.5, fixed=true);
  Real lambda;
equation
  der(x) = lambda*x;
  der(z) = 1 - z;
  x*x*x = 1;
end PinnedSeed;
"#;

const DEVICE_UNIT: &str = r#"{%- from "fmi-me-projection.jinja" import device_instance, device_kernel, binding_updates -%}
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
{{ device_instance(solve, me_refresh, "rmc") }}
static __device__ rmcStatus update_discrete_equations(ModelInstance* m) { (void)m; return rmcOK; }
{{ binding_updates(solve, me_bindings, "rmc", device=true) }}
{{ real_extremum_prelude("cuda", true) }}
{{ device_kernel(solve, me_refresh, "rmc", solve_blocks.state_manifold) }}
"#;

/// Start guesses: the model's own, a scaled one, and one off the manifold
/// that the initialization moves back onto it.
const GUESSES: [f64; 3] = [1.0, 1.3, 0.6];

fn literals(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One lane per guess: `rmc_initialize` from the start values scaled by the
/// guess, then the lane's status, coordinate and parameters.
fn harness(unit: &str, y0: &[f64], p0: &[f64], initial_flag: Option<usize>) -> String {
    let flag = initial_flag.map_or_else(String::new, |index| format!("m->p[{index}] = 1.0;"));
    let mut main = String::from(
        "int main(void) {\n    ModelInstance* m = (ModelInstance*)calloc(1, sizeof *m);\n    if (!m) return 2;\n",
    );
    for scale in GUESSES {
        main.push_str(&format!(
            "    {{ rmc_lane_reset(m); m->time = 0.0; m->parameters_dirty = false; m->rmc_nonfinite = false; m->assertion_failed = false;\n\
             \x20     for (size_t i = 0; i < Y_LEN; ++i) m->y[i] = rmc_y0[i] * {scale:?};\n\
             \x20     for (size_t i = 0; i < P_LEN; ++i) m->p[i] = rmc_p0[i];\n\
             \x20     {flag}\n\
             \x20     int ok = rmc_initialize(m, {TOLERANCE:?}) == rmcOK;\n\
             \x20     if (m->rmc_dtop != 0 || m->rmc_stop != 0) return 3;\n\
             \x20     printf(\"%d\", ok); for (size_t i = 0; i < Y_LEN; ++i) printf(\" %.17g\", m->y[i]);\n\
             \x20     printf(\" |\"); for (size_t i = 0; i < P_LEN; ++i) printf(\" %.17g\", m->p[i]); printf(\"\\n\"); }}\n"
        ));
    }
    main.push_str("    free(m);\n    return 0;\n}\n");
    format!(
        "{unit}\nstatic const double rmc_y0[Y_LEN] = {{ {} }};\nstatic const double rmc_p0[P_LEN] = {{ {} }};\n{main}",
        literals(y0),
        literals(if p0.is_empty() { &[0.0] } else { p0 }),
    )
}

type Lane = (bool, Vec<f64>, Vec<f64>);

fn run(source: &str) -> Vec<Lane> {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("initialization.cpp");
    std::fs::write(&path, source).expect("write the harness");
    let binary = directory.path().join("initialization");
    let compile = std::process::Command::new("c++")
        .args([
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            "-D__device__=",
            "-D__host__=",
        ])
        .arg(&path)
        .args(["-lm", "-o"])
        .arg(&binary)
        .output()
        .expect("start the host C++ compiler");
    assert!(
        compile.status.success(),
        "the device unit did not compile:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let output = std::process::Command::new(&binary)
        .output()
        .expect("run the lanes");
    assert!(output.status.success(), "the lanes fail: {output:?}");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            let (status, rest) = line.split_once(' ').expect("a lane line");
            let (y, p) = rest.split_once('|').expect("y and p");
            let values = |text: &str| {
                text.split_whitespace()
                    .map(|value| value.parse::<f64>().expect("a printed double"))
                    .collect::<Vec<_>>()
            };
            (status == "1", values(y), values(p))
        })
        .collect()
}

fn close(device: &[f64], runtime: &[f64]) -> bool {
    runtime
        .iter()
        .zip(device)
        .all(|(r, d)| (r - d).abs() <= 1.0e-8 * (1.0 + r.abs()))
}

#[test]
fn a_lane_initialization_certifies_the_retained_manifold_as_the_linked_kernel() {
    check("PinnedStart", PINNED_START);
    check("PinnedSeed", PINNED_SEED);
}

fn check(name: &str, source: &str) {
    let compiled = Compiler::new()
        .model(name)
        .compile_str(source, "state_manifold_initialization.mo")
        .unwrap();
    let lowered =
        lower_correlated_for_simulation_with_overrides(&compiled.dae, &SimOptions::default())
            .unwrap();
    let model = lowered.model().clone();
    assert!(
        model.problem.initialization.manifold_row_count() > 0,
        "the initialization residual closes with the retained manifold row"
    );
    let unit = SolveTemplateRenderer::new_owned_model_with_me_projection(
        Arc::new(model.clone()),
        lowered.prepared_dae(),
    )
    .expect("a lane target admits the retained manifold rows of its initialization")
    .render(DEVICE_UNIT)
    .unwrap();
    assert!(
        unit.contains("rmc_certify_state_manifold(m, m->rmc_tol)"),
        "rmc_initialize certifies the manifold"
    );
    let y0 = model.initial_y.to_vec();
    let p0 = model.parameters.to_vec();
    let flag = model.problem.solve_layout.initial_event_parameter_index;
    let lanes = run(&harness(&unit, &y0, &p0, flag));
    let runtime = SolveRuntime::new(&model).unwrap();
    assert_eq!(lanes.len(), GUESSES.len());
    for ((ok, y, p), scale) in lanes.iter().zip(GUESSES) {
        let mut ry = y0.iter().map(|value| value * scale).collect::<Vec<_>>();
        let mut rp = p0.clone();
        runtime.set_initial_event_flag(&mut rp, true);
        runtime
            .settle_initialization_system(
                &mut ry,
                &mut rp,
                0.0,
                TOLERANCE,
                ALGEBRAIC_REFRESH_MAX_ITERS,
            )
            .unwrap_or_else(|error| panic!("{name} guess {scale}: the runtime settles: {error}"));
        runtime
            .certify_state_manifold(&ry, &rp, 0.0, TOLERANCE)
            .unwrap_or_else(|error| panic!("{name} guess {scale}: the runtime certifies: {error}"));
        assert!(
            *ok,
            "{name} guess {scale}: the lane initializes and certifies"
        );
        assert!(
            close(y, &ry) && close(p, &rp),
            "{name} guess {scale}: the lane differs from the runtime\n lane    y {y:?} p {p:?}\n runtime y {ry:?} p {rp:?}"
        );
        assert!(
            (y[0].powi(3) - 1.0).abs() <= 1.0e-9,
            "{name} guess {scale}: the lane's x is on the manifold: {y:?}"
        );
    }
}
