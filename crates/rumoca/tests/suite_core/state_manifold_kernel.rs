//! The shared C kernel projects onto the retained state manifold as the
//! linked kernel does.
//!
//! Index reduction retains `q1^2 + q2^2 + q3^2 = 1` and `q1 + q2 + q3 = 1` of
//! `Sphere` as one manifold block of two rows over three states, whose
//! correction is the minimum-norm scaled Newton step, and `x^3 = 1` of
//! `Pinned` as a square block. The test renders `manifold_kernel` of the
//! shared kernel from `solve_blocks.state_manifold` alone (no FMI component),
//! compiles it with a minimal instance, and compares every projection and
//! certification with `SolveRuntime::project_state_manifold` and
//! `SolveRuntime::certify_state_manifold` on the same points. It does so twice:
//! as C, and in the kernel's device mode (`device=true`, every function
//! `static __device__`) as host C++ with the CUDA qualifiers defined away.

use std::sync::Arc;

use rumoca::Compiler;
use rumoca_eval_solve::projection_policy::ALGEBRAIC_REFRESH_TOLERANCE as TOL;
use rumoca_phase_codegen::SolveTemplateRenderer;
use rumoca_sim::{SimOptions, lower_correlated_for_simulation_with_overrides};
use rumoca_solver::SolveRuntime;

const SPHERE: &str = r#"
model Sphere
  Real q1(start=1,fixed=true);
  Real q2(start=0,fixed=true);
  Real q3(start=0,fixed=true);
  Real a;
  Real b;
equation
  der(q1) = -q2 + a*q1 + b;
  der(q2) = q1 - q3 + a*q2 + b;
  der(q3) = q2 + a*q3 + b;
  q1*q1 + q2*q2 + q3*q3 = 1;
  q1 + q2 + q3 = 1;
end Sphere;
"#;

const PINNED: &str = r#"
model Pinned
  Real x(start=1,fixed=true);
  Real z(start=0,fixed=true);
  Real lambda;
equation
  der(x) = lambda*x;
  der(z) = 1 - z;
  x*x*x = 1;
end Pinned;
"#;

/// A minimal instance around the shared numerics and the manifold kernel,
/// `DEVICE` naming the kernel's mode.
const KERNEL: &str = r#"
{%- from "fmi-me-projection.jinja" import projection_numerics, manifold_kernel %}
{%- set mf = solve_blocks.state_manifold %}
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RMC_COLD
#define RMC_API
#define Y_LEN {{ y_len }}
#define P_LEN {{ p_len }}
typedef struct {
    double time; double y[Y_LEN]; double p[P_LEN > 0 ? P_LEN : 1];
    double rmc_d[{{ mf.work.doubles }}]; size_t rmc_s[{{ mf.work.sizes }}];
    size_t rmc_dtop; size_t rmc_stop; double rmc_tol;
} ModelInstance;
#define RMC_LOG(m, message) ((void)(m), fprintf(stderr, "%s\n", (message)))
static inline double rmc_now(const ModelInstance* m) { return m->time; }
{{ projection_numerics(mf.work, device=DEVICE) }}
{{ manifold_kernel(mf, device=DEVICE) }}
"#;

fn decimal(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Each case projects a copy of the point, prints its status, whether a state
/// moved, and the states, then certifies the original point.
fn harness(kernel: &str, points: &[Vec<f64>], p: &[f64], t: f64, states: usize) -> String {
    let mut main = String::from(
        "int main(void) {\n    static ModelInstance instance; ModelInstance* m = &instance;\n",
    );
    for point in points {
        main.push_str(&format!(
            "    {{ static const double y0[Y_LEN] = {{ {} }}; static const double p0[] = {{ {} }};\n\
             \x20     memcpy(m->y, y0, sizeof y0); memcpy(m->p, p0, sizeof m->p); m->time = {t:.17e};\n\
             \x20     m->rmc_dtop = 0; m->rmc_stop = 0; m->rmc_tol = 0.0; bool changed = false;\n\
             \x20     int status = rmc_project_state_manifold(m, {TOL:.17e}, &changed);\n\
             \x20     if (m->rmc_dtop != 0 || m->rmc_stop != 0) return 2;\n\
             \x20     printf(\"%d %d\", status, changed ? 1 : 0);\n\
             \x20     for (size_t i = 0; i < {states}; ++i) printf(\" %.17e\", m->y[i]);\n\
             \x20     memcpy(m->y, y0, sizeof y0);\n\
             \x20     printf(\" %d\\n\", rmc_certify_state_manifold(m, {TOL:.17e})); }}\n",
            decimal(point),
            decimal(if p.is_empty() { &[0.0] } else { p }),
        ));
    }
    main.push_str("    return 0;\n}\n");
    format!("{kernel}\n{main}")
}

/// The kernel of `renderer`'s model for `build`.
fn render(renderer: &SolveTemplateRenderer, build: Build, y_len: usize, p_len: usize) -> String {
    let device = match build {
        Build::C => "false",
        Build::DeviceAsHostCxx => "true",
    };
    renderer
        .render(
            &KERNEL
                .replace("{{ y_len }}", &y_len.to_string())
                .replace("{{ p_len }}", &p_len.to_string())
                .replace("DEVICE", device),
        )
        .unwrap()
}

struct Outcome {
    projected: bool,
    changed: bool,
    states: Vec<f64>,
    certified: bool,
}

/// How a harness is built: the shared kernel as C, or the device unit as host
/// C++ with the CUDA qualifiers defined away.
#[derive(Clone, Copy)]
enum Build {
    C,
    DeviceAsHostCxx,
}

fn run_c(source: &str, build: Build) -> Vec<Outcome> {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let (file, compiler, standard) = match build {
        Build::C => ("manifold.c", "cc", "-std=c11"),
        Build::DeviceAsHostCxx => ("manifold.cpp", "c++", "-std=c++17"),
    };
    let path = directory.path().join(file);
    std::fs::write(&path, source).expect("write the harness");
    let binary = directory.path().join("manifold");
    let mut command = std::process::Command::new(compiler);
    command.args([
        standard,
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-function",
    ]);
    match build {
        Build::C => command.arg("-Wvla"),
        Build::DeviceAsHostCxx => command.args(["-D__device__=", "-D__host__="]),
    };
    let compile = command
        .arg(&path)
        .args(["-lm", "-o"])
        .arg(&binary)
        .output()
        .expect("start the C compiler");
    assert!(
        compile.status.success(),
        "the manifold kernel did not compile:\n{}\n{source}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = std::process::Command::new(&binary)
        .output()
        .expect("run the harness");
    assert!(
        run.status.success(),
        "the manifold harness failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout)
        .expect("decimal output")
        .lines()
        .map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            let (last, states) = fields[2..].split_last().expect("a certification status");
            Outcome {
                projected: fields[0] == "0",
                changed: fields[1] == "1",
                states: states
                    .iter()
                    .map(|value| value.parse().expect("a decimal state"))
                    .collect(),
                certified: *last == "0",
            }
        })
        .collect()
}

struct Fixture {
    name: &'static str,
    source: &'static str,
    /// (rows, states) of its one manifold block.
    shape: (usize, usize),
    /// State offsets from the start point: on the manifold, near it, far
    /// from it, and onto a point where its rows lose rank.
    offsets: [&'static [f64]; 4],
}

fn check(fixture: &Fixture) {
    let compiled = Compiler::new()
        .model(fixture.name)
        .compile_str(fixture.source, "state_manifold_kernel.mo")
        .unwrap();
    let lowered =
        lower_correlated_for_simulation_with_overrides(&compiled.dae, &SimOptions::default())
            .unwrap();
    let model = lowered.model().clone();
    let blocks = model
        .problem
        .continuous
        .manifold_projection_plan
        .blocks
        .scalar_blocks();
    assert_eq!(
        blocks
            .iter()
            .map(|block| (block.rows.len(), block.y_indices.len()))
            .collect::<Vec<_>>(),
        [fixture.shape],
        "{}: index reduction retains one manifold block",
        fixture.name
    );
    let states = model.state_scalar_count();
    let renderer = SolveTemplateRenderer::new_owned_model_with_dae(
        Arc::new(model.clone()),
        lowered.prepared_dae(),
    )
    .unwrap();
    let start = model.initial_y.to_vec();
    let p = model.parameters.to_vec();
    let points = fixture
        .offsets
        .iter()
        .map(|offset| {
            let mut point = start.clone();
            for (state, delta) in point.iter_mut().zip(offset.iter()) {
                *state += delta;
            }
            point
        })
        .collect::<Vec<_>>();
    let t = 0.25;
    for build in [Build::C, Build::DeviceAsHostCxx] {
        let unit = render(&renderer, build, start.len(), p.len());
        let generated = run_c(&harness(&unit, &points, &p, t, states), build);
        compare(fixture, &model, &points, &p, t, &generated);
    }
}

/// Every projection and certification of `generated` agrees with the linked
/// kernel at the same points.
fn compare(
    fixture: &Fixture,
    model: &rumoca_ir_solve::SolveModel,
    points: &[Vec<f64>],
    p: &[f64],
    t: f64,
    generated: &[Outcome],
) {
    let runtime = SolveRuntime::new(model).unwrap();
    assert_eq!(generated.len(), points.len());
    for (point, c) in points.iter().zip(generated) {
        let mut linked = point.clone();
        let projected = runtime.project_state_manifold(&mut linked, p, t, TOL);
        let certified = runtime.certify_state_manifold(point, p, t, TOL).is_ok();
        let name = fixture.name;
        assert_eq!(
            c.projected,
            projected.is_ok(),
            "{name}: projection status at {point:?}: linked {projected:?}"
        );
        assert_eq!(c.certified, certified, "{name}: certification at {point:?}");
        if let Ok(changed) = projected {
            assert_eq!(c.changed, changed, "{name}: moved states at {point:?}");
        }
        for (index, (&c_value, &linked_value)) in c.states.iter().zip(&linked).enumerate() {
            assert!(
                (c_value - linked_value).abs() <= 1e-12 * (1.0 + linked_value.abs()),
                "{name}: state {index} at {point:?}: generated {c_value:e}, linked {linked_value:e}"
            );
        }
    }
    let [on, near, far, _] = generated else {
        unreachable!("four points")
    };
    assert!(
        on.projected && !on.changed && on.certified,
        "{}: on the manifold",
        fixture.name
    );
    assert!(
        near.projected && near.changed && !near.certified && far.projected && far.changed,
        "{}: a point off the manifold moves onto it",
        fixture.name
    );
}

#[test]
fn shared_kernel_projects_onto_the_retained_manifold_as_the_linked_kernel_does() {
    check(&Fixture {
        name: "Sphere",
        source: SPHERE,
        shape: (2, 3),
        offsets: [
            &[0.0, 0.0, 0.0],
            &[0.05, 0.03, -0.02],
            &[1.0, -1.0, 0.5],
            &[-0.7, 0.3, 0.3],
        ],
    });
    check(&Fixture {
        name: "Pinned",
        source: PINNED,
        shape: (1, 1),
        offsets: [&[0.0], &[0.2], &[2.0], &[-1.0]],
    });
}

/// At a point where the projection cannot move, it fails in both kernels and
/// leaves the states where they were.
#[test]
fn a_failed_projection_restores_the_states() {
    let compiled = Compiler::new()
        .model("Pinned")
        .compile_str(PINNED, "state_manifold_kernel.mo")
        .unwrap();
    let lowered =
        lower_correlated_for_simulation_with_overrides(&compiled.dae, &SimOptions::default())
            .unwrap();
    let model = lowered.model().clone();
    let renderer = SolveTemplateRenderer::new_owned_model_with_dae(
        Arc::new(model.clone()),
        lowered.prepared_dae(),
    )
    .unwrap();
    let mut point = model.initial_y.to_vec();
    point[0] = 0.0;
    let p = model.parameters.to_vec();
    let states = model.state_scalar_count();
    let runtime = SolveRuntime::new(&model).unwrap();
    let mut linked = point.clone();
    assert!(
        runtime
            .project_state_manifold(&mut linked, &p, 0.0, TOL)
            .is_err()
    );
    assert_eq!(linked, point, "the linked kernel restores its states");
    for build in [Build::C, Build::DeviceAsHostCxx] {
        let unit = render(&renderer, build, point.len(), p.len());
        let generated = run_c(
            &harness(&unit, std::slice::from_ref(&point), &p, 0.0, states),
            build,
        );
        assert!(!generated[0].projected && !generated[0].certified);
        assert_eq!(
            generated[0].states,
            point[..states],
            "the generated kernel restores its states"
        );
    }
}
