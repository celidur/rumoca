//! The ME projection kernel built for a device initializes each lane as the
//! linked runtime does (SPEC_0044 ME-PROJ-001, MLS §8.6).
//!
//! A lane target initializes its lanes through the kernel's
//! `rmc_initialize`: the parameter bindings (`binding_updates` over the
//! model's `SolveBindingExecution`) and the initialization projection
//! alternate until the bindings stop changing. Each test model's device unit
//! (`crate::me_projection_device`), built by a host C++ compiler with the CUDA
//! qualifiers defined away, initializes one lane per start guess; the linked
//! runtime's `settle_initialization_system` settles the same guesses. Every
//! lane must succeed and reach the runtime's coordinate and parameters.

use std::fs;
use std::process::Command;

use rumoca_solver::runtime::solve_runtime::SolveRuntime;

use crate::me_projection_device::device_unit;

/// The start guesses each comparison initializes from: the model's own, then
/// every solver coordinate scaled.
pub(crate) const GUESSES: usize = 3;

/// The tolerance both initializations run at.
pub(crate) const TOLERANCE: f64 = 1.0e-10;

fn compile(source: &str, model: &str) -> rumoca::CompilationResult {
    rumoca::Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("compile {model}: {error:#}"))
}

// The C initialization evaluates every residual row on the settled algebraic
// refresh (`parameter_updates::validate_projection`): each initial equation
// below constrains an algebraic, which the refresh computes from the unknown.

/// A nonlinear initial equation that settles a state through an algebraic.
const STATE_THROUGH_ALGEBRAIC: &str = "model DeviceInitState
  Real x(start = 1);
  Real y;
initial equation
  y = 6;
equation
  y = x * x + x;
  der(x) = -x + 0.1 * y;
end DeviceInitState;";

/// A parameter the initialization solves for (`fixed = false`), beside a
/// binding of another parameter that a state start reads.
const FREE_PARAMETER: &str = "model DeviceInitParameter
  parameter Real g = 3;
  parameter Real k(fixed = false, start = 1);
  parameter Real h = 2 * g;
  Real x(start = h, fixed = true);
  Real w;
initial equation
  w = g + 0.5 * x;
equation
  w = k * k * k + k;
  der(x) = -k * x;
end DeviceInitParameter;";

/// A state settled through a torn algebraic loop: every residual evaluation
/// settles the loop by the kernel's projection.
const TORN_LOOP: &str = "model DeviceInitTorn
  parameter Real u = 1.5;
  Real x(start = 1);
  Real a(start = 1);
  Real c(start = 1);
initial equation
  a = u;
equation
  der(x) = u - a;
  a = x + 0.2 * sin(c);
  c = 1 + 0.1 * a * a - 0.3 * cos(c);
end DeviceInitTorn;";

pub(crate) fn models() -> Vec<(rumoca::CompilationResult, &'static str)> {
    vec![
        (
            compile(STATE_THROUGH_ALGEBRAIC, "DeviceInitState"),
            "DeviceInitState",
        ),
        (
            compile(FREE_PARAMETER, "DeviceInitParameter"),
            "DeviceInitParameter",
        ),
        (compile(TORN_LOOP, "DeviceInitTorn"), "DeviceInitTorn"),
    ]
}

fn literals(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The device unit with one lane per guess: each lane starts from the model's
/// start values, scales its solver coordinates for guess `k` and raises the
/// initial event flag, then runs `rmc_initialize`. The host build runs the
/// lanes in turn; a CUDA build launches them on the device. Both print each
/// lane's status, coordinate and parameters.
pub(crate) fn initialization_harness(
    unit: &str,
    y0: &[f64],
    p0: &[f64],
    initial_flag: Option<usize>,
) -> String {
    let y = literals(y0);
    let p = literals(p0);
    let flag = initial_flag.map_or_else(String::new, |index| format!("m->p[{index}] = 1.0;"));
    format!(
        r#"#include <stdio.h>
#include <stdlib.h>
{unit}
static __device__ const double rmc_y0[Y_LEN] = {{ {y} }};
static __device__ const double rmc_p0[P_LEN] = {{ {p} }};
static __device__ int rmc_guess(ModelInstance* m, int k) {{
    rmc_lane_reset(m); m->time = 0.0; m->parameters_dirty = false; m->rmc_nonfinite = false; m->assertion_failed = false;
    for (size_t i = 0; i < Y_LEN; ++i) m->y[i] = rmc_y0[i] * (1.0 + 0.1 * k);
    for (size_t i = 0; i < P_LEN; ++i) m->p[i] = rmc_p0[i];
    {flag}
    return rmc_initialize(m, {TOLERANCE:?}) == rmcOK;
}}
static void rmc_print(const ModelInstance* m, int ok) {{
    printf("%d", ok);
    for (size_t i = 0; i < Y_LEN; ++i) printf(" %.17g", m->y[i]);
    printf(" |");
    for (size_t i = 0; i < P_LEN; ++i) printf(" %.17g", m->p[i]);
    printf("\n");
}}
#if defined(__CUDACC__)
__global__ void rmc_guesses(ModelInstance* lanes, int* ok) {{
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < {GUESSES}) ok[lane] = rmc_guess(&lanes[lane], lane);
}}
int main(void) {{
    ModelInstance* lanes = NULL; int* ok = NULL;
    if (cudaDeviceSetLimit(cudaLimitStackSize, 65536) != cudaSuccess) return 2;
    if (cudaMalloc((void**)&lanes, {GUESSES} * sizeof(ModelInstance)) != cudaSuccess) return 2;
    if (cudaMalloc((void**)&ok, {GUESSES} * sizeof(int)) != cudaSuccess) return 2;
    rmc_guesses<<<1, {GUESSES}>>>(lanes, ok);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    ModelInstance* host = (ModelInstance*)malloc({GUESSES} * sizeof(ModelInstance)); int status[{GUESSES}];
    if (!host) return 2;
    cudaMemcpy(host, lanes, {GUESSES} * sizeof(ModelInstance), cudaMemcpyDeviceToHost);
    cudaMemcpy(status, ok, sizeof status, cudaMemcpyDeviceToHost);
    for (int k = 0; k < {GUESSES}; ++k) rmc_print(&host[k], status[k]);
    free(host); cudaFree(lanes); cudaFree(ok);
    return 0;
}}
#else
int main(void) {{
    ModelInstance* m = (ModelInstance*)calloc(1, sizeof *m);
    if (!m) return 2;
    for (int k = 0; k < {GUESSES}; ++k) {{ int ok = rmc_guess(m, k); rmc_print(m, ok); }}
    free(m);
    return 0;
}}
#endif
"#
    )
}

/// The model the device unit is rendered from, its start values and its
/// initial event flag.
pub(crate) fn lowered(compiled: &rumoca::CompilationResult) -> rumoca_ir_solve::SolveModel {
    rumoca_sim::lower_fmi_c_export_component(&compiled.dae)
        .expect("lower the FMI C component")
        .runtime_view()
        .model()
        .clone()
}

/// The device lanes' (status, y, p), one per guess.
fn device_lanes(
    compiled: &rumoca::CompilationResult,
    model: &str,
) -> Vec<(bool, Vec<f64>, Vec<f64>)> {
    let solve = lowered(compiled);
    let work = tempfile::tempdir().expect("device initialization work directory");
    let harness = work.path().join(format!("{model}.cpp"));
    fs::write(
        &harness,
        initialization_harness(
            &device_unit(compiled),
            &solve.initial_y.to_vec(),
            &solve.parameters.to_vec(),
            solve.problem.solve_layout.initial_event_parameter_index,
        ),
    )
    .expect("write the device harness");
    let binary = work.path().join(model);
    let output = Command::new("c++")
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
        .arg(&binary)
        .output()
        .expect("start the host C++ compiler");
    assert!(
        output.status.success(),
        "{model}: compile the device unit as host C++:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(&binary)
        .output()
        .expect("run the device lanes");
    assert!(output.status.success(), "{model}: the device lanes fail");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            let (status, rest) = line.split_once(' ').expect("a lane line");
            let (y, p) = rest.split_once('|').expect("a lane line holds y and p");
            let values = |text: &str| {
                text.split_whitespace()
                    .map(|value| value.parse::<f64>().expect("a printed double"))
                    .collect::<Vec<_>>()
            };
            (status == "1", values(y), values(p))
        })
        .collect()
}

/// The linked runtime's settled (y, p) from the same guesses.
fn runtime_lanes(compiled: &rumoca::CompilationResult) -> Vec<(Vec<f64>, Vec<f64>)> {
    let solve = lowered(compiled);
    let runtime = SolveRuntime::new(&solve).expect("build the linked runtime");
    let max_iters = rumoca_eval_solve::projection_policy::ALGEBRAIC_REFRESH_MAX_ITERS;
    (0..GUESSES)
        .map(|k| {
            let mut y: Vec<f64> = solve
                .initial_y
                .to_vec()
                .iter()
                .map(|value| value * (1.0 + 0.1 * k as f64))
                .collect();
            let mut p = solve.parameters.to_vec();
            runtime.set_initial_event_flag(&mut p, true);
            runtime
                .settle_initialization_system(&mut y, &mut p, 0.0, TOLERANCE, max_iters)
                .unwrap_or_else(|error| panic!("guess {k}: the runtime settles: {error}"));
            (y, p)
        })
        .collect()
}

fn close(device: &[f64], runtime: &[f64]) -> bool {
    device.len() >= runtime.len()
        && runtime
            .iter()
            .zip(device)
            .all(|(r, d)| (r - d).abs() <= 1.0e-8 * (1.0 + r.abs()))
}

#[test]
fn a_device_build_initializes_every_lane_as_the_linked_runtime() {
    for (compiled, model) in models() {
        let device = device_lanes(&compiled, model);
        let runtime = runtime_lanes(&compiled);
        assert_eq!(device.len(), GUESSES, "{model}: one line per lane");
        for (k, ((ok, y, p), (ry, rp))) in device.iter().zip(&runtime).enumerate() {
            assert!(*ok, "{model}: lane {k} initializes");
            assert!(
                close(y, ry) && close(p, rp),
                "{model}: lane {k} differs from the runtime\n device  y {y:?} p {p:?}\n runtime y {ry:?} p {rp:?}"
            );
        }
    }
}
