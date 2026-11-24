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

// A row role that reads an algebraic evaluates every residual row on the
// settled algebraic refresh; rows reading no algebraic are evaluated on the
// instance coordinates (`parameter_updates::validate_projection`), with each
// one-row block's isolation of its unknown and the partial forward Jacobian.

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

/// Initial equations over state seeds alone: an affine row the row's own
/// isolation settles, a nonlinear row Newton settles, and a coupled pair.
const STATE_SEEDS: &str = "model DeviceInitSeeds
  Real x(start = 1);
  Real z(start = 2);
  Real a(start = 1);
  Real b(start = 2);
initial equation
  2 * x + 1 = 7;
  z * z * z + z = 10;
  a + b = 3;
  a - b * b = -3;
equation
  der(x) = -x;
  der(z) = -0.5 * z;
  der(a) = b;
  der(b) = -a;
end DeviceInitSeeds;";

/// A `fixed = false` parameter whose initial equation reads a fixed state
/// seed and another parameter's binding.
const PARAMETER_OVER_SEEDS: &str = "model DeviceInitParameterSeed
  parameter Real g = 2;
  parameter Real h = 3 * g;
  parameter Real k(fixed = false, start = 1);
  Real x(start = h, fixed = true);
initial equation
  k * k * k + k = x + g;
equation
  der(x) = -k * x;
end DeviceInitParameterSeed;";

/// A delayed value the initial equation reads through an algebraic: while
/// initializing, `delay(x, 0.2) = x` (MLS §3.7.2), held at its source before
/// each pass and refreshed after the projection.
const DELAYED: &str = "model DeviceInitDelay
  parameter Real g = 3;
  parameter Real k(fixed = false, start = 1);
  Real x(start = 1, fixed = true);
  Real w;
  Real d;
initial equation
  w = g + 0.5 * x;
equation
  d = delay(x, 0.2);
  w = k * k * k + k + d;
  der(x) = -k * x;
end DeviceInitDelay;";

/// A delay of an algebraic: while initializing, `delay(y, 0.1) = y`, the
/// value the continuous equations give `y` (MLS §3.7.2, §8.6), not its
/// stored seed. The initial equations give `y = 2`, so `k^3 + k = 1.25`.
const DELAYED_ALGEBRAIC: &str = "model DeviceInitDelayAlgebraic
  parameter Real g = 3.25;
  parameter Real k(fixed = false, start = 1);
  Real x(start = 0.5);
  Real y;
  Real w;
  Real d;
initial equation
  y = 2;
  w = g;
equation
  y = x + x * x * x;
  d = delay(y, 0.1);
  w = k * k * k + k + d;
  der(x) = -k * x;
end DeviceInitDelayAlgebraic;";

/// A parameter binding that passes a transposed parameter matrix to a
/// function (a `TensorTranspose` operand, as in `inv(transpose(Ti))`), read by
/// the initial equation of a `fixed = false` parameter.
const TRANSPOSED_BINDING: &str = "model DeviceInitTranspose
  function scaled
    input Real M[2, 2];
    input Real s;
    output Real y[2, 2];
  algorithm
    for i in 1:2 loop
      for j in 1:2 loop
        y[i, j] := s * M[i, j] + i;
      end for;
    end for;
  end scaled;
  parameter Real A[2, 2] = {{1, 2}, {3, 4}};
  parameter Real g = 2;
  parameter Real B[2, 2] = scaled(transpose(A), g);
  parameter Real k(fixed = false, start = 1);
  Real x(start = 1, fixed = true);
initial equation
  k * k * k + k = B[1, 2] + B[2, 1] * x;
equation
  der(x) = -k * x;
end DeviceInitTranspose;";

pub(crate) fn models() -> Vec<(rumoca::CompilationResult, &'static str)> {
    vec![
        (
            compile(TRANSPOSED_BINDING, "DeviceInitTranspose"),
            "DeviceInitTranspose",
        ),
        (compile(STATE_SEEDS, "DeviceInitSeeds"), "DeviceInitSeeds"),
        (
            compile(PARAMETER_OVER_SEEDS, "DeviceInitParameterSeed"),
            "DeviceInitParameterSeed",
        ),
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

/// Models the FMI C profile refuses (a delay is a runtime event there), so
/// only the linked runtime is compared with.
fn runtime_only_models() -> Vec<(rumoca::CompilationResult, &'static str)> {
    vec![
        (compile(DELAYED, "DeviceInitDelay"), "DeviceInitDelay"),
        (
            compile(DELAYED_ALGEBRAIC, "DeviceInitDelayAlgebraic"),
            "DeviceInitDelayAlgebraic",
        ),
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
    for (compiled, model) in models().into_iter().chain(runtime_only_models()) {
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

/// The linked runtime holds a delayed algebraic at the value the continuous
/// equations settle, so the parameter its initial equation solves for sees
/// `delay(y, 0.1) = y = 2`.
#[test]
fn a_delayed_algebraic_initializes_at_its_settled_value() {
    let compiled = compile(DELAYED_ALGEBRAIC, "DeviceInitDelayAlgebraic");
    let solve = lowered(&compiled);
    let [delayed] = solve.problem.events.delays.value_parameter_indices[..] else {
        panic!("one delay channel");
    };
    for (k, (_, p)) in runtime_lanes(&compiled).into_iter().enumerate() {
        let gain = p
            .iter()
            .copied()
            .enumerate()
            .find(|(index, value)| *index != delayed && *value != 3.25)
            .map(|(_, value)| value)
            .expect("the solved parameter");
        assert!(
            (p[delayed] - 2.0).abs() <= 1.0e-9,
            "guess {k}: the delayed value is its settled source, got {}",
            p[delayed]
        );
        assert!(
            (gain * gain * gain + gain - 1.25).abs() <= 1.0e-9,
            "guess {k}: k^3 + k + delay(y) = g, got k = {gain}"
        );
    }
}
