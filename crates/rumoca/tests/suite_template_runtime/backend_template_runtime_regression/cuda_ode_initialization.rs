//! cuda-ode initializes each lane through the shared ME projection kernel
//! (`rmc_initialize`, MLS §8.6): initial equations solved for a state and a
//! `fixed = false` parameter, from each lane's own parameter values, through
//! the settled algebraic refresh or over the state seeds themselves, with a
//! delayed value equal to its source while initializing (MLS §3.7.2).

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

const LANE_INIT: &str = r#"
model LaneInit
  parameter Real g = 3.25;
  parameter Real k(fixed = false, start = 1);
  Real x(start = 0.5);
  Real y;
  Real w;
initial equation
  y = 2;
  w = g;
equation
  y = x + x * x * x;
  w = k * k * k + k;
  der(x) = -k * x;
end LaneInit;
"#;

/// The same initialization through a delay: `delay(2 * x, 0.1) = 2 * x`
/// while initializing, so the second initial equation solves
/// `k^3 + k + 2 x = g` for `k`.
const LANE_INIT_DELAY: &str = r#"
model LaneInitDelay
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
  d = delay(2 * x, 0.1);
  w = k * k * k + k + d;
  der(x) = -k * x;
end LaneInitDelay;
"#;

/// CPU emulation of two lanes of model `LaneInit`, included after the
/// generated kernel: lane 1 overrides `g` before initializing, and each lane
/// checks `k^3 + k + DELAYED * x = g`.
const HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < LANEINIT_Y_LEN; ++i) {
        if (std::strcmp(LaneInit_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

static void lane(int index) {
    blockIdx.x = 0;
    blockDim.x = 2;
    threadIdx.x = index;
}

int main() {
    const int x = slot("x");
    if (x < 0) { std::printf("no state x\n"); return 1; }
    double y[2 * LANEINIT_Y_LEN] = {0.0};
    double p[2 * LANEINIT_P_LEN] = {0.0};
    double out[2 * LANEINIT_STATE_COUNT] = {0.0};
    int status[2] = {-1, -1};
    void *lanes = std::malloc(2 * LANEINIT_LANE_WORKSPACE_BYTES);
    for (int k = 0; k < 2; ++k) { lane(k); LaneInit_workspace_init_batch(lanes, 2); }
    for (int k = 0; k < 2; ++k) { lane(k); LaneInit_load_starts_batch(y, p, LANEINIT_Y_LEN, LANEINIT_P_LEN, 2); }
    int g = -1;
    for (int i = 0; i < LANEINIT_P_LEN; ++i) if (p[i] == 3.25) g = i;
    if (g < 0) { std::printf("no parameter g\n"); return 1; }
    p[LANEINIT_P_LEN + g] = 10.0;
    for (int k = 0; k < 2; ++k) { lane(k); LaneInit_initialize_batch(0.0, y, p, status, LANEINIT_Y_LEN, LANEINIT_P_LEN, 2); }
    for (int k = 0; k < 2; ++k) { lane(k); LaneInit_rhs_batch(0.0, y, p, out, status, LANEINIT_Y_LEN, LANEINIT_P_LEN, LANEINIT_STATE_COUNT, 2); }
    const double gains[2] = {3.25, 10.0};
    for (int k = 0; k < 2; ++k) {
        const double xk = y[k * LANEINIT_Y_LEN + x];
        // der(x) = -k * x, so the solved parameter is -der(x) / x.
        const double kk = -out[k * LANEINIT_STATE_COUNT] / xk;
        if (status[k] != 0 || std::fabs(xk + xk * xk * xk - 2.0) > 1e-9
            || std::fabs(kk * kk * kk + kk + DELAYED * xk - gains[k]) > 1e-9 * gains[k]) {
            std::printf("lane %d: status %d, x %.17g, k %.17g\n", k, status[k], xk, kk);
            return 1;
        }
    }
    std::free(lanes);
    return 0;
}
"#;

/// Render `model` for cuda-ode, check that it initializes through the shared
/// kernel, and run `harness` (included after the kernel) in CPU emulation.
fn run_lane_harness(model_source: &str, model: &str, harness: &str) {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(model_source, &format!("{model}.mo"))
        .expect("the compiler accepts the model");
    let (directory, files) = rendered_model_target(&compiled, model, "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    let call = format!("rmc_initialize(m, {}_INIT_TOLERANCE)", model.to_uppercase());
    assert!(
        source.content.contains(&call),
        "the lanes must initialize through the shared kernel"
    );
    assert_eq!(
        source.content.contains("rmc_init_delays(m)"),
        model_source.contains("delay("),
        "the initialization refreshes delayed values exactly when the model has a delay"
    );
    let path = directory.path().join("lane_init.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n";
    fs::write(
        &path,
        format!("{prelude}#include {:?}\n{harness}", source.path),
    )
    .expect("write the initialization harness");
    let executable = directory.path().join("cuda-ode-lane-init");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&path)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE initialization emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE initialization emulation",
    );
}

#[test]
fn cuda_ode_lanes_solve_their_initial_equations_with_the_shared_kernel() {
    run_lane_harness(LANE_INIT, "LaneInit", &HARNESS.replace("DELAYED", "0.0"));
}

#[test]
fn cuda_ode_lanes_initialize_a_delay_at_its_source() {
    let harness = HARNESS
        .replace("LaneInit_", "LaneInitDelay_")
        .replace("LANEINIT_", "LANEINITDELAY_")
        .replace("DELAYED", "2.0");
    run_lane_harness(LANE_INIT_DELAY, "LaneInitDelay", &harness);
}

/// Initial equations over state seeds alone: the affine row settles through
/// its isolation, the cubic one by Newton on the partial Jacobian.
const SEED_INIT: &str = r#"
model SeedInit
  parameter Real c = 10;
  Real x(start = 1);
  Real z(start = 2);
initial equation
  2 * x + 1 = c;
  z * z * z + z = c;
equation
  der(x) = -x;
  der(z) = -z;
end SeedInit;
"#;

/// Two lanes; lane 1 overrides `c` before initializing.
const SEED_HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < SEEDINIT_Y_LEN; ++i) {
        if (std::strcmp(SeedInit_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

static void lane(int index) {
    blockIdx.x = 0;
    blockDim.x = 2;
    threadIdx.x = index;
}

int main() {
    const int x = slot("x"), z = slot("z");
    if (x < 0 || z < 0) { std::printf("no states x, z\n"); return 1; }
    double y[2 * SEEDINIT_Y_LEN] = {0.0};
    double p[2 * SEEDINIT_P_LEN] = {0.0};
    int status[2] = {-1, -1};
    void *lanes = std::malloc(2 * SEEDINIT_LANE_WORKSPACE_BYTES);
    for (int k = 0; k < 2; ++k) { lane(k); SeedInit_workspace_init_batch(lanes, 2); }
    for (int k = 0; k < 2; ++k) { lane(k); SeedInit_load_starts_batch(y, p, SEEDINIT_Y_LEN, SEEDINIT_P_LEN, 2); }
    int c = -1;
    for (int i = 0; i < SEEDINIT_P_LEN; ++i) if (p[i] == 10.0) c = i;
    if (c < 0) { std::printf("no parameter c\n"); return 1; }
    p[SEEDINIT_P_LEN + c] = 30.0;
    for (int k = 0; k < 2; ++k) { lane(k); SeedInit_initialize_batch(0.0, y, p, status, SEEDINIT_Y_LEN, SEEDINIT_P_LEN, 2); }
    const double values[2] = {10.0, 30.0};
    for (int k = 0; k < 2; ++k) {
        const double xk = y[k * SEEDINIT_Y_LEN + x], zk = y[k * SEEDINIT_Y_LEN + z];
        if (status[k] != 0 || std::fabs(2.0 * xk + 1.0 - values[k]) > 1e-9 * values[k]
            || std::fabs(zk * zk * zk + zk - values[k]) > 1e-9 * values[k]) {
            std::printf("lane %d: status %d, x %.17g, z %.17g\n", k, status[k], xk, zk);
            return 1;
        }
    }
    std::free(lanes);
    return 0;
}
"#;

#[test]
fn cuda_ode_lanes_solve_initial_equations_over_state_seeds() {
    run_lane_harness(SEED_INIT, "SeedInit", SEED_HARNESS);
}
