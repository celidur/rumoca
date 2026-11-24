//! cuda-ode initializes each lane through the shared ME projection kernel
//! (`rmc_initialize`, MLS §8.6): initial equations solved for a state and a
//! `fixed = false` parameter, from each lane's own parameter values.

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

/// CPU emulation of two lanes, included after the generated kernel: lane 1
/// overrides `g` before initializing.
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
            || std::fabs(kk * kk * kk + kk - gains[k]) > 1e-9 * gains[k]) {
            std::printf("lane %d: status %d, x %.17g, k %.17g\n", k, status[k], xk, kk);
            return 1;
        }
    }
    std::free(lanes);
    return 0;
}
"#;

#[test]
fn cuda_ode_lanes_solve_their_initial_equations_with_the_shared_kernel() {
    let compiled = Compiler::new()
        .model("LaneInit")
        .compile_str(LANE_INIT, "LaneInit.mo")
        .expect("the compiler accepts the model");
    let (directory, files) = rendered_model_target(&compiled, "LaneInit", "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    assert!(
        source
            .content
            .contains("rmc_initialize(m, LANEINIT_INIT_TOLERANCE)"),
        "the lanes must initialize through the shared kernel"
    );
    let harness = directory.path().join("lane_init.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n";
    fs::write(
        &harness,
        format!("{prelude}#include {:?}\n{HARNESS}", source.path),
    )
    .expect("write the initialization harness");
    let executable = directory.path().join("cuda-ode-lane-init");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
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
