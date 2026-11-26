//! cuda-ode on a ladder of array cells whose node voltages form coupled
//! nonlinear algebraic blocks (`fixtures/cuda_ode/RlcLadder.mo`): each lane
//! settles them with its own instance of the shared ME projection kernel.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

const LADDER: &str = include_str!("../../fixtures/cuda_ode/RlcLadder.mo");

fn ladder_source() -> (tempfile::TempDir, String, String) {
    let compiled = Compiler::new()
        .model("RlcLadder.Ladder")
        .compile_str(LADDER, "RlcLadder.mo")
        .expect("the compiler accepts the ladder fixture");
    let (directory, files) = rendered_model_target(&compiled, "RlcLadder.Ladder", "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    (directory, source.path.clone(), source.content.clone())
}

#[test]
fn cuda_ode_ladder_settles_its_nonlinear_nodes_per_lane() {
    let (directory, path, content) = ladder_source();
    assert!(
        content.contains("rmc_project_block(") && content.contains("_workspace_init_batch("),
        "the node voltages must form projection blocks for this test to bite"
    );
    let harness = directory.path().join("ladder.cpp");
    fs::write(
        &harness,
        format!(
            r#"#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Dim3 {{ int x; }};
static Dim3 blockIdx, blockDim, threadIdx;
#define __global__
#define __host__
#define __device__
// The includer's own `snprintf` and `memcmp` survive the kernel's shims.
#define snprintf snprintf
#define memcmp memcmp
#include {path:?}
#if !defined(snprintf) || !defined(memcmp)
#error "the kernel's shims dropped the includer's snprintf or memcmp"
#endif

static int slot(const char *name) {{
    for (int i = 0; i < RLCLADDER_LADDER_Y_LEN; ++i) {{
        if (std::strcmp(RlcLadder_Ladder_y_names[i], name) == 0) {{
            return i;
        }}
    }}
    return -1;
}}

int main() {{
    double y[RLCLADDER_LADDER_Y_LEN] = {{0.0}};
    double p[RLCLADDER_LADDER_P_LEN] = {{0.0}};
    double out[RLCLADDER_LADDER_STATE_COUNT] = {{0.0}};
    int status = -1;
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    void *lanes = std::malloc(RLCLADDER_LADDER_LANE_WORKSPACE_BYTES);
    RlcLadder_Ladder_workspace_init_batch(lanes, 1);
    RlcLadder_Ladder_load_starts_batch(y, p, RLCLADDER_LADDER_Y_LEN, RLCLADDER_LADDER_P_LEN, 1);
    RlcLadder_Ladder_initialize_batch(0.0, y, p, &status, RLCLADDER_LADDER_Y_LEN, RLCLADDER_LADDER_P_LEN, 1);
    if (status != 0) {{
        std::printf("initialize status %d\n", status);
        return 1;
    }}
    // Charge the inductors so the nodes carry a large, nonlinear load.
    char name[32];
    for (int i = 1; i <= 10; ++i) {{
        std::snprintf(name, sizeof name, "iL[%d]", i);
        y[slot(name)] = 100.0 / i;
    }}
    RlcLadder_Ladder_rhs_batch(1e-3, y, p, out, &status, RLCLADDER_LADDER_Y_LEN,
        RLCLADDER_LADDER_P_LEN, RLCLADDER_LADDER_STATE_COUNT, 1);
    if (status != 0) {{
        std::printf("rhs status %d\n", status);
        return 1;
    }}
    // Kirchhoff's current law at every node, with the fixture's parameters.
    const double G = 1e-3, k = 1e-6, Rm = 10.0;
    for (int i = 1; i <= 10; ++i) {{
        std::snprintf(name, sizeof name, "v[%d]", i);
        const double v = y[slot(name)];
        std::snprintf(name, sizeof name, "vC[%d]", i);
        const double vc = y[slot(name)];
        std::snprintf(name, sizeof name, "iL[%d]", i);
        double through = y[slot(name)];
        if (i < 10) {{
            std::snprintf(name, sizeof name, "iL[%d]", i + 1);
            through -= y[slot(name)];
        }}
        const double residual = through - ((v - vc) / Rm + G * v + k * v * v * v);
        if (!(std::fabs(residual) <= 1e-9 * (1.0 + std::fabs(through)))) {{
            std::printf("node %d: v %.17g, KCL residual %.3g\n", i, v, residual);
            return 1;
        }}
    }}
    std::free(lanes);
    return 0;
}}
"#
        ),
    )
    .expect("write ladder harness");
    let executable = directory.path().join("cuda-ode-ladder");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE ladder emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE ladder emulation",
    );
}

#[test]
fn cuda_ode_torn_block_settles_from_any_tear_guess() {
    // An affine torn block: the lane's coordinate carries whatever the caller
    // left in the tear slot, and the shared kernel must not depend on it.
    let compiled = Compiler::new()
        .model("TornIsolate")
        .compile_str(
            r#"
model TornIsolate
  Real x(start = 0, fixed = true);
  Real a;
  Real b;
  Real c;
  Real d;
equation
  der(x) = -x + a;
  c = a + b;
  d = a - b;
  c + d = 4 + b;
  c - 2 * d = x;
end TornIsolate;
"#,
            "TornIsolate.mo",
        )
        .expect("the compiler accepts the torn model");
    let (directory, files) = rendered_model_target(&compiled, "TornIsolate", "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    assert!(
        source.content.contains("rmc_project_block("),
        "the model must keep a projection block for this test to bite"
    );
    let harness = directory.path().join("torn.cpp");
    fs::write(
        &harness,
        format!(
            r#"#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Dim3 {{ int x; }};
static Dim3 blockIdx, blockDim, threadIdx;
#define __global__
#define __host__
#define __device__
#include {:?}

static int slot(const char *name) {{
    for (int i = 0; i < TORNISOLATE_Y_LEN; ++i) {{
        if (std::strcmp(TornIsolate_y_names[i], name) == 0) {{
            return i;
        }}
    }}
    return -1;
}}

int main() {{
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    void *lanes = std::malloc(TORNISOLATE_LANE_WORKSPACE_BYTES);
    TornIsolate_workspace_init_batch(lanes, 1);
    const double guesses[5] = {{1.0, 1e6, 1e9, 1e12, 1e15}};
    for (double guess : guesses) {{
        double y[TORNISOLATE_Y_LEN] = {{0.0}};
        double out[TORNISOLATE_STATE_COUNT] = {{0.0}};
        int status = -1;
        y[slot("a")] = y[slot("b")] = y[slot("c")] = y[slot("d")] = guess;
        TornIsolate_rhs_batch(0.0, y, nullptr, out, &status, TORNISOLATE_Y_LEN, 0,
            TORNISOLATE_STATE_COUNT, 1);
        // With x = 0: a = 2.4, b = 0.8, and der(x) = a.
        if (status != 0 || std::fabs(out[0] - 2.4) > 1e-12
            || std::fabs(y[slot("b")] - 0.8) > 1e-12) {{
            std::printf("guess %g: status %d, der %.17g\n", guess, status, out[0]);
            return 1;
        }}
    }}
    std::free(lanes);
    return 0;
}}
"#,
            source.path,
        ),
    )
    .expect("write torn harness");
    let executable = directory.path().join("cuda-ode-torn");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE torn-block emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE torn-block emulation",
    );
}
