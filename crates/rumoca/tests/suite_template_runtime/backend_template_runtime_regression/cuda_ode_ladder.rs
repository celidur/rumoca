//! cuda-ode on a ladder of array cells whose node voltages form coupled
//! nonlinear algebraic blocks (`fixtures/cuda_ode/RlcLadder.mo`).

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
        content.contains("_block0_project("),
        "the node voltages must form projection blocks for this test to bite"
    );
    let harness = directory.path().join("ladder.cpp");
    fs::write(
        &harness,
        format!(
            r#"#include <cmath>
#include <cstdio>
#include <cstring>
struct Dim3 {{ int x; }};
static Dim3 blockIdx, blockDim, threadIdx;
#define __global__
#define __host__
#define __device__
#include {path:?}

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
fn cuda_ode_ladder_refreshes_once_after_a_settled_event_pass() {
    // Without events or manifold blocks, a settled event pass leaves the
    // slots settled, and the manifold projection is only a refresh.
    let (_directory, _path, content) = ladder_source();
    assert!(
        content.contains("#define RLCLADDER_LADDER_MANIFOLD_BLOCKS 0"),
        "the ladder has no manifold blocks, so a caller may skip the projection"
    );
    let start = content
        .find("int RlcLadder_Ladder_event_iterate(")
        .expect("cuda-ode emits the event iteration");
    let body = &content[start..];
    let body = &body[..body.find("\n}\n").expect("the event iteration ends")];
    let refreshes = body.matches("RlcLadder_Ladder_refresh(time").count();
    assert_eq!(refreshes, 2, "one refresh per pass and one after the loop");
    assert!(
        body.contains("if (!settled) {\n        (void)RlcLadder_Ladder_refresh(time"),
        "the refresh after the loop runs only after an unsettled last pass"
    );
}
