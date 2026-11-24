//! cuda-ode on a model that retains a state manifold: each lane certifies
//! its initial states and projects its accepted states with its own instance
//! of the shared ME projection kernel.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

/// CPU emulation of one lane, included after the generated kernel.
const HARNESS: &str = r#"static int slot(const char *name) {
    for (int i = 0; i < PINNEDSTART_Y_LEN; ++i) {
        if (std::strcmp(PinnedStart_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

#define CHECK(condition, message) \
    do { if (!(condition)) { std::printf("%s: x %.17g z %.17g status %d\n", message, y[x], y[z], status); return 1; } } while (0)

int main() {
    const int x = slot("x"), z = slot("z");
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    void *lanes = std::malloc(PINNEDSTART_LANE_WORKSPACE_BYTES);
    PinnedStart_workspace_init_batch(lanes, 1);
    double y[PINNEDSTART_Y_LEN];
    double p[PINNEDSTART_P_LEN + 1];
    double out[PINNEDSTART_STATE_COUNT];
    int status = -1;
    PinnedStart_load_starts_batch(y, p, PINNEDSTART_Y_LEN, PINNEDSTART_P_LEN, 1);
    PinnedStart_initialize_batch(0.0, y, p, &status, PINNEDSTART_Y_LEN, PINNEDSTART_P_LEN, 1);
    CHECK(status == 0 && std::fabs(y[x] - 1.0) < 1e-8 && std::fabs(y[z] - 1.0) < 1e-8,
        "the initialization settles on the manifold");

    // An accepted step that drifted off the manifold is moved back onto it.
    y[x] = 1.01;
    status = -1;
    PinnedStart_project_states_batch(0.1, y, p, &status, PINNEDSTART_Y_LEN, PINNEDSTART_P_LEN, 1);
    CHECK(status == 0 && std::fabs(y[x] - 1.0) < 1e-8 && y[z] == y[z],
        "the projection restores the constraint");

    // On the manifold der(x) = 0, and der(z) = 1 - z.
    PinnedStart_rhs_batch(0.1, y, p, out, &status, PINNEDSTART_Y_LEN, PINNEDSTART_P_LEN,
        PINNEDSTART_STATE_COUNT, 1);
    CHECK(status == 0 && std::fabs(out[x]) < 1e-8 && std::fabs(out[z] - (1.0 - y[z])) < 1e-12,
        "the right-hand side holds the state on the manifold");
    std::free(lanes);
    return 0;
}
"#;

#[test]
fn cuda_ode_lane_projects_its_states_onto_the_retained_manifold() {
    // Index reduction keeps `x^3 = 1` as a constraint over the state `x`;
    // the initial equation fixes `w`, so `z = 2 - x^2 = 1`.
    let compiled = Compiler::new()
        .model("PinnedStart")
        .compile_str(
            r#"
model PinnedStart
  Real x(start = 1.2);
  Real z(start = 0.5);
  Real lambda;
  Real w;
initial equation
  w = 2;
equation
  der(x) = lambda * x;
  der(z) = 1 - z;
  w = z + x * x;
  x * x * x = 1;
end PinnedStart;
"#,
            "PinnedStart.mo",
        )
        .expect("the compiler accepts the model");
    let (directory, files) = rendered_model_target(&compiled, "PinnedStart", "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    assert!(
        source
            .content
            .contains("#define PINNEDSTART_MANIFOLD_BLOCKS 1"),
        "the model must retain a state manifold for this test to bite"
    );
    let harness = directory.path().join("manifold.cpp");
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

{}"#,
            source.path, HARNESS,
        ),
    )
    .expect("write manifold harness");
    let executable = directory.path().join("cuda-ode-manifold");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE state-manifold emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE state-manifold emulation",
    );
}
