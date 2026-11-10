//! The per-lane Newton solves of cuda-ode report convergence reached on
//! their last allowed iteration as success.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

#[test]
fn cuda_ode_newton_converging_on_its_last_iteration_succeeds() {
    // An affine torn refresh block and a nonlinear initial equation. With one
    // allowed iteration and a step tolerance every step meets, both Newton
    // solves converge on their last iteration.
    let compiled = Compiler::new()
        .model("LastStep")
        .compile_str(
            r#"
model LastStep
  Real x(start = 1, fixed = false);
  Real a;
  Real b;
initial equation
  x * x + x = 6;
equation
  der(x) = -x + a;
  a + b = 4 + x;
  a - b = 2 * x;
end LastStep;
"#,
            "LastStep.mo",
        )
        .expect("the compiler accepts the model");
    let (directory, files) = rendered_model_target(&compiled, "LastStep", "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    assert!(
        source.content.contains("LastStep_block0_project(")
            && source.content.contains("LastStep_init0_project("),
        "the model must keep a refresh block and an initialization block"
    );
    let harness = directory.path().join("last_step.cpp");
    fs::write(
        &harness,
        format!(
            r#"#include <cmath>
#include <cstdio>
struct Dim3 {{ int x; }};
static Dim3 blockIdx, blockDim, threadIdx;
#define __global__
#define __host__
#define __device__
#define LASTSTEP_NEWTON_MAX_ITERATIONS 1
#define LASTSTEP_NEWTON_STEP_TOLERANCE 1e300
#include {:?}

int main() {{
    double y[LASTSTEP_Y_LEN] = {{0.0}};
    double p[1] = {{0.0}};
    double out[LASTSTEP_STATE_COUNT] = {{0.0}};
    int status = -1;
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    LastStep_load_starts_batch(y, p, LASTSTEP_Y_LEN, 0, 1);
    LastStep_initialize_batch(0.0, y, p, &status, LASTSTEP_Y_LEN, 0, 1);
    if (status != 0) {{
        std::printf("initialize status %d\n", status);
        return 1;
    }}
    y[0] = 1.0;
    y[1] = 0.0;
    y[2] = 0.0;
    LastStep_rhs_batch(0.0, y, p, out, &status, LASTSTEP_Y_LEN, 0, LASTSTEP_STATE_COUNT, 1);
    // a = (4 + 3 x) / 2, so der(x) = -x + a = 2.5 at x = 1.
    if (status != 0 || std::fabs(out[0] - 2.5) > 1e-12) {{
        std::printf("rhs status %d, der %.17g\n", status, out[0]);
        return 1;
    }}
    return 0;
}}
"#,
            source.path,
        ),
    )
    .expect("write last-iteration harness");
    let executable = directory.path().join("cuda-ode-last-step");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE last-iteration emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE last-iteration emulation",
    );
}
