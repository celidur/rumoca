//! cuda-ode lane kernels in CPU emulation: bindings, the initial event,
//! frozen relation memories, event indicators and the event update.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use rumoca_compile::codegen::targets::RenderedTargetFile;
use std::fs;
use std::process::Command;

/// The relay model rendered for cuda-ode: its directory and its kernel.
fn relay_kernel() -> (tempfile::TempDir, RenderedTargetFile) {
    let compiled = Compiler::new()
        .model("Relay")
        .compile_str(
            r#"
model Relay
  parameter Real gain = 2;
  parameter Real limit = gain / 4;
  Real x(start = 1, fixed = true);
  Real drive;
equation
  drive = if x > limit then -1 else 1;
  der(x) = drive;
end Relay;
"#,
            "Relay.mo",
        )
        .expect("the compiler accepts the relay");
    let (directory, files) = rendered_model_target(&compiled, "Relay", "cuda-ode");
    let source = files
        .into_iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    (directory, source)
}

/// CPU emulation of one lane, included after the generated kernel.
const RELAY_HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < RELAY_Y_LEN; ++i) {
        if (std::strcmp(Relay_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

#define CHECK(condition, message) \
    do { if (!(condition)) { std::printf("%s\n", message); return 1; } } while (0)

int main() {
    const int x = slot("x");
    CHECK(x >= 0 && RELAY_EVENT_INDICATORS == 1, "one state and one indicator");
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    double y[RELAY_Y_LEN];
    double p[RELAY_P_LEN];
    double out[RELAY_STATE_COUNT];
    double before = 0.0, after = 0.0;
    int status = -1, passes = -1;
    Relay_load_starts_batch(y, p, RELAY_Y_LEN, RELAY_P_LEN, 1);
    Relay_initialize_batch(0.0, y, p, &status, RELAY_Y_LEN, RELAY_P_LEN, 1);
    CHECK(status == 0, "the initial event settles");
    Relay_rhs_batch(0.0, y, p, out, &status, RELAY_Y_LEN, RELAY_P_LEN, RELAY_STATE_COUNT, 1);
    CHECK(status == 0 && out[0] == -1.0, "above the bound limit = gain / 4 the relay drives down");
    Relay_event_indicators_batch(0.0, y, p, &before, RELAY_Y_LEN, RELAY_P_LEN, 1, 1);

    // Crossing the limit changes the indicator's sign, not the derivative:
    // the relation memory is frozen between events.
    y[x] = 0.25;
    Relay_rhs_batch(0.0, y, p, out, &status, RELAY_Y_LEN, RELAY_P_LEN, RELAY_STATE_COUNT, 1);
    CHECK(out[0] == -1.0, "the relation memory is frozen between events");
    Relay_event_indicators_batch(0.0, y, p, &after, RELAY_Y_LEN, RELAY_P_LEN, 1, 1);
    CHECK(before * after < 0.0, "the indicator changes sign across the limit");

    Relay_event_update_batch(0.0, y, p, &passes, RELAY_Y_LEN, RELAY_P_LEN, 1);
    CHECK(passes > 0, "the event iteration settles");
    Relay_rhs_batch(0.0, y, p, out, &status, RELAY_Y_LEN, RELAY_P_LEN, RELAY_STATE_COUNT, 1);
    CHECK(status == 0 && out[0] == 1.0, "the event update switches the relay");
    return 0;
}
"#;

#[test]
fn cuda_ode_lane_switches_its_relation_only_in_the_event_update() {
    let (directory, source) = relay_kernel();
    let harness = directory.path().join("relay.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n";
    fs::write(
        &harness,
        format!("{prelude}#include {:?}\n{RELAY_HARNESS}", source.path),
    )
    .expect("write relay harness");
    let executable = directory.path().join("cuda-ode-relay");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE relay emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE relay emulation",
    );
}

#[test]
fn cuda_ode_refuses_an_initialization_it_would_have_to_solve() {
    // The initial equation is implicit in the start value; solving it belongs
    // to the one projection kernel, which this target does not embed.
    let compiled = Compiler::new()
        .model("ImplicitStart")
        .compile_str(
            r#"
model ImplicitStart
  Real x;
initial equation
  x + x * x * x = 2;
equation
  der(x) = -x;
end ImplicitStart;
"#,
            "ImplicitStart.mo",
        )
        .expect("the compiler accepts the model");
    let error = rumoca::render_target_files(&compiled, "ImplicitStart", "cuda-ode", None)
        .expect_err("cuda-ode holds no initialization solver");
    assert!(
        format!("{error:#}").contains("unsupported-feature:initialization"),
        "unexpected refusal: {error:#}"
    );
}
