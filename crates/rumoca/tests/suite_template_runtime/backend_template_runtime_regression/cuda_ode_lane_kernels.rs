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
    // Without delay() the lane keeps no history, so it needs no buffer.
    Relay_delay_init_batch(0.0, y, p, nullptr, RELAY_Y_LEN, RELAY_P_LEN, 0, 1);
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

    // A slot changes when its bits do, as the runtime compares them: a NaN
    // that stays NaN settles, a zero that changes sign does not.
    CHECK(!Relay_changes(NAN, NAN) && Relay_changes(0.0, -0.0) && !Relay_changes(1.0, 1.0),
        "a slot changes when its bits change");
    Relay_event_update_batch(0.0, y, p, &passes, RELAY_Y_LEN, RELAY_P_LEN, 1);
    CHECK(passes > 0, "the event iteration settles");
    Relay_rhs_batch(0.0, y, p, out, &status, RELAY_Y_LEN, RELAY_P_LEN, RELAY_STATE_COUNT, 1);
    CHECK(status == 0 && out[0] == 1.0, "the event update switches the relay");
    return 0;
}
"#;

#[test]
fn cuda_ode_encodes_a_relation_from_its_root_alone() {
    // As the runtime and the FMI kernels do, an exact zero the domain leaves
    // unoriented encodes false; the lane never keeps the previous memory.
    let (_directory, source) = relay_kernel();
    let start = source
        .content
        .find("int Relay_event_iterate(")
        .expect("cuda-ode emits the event iteration");
    let body = &source.content[start..];
    let body = &body[..body.find("\n}\n").expect("the event iteration ends")];
    assert!(body.contains("double value = root < 0.0 ? 1.0 : 0.0;"));
    assert!(
        !body.contains("value = batch_p[memory[k]]"),
        "a zero root does not keep the previous memory"
    );
}

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
fn cuda_ode_refreshes_once_after_a_settled_event_pass() {
    let (_directory, source) = relay_kernel();
    // A settled event pass leaves the slots settled: one refresh per pass,
    // and one after the loop only when the last pass did not settle.
    let start = source
        .content
        .find("int Relay_event_iterate(")
        .expect("cuda-ode emits the event iteration");
    let body = &source.content[start..];
    let body = &body[..body.find("\n}\n").expect("the event iteration ends")];
    assert_eq!(body.matches("Relay_refresh(time").count(), 2);
    assert!(
        body.contains("if (!settled) {\n        Relay_refresh(time"),
        "the refresh after the loop runs only after an unsettled last pass"
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

/// A delay of an algebraic source, `d = delay(s, tau, tmax)` with `s = x^2`.
const LAG: &str = r#"
model Lag
  parameter Real tau = 0.02;
  parameter Real tmax = 1;
  Real x(start = 0, fixed = true);
  Real s;
  Real d;
equation
  der(x) = 1;
  s = x * x;
  d = delay(s, tau, tmax);
end Lag;
"#;

/// CPU emulation of one lane's delay history: p[0] is tau, p[1] tmax and
/// p[2] the delayed value.
const LAG_HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < LAG_Y_LEN; ++i) {
        if (std::strcmp(Lag_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

#define CHECK(condition, message) \
    do { if (!(condition)) { std::printf("%s: %.17g\n", message, p[2]); return 1; } } while (0)

int main() {
    const int x = slot("x"), s = slot("s");
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    double y[LAG_Y_LEN];
    double p[LAG_P_LEN];
    static double history[LAG_DELAY_HISTORY_STRIDE];
    int status = -1;
    Lag_load_starts_batch(y, p, LAG_Y_LEN, LAG_P_LEN, 1);
    Lag_initialize_batch(0.0, y, p, &status, LAG_Y_LEN, LAG_P_LEN, 1);
    Lag_delay_init_batch(0.0, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(status == 0 && x >= 0 && s >= 0, "the lane initializes");

    // A trial point past the history, its algebraic slot stale: the source is
    // settled from x, and a zero delay reads it.
    y[x] = 0.3;
    y[s] = 99.0;
    p[0] = 0.0;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(std::fabs(p[2] - 0.09) < 1e-15, "a zero delay reads the current source");

    // Past the last accepted point, toward the current source: (0, 0) to
    // (0.3, 0.09) at 0.2.
    p[0] = 0.1;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(std::fabs(p[2] - 0.06) < 1e-15, "past the history the value runs toward the current source");

    // Inside the history, the cubic stencil reads a quadratic back exactly.
    for (int k = 1; k <= 3; ++k) {
        y[x] = 0.1 * k;
        Lag_delay_commit_batch(0.1 * k, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    }
    p[0] = 0.15;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(std::fabs(p[2] - 0.0225) < 1e-15, "the history interpolates a quadratic exactly");

    // A delay time above delayMax, or below zero, has no value.
    p[0] = 2.0;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(std::isnan(p[2]), "a delay time above delayMax reads NaN");
    p[0] = -0.1;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(std::isnan(p[2]), "a negative delay time reads NaN");
    return 0;
}
"#;

#[test]
fn cuda_ode_lane_settles_its_delay_sources_only_when_it_reads_them() {
    // tau and tmax are parameters; only the source s = x * x reads an
    // algebraic slot, and only a channel reading its current source needs it.
    let compiled = Compiler::new()
        .model("Lag")
        .compile_str(LAG, "Lag.mo")
        .expect("the compiler accepts the delay");
    let (_directory, files) = rendered_model_target(&compiled, "Lag", "cuda-ode");
    let source = files
        .into_iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    let start = source
        .content
        .find("void Lag_delay_apply_batch(")
        .expect("cuda-ode emits delay_apply");
    let body = &source.content[start..];
    let body = &body[..body.find("\n}\n").expect("delay_apply ends")];
    assert_eq!(body.matches("Lag_delay_settle(").count(), 1, "{body}");
    assert!(
        body.contains("if (current) {\n        view = Lag_delay_settle("),
        "the sources are settled only for a channel that reads them: {body}"
    );
}

#[test]
fn cuda_ode_lane_reads_its_delays_as_the_runtime() {
    let compiled = Compiler::new()
        .model("Lag")
        .compile_str(LAG, "Lag.mo")
        .expect("the compiler accepts the delay");
    let (directory, files) = rendered_model_target(&compiled, "Lag", "cuda-ode");
    let source = files
        .into_iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    let harness = directory.path().join("lag.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n";
    fs::write(
        &harness,
        format!("{prelude}#include {:?}\n{LAG_HARNESS}", source.path),
    )
    .expect("write delay harness");
    let executable = directory.path().join("cuda-ode-lag");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&harness)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE delay emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE delay emulation",
    );
}

#[test]
fn cuda_ode_refuses_a_reduced_chart_set_it_cannot_switch() {
    // The split circle switches between two reduced charts; a lane would
    // stay on the primary one.
    let compiled = Compiler::new()
        .model("SplitCircleChart")
        .compile_str(
            include_str!("../../fixtures/index_reduction/SplitCircleChart.mo"),
            "SplitCircleChart.mo",
        )
        .expect("the compiler accepts the model");
    let error = rumoca::render_target_files(&compiled, "SplitCircleChart", "cuda-ode", None)
        .expect_err("cuda-ode does not switch charts");
    assert!(
        format!("{error:#}").contains("unsupported-feature:reduced_charts"),
        "unexpected refusal: {error:#}"
    );
}

/// Render `model` for cuda-ode, include its kernel after the emulation
/// prelude, and run `harness`.
fn run_lane_harness(source: &str, model: &str, harness: &str) {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .expect("the compiler accepts the model");
    let (directory, files) = rendered_model_target(&compiled, model, "cuda-ode");
    let kernel = files
        .into_iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    let path = directory.path().join("lane.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n\
        #define CHECK(condition, message) \\\n\
        do { if (!(condition)) { std::printf(\"%s\\n\", message); return 1; } } while (0)\n";
    fs::write(
        &path,
        format!("{prelude}#include {:?}\n{harness}", kernel.path),
    )
    .expect("write lane harness");
    let executable = directory.path().join("cuda-ode-lane");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&path)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE lane emulation",
    );
    run_checked(
        &mut Command::new(executable),
        "execute CUDA ODE lane emulation",
    );
}

#[test]
fn cuda_ode_refuses_a_model_assertion_it_would_not_check() {
    let compiled = Compiler::new()
        .model("Guarded")
        .compile_str(
            r#"
model Guarded
  Real x(start = 1, fixed = true);
equation
  der(x) = -1;
  assert(x > 0, "x must stay positive");
end Guarded;
"#,
            "Guarded.mo",
        )
        .expect("the compiler accepts the model");
    let error = rumoca::render_target_files(&compiled, "Guarded", "cuda-ode", None)
        .expect_err("a lane does not check model assertions");
    assert!(
        format!("{error:#}").contains("assert()"),
        "unexpected refusal: {error:#}"
    );
}

#[test]
fn cuda_ode_lane_fails_where_a_function_assertion_fails() {
    // The assertion inside `tripled` owns a root without a relation memory:
    // the event iteration does not re-encode it, and the lane checks it in the
    // call, whose failure leaves the derivative NaN.
    run_lane_harness(
        r#"
model Tripled
  function tripled
    input Real u;
    output Real y;
  algorithm
    assert(u > 0, "u must be positive");
    y := 0;
    for i in 1:3 loop
      y := y + u;
    end for;
  end tripled;
  Real x(start = 1, fixed = true);
equation
  der(x) = tripled(x);
end Tripled;
"#,
        "Tripled",
        r#"
int main() {
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    double y[TRIPLED_Y_LEN + 1] = {0.0};
    double p[TRIPLED_P_LEN + 1] = {0.0};
    double der[TRIPLED_STATE_COUNT + 1] = {0.0};
    int status = -1;
    Tripled_load_starts_batch(y, p, TRIPLED_Y_LEN, TRIPLED_P_LEN, 1);
    Tripled_initialize_batch(0.0, y, p, &status, TRIPLED_Y_LEN, TRIPLED_P_LEN, 1);
    CHECK(status == 0, "the initial event settles");
    Tripled_rhs_batch(0.0, y, p, der, &status, TRIPLED_Y_LEN, TRIPLED_P_LEN, TRIPLED_STATE_COUNT, 1);
    CHECK(status == 0 && der[0] == 3.0, "the assertion holds at x = 1");
    y[0] = -1.0;
    Tripled_rhs_batch(0.0, y, p, der, &status, TRIPLED_Y_LEN, TRIPLED_P_LEN, TRIPLED_STATE_COUNT, 1);
    CHECK(status == 1 && std::isnan(der[0]), "the failed assertion fails the lane");
    return 0;
}
"#,
    );
}

#[test]
fn cuda_ode_reports_a_crossing_its_rows_read_directly() {
    // `x[i] < 0.5` has no relation memory: the rows read it directly, and
    // the lane reports its root after those of the relation memories.
    run_lane_harness(
        r#"
model Crossing
  Real x[3](start = {1, 2, 3}, each fixed = true);
  Real v[3];
equation
  for i in 1:3 loop
    v[i] = if x[i] < 0.5 then 0 else 1;
    der(x[i]) = -v[i];
  end for;
end Crossing;
"#,
        "Crossing",
        r#"
int main() {
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    double y[CROSSING_Y_LEN + 1] = {0.0};
    double p[CROSSING_P_LEN + 1] = {0.0};
    double z[CROSSING_EVENT_INDICATORS + 1] = {0.0};
    double der[CROSSING_STATE_COUNT + 1] = {0.0};
    int status = -1, passes = -1;
    CHECK(CROSSING_EVENT_INDICATORS == 3, "every crossing is an indicator");
    Crossing_load_starts_batch(y, p, CROSSING_Y_LEN, CROSSING_P_LEN, 1);
    Crossing_initialize_batch(0.0, y, p, &status, CROSSING_Y_LEN, CROSSING_P_LEN, 1);
    CHECK(status == 0, "the initial event settles");
    y[0] = 0.25;
    Crossing_event_indicators_batch(0.0, y, p, z, CROSSING_Y_LEN, CROSSING_P_LEN, CROSSING_EVENT_INDICATORS, 1);
    CHECK(z[0] == -0.25 && z[1] == 1.5 && z[2] == 2.5, "the indicators are the roots");
    Crossing_event_update_batch(0.0, y, p, &passes, CROSSING_Y_LEN, CROSSING_P_LEN, 1);
    CHECK(passes > 0, "the event iteration settles");
    Crossing_rhs_batch(0.0, y, p, der, &status, CROSSING_Y_LEN, CROSSING_P_LEN, CROSSING_STATE_COUNT, 1);
    CHECK(status == 0 && der[0] == 0.0 && der[1] == -1.0, "the rows read the relation");
    return 0;
}
"#,
    );
}

#[test]
fn cuda_ode_event_iteration_refuses_a_non_finite_root() {
    // sqrt(x) - 0.5 has no value for x < 0: the runtime refuses the root,
    // and the lane must not encode its relation as false.
    run_lane_harness(
        r#"
model Rooted
  Real x(start = 1, fixed = true);
  Real drive;
equation
  drive = if sqrt(x) > 0.5 then -1 else 1;
  der(x) = drive;
end Rooted;
"#,
        "Rooted",
        r#"
int main() {
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    double y[ROOTED_Y_LEN + 1] = {0.0};
    double p[ROOTED_P_LEN + 1] = {0.0};
    int status = -1, passes = -1;
    Rooted_load_starts_batch(y, p, ROOTED_Y_LEN, ROOTED_P_LEN, 1);
    Rooted_initialize_batch(0.0, y, p, &status, ROOTED_Y_LEN, ROOTED_P_LEN, 1);
    CHECK(status == 0, "the initial event settles");
    Rooted_event_update_batch(0.0, y, p, &passes, ROOTED_Y_LEN, ROOTED_P_LEN, 1);
    CHECK(passes > 0, "a finite root settles");
    y[0] = -1.0;
    Rooted_event_update_batch(0.0, y, p, &passes, ROOTED_Y_LEN, ROOTED_P_LEN, 1);
    CHECK(passes == -1, "a non-finite root fails the event iteration");
    return 0;
}
"#,
    );
}

/// One lane's ring of capacity `LAG_DELAY_CAPACITY` after the points 0, 0.1,
/// 0.2 and 0.3: a capacity of 3 has dropped the point at 0.
const LAG_RING_HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < LAG_Y_LEN; ++i) {
        if (std::strcmp(Lag_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

int main() {
    const int x = slot("x");
    blockIdx.x = 0;
    blockDim.x = 1;
    threadIdx.x = 0;
    double y[LAG_Y_LEN];
    double p[LAG_P_LEN];
    static double history[LAG_DELAY_HISTORY_STRIDE];
    int status = -1;
    Lag_load_starts_batch(y, p, LAG_Y_LEN, LAG_P_LEN, 1);
    Lag_initialize_batch(0.0, y, p, &status, LAG_Y_LEN, LAG_P_LEN, 1);
    Lag_delay_init_batch(0.0, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(status == 0 && x >= 0, "the lane initializes");
    for (int k = 1; k <= 3; ++k) {
        y[x] = 0.1 * k;
        Lag_delay_commit_batch(0.1 * k, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    }
    // Between the points the ring still holds, the stencil reads the
    // quadratic back exactly.
    p[0] = 0.15;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
    CHECK(std::fabs(p[2] - 0.0225) < 1e-15, "the kept points read back");
    p[0] = 0.25;
    Lag_delay_apply_batch(0.3, y, p, history, LAG_Y_LEN, LAG_P_LEN, LAG_DELAY_HISTORY_STRIDE, 1);
#if LAG_DELAY_CAPACITY == 3
    CHECK(std::isnan(p[2]), "a time older than the kept points reads NaN");
#else
    CHECK(std::isfinite(p[2]), "a ring that dropped nothing reads its history");
#endif
    return 0;
}
"#;

#[test]
fn cuda_ode_lane_reads_nan_past_the_points_its_ring_dropped() {
    let compiled = Compiler::new()
        .model("Lag")
        .compile_str(LAG, "Lag.mo")
        .expect("the compiler accepts the delay");
    let (directory, files) = rendered_model_target(&compiled, "Lag", "cuda-ode");
    let source = files
        .into_iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    let harness = directory.path().join("ring.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n\
        #define CHECK(condition, message) \\\n\
        do { if (!(condition)) { std::printf(\"%s: %.17g\\n\", message, p[2]); return 1; } } while (0)\n";
    fs::write(
        &harness,
        format!("{prelude}#include {:?}\n{LAG_RING_HARNESS}", source.path),
    )
    .expect("write ring harness");
    for capacity in [3, 4] {
        let executable = directory.path().join(format!("cuda-ode-ring-{capacity}"));
        run_checked(
            Command::new("c++")
                .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
                .arg(format!("-DLAG_DELAY_CAPACITY={capacity}"))
                .arg(&harness)
                .arg("-o")
                .arg(&executable)
                .current_dir(directory.path()),
            "compile CUDA ODE ring emulation",
        );
        run_checked(
            &mut Command::new(executable),
            "execute CUDA ODE ring emulation",
        );
    }
}
