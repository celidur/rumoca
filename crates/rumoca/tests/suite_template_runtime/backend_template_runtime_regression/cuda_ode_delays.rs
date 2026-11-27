//! cuda-ode delays: the channels of one delay-time row are read together at
//! the place of their instant in the history's times, with the stencil over
//! them, and rows of a few delay times take the places of the last instants
//! placed.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

/// Channels of eight delay times, each from a parameter of its own, in an
/// order that takes each of the four kept places and asks again at an instant
/// the kernel no longer keeps; three of them ask beside a jump of the history
/// and past its last point. The last two channels share the parameter of an
/// earlier one, and are read with it.
const PLACES: &str = r#"
model DelayPlaces
  parameter Real t1 = 0.002;
  parameter Real t2 = 0.0005;
  parameter Real t3 = 0.002;
  parameter Real t4 = 0.001;
  parameter Real t5 = 0.0015;
  parameter Real t6 = 0.0007;
  parameter Real t7 = 0.0005;
  parameter Real t8 = 0.0015;
  parameter Real t9 = 0.0021;
  parameter Real t10 = 0.0021;
  parameter Real t11 = 0.0015;
  parameter Real t12 = 0.0005;
  parameter Real t13 = 0.0019;
  parameter Real t14 = 0.00003;
  parameter Real t15 = 0.00003;
  Real x(start = 1, fixed = true);
  Real w(start = 0, fixed = true);
  Real d1;
  Real d2;
  Real d3;
  Real d4;
  Real d5;
  Real d6;
  Real d7;
  Real d8;
  Real d9;
  Real d10;
  Real d11;
  Real d12;
  Real d13;
  Real d14;
  Real d15;
  Real d16;
  Real d17;
equation
  der(x) = -x;
  der(w) = 1;
  d1 = delay(x, t1);
  d2 = delay(w, t2);
  d3 = delay(w, t3);
  d4 = delay(x, t4);
  d5 = delay(w, t5);
  d6 = delay(x, t6);
  d7 = delay(x, t7);
  d8 = delay(x, t8);
  d9 = delay(x, t9);
  d10 = delay(w, t10);
  d11 = delay(w, t11);
  d12 = delay(w, t12);
  d13 = delay(x, t13);
  d14 = delay(w, t14);
  d15 = delay(x, t15);
  d16 = delay(w, t1);
  d17 = delay(x, t13);
end DelayPlaces;
"#;

/// Records x = cos(1000 t), w = t^2 at t = k 1e-4 in a ring of 32 points,
/// which wraps, with a jump of both at t = 2e-3 (two points at one time),
/// then reads the delayed values at one time and compares them, bit for bit,
/// with the stencil of the recorded samples written out here.
const PLACES_HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < DELAYPLACES_Y_LEN; ++i) {
        if (std::strcmp(DelayPlaces_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

// The runtime's `delayed_value` after the first sample: the Lagrange stencil
// over the samples around `query` (the last one and the point (now, current)
// past them) and one neighbour on each side across strictly increasing
// times, as offsets from the lower sample.
static double sampled(const double *times, const double *values, int count, double now, double current, double query) {
    int upper = 0;
    while (upper < count && times[upper] <= query) ++upper;
    int lower = upper - 1;
    double next_time = upper < count ? times[upper] : now;
    double next_value = upper < count ? values[upper] : current;
    double stencil_time[4], stencil_value[4];
    int points = 0;
    if (lower >= 1 && times[lower - 1] < times[lower]) {
        stencil_time[points] = times[lower - 1];
        stencil_value[points++] = values[lower - 1];
    }
    stencil_time[points] = times[lower];
    stencil_value[points++] = values[lower];
    stencil_time[points] = next_time;
    stencil_value[points++] = next_value;
    if (upper + 1 < count && times[upper + 1] > next_time) {
        stencil_time[points] = times[upper + 1];
        stencil_value[points++] = values[upper + 1];
    }
    if (points < 3) { std::printf("a stencil of %d points\n", points); std::exit(1); }
    double offset = -0.0;
    for (int i = 0; i < points; ++i) {
        double weight = 1.0;
        for (int j = 0; j < points; ++j) {
            if (j != i) {
                weight *= (query - stencil_time[j]) / (stencil_time[i] - stencil_time[j]);
            }
        }
        offset += weight * (stencil_value[i] - values[lower]);
    }
    return values[lower] + offset;
}

int main() {
    enum { CHANNELS = 17 };
    static const char *names[CHANNELS] = {"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17"};
    static const int of_x[CHANNELS] = {1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1};
    static const double delays[CHANNELS] = {0.002, 0.0005, 0.002, 0.001, 0.0015, 0.0007, 0.0005, 0.0015, 0.0021, 0.0021, 0.0015, 0.0005, 0.0019, 0.00003, 0.00003, 0.002, 0.0019};
    const int x = slot("x"), w = slot("w");
    if (x < 0 || w < 0) { std::printf("missing slots\n"); return 1; }
    enum { STEPS = 40, SAMPLES = STEPS + 1 };
    if (DELAYPLACES_DELAY_CAPACITY != 32) { std::printf("capacity %d\n", DELAYPLACES_DELAY_CAPACITY); return 1; }
    static double y[DELAYPLACES_Y_LEN], p[DELAYPLACES_P_LEN], history[DELAYPLACES_DELAY_HISTORY_STRIDE];
    int status = -1;
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0;
    DelayPlaces_load_starts_batch(y, p, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, 1);
    DelayPlaces_initialize_batch(0.0, y, p, &status, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, 1);
    DelayPlaces_delay_init_batch(0.0, y, p, history, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, DELAYPLACES_DELAY_HISTORY_STRIDE, 1);
    double times[SAMPLES], xs[SAMPLES], ws[SAMPLES];
    int count = 0;
    times[count] = 0.0; xs[count] = y[x]; ws[count++] = y[w];
    for (int k = 1; k < STEPS; ++k) {
        double t = k * 1e-4;
        // The jump: the left values, then the right ones at the same time.
        for (int side = 0; side < (k == 20 ? 2 : 1); ++side) {
            y[x] = std::cos(1000.0 * t) + 0.5 * side; y[w] = t * t + 0.25 * side;
            DelayPlaces_delay_commit_batch(t, y, p, history, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, DELAYPLACES_DELAY_HISTORY_STRIDE, 1);
            times[count] = t; xs[count] = y[x]; ws[count++] = y[w];
        }
    }
    // A trial point past the last accepted one.
    const double now = 0.00397;
    y[x] = std::cos(1000.0 * now); y[w] = now * now;
    const double current_x = y[x], current_w = y[w];
    DelayPlaces_delay_apply_batch(now, y, p, history, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, DELAYPLACES_DELAY_HISTORY_STRIDE, 1);
    DelayPlaces_refresh_batch(now, y, p, &status, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, 1);
    if (status != 0 || count != SAMPLES) { std::printf("status %d, %d samples\n", status, count); return 1; }
    for (int c = 0; c < CHANNELS; ++c) {
        int d = slot(names[c]);
        if (d < 0) { std::printf("missing %s\n", names[c]); return 1; }
        double want = sampled(times, of_x[c] ? xs : ws, count, now, of_x[c] ? current_x : current_w, now - delays[c]);
        if (std::memcmp(&want, &y[d], sizeof want) != 0) {
            std::printf("%s %.17g/%.17g\n", names[c], y[d], want);
            return 1;
        }
    }
    return 0;
}
"#;

#[test]
fn cuda_ode_lanes_place_alternating_delay_times_in_the_history() {
    let compiled = Compiler::new()
        .model("DelayPlaces")
        .compile_str(PLACES, "DelayPlaces.mo")
        .expect("the compiler accepts the delays");
    let (directory, files) = rendered_model_target(&compiled, "DelayPlaces", "cuda-ode");
    let source = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    // Seventeen channels of fifteen delay-time rows.
    assert!(
        source.content.contains("static const int member[17] = ")
            && source.content.contains("static const int group_end[15] = "),
        "the channels of one delay-time row form one group"
    );
    let path = directory.path().join("delays.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n";
    fs::write(
        &path,
        format!("{prelude}#include {:?}\n{PLACES_HARNESS}", source.path),
    )
    .expect("write the delay harness");
    let executable = directory.path().join("cuda-ode-delays");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg("-DDELAYPLACES_DELAY_CAPACITY=32")
            .arg(&path)
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
