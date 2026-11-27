//! cuda-ode delays: channels that ask at one instant share its place in the
//! history's times, and channels of a few delay times take the places of the
//! last instants placed.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

/// Channels of six delay times, in an order that takes each of the four kept
/// places and asks again at an instant the kernel no longer keeps.
const PLACES: &str = r#"
model DelayPlaces
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
equation
  der(x) = -x;
  der(w) = 1;
  d1 = delay(x, 0.002);
  d2 = delay(w, 0.0005);
  d3 = delay(w, 0.002);
  d4 = delay(x, 0.001);
  d5 = delay(w, 0.0015);
  d6 = delay(x, 0.0007);
  d7 = delay(x, 0.0005);
  d8 = delay(x, 0.0015);
  d9 = delay(x, 0.0021);
  d10 = delay(w, 0.0021);
  d11 = delay(w, 0.0015);
  d12 = delay(w, 0.0005);
end DelayPlaces;
"#;

/// Records x = cos(1000 t), w = t^2 at t = k 1e-4 in a ring of 32 points,
/// which wraps, then reads the delayed values at one time and compares them,
/// bit for bit, with the cubic stencil of the recorded samples written out
/// here.
const PLACES_HARNESS: &str = r#"
static int slot(const char *name) {
    for (int i = 0; i < DELAYPLACES_Y_LEN; ++i) {
        if (std::strcmp(DelayPlaces_y_names[i], name) == 0) {
            return i;
        }
    }
    return -1;
}

// The Lagrange stencil over the two samples around `query` and one neighbour
// on each side, as offsets from the lower sample.
static double sampled(const double *times, const double *values, double query) {
    int lower = 0;
    while (times[lower + 1] <= query) ++lower;
    double offset = -0.0;
    for (int i = lower - 1; i <= lower + 2; ++i) {
        double weight = 1.0;
        for (int j = lower - 1; j <= lower + 2; ++j) {
            if (j != i) {
                weight *= (query - times[j]) / (times[i] - times[j]);
            }
        }
        offset += weight * (values[i] - values[lower]);
    }
    return values[lower] + offset;
}

int main() {
    enum { CHANNELS = 12 };
    static const char *names[CHANNELS] = {"d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12"};
    static const int of_x[CHANNELS] = {1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0};
    static const double delays[CHANNELS] = {0.002, 0.0005, 0.002, 0.001, 0.0015, 0.0007, 0.0005, 0.0015, 0.0021, 0.0021, 0.0015, 0.0005};
    const int x = slot("x"), w = slot("w");
    if (x < 0 || w < 0) { std::printf("missing slots\n"); return 1; }
    enum { SAMPLES = 40 };
    if (DELAYPLACES_DELAY_CAPACITY != 32) { std::printf("capacity %d\n", DELAYPLACES_DELAY_CAPACITY); return 1; }
    static double y[DELAYPLACES_Y_LEN], p[DELAYPLACES_P_LEN], history[DELAYPLACES_DELAY_HISTORY_STRIDE];
    int status = -1;
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0;
    DelayPlaces_load_starts_batch(y, p, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, 1);
    DelayPlaces_initialize_batch(0.0, y, p, &status, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, 1);
    DelayPlaces_delay_init_batch(0.0, y, p, history, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, DELAYPLACES_DELAY_HISTORY_STRIDE, 1);
    double times[SAMPLES], xs[SAMPLES], ws[SAMPLES];
    times[0] = 0.0; xs[0] = y[x]; ws[0] = y[w];
    for (int k = 1; k < SAMPLES; ++k) {
        double t = k * 1e-4;
        y[x] = std::cos(1000.0 * t); y[w] = t * t;
        DelayPlaces_delay_commit_batch(t, y, p, history, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, DELAYPLACES_DELAY_HISTORY_STRIDE, 1);
        times[k] = t; xs[k] = y[x]; ws[k] = y[w];
    }
    const double now = 0.00397;
    DelayPlaces_delay_apply_batch(now, y, p, history, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, DELAYPLACES_DELAY_HISTORY_STRIDE, 1);
    DelayPlaces_refresh_batch(now, y, p, &status, DELAYPLACES_Y_LEN, DELAYPLACES_P_LEN, 1);
    if (status != 0) { std::printf("status %d\n", status); return 1; }
    for (int c = 0; c < CHANNELS; ++c) {
        int d = slot(names[c]);
        if (d < 0) { std::printf("missing %s\n", names[c]); return 1; }
        double want = sampled(times, of_x[c] ? xs : ws, now - delays[c]);
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
    let path = directory.path().join("delays.cpp");
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstring>\n\
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
