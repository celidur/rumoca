//! cuda-ode returns a refresh at a recorded fixed point without refreshing
//! (an input the refresh of an event iteration left unchanged, bit for
//! bit): over an integration that evaluates the derivative at each accepted
//! point after its event update, as the GPU integrators do, the trajectory
//! is the same, bit for bit, with fixed points and without them
//! (`<MODEL>_NO_FIXED_POINTS`), on a loop with an event and a delay and on a
//! model that retains a state manifold; and every refresh the record would
//! skip leaves its input unchanged when it runs. The record is copied and
//! compared in runs of words: every word of it counts.

use super::{rendered_model_target, run_checked};
use rumoca::Compiler;
use std::fs;
use std::process::Command;

/// A nonlinear loop, a delayed state and a relation that switches at
/// s = 0.03.
const EVENT_DELAY: (&str, &str) = (
    "FixedEventDelay",
    r#"
model FixedEventDelay
  Real x(start = 1, fixed = true);
  Real s(start = 0, fixed = true);
  Real u(start = 0.5);
  Real d;
equation
  u + 0.5 * sin(u) + 0.1 * u * u * u = x;
  d = delay(x, 0.004);
  der(x) = -u + 0.2 * d + (if s > 0.03 then -1 else 1);
  der(s) = 1;
end FixedEventDelay;
"#,
);

/// A retained state manifold (`x^3 = 1`) beside a nonlinear loop.
const MANIFOLD: (&str, &str) = (
    "FixedManifold",
    r#"
model FixedManifold
  Real x(start = 1.2);
  Real z(start = 0.5);
  Real lambda;
  Real w;
  Real u(start = 0.5);
initial equation
  w = 2;
equation
  der(x) = lambda * x;
  der(z) = 1 - z - 0.1 * u;
  w = z + x * x;
  x * x * x = 1;
  u + 0.1 * u * u * u = z;
end FixedManifold;
"#,
);

/// `@` stands for the model's name and `@@` for its upper-case macro prefix.
const HARNESS: &str = r#"
int main() {
    enum { NY = @@_Y_LEN, NP = @@_P_LEN, NS = @@_STATE_COUNT, HS = @@_DELAY_HISTORY_STRIDE > 0 ? @@_DELAY_HISTORY_STRIDE : 1 };
    static double y[NY], x[NY], p[NP], f0[NS], f1[NS], history[HS], before[NY];
    int status = -1, passes = 0, hits = 0, moved = 0;
    blockIdx.x = 0; blockDim.x = 1; threadIdx.x = 0;
    void *lanes = std::malloc(@@_LANE_WORKSPACE_BYTES);
    const ModelInstance *m = (const ModelInstance *)lanes;
    @_workspace_init_batch(lanes, 1);
    @_load_starts_batch(y, p, NY, NP, 1);
    @_initialize_batch(0.0, y, p, &status, NY, NP, 1);
    @_delay_init_batch(0.0, y, p, history, NY, NP, HS, 1);
    @_delay_apply_batch(0.0, y, p, history, NY, NP, HS, 1);
    @_rhs_batch(0.0, y, p, f0, &status, NY, NP, NS, 1);
    const double h = 1e-3;
    double t = 0.0;
    for (int step = 1; step <= 60; ++step) {
        double t1 = step * h;
        std::memcpy(x, y, sizeof x);
        for (int i = 0; i < NS; ++i) x[i] = y[i] + h * f0[i];
        @_delay_apply_batch(t1, x, p, history, NY, NP, HS, 1);
        @_rhs_batch(t1, x, p, f1, &status, NY, NP, NS, 1);
        std::memcpy(y, x, sizeof y);
        t = t1;
        @_delay_apply_batch(t, y, p, history, NY, NP, HS, 1);
        @_event_update_batch(t, y, p, &passes, NY, NP, 1);
#if @@_MANIFOLD_BLOCKS > 0
        @_project_states_batch(t, y, p, &status, NY, NP, 1);
#endif
        @_delay_apply_batch(t, y, p, history, NY, NP, HS, 1);
        bool fixed = m->rmc_fixed_valid && m->rmc_fixed_time == t
            && std::memcmp(m->rmc_fixed_y, y, sizeof y) == 0 && std::memcmp(m->rmc_fixed_p, p, sizeof p) == 0;
        std::memcpy(before, y, sizeof before);
        @_rhs_batch(t, y, p, f0, &status, NY, NP, NS, 1);
        hits += fixed;
        moved += fixed && std::memcmp(before, y, sizeof y) != 0;
        @_delay_commit_batch(t, y, p, history, NY, NP, HS, 1);
        std::printf("%d", status);
        for (int i = 0; i < NY; ++i) std::printf(" %a", y[i]);
        for (int i = 0; i < NS; ++i) std::printf(" %a", f0[i]);
        std::printf("\n");
    }
    std::printf("fixed %d moved %d\n", hits, moved);
    std::free(lanes);
    return 0;
}
"#;

/// Copies and compares records of every length up to 40 words, the source
/// and the destination each 16-byte aligned or 8 bytes past it (the cases of
/// a device's global pairs: runs of eight, the pairs after the last run, the
/// word before the first pair and after the last; a host build moves words):
/// the copy writes exactly the record's words, a NaN payload included, and
/// the comparison finds a word differing at each position in turn, in either
/// bit pattern of a value that compares equal.
const WORDS: &str = r#"
int main() {
    alignas(16) static double a[44], b[44];
    unsigned long long payload = 0x7ff4000000000123ull;
    for (int sa = 0; sa < 2; ++sa)
    for (int sb = 0; sb < 2; ++sb)
    for (int n = 0; n <= 40; ++n) {
        double *x = a + sa, *y = b + sb;
        for (int i = 0; i < 44; ++i) { a[i] = -7.0; b[i] = 1.0 + i; }
        if (n > 0) std::memcpy(&b[sb + n - 1], &payload, sizeof payload);
        @_copy_words(x, y, n);
        for (int i = 0; i < 44; ++i) {
            bool inside = i >= sa && i < sa + n;
            unsigned long long got, want;
            std::memcpy(&got, &a[i], sizeof got); std::memcpy(&want, inside ? &b[i - sa + sb] : &a[i], sizeof want);
            if (got != want || (!inside && a[i] != -7.0)) {
                std::printf("copy %d %d %d word %d\n", sa, sb, n, i); return 1;
            }
        }
        if (!@_same_bits(x, y, n)) { std::printf("same %d %d %d\n", sa, sb, n); return 1; }
        for (int k = 0; k < n; ++k) {
            unsigned long long kept, flipped;
            std::memcpy(&kept, &x[k], sizeof kept); flipped = kept ^ 1ull;
            std::memcpy(&x[k], &flipped, sizeof flipped);
            if (@_same_bits(x, y, n)) { std::printf("differs %d %d %d at %d\n", sa, sb, n, k); return 1; }
            std::memcpy(&x[k], &kept, sizeof kept);
        }
        if (n > 0) {
            x[0] = -0.0; y[0] = 0.0;
            if (@_same_bits(x, y, n)) { std::printf("signed zero %d %d %d\n", sa, sb, n); return 1; }
        }
    }
    return 0;
}
"#;

fn trajectory(model: &str, source: &str, skip: bool) -> String {
    run_unit(model, source, skip, HARNESS)
}

fn run_unit(model: &str, source: &str, skip: bool, harness: &str) -> String {
    let compiled = Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("compile {model}: {error:?}"));
    let (directory, files) = rendered_model_target(&compiled, model, "cuda-ode");
    let unit = files
        .iter()
        .find(|file| file.path.ends_with("_ode.cu"))
        .expect("cuda-ode emits its kernel");
    assert!(
        unit.content
            .contains("_refresh_recording(time, batch_y, batch_p)"),
        "{model}: the event iteration records the fixed points"
    );
    let prefix = model.to_uppercase();
    let switch = if skip {
        String::new()
    } else {
        format!("#define {prefix}_NO_FIXED_POINTS\n")
    };
    let prelude = "#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n\
        struct Dim3 { int x; };\nstatic Dim3 blockIdx, blockDim, threadIdx;\n\
        #define __global__\n#define __host__\n#define __device__\n";
    let harness = harness.replace("@@", &prefix).replace('@', model);
    let path = directory.path().join("fixed_points.cpp");
    fs::write(
        &path,
        format!("{prelude}{switch}#include {:?}\n{harness}", unit.path),
    )
    .expect("write the fixed-point harness");
    let executable = directory.path().join("cuda-ode-fixed-points");
    run_checked(
        Command::new("c++")
            .args(["-std=c++17", "-Wall", "-Wextra", "-Werror"])
            .arg(&path)
            .arg("-o")
            .arg(&executable)
            .current_dir(directory.path()),
        "compile CUDA ODE fixed-point emulation",
    );
    let output = Command::new(&executable)
        .output()
        .expect("run CUDA ODE fixed-point emulation");
    assert!(
        output.status.success(),
        "{model}: the emulation fails: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn cuda_ode_lanes_skip_a_refresh_at_a_recorded_fixed_point() {
    for (model, source) in [EVENT_DELAY, MANIFOLD] {
        let skipped = trajectory(model, source, true);
        let refreshed = trajectory(model, source, false);
        assert_eq!(
            skipped, refreshed,
            "{model}: the trajectory with fixed points"
        );
        let summary = refreshed.lines().last().expect("the fixed-point summary");
        let counts: Vec<i64> = summary
            .split_whitespace()
            .filter_map(|word| word.parse().ok())
            .collect();
        assert!(
            counts[0] > 0,
            "{model}: the accepted points are fixed points: {summary}"
        );
        assert_eq!(
            counts[1], 0,
            "{model}: a fixed point the refresh moved: {summary}"
        );
    }
}

#[test]
fn cuda_ode_fixed_point_records_copy_and_compare_every_word() {
    let (model, source) = EVENT_DELAY;
    run_unit(model, source, true, WORDS);
}
