//! A coupled affine block keeps its factorization from one call to the next
//! (SPEC_0044 ME-PROJ-001): the retained factors are bound to the bits of the
//! block Jacobian and both scale vectors, and a reuse is the formation it
//! replaces, bit for bit.
//!
//! The FMI 3 component of each model refreshes a sequence of state points on
//! one instance, which reuses its factors while the Jacobian keeps its bits
//! and forms them again when it changes, and on a fresh instance per point,
//! which forms them at every point: both give the same statuses and
//! coordinates. The listed solve of a retained dense factor is checked
//! against `rmc_lu_solve` on its own.

use super::*;

/// A resistive chain of `n` nodes whose branch conductance is `conductance`
/// (`{i}` stands for the node): one coupled affine block. A literal
/// conductance gives it a reduced elimination, which it retains; a
/// conductance that depends on the state leaves it without an elimination
/// layout, and it retains its dense LU.
fn chain(model: &str, n: usize, conductance: &str) -> String {
    let g = |i: usize| conductance.replace("{i}", &i.to_string());
    let mut source = format!(
        "model {model}\n  parameter Real Rm = 10;\n  parameter Real C = 1e-3;\n  \
         Real vC[{n}](start={{0.1*i for i in 1:{n}}}, each fixed=true);\n  Real v[{n}];\nequation\n"
    );
    source.push_str(&format!(
        "  (sin(time) - v[1]) / 0.1 + (v[2] - v[1]) * {} = (v[1] - vC[1]) / Rm;\n",
        g(1)
    ));
    for i in 2..n {
        source.push_str(&format!(
            "  (v[{}] - v[{i}]) * {} + (v[{}] - v[{i}]) * {} = (v[{i}] - vC[{i}]) / Rm;\n",
            i - 1,
            g(i),
            i + 1,
            g(i)
        ));
    }
    source.push_str(&format!(
        "  (v[{}] - v[{n}]) * {} = (v[{n}] - vC[{n}]) / Rm;\n  C * der(vC) = (v - vC) / Rm;\nend {model};\n",
        n - 1,
        g(n)
    ));
    source
}

fn compile(model: &str, source: &str) -> rumoca::CompilationResult {
    rumoca::Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("compile {model}: {error:?}"))
}

/// The prerequisites: the FMI conformance set with its FMI 3 headers.
fn fmi3_headers() -> Option<PathBuf> {
    if !conformance_prerequisites_are_available() {
        return None;
    }
    super::affine_guard_harness::fmi3_header_directory()
}

/// Compile `driver`, which includes the FMI 3 `model.c` of `compiled`, with
/// the component's other units and the `extra` compiler flags, run it, and
/// return what it prints.
fn run_driver(
    work: &Path,
    headers: &Path,
    compiled: &rumoca::CompilationResult,
    model: &str,
    driver: &str,
    extra: &[&str],
) -> String {
    let fmu = build_named_fmu(work, compiled, "fmi3", model);
    let sources = fmu.root.join("sources");
    let source = work.join("retention_driver.c");
    fs::write(&source, driver).expect("write the retention driver");
    let binary = work.join("retention_driver");
    checked_output(
        Command::new("cc")
            .args([
                "-std=c11",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
            ])
            .args(extra)
            .arg(format!("-I{}", sources.display()))
            .arg(format!("-I{}", headers.display()))
            .arg(&source)
            .args(
                [
                    "rmc_assign.c",
                    "rmc_rows.c",
                    "rmc_jacobian.c",
                    "rmc_isolators.c",
                    "rmc_functions.c",
                ]
                .map(|unit| sources.join(unit)),
            )
            .arg("-lm")
            .arg("-o")
            .arg(&binary),
        &format!("compile the {model} retention driver"),
    );
    let output = checked_output(
        &mut Command::new(&binary),
        &format!("run the {model} retention driver"),
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The points one instance refreshes in turn: a repeated point keeps the
/// Jacobian of the point before it, and a point visited again after others
/// meets the factors of another Jacobian. The instance then saves its FMU
/// state, refreshes the last point, restores the state and refreshes the
/// last point again.
const SEQUENCE: [u32; 8] = [1, 2, 2, 3, 1, 4, 5, 5];

/// The instance that keeps its factors, as `fmi3InstantiateModelExchange`
/// allocates them, refreshes the points of `SEQUENCE`, with its FMU state
/// saved before the next to last and restored before the last; then a fresh
/// instance without retained factors forms them at each point. Before a
/// point, the driver sets to 2 each bound key flag, each ready dense flag,
/// and the last pivot slot of each elimination, which a preparation clears
/// and a solve never reads: a key bound again or a dense factor formed writes
/// 1 to its flag, a prepared elimination 0 to that slot. A line of the first
/// part is followed by `F` when the point bound a key again or formed a
/// factor, `R` when it did neither and left the retained doubles as they
/// were, `?` when it changed those doubles without forming anything, and `X`
/// when the FMU state copied or replaced the factors instead of their
/// address; every line gives the statuses and the coordinate.
const SEQUENCE_DRIVER: &str = r#"
#include <stdio.h>
static ModelInstance* fresh(bool keep) {
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m || (keep && !rmc_keep_allocate(m))) exit(1);
    initialize_values(m);
    if (refresh_parameters(m) != fmi3OK) { printf("parameters failed\n"); exit(1); }
    return m;
}
static void point(ModelInstance* m, const double* start, unsigned k) {
    memcpy(m->y, start, sizeof m->y);
    for (size_t i = 0; i < RMC_STATES; ++i) m->y[i] = start[i] * (1.0 + 0.05 * k) + 0.01 * (double)(i + 1);
    int code = refresh_algebraics(m) == fmi3OK;
    code |= (refresh_derivative_values(m) == fmi3OK) << 1;
    printf("%d", code); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
}
static void mark_flags(ModelInstance* m) {
    for (size_t b = 0; b < sizeof rmc_block_desc / sizeof rmc_block_desc[0]; ++b) {
        if (!rmc_block_desc[b].keep) continue;
        size_t* flags = m->rmc_keep_s + rmc_block_desc[b].keep_s;
        if (flags[RMC_KEEP_KEY]) flags[RMC_KEEP_KEY] = 2;
        if (flags[RMC_KEEP_DENSE]) flags[RMC_KEEP_DENSE] = 2;
        if (rmc_block_desc[b].elimination) flags[RMC_KEEP_FLAGS + rmc_block_desc[b].elim_capacity] = 2;
    }
}
static bool formed(const ModelInstance* m) {
    bool any = false;
    for (size_t b = 0; b < sizeof rmc_block_desc / sizeof rmc_block_desc[0]; ++b) {
        if (!rmc_block_desc[b].keep) continue;
        const size_t* flags = m->rmc_keep_s + rmc_block_desc[b].keep_s;
        any = any || flags[RMC_KEEP_KEY] != 2 || flags[RMC_KEEP_DENSE] == 1
            || (rmc_block_desc[b].elimination && flags[RMC_KEEP_FLAGS + rmc_block_desc[b].elim_capacity] != 2);
    }
    return any;
}
int main(void) {
    static const unsigned sequence[] = { RMC_SEQUENCE };
    enum { POINTS = sizeof sequence / sizeof sequence[0] };
    ModelInstance* m = fresh(true);
    double* own = m->rmc_keep_d;
    double start[Y_LEN]; memcpy(start, m->y, sizeof start);
    static double kept[RMC_KEEP_DOUBLES];
    fmi3FMUState state = NULL;
    for (size_t p = 0; p < POINTS; ++p) {
        bool shared = true;
        if (p == POINTS - 2) shared = fmi3GetFMUState(m, &state) == fmi3OK && ((ModelInstance*)state)->rmc_keep_d == own;
        if (p == POINTS - 1) shared = fmi3SetFMUState(m, state) == fmi3OK && m->rmc_keep_d == own;
        memcpy(kept, m->rmc_keep_d, sizeof kept); mark_flags(m);
        point(m, start, sequence[p]);
        printf("%c\n", !shared ? 'X' : formed(m) ? 'F' : memcmp(kept, m->rmc_keep_d, sizeof kept) != 0 ? '?' : 'R');
    }
    fmi3FreeFMUState(m, &state);
    rmc_keep_release(m); free(m);
    printf("fresh\n");
    for (size_t p = 0; p < POINTS; ++p) { m = fresh(false); point(m, start, sequence[p]); free(m); }
    return 0;
}
"#;

/// The kept instance's lines with their formation marks, and the fresh ones.
fn sequence_points(
    headers: &Path,
    compiled: &rumoca::CompilationResult,
    model: &str,
) -> (Vec<(String, char)>, Vec<String>) {
    let work = tempdir().expect("retention work directory");
    let states = rumoca_sim::lower_fmi_c_export_component(&compiled.dae)
        .expect("lower the FMI C component")
        .problem()
        .solve_layout
        .state_scalar_count();
    let sequence = SEQUENCE.map(|k| k.to_string()).join(", ");
    let driver = format!(
        "#include \"model.c\"\n#define RMC_STATES {states}\n#define RMC_SEQUENCE {sequence}\n{SEQUENCE_DRIVER}"
    );
    let output = run_driver(work.path(), headers, compiled, model, &driver, &[]);
    let (kept, fresh) = output
        .split_once("fresh\n")
        .unwrap_or_else(|| panic!("{model}: the driver prints both parts: {output}"));
    let kept: Vec<&str> = kept.lines().collect();
    let kept = kept
        .chunks(2)
        .map(|pair| {
            let mark = pair[1].chars().next().expect("a formation mark");
            (pair[0].to_owned(), mark)
        })
        .collect();
    (kept, fresh.lines().map(str::to_owned).collect())
}

#[test]
fn a_retained_factorization_solves_as_a_fresh_one() {
    let Some(headers) = fmi3_headers() else {
        return;
    };
    // Each model with the marks its kept instance must show: the constant
    // Jacobian of the literal chain prepares its elimination at the first
    // point and reuses it at every later one; a Jacobian that moves with the
    // state, as in the dense chain and the promoting elimination, forms the
    // factors again at every new point and reuses them at a repeated one,
    // the restored FMU state included, whose factors stay the instance's own.
    let constant = "RetainedConstantChain";
    let varying = "RetainedVaryingChain";
    let cases = [
        (
            compile(constant, &chain(constant, 12, "1.0")),
            constant,
            "FRRRRRRR",
        ),
        (
            compile(varying, &chain(varying, 12, "(1 + vC[{i}]^2)")),
            varying,
            "FFRFFFFR",
        ),
        (
            super::affine_promotion::compile(),
            super::affine_promotion::MODEL,
            "FFRFFFFR",
        ),
    ];
    for (compiled, model, marks) in cases {
        let (kept, fresh) = sequence_points(&headers, &compiled, model);
        assert_eq!(kept.len(), SEQUENCE.len(), "{model}: {kept:?}");
        let (kept, seen): (Vec<String>, String) = kept.into_iter().unzip();
        assert_eq!(
            kept, fresh,
            "{model}: the instance that keeps its factors refreshes as fresh instances"
        );
        assert_eq!(seen, marks, "{model}: the points that formed the factors");
    }
}

/// The listed solve of a retained factor is `rmc_lu_solve` bit for bit, the
/// sign of every zero included: on a factor with zero entries in L and U,
/// every right-hand side drawn from -0.0, +0.0, two nonzero values, an
/// infinity and a NaN solves to the same bits both ways.
#[test]
fn the_listed_lu_solve_keeps_the_sign_of_each_zero() {
    let Some(headers) = fmi3_headers() else {
        return;
    };
    let model = "RetainedConstantChain";
    let compiled = compile(model, &chain(model, 12, "1.0"));
    let driver = r#"#include "model.c"
#include <stdio.h>
int main(void) {
    enum { N = 5, VALUES = 6 };
    const double a[N * N] = {
        0.5, 2.0, 0.0, 0.0, 0.0,
        3.0, 1.0, -1.0, 0.0, 0.0,
        0.0, -2.0, 4.0, 1.5, 0.0,
        0.0, 0.0, 1.0, -3.0, 2.0,
        0.0, 0.0, 0.0, 0.5, 1.0 };
    double lu[N * N]; size_t ipiv[N], columns[N], lists[2 * (N + 1) + N * (N - 1)], zeros[N];
    for (size_t i = 0; i < N * N; ++i) lu[i] = a[i];
    rmc_lu(lu, N, ipiv, columns);
    rmc_lu_lists(lu, N, lists);
    const double values[VALUES] = { -0.0, 0.0, -1.5, 2.0, INFINITY, NAN };
    unsigned cases = 1; for (size_t i = 0; i < N; ++i) cases *= VALUES;
    int differ = 0;
    for (unsigned c = 0; c < cases; ++c) {
        double dense[N], listed[N]; unsigned k = c;
        for (size_t i = 0; i < N; ++i) { dense[i] = listed[i] = values[k % VALUES]; k /= VALUES; }
        bool solved = rmc_lu_solve(lu, N, ipiv, dense);
        if (solved != rmc_lu_solve_lists(lu, N, ipiv, lists, listed, zeros) || memcmp(dense, listed, sizeof dense) != 0) ++differ;
    }
    printf("%d\n", differ);
    return 0;
}
"#;
    let work = tempdir().expect("listed solve work directory");
    let output = run_driver(work.path(), &headers, &compiled, model, driver, &[]);
    assert_eq!(output.trim(), "0");
}

/// A CUDA lane of the FMI 3 co-simulation step (`rmc_cuda_step.c.inc`) runs
/// on a copy of a host instance, whose retained factors are host storage:
/// the lane steps without them and hands the host its addresses back. Here
/// the entry runs on the host with the lane index stubbed and the device
/// address space ignored. Over three steps of two lanes from two initialized
/// instances, the lane writes none of the host's retained storage, its copy
/// keeps the host's addresses, and it reaches the coordinate the host
/// instance reaches with its retained factors, bit for bit.
#[test]
fn a_cuda_lane_steps_without_the_host_retained_factors() {
    let Some(headers) = fmi3_headers() else {
        return;
    };
    let driver = r#"
#include <stdio.h>
static unsigned rmc_test_lane;
static unsigned __nvvm_read_ptx_sreg_ctaid_x(void) { return 0; }
static unsigned __nvvm_read_ptx_sreg_ntid_x(void) { return 1; }
static unsigned __nvvm_read_ptx_sreg_tid_x(void) { return rmc_test_lane; }
#include "rmc_cuda_step.c.inc"
int main(void) {
    enum { N = 2, STEPS = 3 };
    static ModelInstance lanes[N];
    ModelInstance* host[N];
    for (unsigned l = 0; l < N; ++l) {
        host[l] = fmi3InstantiateCoSimulation("lane", rmc_fmi3_token, NULL, false, false, false, false, NULL, 0, NULL, NULL, NULL);
        if (!host[l] || !host[l]->rmc_keep_d || fmi3EnterInitializationMode(host[l], false, 0.0, 0.0, false, 0.0) != fmi3OK) return 2;
        host[l]->y[0] += 0.05 * l;
        if (fmi3ExitInitializationMode(host[l]) != fmi3OK) return 2;
    }
    static double kept_d[RMC_KEEP_DOUBLES]; static size_t kept_s[RMC_KEEP_SIZES];
    unsigned touched = 0, moved = 0, differ = 0, failed = 0;
    for (unsigned s = 0; s < STEPS; ++s) {
        const double current = 0.01 * s, dt = 0.01; int statuses[N];
        for (unsigned l = 0; l < N; ++l) lanes[l] = *host[l];
        for (unsigned l = 0; l < N; ++l) {
            memcpy(kept_d, host[l]->rmc_keep_d, sizeof kept_d); memcpy(kept_s, host[l]->rmc_keep_s, sizeof kept_s);
            rmc_test_lane = l; rmc_fmi3_step(lanes, current, dt, statuses, N);
            touched += memcmp(kept_d, host[l]->rmc_keep_d, sizeof kept_d) != 0 || memcmp(kept_s, host[l]->rmc_keep_s, sizeof kept_s) != 0;
            moved += lanes[l].rmc_keep_d != host[l]->rmc_keep_d || lanes[l].rmc_keep_s != host[l]->rmc_keep_s;
            bool event = false, terminate = false, early = false; double last = current;
            failed += statuses[l] != fmi3OK || fmi3DoStep(host[l], current, dt, true, &event, &terminate, &early, &last) != fmi3OK;
            differ += memcmp(lanes[l].y, host[l]->y, sizeof lanes[l].y) != 0;
        }
    }
    printf("%u %u %u %u\n", touched, moved, differ, failed);
    for (unsigned l = 0; l < N; ++l) fmi3FreeInstance(host[l]);
    return 0;
}
"#;
    for (model, conductance) in [
        ("RetainedConstantChain", "1.0"),
        ("RetainedVaryingChain", "(1 + vC[{i}]^2)"),
    ] {
        let compiled = compile(model, &chain(model, 12, conductance));
        let work = tempdir().expect("CUDA lane work directory");
        let output = run_driver(
            work.path(),
            &headers,
            &compiled,
            model,
            driver,
            &["-Wno-attributes"],
        );
        assert_eq!(
            output.trim(),
            "0 0 0 0",
            "{model}: touched host storage, moved addresses, differing lanes, failed steps"
        );
    }
}
