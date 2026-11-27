//! A certified dense or torn block applies the correction it computed once
//! its residual and that correction meet tolerance (SPEC_0044 ME-PROJ-001), so
//! the generated component's refresh follows a change of a state inside the
//! tolerance band.
//!
//! The FMI 3 component of `ImplicitScalar` (one implicit row without an
//! isolator, which takes the dense block Newton) and of `TornLoop` (a torn
//! block with causal steps) refreshes at a state, then from there at the state
//! moved by a step: a step a few hundredths of the refresh tolerance band, and
//! a step far outside it. Every algebraic coordinate must move by the same
//! sensitivity for both steps.

use super::*;

const MODELS: [(&str, &str); 2] = [
    (
        "ImplicitScalar",
        "model ImplicitScalar
  output Real x(start=1, fixed=true);
  output Real u(start=0.5);
equation
  der(x) = -u;
  u + 0.5*sin(u) + 0.1*u*u*u = x;
end ImplicitScalar;",
    ),
    (
        "TornLoop",
        "model TornLoop
  output Real x(start=1, fixed=true);
  output Real a(start=1);
  output Real b(start=1);
  output Real c(start=1);
equation
  der(x) = -0.5*a;
  a = x + 0.2*sin(c);
  b = a*a + 0.1*a;
  c = 1 + b - 0.3*cos(b) + 0.1*sin(c);
end TornLoop;",
    ),
];

/// Prints, for the small and then the large step, the change of every
/// algebraic slot of `y` after the state `y[0]` moved, divided by the step.
const DRIVER: &str = r#"
#include <stdio.h>
static ModelInstance* fresh(void) {
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m) exit(1);
    initialize_values(m);
    if (refresh_parameters(m) != fmi3OK) { printf("parameters failed\n"); exit(1); }
    return m;
}
static void at(ModelInstance* m, double state) {
    m->y[0] = state;
    if (refresh_algebraics(m) != fmi3OK || refresh_derivative_values(m) != fmi3OK) { printf("refresh failed\n"); exit(1); }
}
int main(void) {
    const double steps[2] = { 1.0e-11, 1.0e-6 };
    for (int k = 0; k < 2; ++k) {
        ModelInstance* m = fresh();
        at(m, 1.0);
        double before[Y_LEN]; memcpy(before, m->y, sizeof before);
        at(m, 1.0 + steps[k]);
        printf("sensitivity"); for (size_t i = 1; i < Y_LEN; ++i) printf(" %.17g", (m->y[i] - before[i]) / steps[k]); printf("\n");
        free(m);
    }
    return 0;
}
"#;

/// The sensitivities the driver prints for the small and the large step.
fn sensitivities(headers: &Path, model: &str, source: &str) -> (Vec<f64>, Vec<f64>) {
    let compiled = rumoca::Compiler::new()
        .model(model)
        .compile_str(source, &format!("{model}.mo"))
        .unwrap_or_else(|error| panic!("compile {model}: {error:?}"));
    let work = tempdir().expect("refresh sensitivity work directory");
    let fmu = build_named_fmu(work.path(), &compiled, "fmi3", model);
    let sources = fmu.root.join("sources");
    let driver = work.path().join("sensitivity.c");
    fs::write(&driver, format!("#include \"model.c\"\n{DRIVER}")).expect("write the driver");
    let binary = work.path().join("sensitivity");
    checked_output(
        Command::new("cc")
            .args([
                "-std=c11",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
            ])
            .arg(format!("-I{}", sources.display()))
            .arg(format!("-I{}", headers.display()))
            .arg(&driver)
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
        &format!("compile the {model} sensitivity driver"),
    );
    let output = checked_output(
        &mut Command::new(&binary),
        &format!("run the {model} sensitivity driver"),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut rows = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("sensitivity "))
        .map(|line| {
            line.split(' ')
                .map(|value| value.parse().expect("a sensitivity"))
                .collect::<Vec<f64>>()
        });
    match (rows.next(), rows.next()) {
        (Some(small), Some(large)) => (small, large),
        _ => panic!("{model}: the driver prints two sensitivity rows: {stdout}"),
    }
}

#[test]
fn the_generated_refresh_follows_a_state_change_inside_the_tolerance_band() {
    if !conformance_prerequisites_are_available() {
        return;
    }
    let Some(headers) = super::affine_guard_harness::fmi3_header_directory() else {
        return;
    };
    for (model, source) in MODELS {
        let (small, large) = sensitivities(&headers, model, source);
        let moved: Vec<usize> = (0..large.len())
            .filter(|&i| large[i].abs() > 1.0e-3)
            .collect();
        assert_eq!(
            moved.len(),
            large.len(),
            "{model}: every algebraic coordinate depends on the state: {large:?}"
        );
        for i in moved {
            assert!(
                (small[i] - large[i]).abs() <= 1.0e-3 * large[i].abs(),
                "{model} slot {}: sensitivity {} inside the band, {} outside it",
                i + 1,
                small[i],
                large[i]
            );
        }
    }
}
