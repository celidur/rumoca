//! A one-unknown dense block takes the scalar form of the dense block Newton
//! (`rmc_project_dense_scalar`), whose arithmetic is the general path's
//! operation for operation: the FMI 3 component refreshes every point to the
//! same bits with the scalar form and without it (`RMC_DENSE_SCALAR_OFF`), on
//! an implicit row and on one that starts on a vanished column, which the
//! SVD fallback and the singular-seed nudge settle.

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
        "ImplicitSquare",
        "model ImplicitSquare
  output Real x(start=1, fixed=true);
  output Real u(start=0);
equation
  der(x) = -0.5*u;
  u*u = 1 + x*x;
end ImplicitSquare;",
    ),
];

/// Each of twenty states from a fresh instance, then the same states in turn
/// on one instance: the statuses and coordinates, exactly.
const DRIVER: &str = r#"
#include <stdio.h>
static ModelInstance* fresh(void) {
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m) exit(1);
    initialize_values(m);
    if (refresh_parameters(m) != fmi3OK) { printf("parameters failed\n"); exit(1); }
    return m;
}
static void point(ModelInstance* m, int k) {
    m->y[0] = -2.0 + 0.21 * k;
    int code = refresh_algebraics(m) == fmi3OK;
    code |= (refresh_derivative_values(m) == fmi3OK) << 1;
    printf("%d", code); for (size_t i = 0; i < Y_LEN; ++i) printf(" %a", m->y[i]); printf("\n");
}
int main(void) {
    for (int k = 0; k < 20; ++k) { ModelInstance* m = fresh(); point(m, k); free(m); }
    ModelInstance* m = fresh();
    for (int k = 0; k < 20; ++k) point(m, k);
    free(m);
    return 0;
}
"#;

fn refreshed(headers: &Path, work: &Path, sources: &Path, scalar: bool) -> String {
    let driver = work.join("dense_scalar.c");
    fs::write(&driver, format!("#include \"model.c\"\n{DRIVER}")).expect("write the driver");
    let binary = work.join(if scalar { "scalar" } else { "general" });
    let mut command = Command::new("cc");
    command
        .args([
            "-std=c11",
            "-ffp-contract=off",
            "-Wall",
            "-Wextra",
            "-Werror",
        ])
        .arg(format!("-I{}", sources.display()))
        .arg(format!("-I{}", headers.display()));
    if !scalar {
        command.arg("-DRMC_DENSE_SCALAR_OFF");
    }
    checked_output(
        command
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
        "compile the dense scalar driver",
    );
    let output = checked_output(&mut Command::new(&binary), "run the dense scalar driver");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn a_one_unknown_dense_block_refreshes_as_the_general_dense_path() {
    if !conformance_prerequisites_are_available() {
        return;
    }
    let Some(headers) = super::affine_guard_harness::fmi3_header_directory() else {
        return;
    };
    for (model, source) in MODELS {
        let compiled = rumoca::Compiler::new()
            .model(model)
            .compile_str(source, &format!("{model}.mo"))
            .unwrap_or_else(|error| panic!("compile {model}: {error:?}"));
        let work = tempdir().expect("dense scalar work directory");
        let fmu = build_named_fmu(work.path(), &compiled, "fmi3", model);
        let sources = fmu.root.join("sources");
        let model_c = fs::read_to_string(sources.join("model.c")).expect("read model.c");
        assert!(
            model_c.contains("rmc_project_dense_scalar(m, b, known, update)"),
            "{model}: the component carries the scalar form"
        );
        let scalar = refreshed(&headers, work.path(), &sources, true);
        let general = refreshed(&headers, work.path(), &sources, false);
        assert_eq!(scalar.lines().count(), 40, "{model}: {scalar}");
        assert_eq!(
            scalar, general,
            "{model}: the scalar form refreshes as the general path"
        );
    }
}
