//! The kernel built for a device initializes as the FMI component does.
//!
//! For each initialization test model, the FMI 3 component prints its start
//! values, then initializes from each start guess through `rmc_initialize`
//! (the initialization `fmi3ExitInitializationMode` runs) and prints its
//! status, coordinate and parameters. The device unit of the same model
//! (`crate::me_projection_device_initialization`), compiled by a host C++
//! compiler with the CUDA qualifiers defined away, initializes one lane per
//! guess from those start values. Every line must be the same, bit for bit.

use super::*;
use crate::me_projection_device::device_unit;
use crate::me_projection_device_initialization::{
    GUESSES, TOLERANCE, initialization_harness, lowered, models,
};

fn fmi_driver(initial_flag: Option<usize>) -> String {
    let flag = initial_flag.map_or_else(String::new, |index| format!("m->p[{index}] = 1.0;"));
    format!(
        r#"
#include <stdio.h>
static void rmc_print(const ModelInstance* m, int ok) {{
    printf("%d", ok);
    for (size_t i = 0; i < Y_LEN; ++i) printf(" %.17g", m->y[i]);
    printf(" |");
    for (size_t i = 0; i < P_LEN; ++i) printf(" %.17g", m->p[i]);
    printf("\n");
}}
int main(void) {{
    ModelInstance* m = calloc(1, sizeof *m);
    if (!m) return 1;
    initialize_values(m);
    double y0[Y_LEN], p0[P_LEN];
    memcpy(y0, m->y, sizeof y0); memcpy(p0, m->p, sizeof p0);
    rmc_print(m, 2);
    for (int k = 0; k < {GUESSES}; ++k) {{
        for (size_t i = 0; i < Y_LEN; ++i) m->y[i] = y0[i] * (1.0 + 0.1 * k);
        memcpy(m->p, p0, sizeof p0);
        m->parameters_dirty = false;
        {flag}
        rmc_print(m, rmc_initialize(m, {TOLERANCE:?}) == fmi3OK);
    }}
    free(m);
    return 0;
}}
"#
    )
}

fn values(line: &str) -> (Vec<f64>, Vec<f64>) {
    let (_, rest) = line.split_once(' ').expect("a start line");
    let (y, p) = rest.split_once('|').expect("a start line holds y and p");
    let parse = |text: &str| {
        text.split_whitespace()
            .map(|value| value.parse::<f64>().expect("a printed double"))
            .collect::<Vec<_>>()
    };
    (parse(y), parse(p))
}

fn compile_and_run(mut command: Command, binary: &Path, label: &str) -> String {
    checked_output(command.arg("-o").arg(binary), &format!("compile {label}"));
    let output = checked_output(&mut Command::new(binary), &format!("run {label}"));
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn a_device_build_initializes_every_lane_as_the_fmi_component() {
    if !conformance_prerequisites_are_available() {
        return;
    }
    let headers = standard_roots().1.root.join("headers");
    for (compiled, model) in models() {
        let work = tempdir().expect("device initialization work directory");
        let flag = lowered(&compiled)
            .problem
            .solve_layout
            .initial_event_parameter_index;
        let fmu = build_named_fmu(work.path(), &compiled, "fmi3", model);
        let sources = fmu.root.join("sources");
        let driver = work.path().join("fmi_init.c");
        fs::write(
            &driver,
            format!("#include \"model.c\"\n{}", fmi_driver(flag)),
        )
        .expect("write the FMI driver");
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
            .arg(format!("-I{}", headers.display()))
            .arg(&driver)
            .args(
                [
                    "rmc_rows.c",
                    "rmc_jacobian.c",
                    "rmc_isolators.c",
                    "rmc_assign.c",
                    "rmc_functions.c",
                ]
                .iter()
                .map(|unit| sources.join(unit)),
            )
            .arg("-lm");
        let fmi = compile_and_run(command, &work.path().join("fmi_init"), "the FMI driver");
        let (start, lanes) = fmi
            .split_once('\n')
            .expect("the FMI driver prints its start");
        assert_eq!(lanes.lines().count(), GUESSES, "{model}: {fmi}");
        assert!(
            lanes.lines().all(|line| line.starts_with('1')),
            "{model}: the FMI component initializes every guess\n{fmi}"
        );
        let (y0, p0) = values(start);
        let harness = work.path().join("device_init.cpp");
        fs::write(
            &harness,
            initialization_harness(&device_unit(&compiled), &y0, &p0, flag),
        )
        .expect("write the device harness");
        let mut command = Command::new("c++");
        command
            .args([
                "-std=c++17",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-unused-function",
            ])
            .args(["-D__device__=", "-D__host__="])
            .arg(&harness)
            .arg("-lm");
        let device = compile_and_run(
            command,
            &work.path().join("device_init"),
            "the device lanes",
        );
        assert_eq!(
            device, lanes,
            "{model}: the device lanes differ from the FMI component"
        );
    }
}
