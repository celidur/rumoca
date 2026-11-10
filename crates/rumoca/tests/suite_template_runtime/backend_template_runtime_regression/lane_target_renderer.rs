//! A Solve target that declares `algebraic_projection` renders against the
//! lowering its capability gate checked, whatever its name.

use rumoca::Compiler;
use std::fs;

#[test]
fn a_custom_lane_target_renders_the_lowering_its_gate_checked() {
    let compiled = Compiler::new()
        .model("CoupledPair")
        .compile_str(
            r#"
model CoupledPair
  Real x(start = 1, fixed = true);
  Real a;
  Real b;
equation
  der(x) = -x + a;
  a + b = 4 + x;
  a * a - b = 2 * x;
end CoupledPair;
"#,
            "CoupledPair.mo",
        )
        .expect("the compiler accepts the model");
    let directory = tempfile::tempdir().expect("temporary target directory");
    let target = directory.path().join("lane-probe");
    fs::create_dir_all(&target).expect("create target directory");
    // The probe reads the start values the per-lane renderer exposes; the
    // generic Solve renderer has none.
    fs::write(
        target.join("probe.txt.jinja"),
        "{{ solve_blocks.lanes.starts.y | length }}\n",
    )
    .expect("write probe template");
    fs::write(
        target.join("target.toml"),
        r#"version = 1
ir = "solve"
name = "lane-probe"
description = "per-lane Solve target with a scalar fallback"

[capabilities]
continuous_states = true
residual_equations = false
exact_algebraic_assignments = true
algebraic_projection = true
parameter_initialization = true
scalar_fallback = true
external_functions = false
external_tables = false
random = false
initialization = true
events = true
runtime_events = true
clocks = false
dynamic_ranges = false
dynamic_derivative_subscripts = false
forward_ad = false
reverse_ad = false
dynamic_control_flow = false
host_callbacks = false

[capabilities.tensor]
matmul = "scalar"
linsolve = "unsupported"
elementwise = "scalar"
stencil = "scalar"
reductions = "scalar"
layout = "row-major"
supports_dynamic_shapes = false
sparse = false
dtypes = ["f64"]

[[files]]
path = "probe.txt"
template = "probe.txt.jinja"
"#,
    )
    .expect("write target manifest");
    let files = rumoca::render_target_files(
        &compiled,
        "CoupledPair",
        target.to_str().expect("utf-8 target path"),
        None,
    )
    .expect("the per-lane target renders the coupled model");
    assert_eq!(
        files[0].content.trim(),
        "3",
        "the renderer must expose the start values of x, a and b"
    );
}
