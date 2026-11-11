//! MLS 3.7 sections 10.1 and 12.4.1: an output dimension such as
//! `size(v, 1) + extra` is evaluated with the call's arguments bound to the
//! formal inputs, so a `[:]` binding takes its shape from the call even when
//! the array argument's contents are unknown at translation.

use rumoca::Compiler;
use rumoca_core::VarName;

const SOURCE: &str = "
package P
  function pad
    input Real v[:];
    input Integer extra = 1;
    output Real y[size(v, 1) + extra];
  algorithm
    y := cat(1, v, fill(0.0, extra));
  end pad;
  function first
    input Real v[:];
    input Integer extra;
    output Real y[size(v, 1)];
  algorithm
    y := v;
  end first;
  model M
    parameter Integer n = 2;
    parameter Integer m(fixed = false);
    parameter Real p[3] = {1, 2, 3};
    parameter Real q[3](each fixed = false);
    parameter Real known[:] = pad(p);
    parameter Real unknown[:] = pad(q);
    parameter Real named[:] = pad(extra = n, v = q);
    parameter Real scaled[:] = pad(2 * p);
    parameter Real unread[:] = first(q, m);
  initial equation
    m = 1;
    q = {1, 2, 3};
  end M;
  model Matrix
    parameter Real r[3, 2](each fixed = false);
    parameter Real z[:] = pad(r);
  initial equation
    r = fill(1, 3, 2);
  end Matrix;
end P;
";

fn flat(model: &str) -> Result<rumoca_ir_flat::Model, String> {
    Compiler::new()
        .model(model)
        .compile_str_flat(SOURCE, "P.mo")
        .map_err(|error| format!("{error:#}"))
}

#[test]
fn output_dimensions_bind_the_shape_of_each_array_argument() {
    let model = flat("P.M").expect("every output shape follows from the call");
    for (name, extent) in [
        ("known", 4),
        ("unknown", 4),
        ("named", 5),
        ("scaled", 4),
        ("unread", 3),
    ] {
        let variable = model
            .variables
            .get(&VarName::new(name))
            .unwrap_or_else(|| panic!("flat variable {name}"));
        assert_eq!(variable.dims, vec![extent], "{name}");
    }
}

#[test]
fn an_array_argument_of_another_rank_gives_no_output_shape() {
    let Err(error) = flat("P.Matrix") else {
        panic!("a matrix is not a vector argument");
    };
    assert!(error.contains("unresolved component dimension"), "{error}");
}
