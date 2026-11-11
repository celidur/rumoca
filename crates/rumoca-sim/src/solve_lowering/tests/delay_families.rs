//! A delay inside a compact state-derivative family must reach Solve with a
//! concrete source and timing per occurrence (MLS §3.7.2).

use super::super::entry::lower_dae_for_simulation;
use super::tangent_jacobian::compile_with_roots;
use crate::SimOptions;

#[test]
fn state_derivative_delay_family_compiles_and_lowers() {
    let dae = compile_with_roots(
        "model M\n  Real x[4](each start = 1);\nequation\n  for i in 1:4 loop\n    \
         der(x[i]) = delay(x[i], 0.1);\n  end for;\nend M;",
        "M",
        &[],
    );
    lower_dae_for_simulation(&dae, &SimOptions::default()).expect("delay family lowers");
}
