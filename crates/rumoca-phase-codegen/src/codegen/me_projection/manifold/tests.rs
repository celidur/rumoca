//! Render-time refusals of the state-manifold view.

use rumoca_ir_solve as solve;

use super::program_reach;

const STATES: usize = 2;

#[test]
fn a_constraint_over_the_states_is_admitted() {
    let program = vec![
        solve::LinearOp::LoadY { dst: 0, index: 1 },
        solve::LinearOp::LoadSeed { dst: 1, index: 0 },
    ];
    assert_eq!(program_reach(&[program], STATES, STATES), None);
}

/// The linked kernel evaluates the retained rows on the state prefix, so a row
/// that reads an algebraic coordinate, or a direction seeded past the states,
/// has no linked meaning and refuses the view before any byte.
#[test]
fn a_constraint_past_the_states_is_refused() {
    let algebraic = vec![solve::LinearOp::LoadY {
        dst: 0,
        index: STATES,
    }];
    let range = vec![solve::LinearOp::TensorLoad {
        dst_start: 0,
        input: solve::TensorInputKind::Y,
        input_start: 1,
        count: 2,
        seed_start: None,
        lanes: 1,
    }];
    let seed = vec![solve::LinearOp::LoadSeed {
        dst: 0,
        index: STATES,
    }];
    for (program, reason) in [
        (algebraic, "reads an algebraic coordinate"),
        (range, "reads an algebraic coordinate range"),
        (seed, "seeds a coordinate outside the states"),
    ] {
        let refusal =
            program_reach(&[program], STATES, STATES).expect("a reach past the states is refused");
        assert!(refusal.contains(reason), "{refusal}");
    }
}
