use super::{reads_any_y, require_lane_events};
use rumoca_ir_solve as solve;
use std::collections::BTreeSet;

#[test]
fn lane_events_admit_a_model_without_events() {
    assert!(require_lane_events(&solve::SolveProblem::default()).is_ok());
}

#[test]
fn lane_events_refuse_misaligned_discrete_row_modes() {
    // Checked Solve IR aligns both lists with the discrete rows; the lane
    // view refuses rather than reading the shorter one.
    let mut problem = solve::SolveProblem::default();
    problem.discrete.row_roles = vec![solve::DiscreteRowRole::Equation];
    let error = require_lane_events(&problem).expect_err("misaligned rows are refused");
    assert!(
        error.to_string().contains("not aligned"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_parameter_binding_reading_an_earlier_start_value_is_detected() {
    // The lane view runs parameter bindings before start bindings, so it
    // refuses a parameter binding that reads a start value set earlier.
    let written = BTreeSet::from([3]);
    let reads_written = [solve::LinearOp::LoadY { dst: 0, index: 3 }];
    let reads_other = [solve::LinearOp::LoadY { dst: 0, index: 4 }];
    let tensor_over_written = [solve::LinearOp::TensorLoad {
        dst_start: 0,
        input: solve::TensorInputKind::Y,
        input_start: 2,
        count: 2,
        seed_start: None,
        lanes: 1,
    }];
    let reads_parameter = [solve::LinearOp::LoadP { dst: 0, index: 3 }];
    assert!(reads_any_y(&reads_written, &written));
    assert!(!reads_any_y(&reads_other, &written));
    assert!(reads_any_y(&tensor_over_written, &written));
    assert!(!reads_any_y(&reads_parameter, &written));
}
