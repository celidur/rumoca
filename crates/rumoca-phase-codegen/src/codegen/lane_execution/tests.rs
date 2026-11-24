use super::{dense_rows_value, events_value, ordered_bindings, reads_slot, require_lane_events};
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
fn lane_events_refuse_an_error_level_assertion() {
    let mut problem = solve::SolveProblem::default();
    problem.events.actions = vec![solve::SolveEventAction {
        kind: solve::SolveEventActionKind::Assert,
        message: solve::SolveEventMessage::default(),
        span: fixture_span(),
        origin: String::new(),
        clock_owner: None,
        assertion_projection: None,
    }];
    let error = require_lane_events(&problem).expect_err("a lane does not check assertions");
    assert!(
        error.to_string().contains("assert()"),
        "unexpected refusal: {error}"
    );
}

fn fixture_span() -> rumoca_core::Span {
    rumoca_core::Span::from_offsets(
        rumoca_core::SourceId::from_source_name("lane_execution_fixture.mo"),
        1,
        2,
    )
}

fn p_slot(index: usize) -> solve::ScalarSlot {
    solve::ScalarSlot::P {
        index,
        byte_offset: 8 * index,
    }
}

/// `p[target] := factor * p[source] + offset` as one program.
fn affine_binding(source: usize, factor: f64, offset: f64) -> Vec<solve::LinearOp> {
    vec![
        solve::LinearOp::LoadP {
            dst: 0,
            index: source,
        },
        solve::LinearOp::Const {
            dst: 1,
            value: factor,
        },
        solve::LinearOp::Binary {
            dst: 2,
            op: solve::BinaryOp::Mul,
            lhs: 0,
            rhs: 1,
        },
        solve::LinearOp::Const {
            dst: 3,
            value: offset,
        },
        solve::LinearOp::Binary {
            dst: 4,
            op: solve::BinaryOp::Add,
            lhs: 2,
            rhs: 3,
        },
        solve::LinearOp::StoreOutput { src: 4 },
    ]
}

/// Bindings whose `k`-th program stores at `rows[k]`, the targets `targets`.
fn initialization(
    programs: Vec<Vec<solve::LinearOp>>,
    rows: Vec<usize>,
    targets: Vec<solve::ScalarSlot>,
) -> solve::InitializationSolveSystem {
    let spans = vec![fixture_span(); programs.len()];
    solve::InitializationSolveSystem::construct(solve::InitializationSystemInput {
        update_rhs: solve::ScalarProgramBlock::with_output_indices(programs, spans, rows)
            .expect("one output per program"),
        update_targets: targets,
        ..Default::default()
    })
    .expect("each binding owns its own target")
}

#[test]
fn bindings_run_after_the_bindings_they_read() {
    // p1 := 2 * p2 is issued before p2 := p0 + 1: one pass in issued order
    // would read the start value of p2, the runtime's sweeps its binding.
    let init = initialization(
        vec![affine_binding(2, 2.0, 0.0), affine_binding(0, 1.0, 1.0)],
        vec![0, 1],
        vec![p_slot(1), p_slot(2)],
    );
    let order = ordered_bindings(&init, "initialization").expect("the bindings are acyclic");
    let outputs = order
        .iter()
        .map(|binding| binding.outputs.clone())
        .collect::<Vec<_>>();
    assert_eq!(outputs, vec![vec![2], vec![1]]);
}

#[test]
fn a_binding_stores_at_the_target_its_output_index_names() {
    // The first program stores row 1 (p2), the second row 0 (p1).
    let init = initialization(
        vec![affine_binding(0, 1.0, 1.0), affine_binding(0, 1.0, 2.0)],
        vec![1, 0],
        vec![p_slot(1), p_slot(2)],
    );
    let order = ordered_bindings(&init, "initialization").expect("the bindings are acyclic");
    let outputs = order
        .iter()
        .map(|binding| binding.outputs.clone())
        .collect::<Vec<_>>();
    assert_eq!(outputs, vec![vec![2], vec![1]]);
}

#[test]
fn bindings_reading_each_other_in_a_cycle_are_refused() {
    let init = initialization(
        vec![affine_binding(2, 1.0, 0.0), affine_binding(1, 1.0, 0.0)],
        vec![0, 1],
        vec![p_slot(1), p_slot(2)],
    );
    let error = ordered_bindings(&init, "initialization").expect_err("a cycle is refused");
    assert!(
        error.to_string().contains("cycle"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_row_block_keeps_the_row_of_each_output() {
    let block = solve::ScalarProgramBlock::with_output_indices(
        vec![affine_binding(0, 1.0, 0.0), affine_binding(0, 2.0, 0.0)],
        vec![fixture_span(); 2],
        vec![1, 0],
    )
    .expect("two rows");
    let (value, count) = dense_rows_value(&block, None, "events").expect("scalar rows");
    assert_eq!(count, 2);
    let json = serde_json::to_value(&value).expect("the view serializes");
    let stored = json["scalar_plan"]["programs"]
        .as_array()
        .expect("programs")
        .iter()
        .map(|program| {
            program["ops"]
                .as_array()
                .expect("ops")
                .iter()
                .find(|op| op["kind"] == "StoreOutput")
                .expect("a store")["output_index"]
                .clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(stored, vec![serde_json::json!(1), serde_json::json!(0)]);
}

#[test]
fn a_root_without_a_relation_memory_is_refused() {
    // A plain crossing is never re-encoded by the event iteration, and no
    // lane kernel would report it.
    let mut problem = solve::SolveProblem::default();
    problem.events.root_conditions = solve::ScalarProgramBlock::with_output_indices(
        vec![affine_binding(0, 1.0, -1.0)],
        vec![fixture_span()],
        vec![0],
    )
    .expect("one root");
    problem.events.root_relation_memory_targets = vec![None];
    problem.events.root_zero_domains = vec![solve::RootZeroDomain::Previous];
    let error = events_value(&problem, None).expect_err("the root is refused");
    assert!(
        error.to_string().contains("no relation memory"),
        "unexpected refusal: {error}"
    );
}

#[test]
fn a_binding_reading_a_written_slot_is_detected() {
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
    assert!(reads_slot(&reads_written, "y", &written));
    assert!(!reads_slot(&reads_other, "y", &written));
    assert!(reads_slot(&tensor_over_written, "y", &written));
    assert!(!reads_slot(&reads_parameter, "y", &written));
    assert!(reads_slot(&reads_parameter, "p", &written));
}

#[test]
fn a_binding_reading_a_written_slot_inside_a_conditional_is_detected() {
    // The load sits in the fallback region of a function conditional.
    let written = BTreeSet::from([3]);
    let conditional = |fallback: Vec<solve::LinearOp>| {
        [solve::LinearOp::FunctionConditional {
            dst_start: 0,
            capture_start: 0,
            program: std::sync::Arc::new(solve::FunctionConditionalProgram {
                owner: None,
                capture_count: 0,
                target_widths: Box::new([1]),
                result_count: 1,
                arms: Box::new([]),
                fallback_register_count: 1,
                fallback,
            }),
        }]
    };
    let reads_written = conditional(vec![solve::LinearOp::LoadP { dst: 0, index: 3 }]);
    let reads_other = conditional(vec![solve::LinearOp::LoadP { dst: 0, index: 4 }]);
    assert!(reads_slot(&reads_written, "p", &written));
    assert!(!reads_slot(&reads_other, "p", &written));
}
