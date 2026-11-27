//! The parameter-static gradient certificate (SPEC_0043) over primal tensor
//! operations: each operation certifies a gradient that reads parameters
//! alone and refuses one that reads solver Y or time, and a certified
//! program's Jacobian is the same, bit for bit, at any solver-Y point and
//! time.

use rumoca_ir_solve::{TensorConcatenateSource, TensorInputKind};

use super::*;

fn load(dst: u32, input: TensorInputKind, input_start: usize, count: usize) -> LinearOp {
    LinearOp::TensorLoad {
        dst_start: dst,
        input,
        input_start,
        count,
        seed_start: None,
        lanes: 1,
    }
}

fn elementwise(dst: u32, op: BinaryOp, lhs: u32, rhs: u32, count: usize) -> LinearOp {
    LinearOp::TensorBinary {
        dst_start: dst,
        op,
        lhs_start: lhs,
        rhs_start: rhs,
        count,
        lhs_stride: 1,
        rhs_stride: 1,
        lanes: 1,
    }
}

fn store(start: u32, count: usize) -> LinearOp {
    LinearOp::StoreOutputRange {
        start,
        count,
        stride: 1,
    }
}

/// `p[0..3] .* y[0..3]`, or `y .* y` when `nonlinear`.
fn scaled(nonlinear: bool) -> Vec<LinearOp> {
    vec![
        load(0, TensorInputKind::Y, 0, 3),
        load(3, TensorInputKind::P, 0, 3),
        elementwise(6, BinaryOp::Mul, if nonlinear { 0 } else { 3 }, 0, 3),
        store(6, 3),
    ]
}

/// A 2x3 matrix (of parameters, or of solver Y when `nonlinear`) times
/// `y[3..6]`.
fn product(nonlinear: bool) -> Vec<LinearOp> {
    let matrix = if nonlinear {
        TensorInputKind::Y
    } else {
        TensorInputKind::P
    };
    vec![
        load(0, matrix, 6, 6),
        load(6, TensorInputKind::Y, 3, 3),
        LinearOp::MatrixMultiply {
            dst_start: 9,
            lhs_start: 0,
            rhs_start: 6,
            rows: 2,
            inner: 3,
            columns: 1,
            lanes: 1,
        },
        store(9, 2),
    ]
}

/// The transpose of a 2x3 elementwise product of solver Y and parameters
/// (of solver Y with itself when `nonlinear`).
fn transposed(nonlinear: bool) -> Vec<LinearOp> {
    vec![
        load(0, TensorInputKind::Y, 0, 6),
        load(6, TensorInputKind::P, 0, 6),
        elementwise(12, BinaryOp::Mul, if nonlinear { 0 } else { 6 }, 0, 6),
        LinearOp::TensorTranspose {
            dst_start: 18,
            src_start: 12,
            rows: 3,
            columns: 2,
            element_width: 1,
            lanes: 1,
        },
        store(18, 6),
    ]
}

/// `[y[0..2]; p[0..2] .* y[2..4]]`, the second part `y .* y` when
/// `nonlinear`.
fn concatenated(nonlinear: bool) -> Vec<LinearOp> {
    vec![
        load(0, TensorInputKind::Y, 0, 4),
        load(4, TensorInputKind::P, 0, 2),
        elementwise(6, BinaryOp::Mul, if nonlinear { 2 } else { 4 }, 2, 2),
        LinearOp::TensorConcatenate {
            dst_start: 8,
            sources: vec![
                TensorConcatenateSource {
                    start: 0,
                    dimensions: vec![2].into(),
                },
                TensorConcatenateSource {
                    start: 6,
                    dimensions: vec![2].into(),
                },
            ]
            .into(),
            dimensions: vec![4].into(),
            axis: 0,
            lanes: 1,
        },
        store(8, 4),
    ]
}

/// `p .* y` and `y .* y` stored as one range of outputs: one output's
/// gradient reads solver Y.
fn mixed_outputs() -> Vec<LinearOp> {
    vec![
        load(0, TensorInputKind::Y, 0, 2),
        load(2, TensorInputKind::P, 0, 2),
        elementwise(4, BinaryOp::Mul, 2, 0, 1),
        elementwise(5, BinaryOp::Mul, 1, 1, 1),
        store(4, 2),
    ]
}

/// `t .* y`: a time coefficient, broadcast over the range.
fn time_scaled() -> Vec<LinearOp> {
    vec![
        LinearOp::LoadTime { dst: 0 },
        load(1, TensorInputKind::Y, 0, 3),
        LinearOp::TensorBinary {
            dst_start: 4,
            op: BinaryOp::Mul,
            lhs_start: 0,
            rhs_start: 1,
            count: 3,
            lhs_stride: 0,
            rhs_stride: 1,
            lanes: 1,
        },
        store(4, 3),
    ]
}

fn certified(program: &[LinearOp]) -> Option<Vec<usize>> {
    crate::parameter_static_y_gradient_params(program)
}

#[test]
fn each_tensor_operation_certifies_a_parameter_only_gradient() {
    assert_eq!(certified(&scaled(false)), Some(vec![0, 1, 2]));
    assert_eq!(certified(&product(false)), Some(vec![6, 7, 8, 9, 10, 11]));
    assert_eq!(certified(&transposed(false)), Some((0..6).collect()));
    assert_eq!(certified(&concatenated(false)), Some(vec![0, 1]));
}

#[test]
fn each_tensor_operation_refuses_a_gradient_that_reads_solver_y_or_time() {
    assert_eq!(certified(&scaled(true)), None);
    assert_eq!(certified(&product(true)), None);
    assert_eq!(certified(&transposed(true)), None);
    assert_eq!(certified(&concatenated(true)), None);
    assert_eq!(certified(&mixed_outputs()), None);
    assert_eq!(certified(&time_scaled()), None);
}

#[test]
fn a_tensor_program_with_tangent_lanes_or_seeds_is_not_certified() {
    let seeded = vec![
        LinearOp::TensorLoad {
            dst_start: 0,
            input: TensorInputKind::Y,
            input_start: 0,
            count: 2,
            seed_start: Some(0),
            lanes: 2,
        },
        store(0, 2),
    ];
    let dual = vec![
        load(0, TensorInputKind::Y, 0, 2),
        load(2, TensorInputKind::P, 0, 2),
        LinearOp::TensorBinary {
            dst_start: 4,
            op: BinaryOp::Mul,
            lhs_start: 0,
            rhs_start: 2,
            count: 1,
            lhs_stride: 1,
            rhs_stride: 1,
            lanes: 2,
        },
        store(4, 2),
    ];
    assert_eq!(certified(&seeded), None);
    assert_eq!(certified(&dual), None);
}

/// The forward-mode Jacobian of `program` (outputs by solver-Y columns) at
/// `y` and `t`, as bit patterns.
fn jacobian_bits(program: &[LinearOp], outputs: usize, y: &[f64], p: &[f64], t: f64) -> Vec<u64> {
    let primal =
        ScalarProgramBlock::with_program_spans(vec![program.to_vec()], vec![fixture_span()])
            .expect("tensor fixture is source-backed");
    let dual = crate::ad::lower_scalar_program_block_full_jvp(&primal, y.len())
        .expect("tensor fixture has a forward-mode lowering");
    let mut bits = Vec::new();
    for column in 0..y.len() {
        let mut seed = vec![0.0; y.len() + p.len()];
        seed[column] = 1.0;
        let mut tangent = vec![0.0; outputs];
        crate::eval_scalar_program_block(&dual, y, p, t, Some(&seed), &mut tangent)
            .expect("tensor fixture evaluates");
        bits.extend(tangent.iter().map(|value| value.to_bits()));
    }
    bits
}

#[test]
fn a_certified_tensor_program_has_the_same_jacobian_at_every_point_and_time() {
    let p: Vec<f64> = (0..12).map(|i| 0.3 + 0.7 * f64::from(i)).collect();
    let first: Vec<f64> = (0..8).map(|i| 1.5 - 0.4 * f64::from(i)).collect();
    let second: Vec<f64> = (0..8).map(|i| -2.25 + 1.1 * f64::from(i)).collect();
    for (program, outputs) in [
        (scaled(false), 3),
        (product(false), 2),
        (transposed(false), 6),
        (concatenated(false), 4),
    ] {
        assert!(certified(&program).is_some());
        assert_eq!(
            jacobian_bits(&program, outputs, &first, &p, 0.0),
            jacobian_bits(&program, outputs, &second, &p, 7.5),
            "{program:?}"
        );
    }
    // A product of two solver-Y reads, refused, changes its Jacobian.
    let nonlinear = scaled(true);
    assert!(certified(&nonlinear).is_none());
    assert_ne!(
        jacobian_bits(&nonlinear, 3, &first, &p, 0.0),
        jacobian_bits(&nonlinear, 3, &second, &p, 7.5)
    );
}
