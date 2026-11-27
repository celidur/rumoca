//! A simultaneous plan at a settled point settles in one sweep under the
//! converged correction (SPEC_0044 ME-PROJ-001): a dense block's correction,
//! at the roundoff level there, marks no change, and the torn block it feeds
//! must not see a change either.

use std::cell::Cell;

use super::super::*;

/// A dense block (rows 0 and 1, unknowns 0 and 1) that reads the causal
/// coordinate of a torn block (rows 2 and 3, tear 2, causal 3) that reads the
/// dense block back: one simultaneous plan whose later block invalidates the
/// earlier one.
struct CoupledBlocks {
    plan: solve::AlgebraicProjectionPlan,
    visits: Cell<usize>,
}

impl CoupledBlocks {
    fn new() -> Self {
        let dense = solve::AlgebraicProjectionBlock {
            rows: vec![0, 1],
            y_indices: vec![0, 1],
            tearing: None,
            alternate_charts: Vec::new(),
        };
        let torn = solve::AlgebraicProjectionBlock {
            rows: vec![2, 3],
            y_indices: vec![2, 3],
            tearing: Some(solve::BlockTearing {
                tear_y_indices: vec![2],
                residual_rows: vec![3],
                causal_steps: vec![solve::CausalStep {
                    row: 2,
                    y_index: 3,
                    ..Default::default()
                }],
            }),
            alternate_charts: Vec::new(),
        };
        Self {
            plan: solve::AlgebraicProjectionPlan {
                blocks: vec![dense, torn].into(),
            },
            visits: Cell::new(0),
        }
    }
}

impl ImplicitProjectionModel for CoupledBlocks {
    fn eval_residual(
        &self,
        y: &[f64],
        p: &[f64],
        _t: f64,
        out: &mut [f64],
    ) -> Result<(), RuntimeSolveError> {
        out[0] = y[0] + 0.1 * y[0] * y[0] * y[0] + y[1] - p[0] - 0.1 * y[3];
        out[1] = y[1] - 0.5 * y[0] * y[0] - 0.2;
        out[2] = y[3] - (y[0] - 0.5 * y[2]);
        out[3] = y[2] + 0.1 * y[2] * y[2] * y[2] - y[3];
        Ok(())
    }

    fn eval_implicit_residual_row(
        &self,
        row: usize,
        y: &[f64],
        p: &[f64],
        t: f64,
    ) -> Result<Option<f64>, RuntimeSolveError> {
        let mut residual = [0.0; 4];
        self.eval_residual(y, p, t, &mut residual)?;
        Ok(residual.get(row).copied())
    }

    fn eval_jacobian_v(
        &self,
        y: &[f64],
        _p: &[f64],
        _t: f64,
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), RuntimeSolveError> {
        out[0] = (1.0 + 0.3 * y[0] * y[0]) * v[0] + v[1] - 0.1 * v[3];
        out[1] = -y[0] * v[0] + v[1];
        out[2] = v[3] - v[0] + 0.5 * v[2];
        out[3] = (1.0 + 0.3 * y[2] * y[2]) * v[2] - v[3];
        Ok(())
    }

    fn implicit_target(&self, row: usize) -> Option<solve::ScalarSlot> {
        [0, 1, 3, 2].get(row).copied().map(solve::scalar_slot_y)
    }

    fn algebraic_projection_plan(&self) -> &solve::AlgebraicProjectionPlan {
        &self.plan
    }

    fn implicit_target_assignment_is_exact(&self, row: usize, target: usize) -> bool {
        (row, target) == (2, 3)
    }

    fn eval_implicit_target_value(
        &self,
        row: usize,
        target: usize,
        y: &[f64],
        _p: &[f64],
        _t: f64,
    ) -> Result<Option<f64>, RuntimeSolveError> {
        Ok(self
            .implicit_target_assignment_is_exact(row, target)
            .then(|| y[0] - 0.5 * y[2]))
    }

    fn linked_kernel(&self, request: KernelRequest<'_>) -> Result<KernelAnswer, RuntimeSolveError> {
        if let KernelRequest::BeginBlock { .. } = request {
            self.visits.set(self.visits.get() + 1);
        }
        Ok(KernelAnswer::Declined)
    }
}

/// One certified refresh of the plan at `drive`; the block visits it made.
fn refresh(model: &CoupledBlocks, y: &mut [f64], drive: f64) -> usize {
    model.visits.set(0);
    project_algebraics_with_plan_certified(
        model,
        model.algebraic_projection_plan(),
        y,
        AlgebraicProjectionArgs {
            parameters: &[drive],
            time: 0.0,
            state_count: 0,
            tolerance: 1.0e-10,
        },
        ALGEBRAIC_PROJECTION_MAX_ITERS,
    )
    .unwrap_or_else(|error| panic!("drive {drive}: {error:?}"));
    model.visits.get()
}

#[test]
fn a_settled_simultaneous_plan_settles_again_in_one_sweep() {
    // The corrections at a settled point are at the roundoff level, which
    // depends on the point: sweep the drive over a range.
    for step in 0..400 {
        let drive = 0.5 + 0.00687 * f64::from(step);
        let model = CoupledBlocks::new();
        let mut y = vec![1.0; 4];
        refresh(&model, &mut y, drive);
        refresh(&model, &mut y, drive);
        for call in 0..6 {
            let visits = refresh(&model, &mut y, drive);
            assert_eq!(visits, 2, "drive {drive}, call {call} swept again: {y:?}");
        }
    }
}
