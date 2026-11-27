//! A certified dense block applies the correction it computed once its
//! residual and that correction meet tolerance, so a change of its inputs
//! inside the tolerance band still moves its coordinates (SPEC_0044
//! ME-PROJ-001).

use super::super::*;

/// A nonlinear two-row block driven by `p[0]`: no row is affine and the block
/// has no tearing, so it takes the dense block Newton.
struct DrivenDenseBlock;

impl ImplicitProjectionModel for DrivenDenseBlock {
    fn eval_residual(
        &self,
        y: &[f64],
        p: &[f64],
        _t: f64,
        out: &mut [f64],
    ) -> Result<(), RuntimeSolveError> {
        out[0] = y[0] + 0.1 * y[0] * y[0] * y[0] + y[1] - p[0];
        out[1] = y[1] - 0.5 * y[0] * y[0] - 0.2;
        Ok(())
    }

    fn eval_jacobian_v(
        &self,
        y: &[f64],
        _p: &[f64],
        _t: f64,
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), RuntimeSolveError> {
        out[0] = (1.0 + 0.3 * y[0] * y[0]) * v[0] + v[1];
        out[1] = -y[0] * v[0] + v[1];
        Ok(())
    }

    fn implicit_target(&self, row_idx: usize) -> Option<solve::ScalarSlot> {
        Some(solve::scalar_slot_y(row_idx))
    }

    fn algebraic_projection_plan(&self) -> &solve::AlgebraicProjectionPlan {
        static PLAN: std::sync::OnceLock<solve::AlgebraicProjectionPlan> =
            std::sync::OnceLock::new();
        PLAN.get_or_init(|| solve::AlgebraicProjectionPlan {
            blocks: (vec![solve::AlgebraicProjectionBlock {
                rows: vec![0, 1],
                y_indices: vec![0, 1],
                tearing: None,
                alternate_charts: Vec::new(),
            }])
            .into(),
        })
    }
}

const TOLERANCE: f64 = 1.0e-6;

/// Project at `drive`, then again from that point at `drive + step`.
fn projected_after_step(drive: f64, step: f64) -> (Vec<f64>, Vec<f64>) {
    let model = DrivenDenseBlock;
    let mut y = vec![1.0, 1.0];
    let project = |y: &mut Vec<f64>, drive: f64| {
        project_algebraics_with_plan_certified(
            &model,
            model.algebraic_projection_plan(),
            y,
            AlgebraicProjectionArgs {
                parameters: &[drive],
                time: 0.0,
                state_count: 0,
                tolerance: TOLERANCE,
            },
            ALGEBRAIC_PROJECTION_MAX_ITERS,
        )
        .expect("the driven block projects");
    };
    project(&mut y, drive);
    let before = y.clone();
    project(&mut y, drive + step);
    (before, y)
}

#[test]
fn a_change_inside_the_tolerance_band_moves_a_certified_dense_block() {
    // The drive moves the block's first residual by 1e-7, a tenth of its
    // tolerance band; the reference sensitivity comes from a step far outside
    // the band.
    let small = 1.0e-7;
    let large = 1.0e-5;
    let (before, after) = projected_after_step(2.0, small);
    let (reference_before, reference_after) = projected_after_step(2.0, large);
    for index in 0..2 {
        let sensitivity = (after[index] - before[index]) / small;
        let reference = (reference_after[index] - reference_before[index]) / large;
        assert!(
            reference.abs() > 0.1,
            "coordinate {index} must depend on the drive: {reference}"
        );
        assert!(
            (sensitivity - reference).abs() <= 1.0e-3 * reference.abs(),
            "coordinate {index}: sensitivity {sensitivity} inside the band, {reference} outside it"
        );
    }
}

#[test]
fn a_converged_correction_settles_without_marking_a_change() {
    // From the point settled at one drive, the block settles a drive moved
    // inside the band: it applies its correction and reports no change, and
    // so does every refresh at that point after it, whose correction is at
    // the roundoff level and never vanishes exactly.
    let model = DrivenDenseBlock;
    let (_, mut y) = projected_after_step(2.0, 0.0);
    let block = &model.algebraic_projection_plan().blocks.scalar_blocks()[0];
    let settled = y.clone();
    for call in 0..4 {
        let update = project_algebraic_block(
            &model,
            &mut y,
            &[2.0 + 1.0e-7],
            0.0,
            block,
            0,
            AlgebraicBlockProjectionPolicy {
                tolerance: TOLERANCE,
                step_limit: StepLimit::None,
                certify_coordinates: true,
                owner: None,
            },
        )
        .expect("the driven block projects");
        assert!(update.settled && !update.changed, "call {call}: {update:?}");
    }
    assert!(
        y.iter()
            .zip(&settled)
            .all(|(after, before)| (after - before).abs() > 1.0e-9),
        "the correction moved the block: {settled:?} to {y:?}"
    );
}

/// A nonlinear torn block driven by `p[0]`: the tear `y[0]` meets the
/// residual row 1 once the causal step of row 0 recovers `y[1]` exactly.
struct DrivenTornBlock {
    plan: solve::AlgebraicProjectionPlan,
}

impl DrivenTornBlock {
    fn new() -> Self {
        Self {
            plan: solve::AlgebraicProjectionPlan {
                blocks: (vec![solve::AlgebraicProjectionBlock {
                    rows: vec![0, 1],
                    y_indices: vec![0, 1],
                    tearing: Some(solve::BlockTearing {
                        tear_y_indices: vec![0],
                        residual_rows: vec![1],
                        causal_steps: vec![solve::CausalStep {
                            row: 0,
                            y_index: 1,
                            ..Default::default()
                        }],
                    }),
                    alternate_charts: Vec::new(),
                }])
                .into(),
            },
        }
    }
}

impl ImplicitProjectionModel for DrivenTornBlock {
    fn eval_residual(
        &self,
        y: &[f64],
        p: &[f64],
        _t: f64,
        out: &mut [f64],
    ) -> Result<(), RuntimeSolveError> {
        out[0] = y[1] - (p[0] - 0.5 * y[0]);
        out[1] = y[0] + 0.1 * y[0] * y[0] * y[0] - y[1];
        Ok(())
    }

    fn eval_implicit_residual_row(
        &self,
        row: usize,
        y: &[f64],
        p: &[f64],
        t: f64,
    ) -> Result<Option<f64>, RuntimeSolveError> {
        let mut residual = [0.0; 2];
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
        out[0] = v[1] + 0.5 * v[0];
        out[1] = (1.0 + 0.3 * y[0] * y[0]) * v[0] - v[1];
        Ok(())
    }

    fn implicit_target(&self, row: usize) -> Option<solve::ScalarSlot> {
        [1, 0].get(row).copied().map(solve::scalar_slot_y)
    }

    fn algebraic_projection_plan(&self) -> &solve::AlgebraicProjectionPlan {
        &self.plan
    }

    fn implicit_target_assignment_is_exact(&self, row: usize, target: usize) -> bool {
        (row, target) == (0, 1)
    }

    fn eval_implicit_target_value(
        &self,
        row: usize,
        target: usize,
        y: &[f64],
        p: &[f64],
        _t: f64,
    ) -> Result<Option<f64>, RuntimeSolveError> {
        Ok(self
            .implicit_target_assignment_is_exact(row, target)
            .then(|| p[0] - 0.5 * y[0]))
    }
}

/// Settle the torn block at `drive`, then again from that point at
/// `drive + step`: the coordinates before the step and after it, with the
/// second update.
fn torn_after_step(drive: f64, step: f64) -> (Vec<f64>, Vec<f64>, ProjectionBlockUpdate) {
    let model = DrivenTornBlock::new();
    let tearing = model.plan.blocks.scalar_blocks()[0]
        .tearing
        .as_ref()
        .expect("the block is torn");
    let mut y = vec![1.0, 1.0];
    let project = |y: &mut Vec<f64>, drive: f64| {
        tearing::project_torn_algebraic_block(
            &model,
            y,
            &[drive],
            0.0,
            TornBlock { index: 0, tearing },
            TOLERANCE,
            true,
        )
        .expect("the torn block projects")
        .expect("the torn solve settles the block")
    };
    project(&mut y, drive);
    // Settle the start exactly: its last correction is then quadratic.
    project(&mut y, drive);
    let before = y.clone();
    let update = project(&mut y, drive + step);
    (before, y, update)
}

#[test]
fn a_change_inside_the_tolerance_band_moves_a_certified_torn_block() {
    let small = 1.0e-7;
    let large = 1.0e-5;
    let (before, after, _) = torn_after_step(2.0, small);
    let (reference_before, reference_after, _) = torn_after_step(2.0, large);
    for index in 0..2 {
        let sensitivity = (after[index] - before[index]) / small;
        let reference = (reference_after[index] - reference_before[index]) / large;
        assert!(
            reference.abs() > 0.1,
            "coordinate {index} must depend on the drive: {reference}"
        );
        assert!(
            (sensitivity - reference).abs() <= 1.0e-3 * reference.abs(),
            "coordinate {index}: sensitivity {sensitivity} inside the band, {reference} outside it"
        );
    }
}

#[test]
fn a_torn_correction_marks_no_change() {
    // At the drive it settled, the block moves only by its converged
    // correction, which marks no change.
    let (_, _, update) = torn_after_step(2.0, 0.0);
    assert!(update.settled && !update.changed, "{update:?}");
}
