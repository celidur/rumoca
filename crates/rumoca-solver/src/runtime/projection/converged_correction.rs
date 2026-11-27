//! The correction a certified block applies once both its residual and its
//! Newton correction meet tolerance: a dense block adds it to its unknowns, a
//! torn block keeps its undamped step (`tearing::causal_correction_converged`).
//!
//! Settling such a block without its correction hides every change of the
//! block's inputs that keeps the residual inside the tolerance band: the
//! coordinates stay at the warm start, so a finite-difference probe of the
//! refresh sees no sensitivity at all below `tol` times the row scale. One
//! Newton step from a converged point is the first-order sensitivity
//! `-J^{-1} F` (SPEC_0044 ME-PROJ-001).
//!
//! A correction applies only above its floor
//! (`projection_policy::CONVERGED_CORRECTION_FLOOR`): one at the roundoff level
//! never vanishes exactly, and a later block recovering its coordinates from
//! the moving ones would change at every sweep. Applied, it marks no change:
//! within tolerance it invalidates no other block of a simultaneous plan.

use rumoca_eval_solve::projection_policy::CONVERGED_CORRECTION_FLOOR;
use rumoca_ir_solve as solve;

use super::ProjectionBlockUpdate;
use super::scaling::{scaled_correction_converged, scaled_tolerance};

/// A block's Newton correction with the variable scales and tolerance that
/// judge it.
pub(super) struct Correction<'a> {
    delta: &'a [f64],
    scales: &'a [f64],
    tol: f64,
}

impl<'a> Correction<'a> {
    pub(super) fn new(delta: &'a [f64], scales: &'a [f64], tol: f64) -> Self {
        Self { delta, scales, tol }
    }

    /// Whether every coordinate of the correction meets tolerance.
    pub(super) fn converged(&self) -> bool {
        scaled_correction_converged(self.delta, self.scales, self.tol)
    }

    /// Whether a coordinate of the correction exceeds its floor.
    pub(super) fn significant(&self) -> bool {
        self.delta.iter().zip(self.scales).any(|(&step, &scale)| {
            step.abs() > CONVERGED_CORRECTION_FLOOR * scaled_tolerance(self.tol, scale)
        })
    }

    /// Settle `block`, adding the correction to its unknowns in `y` when it
    /// is significant; the block changed only when an earlier stage of this
    /// call moved it (`changed`).
    pub(super) fn apply(
        &self,
        y: &mut [f64],
        block: &solve::AlgebraicProjectionBlock,
        changed: bool,
    ) -> ProjectionBlockUpdate {
        let update = ProjectionBlockUpdate {
            changed,
            settled: true,
        };
        if !self.significant() {
            return update;
        }
        for (&index, &step) in block.y_indices.iter().zip(self.delta) {
            if let Some(slot) = y.get_mut(index) {
                *slot += step;
            }
        }
        update
    }
}
