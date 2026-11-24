//! The checked models the ME refresh view is built from.
//!
//! The view reads the same facts of either source: the checked FMI C
//! component, whose C settles the algebraic coordinates of one instance, or a
//! complete lowered model, whose target compiles the same kernel for a device
//! and runs it once per lane (SPEC_0044 ME-PROJ-001).

use rumoca_ir_solve as solve;

use crate::errors::CodegenError;

/// What the ME refresh view reads of one checked model.
pub(crate) trait RefreshSource {
    fn problem(&self) -> &solve::SolveProblem;
    fn artifacts(&self) -> &solve::SolveArtifacts;
    fn pure_calls(&self) -> &solve::SolvePureCallTable;
    /// The finite positive characteristic scale of one solver variable, the
    /// scale the linked algebraic projection uses.
    fn solver_variable_scale(&self, index: usize) -> f64;
    /// The initial solver vector and parameter values, the point at which the
    /// linked kernel resolves each reduced chart's state-binding rows.
    fn instantiation_point(&self) -> (&[f64], &[f64]);
    /// The dependency levels of the parameter bindings the kernel settles.
    fn parameter_binding_levels(&self) -> Result<usize, CodegenError>;
}

impl RefreshSource for solve::fmi::FmiCCodegenView {
    fn problem(&self) -> &solve::SolveProblem {
        Self::problem(self)
    }
    fn artifacts(&self) -> &solve::SolveArtifacts {
        Self::artifacts(self)
    }
    fn pure_calls(&self) -> &solve::SolvePureCallTable {
        Self::pure_calls(self)
    }
    fn solver_variable_scale(&self, index: usize) -> f64 {
        Self::solver_variable_scale(self, index)
    }
    fn instantiation_point(&self) -> (&[f64], &[f64]) {
        Self::instantiation_point(self)
    }
    fn parameter_binding_levels(&self) -> Result<usize, CodegenError> {
        Ok(Self::parameter_binding_levels(self))
    }
}

/// A complete lowered model, without an FMI component. Its binding levels
/// are those the C component computes, from the same parameter bindings.
impl RefreshSource for solve::SolveModel {
    fn problem(&self) -> &solve::SolveProblem {
        &self.problem
    }
    fn artifacts(&self) -> &solve::SolveArtifacts {
        &self.artifacts
    }
    fn pure_calls(&self) -> &solve::SolvePureCallTable {
        &self.pure_calls
    }
    fn solver_variable_scale(&self, index: usize) -> f64 {
        Self::solver_variable_scale(self, index)
    }
    fn instantiation_point(&self) -> (&[f64], &[f64]) {
        (&self.initial_y, &self.parameters)
    }
    fn parameter_binding_levels(&self) -> Result<usize, CodegenError> {
        solve::fmi::SolveBindingExecution::checked(self)
            .map(|execution| execution.levels())
            .map_err(|reason| {
                CodegenError::dae_preparation_failed(
                    format!("unsupported-feature:initialization: {reason}"),
                    None,
                )
            })
    }
}
