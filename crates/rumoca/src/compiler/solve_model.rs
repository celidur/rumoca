//! The renderer over the complete lowered model, for targets that seed
//! their own lanes (`solve_model`).

use std::sync::Arc;

use rumoca_compile::codegen::{CodegenError, SolveTemplateRenderer};
use rumoca_compile::compile::Dae;

use crate::error::CompilerError;

/// A renderer over the complete lowered model: the same checked problem a
/// solve target renders, plus the model's start values (`solve_model`).
pub(super) fn build_solve_model_template_renderer(
    dae_model: &Dae,
) -> Result<SolveTemplateRenderer, CompilerError> {
    let lowered = rumoca_sim::lower_correlated_for_simulation_with_overrides(
        dae_model,
        &rumoca_sim::SimOptions::default(),
    )
    .map_err(|err| CompilerError::TemplateError(CodegenError::template(err.to_string())))?;
    let prepared = lowered.prepared_dae();
    let model = Arc::new(lowered.model().clone());
    // A refresh made of exact assignments and an initialization made of
    // bindings need no projection kernel; one that retains a projection stage
    // or initialization residual rows is rendered with the ME refresh view,
    // from which the target compiles the shared kernel per lane.
    let initial_rows = model
        .problem
        .initialization
        .residual()
        .len()
        .is_ok_and(|rows| rows > 0);
    if rumoca_phase_codegen::lane_refresh_admissible(&model.problem) && !initial_rows {
        SolveTemplateRenderer::new_owned_model_with_dae(model, prepared)
    } else {
        SolveTemplateRenderer::new_owned_model_with_me_projection(model, prepared)
    }
    .map_err(CompilerError::TemplateError)
}
