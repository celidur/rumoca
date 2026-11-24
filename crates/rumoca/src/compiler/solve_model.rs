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
    SolveTemplateRenderer::new_owned_model_with_dae(Arc::new(lowered.model().clone()), prepared)
        .map_err(CompilerError::TemplateError)
}
