//! Capability gate of Solve targets that project algebraic blocks per lane.

use anyhow::{Context, Result};
use rumoca_compile::codegen::targets::{
    TargetCapabilities, TargetManifest, TargetTemplateIr,
    validate_solve_target_capabilities_with_calls,
};

/// A Solve target that declares `algebraic_projection` projects its algebraic
/// blocks per lane. Its capability gate and its renderer
/// (`ManifestRenderer::SolveModel`) both follow this one predicate, so the
/// gate always reads the lowering the renderer renders.
pub(super) fn projects_per_lane(manifest: &TargetManifest) -> bool {
    manifest.ir == TargetTemplateIr::Solve
        && manifest
            .capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.algebraic_projection)
            == Some(true)
}

/// A target that projects algebraic blocks per lane renders the per-scalar
/// refresh owners of the scalar simulation profile
/// (`render_solve_model_template_str`), so the gate reads the same lowering.
pub(super) fn validate(
    result: &crate::CompilationResult,
    manifest: &TargetManifest,
    capabilities: &TargetCapabilities,
) -> Result<()> {
    let lowered = rumoca_sim::lower_correlated_for_simulation_with_overrides(
        &result.dae,
        &rumoca_sim::SimOptions::default(),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))
    .context("Lower Solve IR for target capability validation")?;
    validate_solve_target_capabilities_with_calls(
        &lowered.model().problem,
        Some(&lowered.model().pure_calls),
        manifest,
        capabilities,
    )?;
    Ok(())
}
