//! Capability gate of Solve targets that initialize their own lanes.

use anyhow::{Context, Result};
use rumoca_compile::codegen::targets::{
    TargetCapabilities, TargetManifest, TargetTemplateIr,
    validate_solve_target_capabilities_with_calls,
};

/// A Solve target that declares `initialization` seeds and initializes each
/// lane itself, so it needs the start values of the complete lowered model.
/// Its capability gate and its renderer (`ManifestRenderer::SolveModel`) both
/// follow this one predicate, so the gate always reads the lowering the
/// renderer renders. Such a target executes the per-lane view
/// (`solve_blocks.lanes`): the gate admits its staged refresh without
/// residual export.
pub(super) fn initializes_lanes(manifest: &TargetManifest) -> bool {
    manifest.ir == TargetTemplateIr::Solve
        && manifest
            .capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.initialization)
            == Some(true)
}

/// A target that initializes its lanes renders the per-scalar refresh owners
/// of the scalar simulation profile
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
    // The lane kernels run a staged refresh made of exact-assignment stages
    // themselves (`solve_blocks.lanes.refresh`) and one that retains
    // projection stages with the shared ME projection kernel, built per lane
    // (SPEC_0044 ME-PROJ-001), so neither needs residual export. That kernel
    // also projects the lane's states onto a retained state manifold.
    let problem = &lowered.model().problem;
    // A lane refreshes on the primary chart alone: no lane kernel switches
    // among the executable alternate charts of a reduced chart set (SPEC_0040
    // STRUCT-T07), so such a model is refused rather than integrated on one.
    if problem
        .continuous
        .reduced_chart_set
        .charts
        .iter()
        .any(|chart| chart.plan.is_some())
    {
        anyhow::bail!(
            "unsupported-feature:reduced_charts: a lane cannot switch among the alternate charts of a reduced chart set"
        );
    }
    let mut capabilities = capabilities.clone();
    if capabilities.exact_algebraic_assignments == Some(true)
        && (rumoca_phase_codegen::lane_refresh_admissible(problem)
            || rumoca_phase_codegen::me_refresh_admissible_with_state_manifold(problem))
    {
        capabilities.residual_equations = Some(true);
    }
    validate_solve_target_capabilities_with_calls(
        problem,
        Some(&lowered.model().pure_calls),
        manifest,
        &capabilities,
    )?;
    Ok(())
}
