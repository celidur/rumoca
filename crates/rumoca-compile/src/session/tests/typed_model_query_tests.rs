use super::*;

/// Drop only the final compile result; DAE and flat artifacts stay cached.
fn evict_compile_result(session: &mut Session, model: &str) {
    session.query_state.dae.compile_results.shift_remove(model);
}

/// Drop only the cached flat artifact for `model`.
fn evict_flat_model(session: &mut Session, model: &str) {
    let flat_key = standard_flat_cache_key(session, model);
    session
        .query_state
        .flat
        .flat_models
        .artifacts
        .shift_remove(&flat_key);
}

#[test]
fn unit_metadata_warnings_survive_typed_cache_hits_and_clear_on_invalidation() {
    let mut session = Session::new(SessionConfig {
        unit_metadata_policy: UnitMetadataPolicy::Warn,
        ..SessionConfig::default()
    });
    session
        .add_document(
            "unit-warning.mo",
            "model UnitWarning\n  Real x(unit = \"kV RMSLL\");\nequation\n  x = 1;\nend UnitWarning;\n",
        )
        .expect("source should parse");

    let first = session
        .compile_model_phases("UnitWarning")
        .expect("compile should return phase result");
    assert!(matches!(first, PhaseResult::Success(_)), "{first:?}");
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);
    assert_eq!(
        session.typecheck_warnings("UnitWarning")[0].code.as_deref(),
        Some("ET010")
    );

    session
        .add_document(
            "unrelated.mo",
            "model Unrelated\n  Real y;\nend Unrelated;\n",
        )
        .expect("unrelated source should parse");
    assert!(
        session
            .update_document(
                "unrelated.mo",
                "model Unrelated\n  Real y;\n  Real z;\nend Unrelated;\n",
            )
            .is_none()
    );
    let cached_compile = session
        .compile_model_phases("UnitWarning")
        .expect("unchanged target compile should hit cache");
    assert!(matches!(cached_compile, PhaseResult::Success(_)));
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);

    // Evict only the final compile result. The cached DAE and flat artifacts
    // remain, so this compile must restore warnings on the DAE-cache-hit path.
    evict_compile_result(&mut session, "UnitWarning");
    let dae_hit = session
        .compile_model_phases("UnitWarning")
        .expect("DAE-cache-hit compile should succeed");
    assert!(matches!(dae_hit, PhaseResult::Success(_)));
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);

    // Evict the top-level and DAE artifacts only. Keep the flat artifact so
    // the compile traverses the flat-cache-hit branch.
    evict_compile_result(&mut session, "UnitWarning");
    let dae_key = standard_dae_cache_key(&mut session, "UnitWarning");
    session.query_state.dae.dae_models.shift_remove(&dae_key);
    let flat_hit = session
        .compile_model_phases("UnitWarning")
        .expect("flat-cache-hit compile should succeed");
    assert!(matches!(flat_hit, PhaseResult::Success(_)));
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);

    let uncached_dae =
        session.compile_model_dae_strict_reachable_uncached_with_recovery_detailed("UnitWarning");
    assert!(uncached_dae.is_ok(), "{uncached_dae:?}");
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);

    let uncached_report =
        session.compile_model_strict_reachable_uncached_with_recovery("UnitWarning");
    assert!(uncached_report.requested_succeeded());
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);

    // Clear only later phase caches so the next request exercises the typed
    // model cache while preserving the same source fingerprint.
    evict_compile_result(&mut session, "UnitWarning");
    let dae_key = standard_dae_cache_key(&mut session, "UnitWarning");
    session.query_state.dae.dae_models.shift_remove(&dae_key);
    evict_flat_model(&mut session, "UnitWarning");
    let warm = session
        .compile_model_phases("UnitWarning")
        .expect("cached typed model should compile");
    assert!(matches!(warm, PhaseResult::Success(_)), "{warm:?}");
    assert_eq!(session.typecheck_warnings("UnitWarning").len(), 1);

    assert!(session
        .update_document(
            "unit-warning.mo",
            "model UnitWarning\n  Real x(unit = \"V\");\nequation\n  x = 1;\nend UnitWarning;\n",
        )
        .is_none());
    let rebuilt = session
        .compile_model_phases("UnitWarning")
        .expect("updated source should compile");
    assert!(matches!(rebuilt, PhaseResult::Success(_)), "{rebuilt:?}");
    assert!(
        session.typecheck_warnings("UnitWarning").is_empty(),
        "stale warning must not survive input invalidation"
    );

    // A later parse failure happens before typed-model query. The getter must
    // still reflect the current failed compile, not retain the prior result.
    assert!(
        session
            .update_document("unit-warning.mo", "model UnitWarning\n  Real x(\n")
            .is_some()
    );
    assert!(session.compile_model_phases("UnitWarning").is_err());
    assert!(session.typecheck_warnings("UnitWarning").is_empty());
}

#[test]
fn session_exposes_unit_warnings_even_when_other_type_errors_fail_compile() {
    let mut session = Session::new(SessionConfig {
        unit_metadata_policy: UnitMetadataPolicy::Warn,
        ..SessionConfig::default()
    });
    session
        .add_document(
            "unit-warning-error.mo",
            "model UnitWarningError\n  Real x(unit = \"kV RMSLL\");\n  Boolean b;\nequation\n  x = b;\nend UnitWarningError;\n",
        )
        .expect("source should parse");
    let result = session
        .compile_model_phases("UnitWarningError")
        .expect("compile should return phase diagnostics");
    assert!(
        matches!(
            result,
            PhaseResult::Failed {
                phase: FailedPhase::Typecheck,
                ..
            }
        ),
        "{result:?}"
    );
    let warnings = session.typecheck_warnings("UnitWarningError");
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code.as_deref(), Some("ET010"));
}

fn assert_typecheck_failure(result: PhaseResult, expected_code: &str) {
    match result {
        PhaseResult::Failed {
            phase, error_code, ..
        } => {
            assert_eq!(phase, FailedPhase::Typecheck);
            assert_eq!(error_code.as_deref(), Some(expected_code));
        }
        other => panic!("expected typecheck failure, got {other:?}"),
    }
}

#[test]
fn typed_model_query_cache_is_reused_by_compile_and_diagnostics() {
    let mut session = Session::default();
    session
        .add_document(
            "target.mo",
            "model Target\n  Real x(startd = 1.0);\nequation\n  der(x) = -x;\nend Target;\n",
        )
        .expect("target should parse");
    session
        .add_document("other.mo", "model Other\n  Real y;\nend Other;\n")
        .expect("other should parse");

    let first = session
        .compile_model_phases("Target")
        .expect("compile should return phase result");
    assert_typecheck_failure(first, "ET001");

    let cache_key = standard_typed_cache_key(&mut session, "Target");
    match session
        .query_state
        .flat
        .typed_models
        .artifacts
        .get(&cache_key)
        .expect("typed query should be cached")
        .outcome
        .clone()
    {
        TypedModelOutcome::TypecheckError(diags) => {
            assert!(
                diags
                    .iter()
                    .any(|diag| diag.code.as_deref() == Some("ET001")),
                "initial typed artifact should preserve the real typecheck failure"
            );
        }
        other => panic!("expected cached typecheck error, got {other:?}"),
    }

    session
        .query_state
        .flat
        .typed_models
        .artifacts
        .get_mut(&cache_key)
        .expect("typed query should be cached")
        .outcome = TypedModelOutcome::TypecheckError(vec![CommonDiagnostic::error(
        "TTEST",
        "typed cache sentinel",
        PrimaryLabel::new(Span::DUMMY).with_message("cache sentinel"),
    )]);

    let parse_error = session.update_document(
        "other.mo",
        "model Other\n  Real y;\n  Real z;\nend Other;\n",
    );
    assert!(parse_error.is_none(), "unrelated edit should remain valid");

    session
        .query_state
        .dae
        .compile_results
        .shift_remove("Target");
    let dae_cache_key = standard_dae_cache_key(&mut session, "Target");
    session
        .query_state
        .dae
        .dae_models
        .shift_remove(&dae_cache_key);
    let flat_cache_key = standard_flat_cache_key(&mut session, "Target");
    session
        .query_state
        .flat
        .flat_models
        .artifacts
        .shift_remove(&flat_cache_key);

    let warm_compile = session
        .compile_model_phases("Target")
        .expect("warm compile should return phase result");
    assert_typecheck_failure(warm_compile, "TTEST");

    let warm_diagnostics = session.compile_model_diagnostics("Target");
    assert_diagnostics_have_code(&warm_diagnostics, "TTEST");

    let parse_error = session.update_document(
        "target.mo",
        "model Target\n  Real x(start = 1.0);\nequation\n  der(x) = -x;\nend Target;\n",
    );
    assert!(parse_error.is_none(), "target edit should remain valid");

    let rebuilt = session
        .compile_model_phases("Target")
        .expect("rebuild should return phase result");
    assert!(
        matches!(rebuilt, PhaseResult::Success(_)),
        "target edit should rebuild the typed model instead of reusing the stale typecheck failure"
    );

    let rebuilt_diagnostics = session.compile_model_diagnostics("Target");
    assert_diagnostics_lack_code(&rebuilt_diagnostics, "TTEST");
}
