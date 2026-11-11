//! End-to-end `typecheck_instanced` checks over overlay bodies: equation and
//! algorithm shapes per instance scope, and modifier diagnostics reported
//! through the instanced pipeline.

use super::*;

#[test]
fn test_typecheck_instanced_checks_compound_equation_types_and_builtin_arguments() {
    let source = r#"
        function takesReal
            input Real u;
            output Real y;
        algorithm
            y := u;
        end takesReal;
        model Test
            Real x;
            Boolean b;
            Boolean c;
            String s;
        equation
            x = b and c;
            x = sin(s);
            x = takesReal(s);
        end Test;
    "#;
    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let model = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");
    let mut overlay = InstanceOverlay::new();
    for name in ["x", "b", "c", "s"] {
        add_instanced_component(
            &mut overlay,
            &format!("Test.{name}"),
            model.components.get(name).expect("component"),
            true,
        );
    }

    let diagnostics = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("compound type errors must fail instanced typecheck");
    let mismatches = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code.as_deref() == Some("ET002"))
        .count();
    assert!(
        mismatches >= 3,
        "expected equation, builtin, and user-function argument mismatches, got: {diagnostics:?}"
    );
}

#[test]
fn unit_metadata_warn_policy_preserves_et010_and_keeps_legacy_strict() {
    let source = r#"
        model Test
            Real x(unit = "kV RMSLL");
        end Test;
    "#;
    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let model = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");
    let mut strict_overlay = InstanceOverlay::new();
    add_instanced_component(
        &mut strict_overlay,
        "Test.x",
        model.components.get("x").expect("x component"),
        true,
    );
    let strict = typecheck_instanced(&tree, &mut strict_overlay, "Test")
        .expect_err("legacy entry point must remain strict");
    assert!(strict.iter().any(|d| d.code.as_deref() == Some("ET010")));

    let mut warn_overlay = InstanceOverlay::new();
    add_instanced_component(
        &mut warn_overlay,
        "Test.x",
        model.components.get("x").expect("x component"),
        true,
    );
    let diagnostics = typecheck_instanced_with_unit_metadata_policy(
        &tree,
        &mut warn_overlay,
        "Test",
        UnitMetadataPolicy::Warn,
    )
    .expect("invalid unit metadata must not fail Warn policy");
    let warning = diagnostics
        .iter()
        .find(|d| d.code.as_deref() == Some("ET010"))
        .expect("ET010 warning should be retained");
    assert_eq!(warning.severity, rumoca_core::DiagnosticSeverity::Warning);
    assert!(
        !warning.labels.is_empty(),
        "source labels should be preserved"
    );
    let unit_expr = model
        .components
        .get("x")
        .and_then(|component| component.modifications.get("unit"))
        .expect("source unit modifier should remain in AST");
    let Expression::Terminal { token, .. } = unit_expr else {
        panic!("unit metadata should remain a string terminal");
    };
    assert_eq!(token.text.as_ref(), "kV RMSLL");
}

#[test]
fn unit_metadata_warn_policy_does_not_suppress_other_modifier_errors() {
    let source = r#"
        model Test
            Real x(unit = "kV RMSLL", quantity = 1, fixed = 1);
        end Test;
    "#;
    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let model = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");
    let mut overlay = InstanceOverlay::new();
    add_instanced_component(
        &mut overlay,
        "Test.x",
        model.components.get("x").expect("x component"),
        true,
    );
    let diagnostics = typecheck_instanced_with_unit_metadata_policy(
        &tree,
        &mut overlay,
        "Test",
        UnitMetadataPolicy::Warn,
    )
    .expect_err("Boolean modifier mismatch remains an error");
    assert!(diagnostics.iter().any(|d| {
        d.code.as_deref() == Some("ET010") && d.severity == rumoca_core::DiagnosticSeverity::Warning
    }));
    assert!(diagnostics.iter().any(|d| {
        d.code.as_deref() == Some("ET002") && d.severity == rumoca_core::DiagnosticSeverity::Error
    }));
    assert!(
        diagnostics
            .iter()
            .any(|d| { d.code.as_deref() == Some("ET002") && d.message.contains("fixed") })
    );
    assert!(
        diagnostics
            .iter()
            .any(|d| { d.code.as_deref() == Some("ET002") && d.message.contains("quantity") })
    );
}

#[test]
fn valid_unit_metadata_has_same_typecheck_result_under_both_policies() {
    let source = r#"
        model Test
            Real x(unit = "V", quantity = "Voltage", start = 2);
        equation
            x = 2;
        end Test;
    "#;
    let resolved = resolve(parse(source)).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let model = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");

    let mut strict_overlay = InstanceOverlay::new();
    add_instanced_component(
        &mut strict_overlay,
        "Test.x",
        model.components.get("x").expect("x component"),
        true,
    );
    let strict = typecheck_instanced_with_unit_metadata_policy(
        &tree,
        &mut strict_overlay,
        "Test",
        UnitMetadataPolicy::Strict,
    )
    .expect("valid metadata should pass strict policy");

    let mut warn_overlay = InstanceOverlay::new();
    add_instanced_component(
        &mut warn_overlay,
        "Test.x",
        model.components.get("x").expect("x component"),
        true,
    );
    let warn = typecheck_instanced_with_unit_metadata_policy(
        &tree,
        &mut warn_overlay,
        "Test",
        UnitMetadataPolicy::Warn,
    )
    .expect("valid metadata should pass Warn policy");
    assert!(strict.is_empty());
    assert!(warn.is_empty());
    assert_eq!(
        strict_overlay.components.len(),
        warn_overlay.components.len()
    );
    assert_eq!(strict_overlay.classes.len(), warn_overlay.classes.len());
}

#[test]
fn test_typecheck_instanced_checks_algorithms_and_bindings() {
    let source = r#"
        model Test
            Boolean b;
            Real bound = b;
            Real assigned;
        algorithm
            assigned := b;
        end Test;
    "#;
    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let model = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");
    let mut overlay = InstanceOverlay::new();
    for name in ["b", "bound", "assigned"] {
        add_instanced_component(
            &mut overlay,
            &format!("Test.{name}"),
            model.components.get(name).expect("component"),
            true,
        );
    }

    let diagnostics = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("algorithm and binding mismatches must fail instanced typecheck");
    let mismatches = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code.as_deref() == Some("ET002"))
        .count();
    assert!(
        mismatches >= 2,
        "expected algorithm and binding mismatches, got: {diagnostics:?}"
    );
}

#[test]
fn test_typecheck_instanced_checks_reachable_component_class_equations() {
    let source = r#"
        model Worker
            Real x;
            Boolean b;
        equation
            x = b;
        end Worker;
        model Test
            Worker worker;
        end Test;
    "#;
    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let test = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");
    let worker = tree
        .get_class_by_qualified_name("Worker")
        .expect("Worker class");
    let mut overlay = InstanceOverlay::new();
    add_instanced_component(
        &mut overlay,
        "Test.worker",
        test.components.get("worker").expect("worker component"),
        false,
    );
    for name in ["x", "b"] {
        add_instanced_component(
            &mut overlay,
            &format!("Test.worker.{name}"),
            worker.components.get(name).expect("worker member"),
            true,
        );
    }

    let diagnostics = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("reachable component-class equations must be checked");
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_deref() == Some("ET002")),
        "expected component-class equation mismatch, got: {diagnostics:?}"
    );
}

/// MLS §4.7 / §10: the same class equation is elaborated once per instance
/// scope and must be shape-consistent in *every* one of them. `Worker`'s
/// `x = {1.0}` is correct inside `first` (where `n` keeps its default 1) and
/// wrong inside `second` (where the modification makes `n` 2).
#[test]
fn test_typecheck_instanced_checks_each_concrete_component_scope() {
    let source = r#"
        model Worker
            parameter Integer n = 1;
            Real x[n];
        equation
            x = {1.0};
        end Worker;
        model Test
            Worker first;
            Worker second(n = 2);
        end Test;
    "#;
    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let test = tree
        .get_class_by_qualified_name("Test")
        .expect("Test class");
    let worker = tree
        .get_class_by_qualified_name("Worker")
        .expect("Worker class");
    let mut overlay = InstanceOverlay::new();
    for instance in ["first", "second"] {
        add_instanced_component(
            &mut overlay,
            &format!("Test.{instance}"),
            test.components.get(instance).expect("worker component"),
            false,
        );
        for member in ["n", "x"] {
            add_instanced_component(
                &mut overlay,
                &format!("Test.{instance}.{member}"),
                worker.components.get(member).expect("worker member"),
                true,
            );
        }
    }
    // The instantiate phase folds `second(n = 2)` into the instance binding of
    // `Test.second.n`. Mirror exactly that, rather than overwriting `dims`
    // directly: a hand-set `dims` that disagrees with `dims_expr` is not a
    // shape the real overlay can contain, and dimension evaluation would
    // legitimately recompute it from `dims_expr`.
    let modified_n = test
        .components
        .get("second")
        .expect("second component")
        .modifications
        .get("n")
        .expect("`second(n = 2)` modification")
        .clone();
    overlay
        .components
        .values_mut()
        .find(|data| data.qualified_name.to_flat_string() == "Test.second.n")
        .expect("second worker parameter")
        .binding = Some(modified_n);

    let diagnostics = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("the second concrete Worker scope has a shape mismatch");
    let mismatches: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code.as_deref() == Some("ET002"))
        .collect();
    assert_eq!(
        mismatches.len(),
        1,
        "expected exactly the second-instance shape mismatch, got: {diagnostics:?}"
    );
    assert!(
        mismatches[0].message.contains("[2]"),
        "expected the mismatch to report the second scope's `[2]` shape, got: {:?}",
        mismatches[0]
    );
}

#[test]
fn test_typecheck_instanced_reports_unknown_builtin_modifier() {
    let source = r#"
        model Test
            Real x(startd = 1.0);
        end Test;
    "#;

    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let mut overlay = rumoca_ir_ast::InstanceOverlay::new();

    let err = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("instanced typecheck should reject unknown builtin modifiers");
    assert!(
        err.iter().any(|d| d.code.as_deref() == Some("ET001")
            && d.message.contains("unknown modifier `startd`")),
        "expected unknown modifier diagnostic in instanced pipeline, got: {:?}",
        err
    );
}

#[test]
fn test_typecheck_instanced_reports_unknown_class_component_modifier() {
    let source = r#"
        model PID
            parameter Real kp = 1.0;
        end PID;

        model Test
            PID pid(kps = 10.0);
        end Test;
    "#;

    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let mut overlay = rumoca_ir_ast::InstanceOverlay::new();

    let err = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("instanced typecheck should reject unknown class modifiers");
    assert!(
        err.iter()
            .any(|d| d.code.as_deref() == Some("ET001")
                && d.message.contains("unknown modifier `kps`")),
        "expected unknown class modifier diagnostic in instanced pipeline, got: {:?}",
        err
    );
}

#[test]
fn test_typecheck_instanced_reports_unknown_class_component_start_modifier() {
    let source = r#"
        model Main
            Test t1(start=1), t2(start=2);
        end Main;

        model Test
            Real x;
        end Test;
    "#;

    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let mut overlay = rumoca_ir_ast::InstanceOverlay::new();

    let err = typecheck_instanced(&tree, &mut overlay, "Main")
        .expect_err("instanced typecheck should reject unknown class start modifiers");
    assert!(
        err.iter().any(|d| d.code.as_deref() == Some("ET001")
            && d.message.contains("unknown modifier `start`")),
        "expected unknown class start modifier diagnostic in instanced pipeline, got: {:?}",
        err
    );
}

#[test]
fn test_typecheck_instanced_reports_unknown_nested_builtin_modifier() {
    let source = r#"
        model Plane
            Real x;
            Real y;
            Real theta;
        end Plane;

        model Test
            Plane p1(x.star88t = 1.0);
        end Test;
    "#;

    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let mut overlay = rumoca_ir_ast::InstanceOverlay::new();

    let err = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("instanced typecheck should reject unknown nested builtin modifiers");
    assert!(
        err.iter().any(|d| d.code.as_deref() == Some("ET001")
            && d.message.contains("unknown modifier `x.star88t`")),
        "expected unknown nested builtin modifier diagnostic in instanced pipeline, got: {:?}",
        err
    );
}

// The instanced pipeline never observes an unknown dotted record member either:
// resolve rejects it first with its own `ER002` diagnostic, covered by
// `rumoca-phase-resolve`'s `tests::component_lookup`.

#[test]
fn test_typecheck_instanced_reports_builtin_modifier_type_mismatch() {
    let source = r#"
        model Test
            Boolean df = true;
            Real v(start = df);
        equation
            der(v) = -v;
        end Test;
    "#;

    let parsed = parse(source);
    let resolved = resolve(parsed).expect("resolve should succeed");
    let tree = resolved.into_inner();
    let mut overlay = rumoca_ir_ast::InstanceOverlay::new();

    let err = typecheck_instanced(&tree, &mut overlay, "Test")
        .expect_err("instanced typecheck should reject incompatible modifier types");
    assert!(
        err.iter().any(|d| d.code.as_deref() == Some("ET002")
            && d.message.contains("modifier `start`")
            && d.message.contains("expects `Real`, found `Boolean`")),
        "expected modifier type mismatch diagnostic in instanced pipeline, got: {:?}",
        err
    );
}
