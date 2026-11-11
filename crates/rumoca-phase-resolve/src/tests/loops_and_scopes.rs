//! Loop and nested-class scoping: for/while index scopes, range expression
//! resolution, and nested class DefId assignment.

use super::*;

#[test]
fn test_for_loop_scope() {
    let source = r#"
model Test
Real x[3];
equation
for i in 1:3 loop
    x[i] = i;
end for;
end Test;
"#;
    let result = resolve_parsed_tree_source(source);
    assert!(result.is_ok(), "resolution should succeed");
}

#[test]
fn loop_iterator_identity_does_not_replace_a_top_level_class_entry() {
    let source = r#"
model i
end i;

model Test
  Real x[2];
equation
  for i in 1:2 loop
    x[i] = i;
  end for;
end Test;
"#;
    let tree = resolve_parsed_tree_source(source)
        .unwrap_or_else(|diagnostics| panic!("resolution failed: {diagnostics:#?}"))
        .into_inner();
    let class_id = tree
        .definitions
        .classes
        .get("i")
        .and_then(|class| class.def_id)
        .expect("top-level class identity");

    assert_eq!(tree.name_map.get("i"), Some(&class_id));
    assert_eq!(
        tree.def_map
            .values()
            .filter(|name| name.as_str() == "i")
            .count(),
        1,
        "the scope-local iterator must not enter the qualified class-name index"
    );
}

#[test]
fn test_for_equation_range_resolves() {
    let source = r#"
model Test
parameter Integer n = 3;
Real x[n];
equation
for i in 1:n loop
    x[i] = i;
end for;
end Test;
"#;
    let tree = resolve_tree_source(source).into_inner();
    let model = tree
        .definitions
        .classes
        .get("Test")
        .expect("Test should exist");
    let rumoca_ir_ast::Equation::For { indices, .. } = &model.equations[0] else {
        panic!("expected for-equation");
    };
    let range_expr = &indices[0].range;
    assert!(
        find_comp_ref_def_id(range_expr).is_some(),
        "range expression should resolve component references"
    );
}

#[test]
fn test_for_statement_range_resolves() {
    let source = r#"
model Test
parameter Integer n = 3;
Integer x;
algorithm
for i in 1:n loop
    x := i;
end for;
end Test;
"#;
    let tree = resolve_tree_source(source).into_inner();
    let model = tree
        .definitions
        .classes
        .get("Test")
        .expect("Test should exist");
    let stmt = model.algorithms[0].first().expect("for statement");
    let rumoca_ir_ast::Statement::For { indices, .. } = stmt else {
        panic!("expected for-statement");
    };
    let range_expr = &indices[0].range;
    assert!(
        find_comp_ref_def_id(range_expr).is_some(),
        "range expression should resolve component references"
    );
}

#[test]
fn test_while_condition_resolves() {
    let source = r#"
model Test
Integer n = 3;
algorithm
while n > 0 loop
    n := n - 1;
end while;
end Test;
"#;
    let tree = resolve_tree_source(source).into_inner();
    let model = tree
        .definitions
        .classes
        .get("Test")
        .expect("Test should exist");
    let stmt = model.algorithms[0].first().expect("while statement");
    let rumoca_ir_ast::Statement::While(block) = stmt else {
        panic!("expected while-statement");
    };
    assert!(
        find_comp_ref_def_id(&block.cond).is_some(),
        "while condition should resolve component references"
    );
}

#[test]
fn test_nested_class_resolution() {
    let source = r#"
package TestPkg
model Inner
    Real x;
end Inner;
end TestPkg;
"#;
    let result = resolve_parsed_tree_source(source);
    assert!(result.is_ok(), "resolution should succeed");

    let tree = result.unwrap().into_inner();
    let pkg = tree
        .definitions
        .classes
        .get("TestPkg")
        .expect("TestPkg should exist");
    assert!(pkg.def_id.is_some());

    let inner = pkg.classes.get("Inner").expect("Inner should exist");
    assert!(inner.def_id.is_some());
}

#[test]
fn delay_parameter_accepts_for_equation_array_subscripts() {
    resolve_test_source(
        r#"
model DelayLoop
  parameter Integer m = 2;
  parameter Real tau[m] = {0.1, 0.2};
  Real y[m];
equation
  for i in 1:m loop
    if time < tau[i] then
      y[i] = 0;
    else
      y[i] = delay(time, tau[i]);
    end if;
  end for;
end DelayLoop;
"#,
    )
    .expect("parameter array indexed by a structural iterator is a parameter expression");
}

#[test]
fn delay_parameter_accepts_three_argument_for_equation() {
    resolve_test_source(
        r#"
model DelayLoop3arg
  parameter Real tau[2] = {0.1, 0.2};
  Real y[2];
equation
  for i in 1:2 loop
    y[i] = delay(time, 0.05 + 0.01 * sin(time), tau[i]);
  end for;
end DelayLoop3arg;
"#,
    )
    .expect("delayMax may use a parameter iterator; delayTime may vary");
}

#[test]
fn delay_parameter_accepts_for_statement_array_subscripts() {
    resolve_test_source(
        r#"
model Test
  parameter Integer m = 2;
  parameter Real tau[m] = {0.1, 0.2};
  Integer i = 1;
  Real y[m];
algorithm
  for i in 1:m loop
    y[i] := delay(time, tau[i]);
  end for;
end Test;
"#,
    )
    .expect("parameter-range iterator must hide a non-parameter class component");
}

#[test]
fn delay_parameter_accepts_array_comprehension_subscripts() {
    resolve_test_source(
        r#"
model Test
  parameter Integer m = 2;
  parameter Real tau[m] = {0.1, 0.2};
  Real y[m];
equation
  y = {delay(time, tau[i]) for i in 1:m};
end Test;
"#,
    )
    .expect("array comprehension supplies its own parameter-range iterator scope");
}

#[test]
fn delay_parameter_accepts_nested_shadowing_and_enclosing_range() {
    resolve_test_source(
        r#"
model Test
  parameter Integer i = 2;
  parameter Real tau[2] = {0.1, 0.2};
  Real y[2,2];
equation
  for i in 1:i loop
    for i in 1:i loop
      y[i,i] = delay(time, tau[i]);
    end for;
  end for;
end Test;
"#,
    )
    .expect("each range resolves before binding the new iterator of the same name");
}

fn assert_delay_parameter_error_count(source: &str, expected: usize) {
    let diagnostics = resolve_test_source(source).expect_err("invalid delay must fail resolution");
    let count = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code.as_deref() == Some("ER055"))
        .count();
    assert_eq!(count, expected, "unexpected diagnostics: {diagnostics:#?}");
}

#[test]
fn delay_parameter_rejects_dynamic_for_equation_range() {
    assert_delay_parameter_error_count(
        r#"
model Test
  Integer n = 2;
  parameter Real tau[2] = {0.1, 0.2};
  Real y[2];
equation
  for i in 1:n loop
    y[i] = delay(time, tau[i]);
  end for;
end Test;
"#,
        1,
    );
}

#[test]
fn delay_parameter_rejects_dynamic_nested_index_and_restores_outer_scope() {
    assert_delay_parameter_error_count(
        r#"
model Test
  parameter Real tau[2] = {0.1, 0.2};
  Integer n = 2;
  Real y[2];
algorithm
  for i in 1:2 loop
    for i in 1:n loop
      y[i] := delay(time, tau[i]);
    end for;
    y[i] := delay(time, tau[i]);
  end for;
end Test;
"#,
        1,
    );
}

#[test]
fn delay_parameter_rejects_dynamic_comprehension_shadowing_parameter() {
    assert_delay_parameter_error_count(
        r#"
model Test
  parameter Integer i = 1;
  parameter Real tau[2] = {0.1, 0.2};
  Integer n = 2;
  Real y[2];
equation
  y = {delay(time, 0.05, tau[i]) for i in 1:n};
end Test;
"#,
        1,
    );
}

#[test]
fn delay_parameter_rejects_dynamic_subscript_inside_parameter_loop() {
    assert_delay_parameter_error_count(
        r#"
model Test
  parameter Real tau[2] = {0.1, 0.2};
  Integer j = 1;
  Real y[2];
equation
  for i in 1:2 loop
    y[i] = delay(time, tau[j]);
  end for;
end Test;
"#,
        1,
    );
}

#[test]
fn delay_parameter_iterator_does_not_leak_into_following_class() {
    assert_delay_parameter_error_count(
        r#"
package Test
  model First
    parameter Real tau[2] = {0.1, 0.2};
    Real y[2];
  equation
    for i in 1:2 loop
      y[i] = delay(time, tau[i]);
    end for;
  end First;
  model Second
    parameter Real tau[2] = {0.1, 0.2};
    Integer i = 1;
    Real y;
  equation
    y = delay(time, tau[i]);
  end Second;
end Test;
"#,
        1,
    );
}

#[test]
fn delay_parameter_for_statement_later_range_sees_earlier_iterator() {
    resolve_test_source(
        r#"
model Test
  parameter Real tau[2] = {0.1, 0.2};
  Real y;
algorithm
  for i in 1:2, j in {integer(delay(time, tau[i]))} loop
    y := j;
  end for;
end Test;
"#,
    )
    .expect("later algorithm range must see the preceding parameter iterator");
}

#[test]
fn delay_parameter_for_equation_later_range_sees_earlier_iterator() {
    resolve_test_source(
        r#"
model Test
  parameter Real tau[2] = {0.1, 0.2};
  Real y[2];
equation
  for i in 1:2, j in {integer(delay(1.0, tau[i]))} loop
    y[i] = j;
  end for;
end Test;
"#,
    )
    .expect("later equation range must see the preceding parameter iterator");
}

#[test]
fn delay_parameter_later_range_sees_outer_same_name_before_binding() {
    resolve_test_source(
        r#"
model Test
  parameter Real tau[2] = {0.1, 0.2};
  Real y;
algorithm
  for i in 1:2 loop
    for j in 1:2, i in {integer(delay(time, tau[i] + tau[j]))} loop
      y := j;
    end for;
  end for;
end Test;
"#,
    )
    .expect("new i range must see outer i and preceding j before shadowing i");
}

#[test]
fn delay_parameter_iterator_is_not_bound_in_its_own_range() {
    assert_delay_parameter_error_count(
        r#"
model Test
  parameter Real tau[2] = {0.1, 0.2};
  Integer i = 1;
  Real y;
algorithm
  for j in 1:2, i in {integer(delay(time, tau[i]))} loop
    y := j;
  end for;
end Test;
"#,
        1,
    );
}
