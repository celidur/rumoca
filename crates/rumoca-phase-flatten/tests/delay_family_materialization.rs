//! Delay ownership needs concrete occurrences even with compact families enabled.

use std::collections::HashSet;

use rumoca_core::{BuiltinFunction, Expression, ExpressionVisitor, Reference, Subscript};
use rumoca_ir_ast as ast;
use rumoca_ir_flat as flat;
use rumoca_phase_flatten::FlattenOptions;

fn flatten_source(source: &str) -> flat::Model {
    let file_name = "<delay_family_materialization>";
    let stored = rumoca_phase_parse::parse_to_ast(source, file_name).expect("source parses");
    let mut tree = ast::ClassTree::from_parsed(stored);
    tree.source_map.add(file_name, source);
    let resolved =
        rumoca_phase_resolve::resolve(ast::ParsedTree::new(tree)).expect("source resolves");
    let instanced =
        rumoca_phase_instantiate::instantiate(resolved, "M").expect("model instantiates");
    let ast::InstancedTree { tree, mut overlay } = instanced;
    rumoca_phase_typecheck::typecheck_instanced(&tree, &mut overlay, "M")
        .expect("instanced model typechecks");
    rumoca_phase_flatten::flatten_ref_with_options(
        &tree,
        &overlay,
        "M",
        FlattenOptions {
            materialize_structured_families: false,
            ..FlattenOptions::default()
        },
    )
    .expect("model flattens")
}

#[derive(Default)]
struct DelayCalls {
    args: Vec<Vec<Expression>>,
    references: Vec<String>,
}

impl ExpressionVisitor for DelayCalls {
    fn visit_builtin_call(&mut self, function: &BuiltinFunction, args: &[Expression]) {
        if *function == BuiltinFunction::Delay {
            self.args.push(args.to_vec());
        }
        self.walk_builtin_call(function, args);
    }

    fn visit_var_ref(&mut self, name: &Reference, subscripts: &[Subscript]) {
        self.references.push(name.as_str().to_owned());
        self.walk_var_ref(name, subscripts);
    }
}

fn assert_concrete_delay_rows(model: &flat::Model, rows: usize, arity: usize) -> DelayCalls {
    assert_eq!(model.equations.len(), rows);
    let mut owned_rows = vec![false; rows];
    for family in &model.structured_equations {
        assert!(family.template.is_none(), "delay template must not survive");
        assert!(
            family.interiors_materialized(),
            "delay rows must be complete"
        );
        let family_rows = family
            .domain
            .index_tuple_iter()
            .expect("valid domain")
            .len()
            * family.equations_per_point;
        let end = family.first_equation_index + family_rows;
        assert!(end <= rows, "family ownership stays within scalar rows");
        for owned in &mut owned_rows[family.first_equation_index..end] {
            assert!(!*owned, "family row ownership must not overlap");
            *owned = true;
        }
        assert!(!family.span.is_dummy(), "keep source span");
    }
    assert!(
        owned_rows.iter().all(|owned| *owned),
        "every row has an owner"
    );
    let mut calls = DelayCalls::default();
    for equation in &model.equations {
        let before = calls.args.len();
        calls.visit_expression(&equation.residual);
        assert_eq!(calls.args.len(), before + 1, "every row retains its delay");
    }
    assert!(calls.args.iter().all(|args| args.len() == arity));
    for name in &calls.references {
        assert!(
            name != "i"
                && name != "j"
                && !name.contains("[i")
                && !name.contains(",i")
                && !name.contains("[j")
                && !name.contains(",j"),
            "unbound loop index in {name}"
        );
    }
    calls
}

#[test]
fn parameter_delay_time_is_bound_without_a_compact_template() {
    let model = flatten_source(
        r#"model M
          parameter Real tau[4] = {0.1, 0.2, 0.3, 0.4};
          Real y[4];
        equation
          for i in 1:4 loop
            y[i] = delay(time, tau[i]);
          end for;
        end M;"#,
    );
    let calls = assert_concrete_delay_rows(&model, 4, 2);
    let timings: HashSet<_> = calls
        .args
        .iter()
        .map(|args| format!("{:?}", args[1]))
        .collect();
    assert_eq!(
        timings.len(),
        4,
        "each timing argument uses its bound element"
    );
}

#[test]
fn derivative_delay_family_keeps_every_indexed_source() {
    let model = flatten_source(
        r#"model M
          Real x[4](each start = 1);
        equation
          for i in 1:4 loop
            der(x[i]) = delay(x[i], 0.1);
          end for;
        end M;"#,
    );
    let calls = assert_concrete_delay_rows(&model, 4, 2);
    let sources: HashSet<_> = calls
        .args
        .iter()
        .map(|args| format!("{:?}", args[0]))
        .collect();
    assert_eq!(
        sources.len(),
        4,
        "all scalar sources survive corner cheapening"
    );
    assert!(
        model
            .structured_equations
            .iter()
            .all(|family| family.regular.is_some())
    );
}

#[test]
fn nested_delay_family_keeps_both_bound_indices_and_maximum_delay() {
    let model = flatten_source(
        r#"model M
          Real x[3, 4](each start = 1);
        equation
          for i in 1:3 loop
            for j in 1:4 loop
              der(x[i, j]) = delay(x[i, j], 0.1, 0.5);
            end for;
          end for;
        end M;"#,
    );
    let calls = assert_concrete_delay_rows(&model, 12, 3);
    let sources: HashSet<_> = calls
        .args
        .iter()
        .map(|args| format!("{:?}", args[0]))
        .collect();
    assert_eq!(sources.len(), 12);
    assert!(calls.args.iter().all(|args| matches!(
        &args[2], Expression::Literal { value: rumoca_core::Literal::Real(value), .. }
            if *value == 0.5
    )));
    // Without a template, the existing lifting proof retains each inner
    // family. Their concrete row intervals retain the outer-index ownership.
    assert_eq!(model.structured_equations.len(), 3);
    for (outer_index, family) in model.structured_equations.iter().enumerate() {
        assert_eq!(family.first_equation_index, outer_index * 4);
        assert_eq!(family.equations_per_point, 1);
        assert_eq!(family.domain.binders.len(), 1);
        let binder = &family.domain.binders[0];
        assert_eq!(binder.display_name, "j");
        assert_eq!((binder.lower, binder.upper, binder.step), (1, 4, 1));
    }
}

#[test]
fn conditional_tlm_delay_family_keeps_concrete_sources() {
    let model = flatten_source(
        r#"model M
          parameter Real tau[4] = {0.1, 0.2, 0.3, 0.4};
          input Real x[4];
          Real y[4];
        equation
          for i in 1:4 loop
            if time < tau[i] then
              y[i] = x[i];
            else
              y[i] = delay(x[i], tau[i], 0.5);
            end if;
          end for;
        end M;"#,
    );
    let calls = assert_concrete_delay_rows(&model, 4, 3);
    let sources: HashSet<_> = calls
        .args
        .iter()
        .map(|args| format!("{:?}", args[0]))
        .collect();
    assert_eq!(sources.len(), 4);
}

#[test]
fn non_delay_derivative_family_keeps_compact_lowering() {
    let model = flatten_source(
        r#"model M
          Real x[4](each start = 1);
        equation
          for i in 1:4 loop
            der(x[i]) = -x[i];
          end for;
        end M;"#,
    );
    assert_eq!(model.structured_equations.len(), 1);
    let family = &model.structured_equations[0];
    assert!(family.regular.is_some());
    assert!(family.template.is_some());
    assert!(!family.interiors_materialized());
}
