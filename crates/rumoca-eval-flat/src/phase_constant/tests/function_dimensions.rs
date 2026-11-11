use super::*;

fn copy_function(default_extra: rumoca_core::Expression) -> rumoca_core::Function {
    let mut function = rumoca_core::Function::new("Pkg.copy", test_span());
    function.add_input(function_param("v", TestScalar::Real, vec![0]));
    function.add_input(
        function_param("extra", TestScalar::Integer, vec![]).with_default(default_extra),
    );
    function.add_output(
        function_param("y", TestScalar::Real, vec![0]).with_shape_expr(vec![
            rumoca_core::Subscript::expr(
                Box::new(binary(
                    rumoca_core::OpBinary::Add,
                    call(rumoca_core::BuiltinFunction::Size, vec![var("v"), int(1)]),
                    var("extra"),
                )),
                test_span(),
            ),
        ]),
    );
    function
}

fn named(name: &str, value: rumoca_core::Expression) -> rumoca_core::Expression {
    rumoca_core::Expression::FunctionCall {
        name: rumoca_core::Reference::new(format!(
            "{}{name}",
            rumoca_core::NAMED_FUNCTION_ARG_PREFIX
        )),
        args: vec![value],
        is_constructor: true,
        span: test_span(),
    }
}

fn infer_copy(
    args: Vec<rumoca_core::Expression>,
    default_extra: rumoca_core::Expression,
    array_dims: FxHashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let functions = FxHashMap::from_iter([("Pkg.copy".into(), copy_function(default_extra))]);
    // Same-named caller parameters must not replace function defaults.
    let known_ints = FxHashMap::from_iter([("extra".into(), 99)]);
    infer_array_dimensions_full_with_functions(
        &function_call("Pkg.copy", args),
        &ParamEvalContext::new(
            &known_ints,
            &FxHashMap::default(),
            &FxHashMap::default(),
            &FxHashMap::default(),
            &array_dims,
            &functions,
            Some("plant.y"),
        ),
    )
}

#[test]
fn array_formal_shape_uses_literal_and_default_scalar() {
    let array = rumoca_core::Expression::Array {
        elements: vec![real(3.0), real(1.0), real(2.0)],
        kind: rumoca_core::ArrayConstructor::Array,
        span: test_span(),
    };
    assert_eq!(
        infer_copy(vec![array], int(1), FxHashMap::default()),
        Some(vec![4])
    );
}

#[test]
fn array_formal_shape_does_not_require_array_contents() {
    let dims = FxHashMap::from_iter([("plant.values".into(), vec![7])]);
    assert_eq!(infer_copy(vec![var("values")], int(1), dims), Some(vec![8]));
}

#[test]
fn array_formal_named_arguments_bind_by_name() {
    let dims = FxHashMap::from_iter([("plant.values".into(), vec![7])]);
    assert_eq!(
        infer_copy(
            vec![named("extra", int(2)), named("v", var("values"))],
            int(1),
            dims
        ),
        Some(vec![9])
    );
}

#[test]
fn array_formal_default_can_read_preceding_formal_shape() {
    let dims = FxHashMap::from_iter([("plant.values".into(), vec![7])]);
    assert_eq!(
        infer_copy(
            vec![var("values")],
            call(rumoca_core::BuiltinFunction::Size, vec![var("v"), int(1)]),
            dims,
        ),
        Some(vec![14])
    );
}

#[test]
fn array_formal_preserves_boolean_default() {
    let mut function = copy_function(int(0));
    let boolean_type = rumoca_core::TypeId::new(3);
    function.add_input(
        rumoca_core::FunctionParam::new(
            "ascending",
            "Boolean",
            rumoca_core::EffectiveType::new(boolean_type, boolean_type, vec![]).unwrap(),
            test_span(),
        )
        .with_default(rumoca_core::Expression::Literal {
            value: rumoca_core::Literal::Boolean(true),
            span: test_span(),
        }),
    );
    let functions = FxHashMap::from_iter([("Pkg.copy".into(), function)]);
    let dims = FxHashMap::from_iter([("values".into(), vec![3])]);
    assert_eq!(
        infer_array_dimensions_full_with_functions(
            &function_call("Pkg.copy", vec![var("values")]),
            &ParamEvalContext::new(
                &FxHashMap::default(),
                &FxHashMap::default(),
                &FxHashMap::default(),
                &FxHashMap::default(),
                &dims,
                &functions,
                None,
            ),
        ),
        Some(vec![3])
    );
}

#[test]
fn array_formal_unknown_values_and_duplicate_arguments_are_refused() {
    let dims = FxHashMap::from_iter([("plant.values".into(), vec![7])]);
    assert_eq!(infer_copy(vec![var("missing")], int(1), dims.clone()), None);
    assert_eq!(
        infer_copy(vec![var("values"), var("dynamic")], int(1), dims.clone()),
        None
    );
    assert_eq!(
        infer_copy(
            vec![var("values"), named("v", var("values"))],
            int(1),
            dims.clone()
        ),
        None
    );
    assert_eq!(
        infer_copy(vec![named("unknown", var("values"))], int(1), dims),
        None
    );
}

#[test]
fn array_formal_shape_infers_cat_of_projected_columns() {
    let dims = FxHashMap::from_iter([("plant.table".into(), vec![2, 2])]);
    let column = rumoca_core::Expression::VarRef {
        name: rumoca_core::Reference::new("table"),
        subscripts: vec![
            rumoca_core::Subscript::colon(test_span()),
            rumoca_core::Subscript::index(1, test_span()),
        ],
        span: test_span(),
    };
    let negated = rumoca_core::Expression::Unary {
        op: rumoca_core::OpUnary::Minus,
        rhs: Box::new(column.clone()),
        span: test_span(),
    };
    let zero = rumoca_core::Expression::Array {
        elements: vec![real(0.0)],
        kind: rumoca_core::ArrayConstructor::Array,
        span: test_span(),
    };
    let concatenated = call(
        rumoca_core::BuiltinFunction::Cat,
        vec![int(1), negated, zero, column],
    );
    assert_eq!(infer_copy(vec![concatenated], int(0), dims), Some(vec![5]));
}

#[test]
fn cat_shape_requires_valid_axis_rank_and_matching_other_extents() {
    let dims = FxHashMap::from_iter([
        ("a".into(), vec![2, 3]),
        ("b".into(), vec![4, 3]),
        ("c".into(), vec![4, 2]),
        ("v".into(), vec![3]),
    ]);
    let known_ints = FxHashMap::default();
    let known_reals = FxHashMap::default();
    let known_bools = FxHashMap::default();
    let known_enums = FxHashMap::default();
    let functions = FxHashMap::default();
    let ctx = ParamEvalContext::new(
        &known_ints,
        &known_reals,
        &known_bools,
        &known_enums,
        &dims,
        &functions,
        None,
    );
    for (axis, rhs, expected) in [
        (int(1), "b", Some(vec![6, 3])),
        (int(2), "a", Some(vec![2, 6])),
        (int(0), "b", None),
        (int(3), "b", None),
        (var("dynamic"), "b", None),
        (int(1), "c", None),
        (int(1), "v", None),
    ] {
        assert_eq!(
            infer_array_dimensions_full_with_functions(
                &call(
                    rumoca_core::BuiltinFunction::Cat,
                    vec![axis, var("a"), var(rhs)]
                ),
                &ctx,
            ),
            expected
        );
    }
}
