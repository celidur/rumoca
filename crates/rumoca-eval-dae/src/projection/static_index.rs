//! Exact static Integer and index projection outside a dependency walk.

use super::*;

/// Evaluate one scalar Integer or enumeration value with checked i64
/// arithmetic and static bindings.
///
/// This is intentionally a static projection: it has no runtime override
/// provider. A caller that permits tunable overrides of index dependencies
/// must resolve those indices through a separate exact-integer policy.
pub fn project_static_integer<'dae>(
    view: dae::DaeView<'dae>,
    expression: dae::ExprId<'dae>,
    scalar_index: usize,
    domain_point: Option<(dae::DomainId<'dae>, &[i64])>,
) -> Result<i64, StaticProjectionError> {
    let node = view
        .expression(expression)
        .expect("branded expression resolves in its owning DAE");
    let span = node.provenance().span();
    // MLS §10.6: `/` and scalar `^` yield Real, so an exact i64 is only the
    // value of an Integer or enumeration expression.
    if !matches!(
        node.value_type().scalar_type(),
        dae::ScalarType::Integer | dae::ScalarType::Enumeration
    ) {
        return Err(StaticProjectionError::NotIntegerExpression { span });
    }
    let mut cache = ScalarCoordinateProjectionCache::default();
    let mut visit = |_: dae::CoordinateView<'dae>, _: usize| {};
    let mut projection = Projection::new(
        view,
        checked_domain_points(view, span, domain_point)?,
        &mut cache,
        ProjectionRequest::default(),
        &mut visit,
    );
    Ok(projection.integer(expression, scalar_index)?)
}

/// Resolve an indexed expression's selected base expression and flat scalar.
///
/// The subscript is evaluated exactly as an i64 and checked against the
/// Modelica one-based axis bounds. Dynamic subscripts are returned as errors;
/// this function does not use the conservative incidence-union fallback.
pub fn project_indexed_scalar<'dae>(
    view: dae::DaeView<'dae>,
    indexed: dae::ExprId<'dae>,
    result_scalar: usize,
    domain_point: Option<(dae::DomainId<'dae>, &[i64])>,
) -> Result<(dae::ExprId<'dae>, usize), StaticProjectionError> {
    let node = view
        .expression(indexed)
        .expect("branded expression resolves in its owning DAE");
    let span = node.provenance().span();
    let dae::ExpressionOperation::Index { base, subscripts } = node.operation() else {
        return Err(StaticProjectionError::NotIndexExpression { span });
    };
    let mut cache = ScalarCoordinateProjectionCache::default();
    let mut visit = |_: dae::CoordinateView<'dae>, _: usize| {};
    let mut projection = Projection::new(
        view,
        checked_domain_points(view, span, domain_point)?,
        &mut cache,
        ProjectionRequest::default(),
        &mut visit,
    );
    projection.expect_scalar_index(node, result_scalar)?;
    let scalar = projection.indexed_base_scalar(
        base,
        subscripts,
        node.value_type().dimensions(),
        result_scalar,
    )?;
    Ok((base, scalar))
}

fn checked_domain_points<'dae>(
    view: dae::DaeView<'dae>,
    span: Span,
    domain_point: Option<(dae::DomainId<'dae>, &[i64])>,
) -> Result<Vec<(dae::DomainId<'dae>, Vec<i64>)>, StaticProjectionError> {
    let Some((domain, indices)) = domain_point else {
        return Ok(Vec::new());
    };
    let domain_view = view
        .domain(domain)
        .expect("branded domain resolves in its owning DAE");
    match domain_view.structured().ordinal_of(indices) {
        Ok(Some(_)) => Ok(vec![(domain, indices.to_vec())]),
        Ok(None) | Err(_) => Err(StaticProjectionError::InvalidDomainPoint { span }),
    }
}
