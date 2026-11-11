//! Evaluation with one structured-domain point bound outside a comprehension.

use super::*;

impl<'dae, F> NumericEvaluator<'dae, F>
where
    F: FnMut(dae::VariableView<'dae>, usize) -> Option<f64>,
{
    /// Evaluate an expression with one checked structured-domain point in scope.
    ///
    /// This evaluates binder-dependent expressions outside a comprehension
    /// while retaining the same coordinate and cache semantics.
    pub fn expression_at_domain_point(
        &mut self,
        id: dae::ExprId<'dae>,
        domain: dae::DomainId<'dae>,
        index_tuple: &[i64],
    ) -> Result<Vec<f64>, NumericEvaluationError> {
        let span = self
            .view
            .expression(id)
            .expect("finalized expression identity resolves")
            .provenance()
            .span();
        let domain_view = self.view.domain(domain).ok_or_else(|| {
            failure(
                NumericEvaluationErrorKind::UnsupportedOperation,
                "domain identity does not resolve",
                span,
            )
        })?;
        match domain_view.structured().ordinal_of(index_tuple) {
            Ok(Some(_)) => {}
            Ok(None) => {
                return Err(failure(
                    NumericEvaluationErrorKind::OutOfBounds,
                    "domain point does not resolve",
                    span,
                ));
            }
            Err(_) => {
                return Err(failure(
                    NumericEvaluationErrorKind::Overflow,
                    "domain projection overflowed",
                    span,
                ));
            }
        }
        self.with_domain_point(domain, index_tuple.to_vec(), id, |this| this.expression(id))
    }

    /// Bind one structured-domain point with point-local caches owned by the
    /// checked function scope of `body`, if any.
    pub(super) fn with_domain_point<T>(
        &mut self,
        domain: dae::DomainId<'dae>,
        point: Vec<i64>,
        body: dae::ExprId<'dae>,
        evaluate: impl FnOnce(&mut Self) -> Result<T, NumericEvaluationError>,
    ) -> Result<T, NumericEvaluationError> {
        self.domain_points.push((domain, point));
        let owner = self
            .view
            .expression(body)
            .and_then(|node| node.function_scope());
        self.scoped_expression_values.push(
            function_states::NumericFunctionFrame::comprehension_frame(owner),
        );
        let result = evaluate(self);
        self.scoped_expression_values.pop();
        self.domain_points.pop();
        result
    }
}
