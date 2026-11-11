//! Keep delay occurrences concrete while retaining structured row ownership.

use std::ops::ControlFlow;

use rumoca_core::{BuiltinFunction, ComprehensionTemplate, Expression, ExpressionVisitor};
use rumoca_ir_ast as ast;

pub(super) fn contains_delay(
    template: Option<&ComprehensionTemplate>,
    equations: &[ast::Equation],
) -> bool {
    if let Some(template) = template {
        let mut finder = FlatDelayFinder(false);
        for residual in &template.body {
            finder.visit_expression(residual);
        }
        return finder.0;
    }

    // Conditional/when bodies cannot be captured as templates. Walk the
    // original equations, including nested bodies, before any corner cheapening.
    let mut finder = AstDelayFinder;
    equations
        .iter()
        .any(|equation| ast::Visitor::visit_equation(&mut finder, equation).is_break())
}

struct FlatDelayFinder(bool);

impl ExpressionVisitor for FlatDelayFinder {
    fn visit_builtin_call(&mut self, function: &BuiltinFunction, args: &[Expression]) {
        if *function == BuiltinFunction::Delay {
            self.0 = true;
        } else {
            self.walk_builtin_call(function, args);
        }
    }
}

struct AstDelayFinder;

impl ast::Visitor for AstDelayFinder {
    fn visit_expr_function_call(
        &mut self,
        comp: &ast::ComponentReference,
        args: &[ast::Expression],
    ) -> ControlFlow<()> {
        // Match the same single-part builtin spelling as AST lowering. A
        // qualified user function with a final part named `delay` is unrelated.
        if comp.parts.len() == 1
            && BuiltinFunction::from_name(&comp.parts[0].ident.text) == Some(BuiltinFunction::Delay)
        {
            return ControlFlow::Break(());
        }
        self.visit_each(args, Self::visit_expression)
    }
}
