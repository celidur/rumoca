//! Keep delay occurrences concrete while retaining structured row ownership.

use rumoca_core::{BuiltinFunction, ComprehensionTemplate, Expression, ExpressionVisitor};

/// True when a residual of `template` calls `delay` (MLS §3.7.2).
pub(super) fn template_contains_delay(template: &ComprehensionTemplate) -> bool {
    let mut finder = DelayFinder(false);
    for residual in &template.body {
        finder.visit_expression(residual);
    }
    finder.0
}

struct DelayFinder(bool);

impl ExpressionVisitor for DelayFinder {
    fn visit_builtin_call(&mut self, function: &BuiltinFunction, args: &[Expression]) {
        if *function == BuiltinFunction::Delay {
            self.0 = true;
        } else {
            self.walk_builtin_call(function, args);
        }
    }
}
