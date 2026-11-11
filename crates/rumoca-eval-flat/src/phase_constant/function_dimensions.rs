//! Function-local scalar values and array shapes used by output dimensions
//! (MLS §12.4.1).

use rustc_hash::{FxHashMap, FxHashSet};

use super::{DimensionScope, ParamEvalContext, infer_dimensions_scoped, named_call_arg};
use crate::constant::Value;

/// The values and array shapes a function call binds to its inputs while its
/// result shape is evaluated.
pub(super) struct DimensionArgs {
    ints: FxHashMap<String, i64>,
    reals: FxHashMap<String, f64>,
    bools: FxHashMap<String, bool>,
    values: FxHashMap<String, Value>,
    array_dims: FxHashMap<String, Vec<i64>>,
}

/// What one formal input is bound to.
enum Binding {
    Integer(i64),
    Real(f64),
    Boolean(bool),
    /// An evaluated String or array value, with the array's shape if known.
    Value(Value, Option<Vec<i64>>),
    /// An array actual whose shape is known but whose contents are not.
    Shape(Vec<i64>),
}

impl Binding {
    /// Classify `expr` as an actual of `param`. A scalar formal never accepts
    /// an array, and an array formal needs at least a known shape.
    fn infer(
        param: &rumoca_core::FunctionParam,
        expr: &rumoca_core::Expression,
        ctx: &DimensionScope<'_, '_>,
    ) -> Option<Self> {
        let shape = infer_dimensions_scoped(expr, ctx).filter(|shape| !shape.is_empty());
        if shape
            .as_ref()
            .is_some_and(|shape| shape.iter().any(|extent| *extent < 0))
        {
            return None;
        }
        if shape.is_none() && !param.dimensions().is_empty() {
            return None;
        }
        Some(match ctx.value_of(expr) {
            Some(Value::Integer(value)) => Self::Integer(value),
            Some(Value::Real(value)) => Self::Real(value),
            Some(Value::Bool(value)) => Self::Boolean(value),
            Some(value @ (Value::String(_) | Value::Array(_))) => Self::Value(value, shape),
            Some(Value::Enum(..) | Value::Record(_)) => return None,
            None => Self::Shape(shape?),
        })
    }
}

impl DimensionArgs {
    /// The caller's bindings with every formal input name shadowed, even when
    /// the formal's value turns out to be unavailable.
    fn shadowing(func: &rumoca_core::Function, ctx: &DimensionScope<'_, '_>) -> Self {
        let mut args = Self {
            ints: ctx.known_ints.clone(),
            reals: ctx.known_reals.clone(),
            bools: ctx.known_bools.clone(),
            values: FxHashMap::default(),
            array_dims: ctx.array_dims.clone(),
        };
        for param in &func.inputs {
            args.ints.remove(&param.name);
            args.reals.remove(&param.name);
            args.bools.remove(&param.name);
            args.array_dims.remove(&param.name);
        }
        args
    }

    /// The evaluation context of the callee's dimension expressions.
    pub(super) fn context<'s, 'b: 's>(
        &'s self,
        ctx: &DimensionScope<'_, 'b>,
    ) -> ParamEvalContext<'s> {
        ParamEvalContext {
            known_ints: &self.ints,
            known_reals: &self.reals,
            known_bools: &self.bools,
            known_enums: ctx.known_enums,
            array_dims: &self.array_dims,
            known_values: &self.values,
            resources: None,
            functions: ctx.functions,
            var_context: None,
        }
    }

    fn insert(&mut self, name: &str, binding: Binding) {
        let name = name.to_string();
        match binding {
            Binding::Integer(value) => {
                self.ints.insert(name, value);
            }
            Binding::Real(value) => {
                self.reals.insert(name, value);
            }
            Binding::Boolean(value) => {
                self.bools.insert(name, value);
            }
            Binding::Value(value, shape) => {
                if let Some(shape) = shape {
                    self.array_dims.insert(name.clone(), shape);
                }
                self.values.insert(name, value);
            }
            Binding::Shape(shape) => {
                self.array_dims.insert(name, shape);
            }
        }
    }
}

/// Bind a call's actual and default arguments to the callee's formal inputs.
///
/// Named arguments bind by formal name; an unknown name or a formal supplied
/// twice is refused. Defaults are evaluated in the callee scope, after the
/// supplied arguments and preceding defaults are bound (MLS §12.4.1).
pub(super) fn bind_function_dimension_args(
    func: &rumoca_core::Function,
    args: &[rumoca_core::Expression],
    ctx: &DimensionScope<'_, '_>,
) -> Option<DimensionArgs> {
    let mut locals = DimensionArgs::shadowing(func, ctx);
    let mut supplied = FxHashSet::default();
    let mut positional = 0usize;
    for arg in args {
        let (param, value) = if let Some((name, value)) = named_call_arg(arg) {
            (func.inputs.iter().find(|param| param.name == name)?, value)
        } else {
            let param = func.inputs.get(positional)?;
            positional += 1;
            (param, arg)
        };
        if !supplied.insert(param.name.as_str()) {
            return None;
        }
        locals.insert(&param.name, Binding::infer(param, value, ctx)?);
    }

    for param in &func.inputs {
        if supplied.contains(param.name.as_str()) {
            continue;
        }
        let default = param.default.as_ref()?;
        let binding = {
            let local_ctx = locals.context(ctx);
            Binding::infer(param, default, &DimensionScope::new(&local_ctx))?
        };
        locals.insert(&param.name, binding);
    }
    Some(locals)
}
