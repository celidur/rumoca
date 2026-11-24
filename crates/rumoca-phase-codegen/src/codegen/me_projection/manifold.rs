//! Execution view of the projection onto the retained state manifold.
//!
//! Index reduction replaces lower-order constraints by their derivatives and
//! retains the originals as `continuous.manifold_residual`, partitioned into
//! blocks by `continuous.manifold_projection_plan`. The linked kernel moves
//! the block states back onto those constraints with a minimum-norm scaled
//! Newton correction (`rumoca-solver` `runtime::projection::manifold`,
//! `project_state_manifold` and `certify_state_manifold`). This module reads
//! the checked plan once and records, for the shared C kernel
//! (`templates/fmi3/me_projection.jinja`, `manifold_kernel`), the residual and
//! directional programs, each block's rows and state coordinates as ranges of
//! one index pool, and the solver variable scale of every block state.
//!
//! The view is built from the Solve problem, its artifacts and the solver
//! variable scales alone, so a target without an FMI component reads it as
//! `solve_blocks.state_manifold`. It derives execution views only; it never
//! records canonical IR.

use std::fmt;
use std::sync::{Arc, OnceLock};

use minijinja::Value;
use minijinja::value::{Enumerator, Object, ObjectRepr};
use rumoca_eval_solve::{projection_policy as policy, to_scalar_program_projection};
use rumoca_ir_solve as solve;
use serde::Serialize;

use super::super::scalar_program_plan::ScalarProgramPlan;
use super::super::solve_lazy::SolveRenderHandle;
use crate::errors::CodegenError;

fn refuse(reason: &str) -> CodegenError {
    CodegenError::dae_preparation_failed(
        format!("unsupported-feature:manifold_projection: {reason}"),
        None,
    )
}

/// One manifold block: its residual row count `m`, its state coordinate count
/// `n` (`m <= n`), and the pool offsets of its rows and coordinates.
#[derive(Serialize)]
struct ManifoldBlockRecord {
    m: usize,
    n: usize,
    rows: usize,
    y: usize,
}

/// The first solver coordinate a manifold program reads at or past `first`,
/// a seed it loads at or past `seed_len`, or a dynamically indexed seed.
fn program_reach(
    programs: &[Vec<solve::LinearOp>],
    first: usize,
    seed_len: usize,
) -> Option<&'static str> {
    programs.iter().flatten().find_map(|op| match op {
        solve::LinearOp::LoadY { index, .. } if *index >= first => {
            Some("a retained constraint reads an algebraic coordinate")
        }
        solve::LinearOp::TensorLoad {
            input: solve::TensorInputKind::Y,
            input_start,
            count,
            ..
        } if input_start
            .checked_add(*count)
            .is_none_or(|end| end > first) =>
        {
            Some("a retained constraint reads an algebraic coordinate range")
        }
        solve::LinearOp::LoadSeed { index, .. } if *index >= seed_len => {
            Some("a constraint direction seeds a coordinate outside the states")
        }
        solve::LinearOp::TensorLoad {
            seed_start: Some(start),
            count,
            ..
        } if start.checked_add(*count).is_none_or(|end| end > seed_len) => {
            Some("a constraint direction seeds a range outside the states")
        }
        solve::LinearOp::LoadIndexedSeed { .. } => {
            Some("a constraint direction loads a dynamically indexed seed")
        }
        _ => None,
    })
}

fn scalar_plan(
    block: &solve::ComputeBlock,
    state_count: usize,
) -> Result<(Value, usize), CodegenError> {
    let projection = to_scalar_program_projection(block)?.into_block();
    if let Some(reason) = program_reach(projection.programs(), state_count, state_count) {
        return Err(refuse(reason));
    }
    let outputs = projection.output_indices().len();
    Ok((
        Value::from_object(ScalarProgramPlan::new(Arc::new(projection))?),
        outputs,
    ))
}

fn float_literal(value: f64) -> String {
    format!("{value:?}")
}

/// The manifold projection's arena `(doubles, sizes)`: the deepest frame
/// chain is the projection's saved states and full residual, one block's
/// residual, Jacobian, scales, saved coordinates and Newton correction, and
/// the larger of the square solve (scaled matrix, LU, right-hand side,
/// solution and SVD factors) and the wide minimum-norm solve (transposed
/// scaled matrix, rotations, right-hand side and solution), plus the JVP seed
/// and slack.
fn arena(state_count: usize, rows: usize, max_m: usize, max_n: usize) -> (usize, usize) {
    let square = 4 * max_n * max_n + 2 * max_n;
    let wide = 2 * max_m * max_n + max_m * max_m + max_m + max_n;
    let block_doubles = 3 * max_m + max_m * max_n + 3 * max_n + square.max(wide);
    let doubles = state_count + 2 * rows + block_doubles + state_count + 64;
    (doubles, 2 * max_n + 8)
}

/// The arena the manifold projection of `problem` needs, `(0, 0)` without
/// retained constraints, for a target whose refresh arena also holds it.
pub(super) fn manifold_work(problem: &solve::SolveProblem) -> Result<(usize, usize), CodegenError> {
    let plan = &problem.continuous.manifold_projection_plan;
    if plan.is_empty() {
        return Ok((0, 0));
    }
    let rows = problem
        .continuous
        .manifold_residual
        .len()
        .map_err(|error| refuse(&error.to_string()))?;
    let (max_m, max_n) = plan
        .blocks
        .scalar_blocks()
        .iter()
        .fold((1, 1), |(m, n), block| {
            (m.max(block.rows.len()), n.max(block.y_indices.len()))
        });
    Ok(arena(
        problem.solve_layout.state_scalar_count(),
        rows,
        max_m,
        max_n,
    ))
}

/// The manifold kernel's view of `problem`, with `scale` the solver variable
/// scale of each solver coordinate (`SolveModel::solver_variable_scale`, the
/// scale the linked kernel's `manifold_variable_scale` reads).
///
/// A model without retained constraints yields `blocks = []`; the kernel then
/// emits nothing.
pub(in crate::codegen) fn manifold_value(
    problem: &solve::SolveProblem,
    artifacts: &solve::SolveArtifacts,
    scale: &dyn Fn(usize) -> f64,
) -> Result<Value, CodegenError> {
    let plan = &problem.continuous.manifold_projection_plan;
    if plan.is_empty() {
        return Ok(minijinja::context! { blocks => Vec::<Value>::new() });
    }
    let state_count = problem.solve_layout.state_scalar_count();
    let rows = problem
        .continuous
        .manifold_residual
        .len()
        .map_err(|error| refuse(&error.to_string()))?;
    let (residual, residual_outputs) =
        scalar_plan(&problem.continuous.manifold_residual, state_count)?;
    let (jvp, jvp_outputs) = scalar_plan(&artifacts.continuous.manifold_jacobian_v, state_count)?;
    if residual_outputs != rows || jvp_outputs != rows {
        return Err(refuse(
            "the constraint residual and its direction differ in row count",
        ));
    }
    let mut pool = Vec::new();
    let mut blocks = Vec::new();
    let (mut max_m, mut max_n) = (1, 1);
    for block in plan.blocks.scalar_blocks() {
        let (m, n) = (block.rows.len(), block.y_indices.len());
        if m == 0 || m > n {
            return Err(refuse("a manifold block has more rows than coordinates"));
        }
        if block.rows.iter().any(|&row| row >= rows)
            || block.y_indices.iter().any(|&index| index >= state_count)
        {
            return Err(refuse(
                "a manifold block reaches outside its residual or the states",
            ));
        }
        let rows_at = pool.len();
        pool.extend(block.rows.iter().copied());
        let y_at = pool.len();
        pool.extend(block.y_indices.iter().copied());
        blocks.push(ManifoldBlockRecord {
            m,
            n,
            rows: rows_at,
            y: y_at,
        });
        max_m = max_m.max(m);
        max_n = max_n.max(n);
    }
    let state_scales = (0..state_count)
        .map(|index| float_literal(scale(index)))
        .collect::<Vec<_>>();
    let (doubles, sizes) = arena(state_count, rows, max_m, max_n);
    Ok(minijinja::context! {
        blocks => Value::from_serialize(&blocks),
        pool => pool,
        rows => rows,
        state_count => state_count,
        max_m => max_m,
        max_n => max_n,
        residual => residual,
        jvp => jvp,
        state_scales => state_scales,
        max_iters => policy::MANIFOLD_PROJECTION_MAX_ITERS,
        work => minijinja::context! { doubles => doubles, sizes => sizes },
    })
}

/// `solve_blocks.state_manifold`: the manifold view of a rendered model,
/// built on first read so a target that never reads it pays nothing and an
/// unsupported constraint only fails the template that reads it.
pub(in crate::codegen) struct StateManifoldBlocks {
    handle: SolveRenderHandle,
    value: OnceLock<Result<Value, CodegenError>>,
}

impl fmt::Debug for StateManifoldBlocks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StateManifoldBlocks")
            .field("ready", &self.value.get().is_some())
            .finish_non_exhaustive()
    }
}

const KEYS: [&str; 11] = [
    "blocks",
    "pool",
    "rows",
    "state_count",
    "max_m",
    "max_n",
    "residual",
    "jvp",
    "state_scales",
    "max_iters",
    "work",
];

impl StateManifoldBlocks {
    pub(in crate::codegen) fn value(handle: SolveRenderHandle) -> Value {
        Value::from_object(Self {
            handle,
            value: OnceLock::new(),
        })
    }

    fn build(&self) -> Result<Value, CodegenError> {
        let problem = self.handle.problem();
        let artifacts = self.handle.artifacts();
        if problem.continuous.manifold_projection_plan.is_empty() {
            return manifold_value(problem, artifacts, &|_| 1.0);
        }
        let scales = self.handle.solver_variable_scales().ok_or_else(|| {
            refuse("the solver variable scales need the retained model or FMI component")
        })?;
        manifold_value(problem, artifacts, &|index| {
            scales.get(index).copied().unwrap_or(1.0)
        })
    }
}

impl Object for StateManifoldBlocks {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Str(&KEYS)
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        if !KEYS.contains(&key) {
            return None;
        }
        match self.value.get_or_init(|| self.build()) {
            Ok(value) => value
                .get_attr(key)
                .ok()
                .filter(|value| !value.is_undefined()),
            Err(error) => Some(Value::from(
                crate::errors::render_err(error.to_string()).with_source(error.clone()),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
