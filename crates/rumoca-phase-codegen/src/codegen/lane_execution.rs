//! Per-lane execution view of a hybrid Solve problem for batched targets.
//!
//! A batched GPU kernel runs one independent model instance per thread and
//! cannot call back into the linked runtime. This module reads the checked
//! Solve owners once and exposes them as straight-line scalar programs plus
//! small tables a per-thread kernel executes on its own:
//!
//! - the continuous algebraic refresh as the ordered exact-assignment
//!   schedules of [`solve::ContinuousRefreshOwners::staged_refresh_steps`];
//! - the event partition: root rows with their relation-memory slots and zero
//!   domains, the discrete rows with their targets and roles, and the static
//!   time-event instants;
//! - the delay partition: source, delay-time and delay-max rows and the
//!   parameter slots that receive delayed values;
//! - the initialization: the ordered parameter and start bindings;
//! - the model's start values when the renderer retains a complete model.
//!
//! It derives execution views only; it never records canonical IR. A shape a
//! per-lane kernel cannot reproduce is refused with an `unsupported-feature`
//! error when a template first reads the view. The view holds no solver: an
//! algebraic projection stage, an initialization residual block and a retained
//! state-manifold projection belong to the one projection kernel
//! (SPEC_0044 ME-PROJ-001) and are refused here.

use std::sync::{Arc, OnceLock};

use minijinja::Value;
use minijinja::value::{Enumerator, Object, ObjectRepr};
use rumoca_eval_solve::lower_scalar_ops;
use rumoca_ir_solve as solve;
use serde::Serialize;

use super::solve_lazy::SolveRenderHandle;
use crate::errors::CodegenError;

fn refuse(feature: &str, reason: impl std::fmt::Display) -> CodegenError {
    CodegenError::dae_preparation_failed(
        format!("unsupported-feature:{feature}: {reason}; the per-lane kernel cannot execute it"),
        None,
    )
}

/// One storage slot a lane program writes or iterates.
#[derive(Serialize, Clone, Copy)]
struct Slot {
    /// `"y"` or `"p"`.
    column: &'static str,
    index: usize,
}

fn slot(target: solve::ScalarSlot, feature: &str) -> Result<Slot, CodegenError> {
    match target {
        solve::ScalarSlot::Y { index, .. } => Ok(Slot { column: "y", index }),
        solve::ScalarSlot::P { index, .. } => Ok(Slot { column: "p", index }),
        _ => Err(refuse(feature, "a target is not Y or P storage")),
    }
}

/// Lower every program of `block` to scalar ops and place its stored outputs
/// at `outputs` (one index per stored output, in store order).
fn scalar_block_value(
    programs: Vec<Vec<solve::LinearOp>>,
    spans: Vec<rumoca_core::Span>,
    outputs: Vec<usize>,
    pure_calls: Option<&solve::SolvePureCallTable>,
    feature: &str,
) -> Result<Value, CodegenError> {
    if programs.is_empty() {
        return Ok(empty_block());
    }
    let lowered = programs
        .iter()
        .map(|program| {
            // A conditional the lowering leaves compact, and a call whose
            // body cannot be inlined (a data-dependent loop), render natively:
            // the latter as a call to its typed device function.
            lower_scalar_ops(program, pure_calls)
                .or_else(|| {
                    program
                        .iter()
                        .any(|op| {
                            matches!(
                                op,
                                solve::LinearOp::FunctionConditional { .. }
                                    | solve::LinearOp::PureCall { .. }
                                    | solve::LinearOp::PureCallObservation { .. }
                            )
                        })
                        .then(|| program.clone())
                })
                .ok_or_else(|| {
                    let ops = program
                        .iter()
                        .map(solve::LinearOp::kind_name)
                        .collect::<std::collections::BTreeSet<_>>();
                    refuse(
                        feature,
                        format!("a program holds an op no scalar kernel renders (ops: {ops:?})"),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let block = solve::ScalarProgramBlock::with_output_indices(lowered, spans, outputs)
        .map_err(|error| CodegenError::template(error.to_string()))?;
    super::solve_template_compute_block_json(&solve::ComputeBlock {
        nodes: vec![solve::ComputeNode::ScalarPrograms(block)],
    })
}

/// The view of a block without programs.
fn empty_block() -> Value {
    minijinja::context! {
        scalar_plan => minijinja::context! {
            programs => Vec::<Value>::new(),
            output_count => 0,
        },
        output_count => 0,
    }
}

/// A whole scalar row block, each output at its row (`output_indices`).
fn dense_rows_value(
    block: &solve::ScalarProgramBlock,
    pure_calls: Option<&solve::SolvePureCallTable>,
    feature: &str,
) -> Result<(Value, usize), CodegenError> {
    let count = block.stored_output_count();
    let value = scalar_block_value(
        block.programs().to_vec(),
        block.program_spans().to_vec(),
        block.output_indices().to_vec(),
        pure_calls,
        feature,
    )?;
    Ok((value, count))
}

/// One exact-assignment schedule as one scalar block whose outputs are the
/// `y` slots it assigns, in checked order.
fn assignment_step(
    problem: &solve::SolveProblem,
    schedule: &solve::ExactRefreshAssignmentSchedule,
    pure_calls: Option<&solve::SolvePureCallTable>,
) -> Result<Value, CodegenError> {
    const FEATURE: &str = "exact_algebraic_assignments";
    let owners = &problem.continuous.refresh_owners;
    let mut programs = Vec::new();
    let mut spans = Vec::new();
    let mut outputs = Vec::new();
    for id in schedule.program_ids() {
        let program = owners
            .exact_assignment_program(*id)
            .ok_or_else(|| refuse(FEATURE, "a schedule names an unissued program"))?;
        let block = program
            .final_scalar_program(&problem.continuous.implicit_rhs)
            .map_err(|error| refuse(FEATURE, error))?;
        let [ops] = block.programs() else {
            return Err(refuse(
                FEATURE,
                "an assignment materializes as several programs",
            ));
        };
        programs.push(ops.clone());
        spans.push(
            block
                .program_span(0)
                .ok_or_else(|| refuse(FEATURE, "an assignment has no provenance"))?,
        );
        outputs.extend(program.target_indices().iter().copied());
    }
    let count = outputs.len();
    let rows = scalar_block_value(programs, spans, outputs, pure_calls, FEATURE)?;
    Ok(minijinja::context! { rows => rows, output_count => count })
}

/// The ordered algebraic refresh of every continuous evaluation: its exact
/// assignment schedules. A projection stage is refused.
fn refresh_value(
    problem: &solve::SolveProblem,
    artifacts: &solve::SolveArtifacts,
    pure_calls: Option<&solve::SolvePureCallTable>,
) -> Result<Value, CodegenError> {
    const FEATURE: &str = "algebraic_projection";
    let owners = &problem.continuous.refresh_owners;
    let plan = owners.algebraic();
    let schedule = plan.stage_schedule(&artifacts.continuous.structural);
    let steps = owners
        .staged_refresh_steps(plan, schedule)
        .map_err(|error| refuse(FEATURE, format!("{error:?}")))?;
    let mut rendered = Vec::new();
    for step in &steps {
        match step {
            solve::StagedRefreshStep::Assignments(schedule) => {
                rendered.push(assignment_step(problem, schedule, pure_calls)?);
            }
            solve::StagedRefreshStep::Project { .. }
            | solve::StagedRefreshStep::ProjectComplete => {
                return Err(refuse(
                    FEATURE,
                    "the algebraic refresh retains a projection stage",
                ));
            }
        }
    }
    Ok(minijinja::context! { steps => rendered })
}

fn zero_domain(domain: solve::RootZeroDomain) -> i8 {
    match domain {
        solve::RootZeroDomain::Positive => 1,
        solve::RootZeroDomain::NonPositive => -1,
        solve::RootZeroDomain::Previous => 0,
    }
}

/// Refuse the event shapes a per-lane event iteration does not execute.
fn require_lane_events(problem: &solve::SolveProblem) -> Result<(), CodegenError> {
    const FEATURE: &str = "events";
    let discrete = &problem.discrete;
    let events = &problem.events;
    let checks: [(bool, &str); 14] = [
        (
            !discrete.event_iteration_plan.runs.is_empty(),
            "the event iteration plan has partition runs",
        ),
        (
            !discrete.runtime_assignment_rhs.is_empty(),
            "runtime discrete assignments are present",
        ),
        (
            !discrete.post_commit_assignment_rhs.is_empty(),
            "post-commit assignments are present",
        ),
        (
            !discrete.guarded_assignments.is_empty(),
            "guarded assignments are present",
        ),
        (
            !discrete.event_transactions.is_empty(),
            "event algorithms are present",
        ),
        (
            !discrete.structured_updates.is_empty(),
            "structured discrete updates are present",
        ),
        (
            discrete.clock_owners.iter().any(Option::is_some),
            "clocked discrete rows are present",
        ),
        (
            !events.scheduled_root_conditions.is_empty(),
            "scheduled root conditions are present",
        ),
        (
            !events.dynamic_time_event_rhs.is_empty(),
            "dynamic time events are present",
        ),
        (events.has_terminal_event, "terminate() is present"),
        (
            events
                .actions
                .iter()
                .any(|action| matches!(action.kind, solve::SolveEventActionKind::Terminate)),
            "terminate() actions are present",
        ),
        // A lane neither checks nor reports a model assertion: an
        // error-level one would let a violated lane run on.
        (
            events
                .actions
                .iter()
                .any(|action| matches!(action.kind, solve::SolveEventActionKind::Assert)),
            "error-level assert() actions are present",
        ),
        (
            discrete.row_roles.len() != discrete.pre_modes.len(),
            "discrete row roles and pre() modes are not aligned",
        ),
        (
            discrete
                .row_roles
                .iter()
                .zip(&discrete.pre_modes)
                .any(|(role, mode)| {
                    !matches!(
                        (role, mode),
                        (
                            solve::DiscreteRowRole::Equation,
                            solve::DiscreteEventPreMode::FollowCurrent
                        ) | (solve::DiscreteRowRole::ConditionMemory, _)
                    )
                }),
            "a discrete row is not a current-value equation or a condition memory",
        ),
    ];
    match checks.iter().find(|(present, _)| *present) {
        Some((_, reason)) => Err(refuse(FEATURE, reason)),
        None => Ok(()),
    }
}

/// A retained state-manifold projection moves accepted states back onto the
/// lower-order constraints index reduction replaced; no lane kernel does.
fn require_no_state_manifold(problem: &solve::SolveProblem) -> Result<(), CodegenError> {
    if problem.continuous.manifold_projection_plan.is_empty() {
        return Ok(());
    }
    Err(refuse(
        "manifold_projection",
        "the model retains a state-manifold projection",
    ))
}

fn events_value(
    problem: &solve::SolveProblem,
    pure_calls: Option<&solve::SolvePureCallTable>,
) -> Result<Value, CodegenError> {
    require_lane_events(problem)?;
    require_no_state_manifold(problem)?;
    let events = &problem.events;
    let discrete = &problem.discrete;
    let (roots, root_count) = lane_roots_value(&events.root_conditions, pure_calls)?;
    let mut event_roots = Vec::new();
    let mut crossings = Vec::new();
    for (root, target) in events.root_relation_memory_targets.iter().enumerate() {
        let zero = events
            .root_zero_domains
            .get(root)
            .copied()
            .map(zero_domain)
            .unwrap_or(0);
        // A root without a relation memory is a crossing the event iteration
        // never re-encodes: the rows read its relation directly, so the lane
        // only reports it to the host, which locates it. That of a function
        // assertion, which observes the call, is not even reported: the lane
        // checks the assertion in the call itself and fails at the first
        // point that violates it.
        let Some(target) = target else {
            let roots = &events.root_conditions;
            if !roots
                .program_index_for_output(root)
                .and_then(|program| roots.program(program))
                .is_some_and(observes_call)
            {
                crossings.push(minijinja::context! { root => root, zero => zero });
            }
            continue;
        };
        let solve::ScalarSlot::P { index, .. } = target else {
            return Err(refuse(
                "events",
                "a relation memory is not parameter storage",
            ));
        };
        event_roots.push(minijinja::context! { root => root, memory => *index, zero => zero });
    }
    let (rows, row_count) = dense_rows_value(&discrete.rhs, pure_calls, "events")?;
    if row_count != discrete.update_targets.len() || row_count != discrete.row_roles.len() {
        return Err(refuse(
            "events",
            "discrete rows and targets are not aligned",
        ));
    }
    let targets = discrete
        .update_targets
        .iter()
        .map(|target| slot(*target, "events"))
        .collect::<Result<Vec<_>, _>>()?;
    let memory = discrete
        .row_roles
        .iter()
        .map(|role| role.is_condition_memory())
        .collect::<Vec<_>>();
    let mut instants = events
        .scheduled_time_events
        .iter()
        .copied()
        .filter(|time| time.is_finite())
        .collect::<Vec<_>>();
    instants.sort_by(f64::total_cmp);
    instants.dedup();
    Ok(minijinja::context! {
        roots => roots,
        root_count => root_count,
        event_roots => event_roots,
        crossings => crossings,
        discrete => rows,
        discrete_count => row_count,
        discrete_targets => targets,
        discrete_memory => memory,
        time_events => instants.iter().map(|time| float_literal(*time)).collect::<Vec<_>>(),
        initial_flag => problem.solve_layout.initial_event_parameter_index,
        terminal_flag => problem.solve_layout.terminal_event_parameter_index,
        homotopy_lambda => problem.solve_layout.initial_homotopy_parameter_index,
    })
}

/// Whether a root program is that of a function assertion: it observes the
/// call that checks the assertion (only an assertion observation may).
fn observes_call(program: &[solve::LinearOp]) -> bool {
    program
        .iter()
        .any(|op| matches!(op, solve::LinearOp::PureCallObservation { .. }))
}

/// The root programs a lane evaluates, and the dense root count: those of
/// function assertions, which no relation memory reads, are left out.
fn lane_roots_value(
    block: &solve::ScalarProgramBlock,
    pure_calls: Option<&solve::SolvePureCallTable>,
) -> Result<(Value, usize), CodegenError> {
    let indices = block.output_indices();
    let mut programs = Vec::new();
    let mut spans = Vec::new();
    let mut outputs = Vec::new();
    for (index, program) in block.programs().iter().enumerate() {
        if observes_call(program) {
            continue;
        }
        let (Some(ordinals), Some(span)) = (
            block.program_output_ordinals(index),
            block.program_span(index),
        ) else {
            return Err(refuse("events", "a root program has no outputs"));
        };
        programs.push(program.clone());
        spans.push(span);
        outputs.extend_from_slice(&indices[ordinals]);
    }
    let value = scalar_block_value(programs, spans, outputs, pure_calls, "events")?;
    Ok((value, block.stored_output_count()))
}

fn delays_value(
    problem: &solve::SolveProblem,
    pure_calls: Option<&solve::SolvePureCallTable>,
) -> Result<Value, CodegenError> {
    let delays = &problem.events.delays;
    let count = delays.value_parameter_indices.len();
    let (source, sources) = dense_rows_value(&delays.source_rhs, pure_calls, "runtime_events")?;
    let (delay_time, times) =
        dense_rows_value(&delays.delay_time_rhs, pure_calls, "runtime_events")?;
    let (delay_max, maxima) =
        dense_rows_value(&delays.delay_max_rhs, pure_calls, "runtime_events")?;
    if sources != count
        || times != count
        || maxima != count
        || delays.source_is_discrete.len() != count
    {
        return Err(refuse(
            "runtime_events",
            "delay rows are not aligned with their slots",
        ));
    }
    Ok(minijinja::context! {
        count => count,
        source => source,
        delay_time => delay_time,
        delay_max => delay_max,
        value_parameters => delays.value_parameter_indices.clone(),
        discrete => delays.source_is_discrete.iter().map(|discrete| u8::from(*discrete)).collect::<Vec<_>>(),
    })
}

/// Whether `program` reads one of `slots` (`column` `"y"` or `"p"`).
fn reads_slot(
    program: &[solve::LinearOp],
    column: &str,
    slots: &std::collections::BTreeSet<usize>,
) -> bool {
    let reads = |start: usize, count: usize| slots.range(start..start + count).next().is_some();
    program.iter().any(|op| match (column, op) {
        ("y", solve::LinearOp::LoadY { index, .. })
        | ("p", solve::LinearOp::LoadP { index, .. }) => slots.contains(index),
        ("p", solve::LinearOp::LoadIndexedP { base, count, .. }) => reads(*base, *count),
        (
            "y",
            solve::LinearOp::TensorLoad {
                input: solve::TensorInputKind::Y,
                input_start,
                count,
                ..
            },
        )
        | (
            "p",
            solve::LinearOp::TensorLoad {
                input: solve::TensorInputKind::P,
                input_start,
                count,
                ..
            },
        ) => reads(*input_start, *count),
        (
            _,
            solve::LinearOp::FunctionFold { program, .. }
            | solve::LinearOp::GuardedFunctionFold { program, .. },
        ) => program
            .regions()
            .any(|region| reads_slot(region, column, slots)),
        (_, solve::LinearOp::FunctionConditional { program, .. }) => {
            program.arms.iter().any(|arm| {
                reads_slot(&arm.condition, column, slots) || reads_slot(&arm.result, column, slots)
            }) || reads_slot(&program.fallback, column, slots)
        }
        _ => false,
    })
}

/// One parameter or start binding: its program and the slots of one column
/// it writes, in store order.
#[derive(Debug)]
struct Binding {
    program: Vec<solve::LinearOp>,
    span: rumoca_core::Span,
    column: &'static str,
    outputs: Vec<usize>,
}

/// The bindings in dependency order: each runs once, after every binding
/// that writes a slot it reads, so that one pass computes the values the
/// runtime's repeated sweeps settle on (`apply_prepared_update_rows_until_stable`).
/// A binding that reads its own output or a cycle is refused.
fn binding_order(bindings: Vec<Binding>, feature: &str) -> Result<Vec<Binding>, CodegenError> {
    let mut pending = bindings;
    let mut order = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let written = |column: &str| {
            pending
                .iter()
                .filter(|binding| binding.column == column)
                .flat_map(|binding| binding.outputs.iter().copied())
                .collect::<std::collections::BTreeSet<_>>()
        };
        let (y, p) = (written("y"), written("p"));
        let (ready, waiting): (Vec<_>, Vec<_>) = pending.into_iter().partition(|binding| {
            !reads_slot(&binding.program, "y", &y) && !reads_slot(&binding.program, "p", &p)
        });
        if ready.is_empty() {
            return Err(refuse(
                feature,
                "the bindings read their own outputs in a cycle",
            ));
        }
        order.extend(ready);
        pending = waiting;
    }
    Ok(order)
}

/// The parameter and start bindings of `init` in dependency order, each
/// storing its outputs at the targets its output indices name.
fn ordered_bindings(
    init: &solve::InitializationSolveSystem,
    feature: &str,
) -> Result<Vec<Binding>, CodegenError> {
    let update = init.update_rhs();
    let targets = init.update_targets();
    if update.stored_output_count() != targets.len() {
        return Err(refuse(feature, "bindings and targets are not aligned"));
    }
    let mut bindings = Vec::new();
    let mut cursor = 0;
    for (index, program) in update.programs().iter().enumerate() {
        let count = solve::ScalarProgramBlock::checked_program_output_count(program)
            .ok_or_else(|| refuse(feature, "a binding has no output count"))?;
        let slots = update
            .output_indices()
            .get(cursor..cursor + count)
            .ok_or_else(|| refuse(feature, "bindings and targets are not aligned"))?
            .iter()
            .map(|row| {
                targets
                    .get(*row)
                    .ok_or_else(|| refuse(feature, "bindings and targets are not aligned"))
                    .and_then(|target| slot(*target, feature))
            })
            .collect::<Result<Vec<_>, _>>()?;
        cursor += count;
        let Some(first) = slots.first() else {
            continue;
        };
        let column = first.column;
        if slots.iter().any(|slot| slot.column != column) {
            return Err(refuse(
                feature,
                "a binding writes a parameter and a start value",
            ));
        }
        bindings.push(Binding {
            program: program.clone(),
            span: update
                .program_span(index)
                .ok_or_else(|| refuse(feature, "a binding has no provenance"))?,
            column,
            outputs: slots.iter().map(|slot| slot.index).collect(),
        });
    }
    binding_order(bindings, feature)
}

fn initialization_value(
    problem: &solve::SolveProblem,
    pure_calls: Option<&solve::SolvePureCallTable>,
) -> Result<Value, CodegenError> {
    const FEATURE: &str = "initialization";
    let init = &problem.initialization;
    if !init.projection_plan().blocks.is_empty() {
        return Err(refuse(
            FEATURE,
            "the initialization retains residual blocks to solve",
        ));
    }
    // Consecutive bindings of one column form one block, written straight
    // into that column.
    let mut runs: Vec<(&'static str, Vec<Binding>)> = Vec::new();
    for binding in ordered_bindings(init, FEATURE)? {
        match runs.last_mut() {
            Some((column, run)) if *column == binding.column => run.push(binding),
            _ => runs.push((binding.column, vec![binding])),
        }
    }
    let runs = runs
        .into_iter()
        .map(|(column, run)| {
            let mut programs = Vec::with_capacity(run.len());
            let mut spans = Vec::with_capacity(run.len());
            let mut outputs = Vec::new();
            for binding in run {
                programs.push(binding.program);
                spans.push(binding.span);
                outputs.extend(binding.outputs);
            }
            let rows = scalar_block_value(programs, spans, outputs, pure_calls, FEATURE)?;
            Ok(minijinja::context! { column => column, rows => rows })
        })
        .collect::<Result<Vec<_>, CodegenError>>()?;
    Ok(minijinja::context! { bindings => runs })
}

fn float_literal(value: f64) -> String {
    if value.is_nan() {
        "NAN".to_string()
    } else if value.is_infinite() {
        if value > 0.0 {
            "INFINITY"
        } else {
            "(-INFINITY)"
        }
        .to_string()
    } else {
        format!("{value:?}")
    }
}

fn start_values(values: &solve::SolveInitialValues) -> Vec<String> {
    values.to_vec().into_iter().map(float_literal).collect()
}

const KEYS: [&str; 6] = [
    "refresh",
    "events",
    "delays",
    "initialization",
    "starts",
    "functions",
];

/// The typed pure-call owners a lane program may call natively: the emitted
/// families (`owners`, the table `functions(table, prefix)` reads), the symbol
/// of each owner id, and each owner's Real-array call shape (`inputs`: scalar
/// count per argument; `outputs`: scalar count of the whole result tuple).
fn functions_value(table: Option<&solve::SolvePureCallTable>) -> Result<Value, CodegenError> {
    let Some(table) = table else {
        return Ok(minijinja::context! {
            table => minijinja::context! { owners => Vec::<Value>::new() },
            symbols => Vec::<u32>::new(),
            shapes => Vec::<Value>::new(),
        });
    };
    let families = super::pure_call_families::PureCallFamilies::new(table)?;
    let shapes = table
        .owners()
        .iter()
        .map(|owner| {
            let inputs = owner
                .inputs()
                .iter()
                .map(|ty| ty.scalar_count() as usize)
                .collect::<Vec<_>>();
            let outputs = owner
                .outputs()
                .iter()
                .map(|output| output.value_type().scalar_count() as usize)
                .sum::<usize>();
            minijinja::context! { inputs => inputs, outputs => outputs }
        })
        .collect::<Vec<_>>();
    Ok(minijinja::context! {
        table => families.owners_value(),
        symbols => families.symbols_value(),
        shapes => shapes,
    })
}

fn lane_part(handle: &SolveRenderHandle, key: &str) -> Result<Value, CodegenError> {
    let problem = handle.problem();
    let pure_calls = handle.inlined_pure_calls();
    match key {
        "refresh" => refresh_value(problem, handle.artifacts(), pure_calls),
        "events" => events_value(problem, pure_calls),
        "delays" => delays_value(problem, pure_calls),
        "initialization" => initialization_value(problem, pure_calls),
        "functions" => functions_value(pure_calls),
        _ => Ok(handle.model().map_or_else(
            || Value::from(()),
            |model| {
                minijinja::context! {
                    y => start_values(&model.initial_y),
                    p => start_values(&model.parameters),
                    // Reporting only: a layout with an unnamed slot emits
                    // no name table.
                    y_names => problem
                        .solver_named_report(0..problem.solver_scalar_count())
                        .ok(),
                }
            },
        )),
    }
}

/// The lane view, each part built on first read so targets that never read
/// it pay nothing and an unsupported part only fails the template reading it.
pub(super) struct LaneBlocks {
    handle: SolveRenderHandle,
    parts: [OnceLock<Result<Value, CodegenError>>; 6],
}

impl std::fmt::Debug for LaneBlocks {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LaneBlocks")
            .field(
                "ready",
                &self
                    .parts
                    .iter()
                    .map(|part| part.get().is_some())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl LaneBlocks {
    pub(super) fn value(handle: SolveRenderHandle) -> Value {
        Value::from_object(Self {
            handle,
            parts: Default::default(),
        })
    }
}

impl Object for LaneBlocks {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Str(&KEYS)
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        let index = KEYS.iter().position(|candidate| *candidate == key)?;
        match self.parts[index].get_or_init(|| lane_part(&self.handle, key)) {
            Ok(value) => Some(value.clone()),
            Err(error) => Some(Value::from(
                crate::errors::render_err(error.to_string()).with_source(error.clone()),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
