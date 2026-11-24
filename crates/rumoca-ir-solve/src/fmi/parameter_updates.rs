//! Parameter binding ownership shared by FMI metadata and C admission.

use super::{FmiCausality, FmiVariable};
use crate::{ScalarSlot, SolveProblem, SolvePureCallTable, SolveVariableStorageRole};

pub(super) fn classify_metadata(problem: &SolveProblem, variables: &mut [FmiVariable]) {
    let owned: std::collections::BTreeSet<_> = problem
        .initialization
        .update_targets()
        .iter()
        .chain(problem.initialization.projection_unknowns())
        .filter_map(|slot| match slot {
            ScalarSlot::P { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    classify_initialized_coordinates(problem, variables);
    for variable in variables {
        if variable.role() != Some(SolveVariableStorageRole::Parameter) {
            continue;
        }
        let Some(storage) = variable.backing.storage() else {
            continue;
        };
        if (storage.base()..storage.base() + storage.scalar_count()).any(|i| owned.contains(&i)) {
            variable.causality = FmiCausality::CalculatedParameter;
            let binding_owned = problem.initialization.update_targets().iter().any(|slot| {
                matches!(slot, ScalarSlot::P { index, .. }
                    if *index >= storage.base() && *index - storage.base() < storage.scalar_count())
            });
            if binding_owned {
                variable.initial = Some(super::FmiInitial::Calculated);
                variable.start_allowed = false;
            } else {
                variable.initial = Some(super::FmiInitial::Approx);
            }
        } else if variable.causality == FmiCausality::Parameter && variable.evaluable {
            // MLS §4.5, §18.3: a final or `Evaluate = true` parameter keeps
            // parameter variability, but its value is fixed at translation, so
            // the environment may not set it: FMI's non-settable parameter
            // form (FMI 3.0.2 §2.4.7.4).
            variable.causality = FmiCausality::CalculatedParameter;
            variable.initial = Some(super::FmiInitial::Calculated);
            variable.start_allowed = false;
        }
    }
}

/// FMI 2.0.5 §2.2.7 and FMI 3.0.2 §2.4.7: a non-parameter coordinate whose
/// start value the initialization may change is `approx` (a state or other
/// coordinate the initialization projection solves for keeps its start as the
/// guess), and one a discrete equation defines or an initialization update
/// assigns from a settable parameter is `calculated` (its start is not its
/// value).
fn classify_initialized_coordinates(problem: &SolveProblem, variables: &mut [FmiVariable]) {
    let projected: std::collections::BTreeSet<_> = problem
        .initialization
        .projection_unknowns()
        .iter()
        .filter_map(|slot| match slot {
            ScalarSlot::Y { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    let discrete = &problem.discrete;
    let defined: std::collections::BTreeSet<_> = discrete
        .update_targets
        .iter()
        .zip(&discrete.row_roles)
        .filter(|(_, role)| **role == crate::DiscreteRowRole::Equation)
        .filter_map(|(slot, _)| match slot {
            ScalarSlot::P { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    // A state whose start reads a settable parameter is assigned from it.
    let assigned: std::collections::BTreeSet<_> = problem
        .initialization
        .update_targets()
        .iter()
        .filter_map(|slot| match slot {
            ScalarSlot::Y { index, .. } => Some(*index),
            _ => None,
        })
        .collect();
    for variable in variables.iter_mut() {
        let Some(storage) = variable.backing.storage() else {
            continue;
        };
        if matches!(
            variable.role(),
            Some(SolveVariableStorageRole::Parameter | SolveVariableStorageRole::Constant) | None
        ) || variable.causality == FmiCausality::Input
        {
            continue;
        }
        let range = storage.base()..storage.base() + storage.scalar_count();
        let column = storage.column();
        let calculated = match column {
            super::FmiStorageColumn::P => range.clone().any(|i| defined.contains(&i)),
            super::FmiStorageColumn::Y => range.clone().any(|i| assigned.contains(&i)),
        };
        if calculated {
            variable.initial = Some(super::FmiInitial::Calculated);
            variable.start_allowed = false;
        } else if column == super::FmiStorageColumn::Y
            && variable.initial == Some(super::FmiInitial::Exact)
            && range.clone().any(|i| projected.contains(&i))
        {
            variable.initial = Some(super::FmiInitial::Approx);
        }
    }
}

/// Admit the parameter bindings and return the order the generated component
/// evaluates their programs in, with the number of dependency levels (the
/// repeated sweeps the runtime needs to settle them).
///
/// The runtime applies the bindings as a Jacobi sweep repeated until no value
/// changes (`eval_and_apply_update_rows`), whatever their issued order. Each
/// program is a function of settled parameters and constants alone, so the
/// fixed point assigns every target the value its program computes from the
/// final values of its reads; evaluating the programs once in dependency
/// order computes exactly those values, bit for bit. A binding that reads
/// time, a continuous coordinate, an input, or anything in a cycle is refused.
pub(super) fn validate(
    problem: &SolveProblem,
    calls: &SolvePureCallTable,
    certifies_state_manifold: bool,
) -> Result<(Vec<usize>, usize), &'static str> {
    let init = &problem.initialization;
    validate_projection(problem, certifies_state_manifold)?;
    let mut p = stable_parameters(problem);
    let y = vec![false; problem.layout.y_scalars()];
    // A parameter target is settled once its binding runs; a state target is a
    // start assigned from settled parameters, which no binding reads.
    let mut targets = Vec::with_capacity(init.update_targets().len());
    for slot in init.update_targets() {
        let target = match *slot {
            ScalarSlot::P { index, .. }
                if owns(problem, SolveVariableStorageRole::Parameter, slot) =>
            {
                Some(index)
            }
            ScalarSlot::Y { .. } if owns(problem, SolveVariableStorageRole::State, slot) => None,
            _ => return Err("C initialization can only assign parameters and state starts"),
        };
        targets.push(target);
    }
    for target in targets.iter().flatten() {
        p[*target] = false;
    }
    let programs = init.update_rhs().programs();
    let mut outputs = Vec::with_capacity(programs.len());
    let mut cursor = 0;
    for program in programs {
        let count = crate::ScalarProgramBlock::program_output_count(program);
        let written = init
            .update_rhs()
            .output_indices()
            .get(cursor..cursor + count)
            .ok_or("missing parameter binding output index")?
            .iter()
            .map(|output| targets.get(*output).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or("missing parameter update target")?;
        outputs.push(written.into_iter().flatten().collect());
        cursor += count;
    }
    if cursor != targets.len() {
        return Err("missing parameter binding output");
    }
    let candidates = (0..programs.len()).collect::<Vec<_>>();
    use super::static_assertions::scalar_dependencies::{OrderError, dependency_order};
    dependency_order(calls, programs, &candidates, &outputs, &y, &mut p).map_err(
        |error| match error {
            OrderError::Unsupported => "unsupported parameter binding dependency operation",
            OrderError::Unsettled => {
                "parameter binding depends on unsettled or non-parameter values"
            }
        },
    )
}

/// Whether `slot` lies in a storage run of `role`.
fn owns(problem: &SolveProblem, role: SolveVariableStorageRole, slot: &ScalarSlot) -> bool {
    problem
        .solve_layout
        .variable_storage_runs
        .iter()
        .any(|run| {
            run.role == role
                && match (run.base, slot) {
                    (ScalarSlot::P { index: base, .. }, ScalarSlot::P { index, .. })
                    | (ScalarSlot::Y { index: base, .. }, ScalarSlot::Y { index, .. }) => {
                        *index >= base && *index - base < run.scalar_count
                    }
                    _ => false,
                }
        })
}

pub(super) fn stable_parameters(problem: &SolveProblem) -> Vec<bool> {
    let mut p = vec![false; problem.layout.p_scalars()];
    for run in &problem.solve_layout.variable_storage_runs {
        if !matches!(
            run.role,
            SolveVariableStorageRole::Parameter | SolveVariableStorageRole::Constant
        ) {
            continue;
        }
        if let ScalarSlot::P { index, .. } = run.base {
            p[index..index + run.scalar_count].fill(true);
        }
    }
    p
}

/// The initialization projection the C kernel runs: the runtime's
/// `settle_initialization_system` without a homotopy sweep. Its residual
/// rows are evaluated on the settled coordinates (the bindings and the
/// algebraic refresh applied first) when a row role reads a reconstructed
/// algebraic, as the runtime does
/// ([`initial_rows_read_settled_view`]); otherwise on the instance
/// coordinates, with single-row isolation and the forward initial Jacobian,
/// which matches the runtime only when no row reads a continuous algebraic.
///
/// The retained state-manifold rows close the residual as ordinary rows,
/// which the runtime settles with the others; the linked kernel then
/// certifies the settled states on the manifold (`certify_state_manifold`).
/// A target that certifies them (`certifies_state_manifold`: the shared
/// kernel's `rmc_initialize` calling `rmc_certify_state_manifold`) admits
/// them; the FMI C component, which never emits that projection, does not.
///
/// Delays need no history while initializing: `delay(u, ...) = u` (MLS
/// §3.7.2), so the kernel holds each delayed value at its source before every
/// pass and refreshes it after the projection, as the runtime's
/// `refresh_delay_values` does before any history is committed. The history
/// itself belongs to the target that steps the model; the FMI C profile
/// still refuses delays as runtime events.
fn validate_projection(
    problem: &SolveProblem,
    certifies_state_manifold: bool,
) -> Result<(), &'static str> {
    let init = &problem.initialization;
    if init.projection_plan().iterates_discretes {
        return Err("C initialization cannot alternate the projection with held discretes");
    }
    let rows = init
        .residual()
        .len()
        .map_err(|_| "the initialization residual has no checked row count")?;
    if rows == 0 {
        return if init.projection_plan().is_empty() {
            Ok(())
        } else {
            Err("C initialization projects blocks without residual rows")
        };
    }
    if init.manifold_row_count() > 0 && !certifies_state_manifold {
        return Err("C initialization cannot certify retained state-manifold rows");
    }
    if problem
        .solve_layout
        .initial_homotopy_parameter_index
        .is_some()
    {
        return Err("C initialization cannot run the homotopy continuation");
    }
    if !initial_rows_read_settled_view(problem) && !residuals_read_no_algebraic(problem) {
        return Err(
            "C initialization evaluates a row that reads a continuous algebraic only on the settled algebraic refresh, which its row roles do not select",
        );
    }
    if problem.solve_layout.solver_scalar_count() != problem.layout.y_scalars() {
        return Err("C initialization needs the solver coordinates to fill the Y storage");
    }
    Ok(())
}

/// Whether the initialization residual rows are evaluated on the settled
/// view: a row role reads a continuous algebraic the refresh reconstructs
/// (`initial_rows_read_settled_view` of the runtime's initial projection).
/// The C profile admits the other view only when no row reads a continuous
/// algebraic, where the runtime's read cones are empty as well.
pub fn initial_rows_read_settled_view(problem: &SolveProblem) -> bool {
    use crate::{InitializationCoordinateKind, InitializationRowRole};
    problem.initialization.row_roles().iter().any(|role| {
        matches!(
            role,
            InitializationRowRole::UnownedCoordinate(InitializationCoordinateKind::Algebraic)
                | InitializationRowRole::SolvedThroughAlgebraicRefresh
                | InitializationRowRole::SurplusAlgebraicCheck
        )
    })
}

/// Whether every initialization residual row is a scalar program reading no
/// continuous algebraic (a solver coordinate past the states), so its value
/// on the instance coordinates is the one the runtime evaluates.
fn residuals_read_no_algebraic(problem: &SolveProblem) -> bool {
    let state_count = problem.solve_layout.state_scalar_count();
    problem.initialization.residual().nodes.iter().all(|node| {
        let crate::ComputeNode::ScalarPrograms(block) = node else {
            return false;
        };
        block
            .programs()
            .iter()
            .all(|program| !reads_algebraic(program, state_count))
    })
}

/// Whether `ops` (nested regions included) loads a solver coordinate at or
/// past `state_count`.
fn reads_algebraic(ops: &[crate::LinearOp], state_count: usize) -> bool {
    struct Reads {
        state_count: usize,
        found: bool,
    }
    impl crate::visitor::SolveVisitor for Reads {
        type Error = std::convert::Infallible;

        fn visit_linear_op(
            &mut self,
            _kind: crate::visitor::LinearOpSliceKind,
            _index: usize,
            op: &crate::LinearOp,
        ) -> Result<(), Self::Error> {
            self.found |= match op {
                crate::LinearOp::LoadY { index, .. } => *index >= self.state_count,
                crate::LinearOp::TensorLoad {
                    input: crate::TensorInputKind::Y,
                    input_start,
                    count,
                    ..
                } => input_start.saturating_add(*count) > self.state_count,
                _ => false,
            };
            Ok(())
        }
    }
    let mut reads = Reads {
        state_count,
        found: false,
    };
    let kind = crate::visitor::LinearOpSliceKind::ScalarProgram {
        program_index: 0,
        span: None,
    };
    let Ok(()) = crate::visitor::walk_linear_op_slice(&mut reads, kind, ops);
    reads.found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearOp;

    #[test]
    fn an_algebraic_read_is_found_through_scalar_and_tensor_loads() {
        assert!(!reads_algebraic(
            &[
                LinearOp::LoadY { dst: 0, index: 1 },
                LinearOp::StoreOutput { src: 0 }
            ],
            2
        ));
        assert!(reads_algebraic(
            &[
                LinearOp::LoadY { dst: 0, index: 2 },
                LinearOp::StoreOutput { src: 0 }
            ],
            2
        ));
        assert!(!reads_algebraic(
            &[
                LinearOp::LoadP { dst: 0, index: 7 },
                LinearOp::StoreOutput { src: 0 }
            ],
            0
        ));
    }
}
