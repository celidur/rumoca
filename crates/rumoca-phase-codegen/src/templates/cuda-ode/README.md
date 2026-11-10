# `cuda-ode`

## Use case

Use this target to evaluate one hybrid ODE model across many independent lanes
on an NVIDIA GPU—for Monte Carlo analysis, parameter sweeps, ensemble filters,
or batched control-design workloads. The host owns allocation, launch policy,
and integration (including event location and stopping at time events).

## Contract

- Input: checked Solve IR, lowered with the scalar simulation profile.
- Every CUDA thread evaluates one checked `(y, p)` lane with explicit strides.
- Algebraic variables: exact assignments run in checked order; coupled blocks
  of the staged refresh are solved per lane by finite-difference Newton over
  their tear unknowns (`<model>_refresh`, `<model>_refresh_batch`).
  `<model>_rhs_batch` settles the algebraics in place, then writes derivatives.
- Manifold constraints: `<model>_project_manifold_batch` applies the
  minimum-norm Gauss-Newton correction after a step.
- Events: `<model>_event_indicators_batch` returns the root functions,
  `<model>_event_update_batch` iterates relation memories and discrete
  equations to a fixed point. Static time events are listed in the host array
  `<model>_time_events`.
- Delays: per-lane history ring buffers (`<model>_DELAY_HISTORY_STRIDE`
  doubles per lane, capacity `<model>_DELAY_CAPACITY`), with
  `<model>_delay_init_batch`, `<model>_delay_apply_batch` (before each RHS)
  and `<model>_delay_commit_batch` (after each accepted step).
- Initialization: `<model>_load_starts_batch` writes the compiled start values,
  `<model>_initialize_batch` runs the bindings, initial equations and the
  initial event.
- Functions: pure-function bodies that cannot be inlined (data-dependent
  loops, such as table interpolation or sorting) are emitted as typed
  `__device__` functions. A failed error-level assertion inside one is
  observed as an event root; a domain error makes that lane's results NaN.
- Tolerances: `<model>_NEWTON_MAX_ITERATIONS`, `<model>_NEWTON_STEP_TOLERANCE`
  and `<model>_EVENT_MAX_PASSES` can be overridden with `-D`.

## Unsupported

This is not an NVRTC executor, an FMU, or a full simulator: there is no
integrator. Linear-solve tensor ops, clocks, external functions, callbacks,
and unsupported operations fail before source is written. No GPU availability
is inferred from successful source generation.

## Verification

- `cuda_ode_generated_kernel_compiles_and_executes_cpu_emulation` checks batched
  indexing and numerical outputs without requiring a GPU.
- `cuda_ode_generated_kernel_compiles_with_required_nvcc` is a required CI source
  compilation with the pinned Nix CUDA compiler.
- `explicit_rhs_targets_reject_implicit_algebraic_models` checks that the
  implicit block is solved per lane.
- `cuda_ode_builtin_target_rejects_linsolve_before_writing_source` is a
  negative control.

## Example

```sh
rumoca compile Ensemble.mo --model Ensemble --target cuda-ode --output generated
nvcc -c generated/Ensemble_ode.cu
```
