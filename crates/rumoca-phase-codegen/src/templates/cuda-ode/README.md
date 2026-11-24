# `cuda-ode`

## Use case

Use this target to evaluate one hybrid ODE model across many independent lanes
on an NVIDIA GPU—for Monte Carlo analysis, parameter sweeps, ensemble filters,
or batched control-design workloads. The host owns allocation, launch policy,
and integration (including event location and stopping at time events).

## Contract

- Input: checked Solve IR, lowered with the scalar simulation profile.
- Every CUDA thread evaluates one checked `(y, p)` lane with explicit strides.
- Algebraic variables: when every algebraic slot has an issued exact
  assignment (`exact_algebraic_assignments`), the kernels assign them in
  checked order (`<model>_refresh`, `<model>_refresh_batch`).
  `<model>_rhs_batch` settles the algebraics in place, then writes derivatives.
  Coupled or implicit algebraic blocks are still rejected
  (`residual_equations`): the target holds no projection solver.
- Events: `<model>_event_indicators_batch` returns the root functions
  (those of the relation memories, then the crossings whose relation the
  rows read directly, such as `time < Tclose`; function assertions have
  none),
  `<model>_event_update_batch` iterates relation memories and discrete
  equations to a fixed point, and reports a non-finite root as `-1` passes.
  It re-encodes each relation from the sign of its root alone, so the host
  launches it where a located crossing has changed the root's sign. Static
  time events are listed in the host array `<model>_time_events`.
- Delays: per-lane history ring buffers (`<model>_DELAY_HISTORY_STRIDE`
  doubles per lane, capacity `<model>_DELAY_CAPACITY`), with
  `<model>_delay_init_batch`, `<model>_delay_apply_batch` (before each RHS)
  and `<model>_delay_commit_batch` (after each accepted step). A lane reads
  its history as the runtime does (`delayed_value`), and reads NaN for a
  delay time outside `[0, delayMax]` or a time older than the points its
  ring kept.
- Initialization: `<model>_load_starts_batch` writes the compiled start values,
  `<model>_initialize_batch` runs the parameter and start bindings and the
  initial event.
- Functions: pure-function bodies that cannot be inlined (data-dependent
  loops, such as table interpolation or sorting) are emitted as typed
  `__device__` functions. A failed error-level assertion inside one fails
  the call, which makes that lane's results NaN as a domain error does; the
  lane does not locate it as an event.
- `<model>_EVENT_MAX_PASSES` can be overridden with `-D`.
- Build option: `-DRUMOCA_OMIT_INITIALIZATION` leaves out what only
  initialization uses (parameter and start initialization, start tables,
  bindings, `<model>_initialize_batch`), for a device build whose host
  initializes the lanes.

## Unsupported

This is not an NVRTC executor, an FMU, or a full simulator: there is no
integrator. Coupled algebraic blocks, initialization residual blocks, a
retained state-manifold projection, a reduced chart set with alternate charts,
error-level model assertions, linear-solve tensor ops, clocks, external
functions, callbacks, and unsupported operations fail before source is
written. No GPU availability
is inferred from successful source generation.

## Verification

- `cuda_ode_generated_kernel_compiles_and_executes_cpu_emulation` checks batched
  indexing and numerical outputs without requiring a GPU.
- `cuda_ode_generated_kernel_compiles_with_required_nvcc` is a required CI source
  compilation with the pinned Nix CUDA compiler.
- `cuda_ode_builtin_target_rejects_linsolve_before_writing_source` and
  `explicit_rhs_targets_reject_implicit_algebraic_models` are negative controls.

## Example

```sh
rumoca compile Ensemble.mo --model Ensemble --target cuda-ode --output generated
nvcc -c generated/Ensemble_ode.cu
```
