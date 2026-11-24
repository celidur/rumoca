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
  A refresh that retains projection stages (coupled or implicit algebraic
  blocks) runs the shared Model-Exchange projection kernel
  (`fmi3/me_projection.jinja`, SPEC_0044 ME-PROJ-001), compiled into the
  source for the device: the target holds no projection solver of its own and
  does not declare `algebraic_projection`. Each lane owns one kernel instance
  (its coordinate, projection workspace and fallback counters) in a device
  buffer of `<model>_LANE_WORKSPACE_BYTES` bytes per lane that the host
  allocates and attaches once with `<model>_workspace_init_batch` before any
  other kernel. `RUMOCA_LANE_INDEX` (default
  `blockIdx.x * blockDim.x + threadIdx.x`) names the calling thread's lane.
  `<model>_refresh_batch` and `<model>_rhs_batch` report a failed refresh in
  `status`.
- Events: `<model>_event_indicators_batch` returns the root functions,
  `<model>_event_update_batch` iterates relation memories and discrete
  equations to a fixed point. Static time events are listed in the host array
  `<model>_time_events`.
- Delays: per-lane history ring buffers (`<model>_DELAY_HISTORY_STRIDE`
  doubles per lane, capacity `<model>_DELAY_CAPACITY`), with
  `<model>_delay_init_batch`, `<model>_delay_apply_batch` (before each RHS)
  and `<model>_delay_commit_batch` (after each accepted step).
- Initialization: `<model>_load_starts_batch` writes the compiled start values,
  `<model>_initialize_batch` runs the parameter and start bindings and the
  initial event.
- Functions: pure-function bodies that cannot be inlined (data-dependent
  loops, such as table interpolation or sorting) are emitted as typed
  `__device__` functions. A failed error-level assertion inside one is
  observed as an event root; a domain error makes that lane's results NaN.
- `<model>_EVENT_MAX_PASSES` can be overridden with `-D`.
- Build option: `-DRUMOCA_OMIT_INITIALIZATION` leaves out what only
  initialization uses (parameter and start initialization, start tables,
  bindings, `<model>_initialize_batch`), for a device build whose host
  initializes the lanes.

## Unsupported

This is not an NVRTC executor, an FMU, or a full simulator: there is no
integrator. Initialization residual blocks, a retained state-manifold
projection, linear-solve tensor ops, clocks, external
functions, callbacks, and unsupported operations fail before source is
written. No GPU availability
is inferred from successful source generation.

## Verification

- `cuda_ode_generated_kernel_compiles_and_executes_cpu_emulation` checks batched
  indexing and numerical outputs without requiring a GPU.
- `cuda_ode_generated_kernel_compiles_with_required_nvcc` is a required CI source
  compilation with the pinned Nix CUDA compiler.
- `cuda_ode_ladder_settles_its_nonlinear_nodes_per_lane` and
  `cuda_ode_torn_block_settles_from_any_tear_guess` run the shared projection
  kernel per lane in CPU emulation.
- `cuda_ode_builtin_target_rejects_linsolve_before_writing_source` and
  `cuda_ode_refuses_an_initialization_it_would_have_to_solve` are negative
  controls.

## Example

```sh
rumoca compile Ensemble.mo --model Ensemble --target cuda-ode --output generated
nvcc -c generated/Ensemble_ode.cu
```
