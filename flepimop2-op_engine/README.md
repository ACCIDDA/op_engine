# flepimop2-op_engine

Provider package that adapts `op_engine` to `flepimop2`.

Install the provider directly, or include its op_system integration extra:

```bash
pip install flepimop2-op-engine
pip install "flepimop2-op-engine[op-system]"
```

For systems supplied by `flepimop2-op_system`, the provider consumes the
compiled `state_names`, `initial_state`, and `axis_labels` options to assemble
the flat solver state. Scalar seeds may be shared by multiple state cells, and
shaped seeds are selected by their named coordinates. Resolved
`ParameterValue` payloads are unwrapped and bound to the system once before
integration; wrappers are not forwarded through every right-hand-side
evaluation.

The numerical array namespace is selected from the initial state rather than
from an engine configuration flag. Initial-state assembly, right-hand-side
evaluation, `ModelCore` history, and the returned `(time, state...)` array stay
in that namespace. Evaluation times remain NumPy arrays because they are static
solver structure. An `op_system` stepper converts bound parameter values into
the state namespace when it evaluates the RHS, so a JAX state can safely consume
static NumPy parameters without moving the evolving state back to the host.

Install the JAX runtime for portable fixed-step JIT and differentiation:

```bash
pip install "flepimop2-op_engine[jax]"
```

Diffrax is not required for JAX differentiation of fixed-step Euler, Heun, RK4,
Dormand--Prince 5(4), or dense IMEX/implicit methods. Fixed-step explicit JAX
trajectories use one `jax.lax.scan`, so the traced program does not grow with
the number of output times. Fixed flat-state Dormand--Prince computes only the
high-order solution, reuses its FSAL stage, and carries storage for requested
outputs rather than returning every hidden step from the scan. RK4 retains its
smaller, compile-sensitive complete-tail scan; method choice therefore remains
an explicit accuracy/compile tradeoff. Neither policy promises constant-memory
reverse-mode differentiation; JAX's transformations and the checkpoint policy
govern gradient residual storage. Typed dense operator descriptors are compiled
in the evolving state's namespace, so descriptor parameters remain traceable
too. Explicit methods apply those descriptors as
additive drift at every Runge--Kutta stage for flat, PyTree, and block state
layouts; the structured paths use small axis-local matrices rather than a
dense full-state operator.
The compiler supports row-source axis-kernel generators, conservative
jump-integral matrices with declared direction and continuous-axis target
quadrature, first-order upwind advection on uniform axes, and centered
finite-volume diffusion on uniform or monotone non-uniform cell-center axes,
including dynamic rates, kernels, signed velocities, and diffusion coefficients.
Non-uniform no-flux diffusion is conservative under
the inferred cell-volume weights; non-uniform periodic diffusion requires more
domain geometry than axis centers provide and is rejected explicitly.
The NumPy path applies op_system's value-dependent generator and jump-kernel
validation plus eager scalar checks. A traced non-NumPy path can validate shapes
and static layout only; producers are responsible for maintaining generator,
jump-kernel, finiteness, and non-negative diffusion-coefficient invariants in
dynamic parameter values.

For explicit methods, `fixed_max_step` separates integration accuracy from the
requested output grid while retaining that compact scan:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    method: rk4
    fixed_max_step: 0.25
```

Each output interval is partitioned into bounded internal steps, including a
final remainder, but the returned trajectory contains only requested output
times. The setting is mutually exclusive with `adaptive: true` and does not
apply to stochastic mode (`tau_max_step` controls fixed tau-leaping).

Long reverse-mode DOPRI5 solves can opt into step or chunk rematerialization:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    method: dopri5
    fixed_max_step: 0.05
    fixed_checkpoint: chunk
    checkpoint_chunk_size: 32
```

`step` stores the least step-local residual state and recomputes each numerical
step during the backward pass. `chunk` rematerializes bounded groups of internal
steps and can trade more residual storage for less recomputation. These policies
currently target deterministic, flat-state, fixed DOPRI5 JAX execution; eager
NumPy execution is unchanged.

### Prepared execution

Repeated inference calls with the same model structure can prepare the provider
once while keeping initial-state and parameter arrays dynamic:

```python
prepared = engine.prepare(
    system,
    times,
    sample_initial_state,
    sample_params,
    model_state=model_state,
)

def final_state(rate, initial):
    trajectory = prepared(
        {"x0": initial},
        {"rate": rate},
    )
    return trajectory[-1, 1]

compiled = jax.jit(jax.value_and_grad(final_state, argnums=(0, 1)))
```

Sample contents do not enter the cache key. They establish the mapping names,
shapes, dtypes, and array namespaces that later calls must preserve. The key
also includes system identity and structural metadata, method and configuration,
the output and internal-step grid, state ordering, and any frozen adaptive
schedule. Calling `prepare` again with the same structure returns the same
`PreparedExecution`; changed values alone do not cause a miss.

`prepared.run(initial_state, params)` accepts ordinary `ParameterValue`
mappings. Calling `prepared(raw_initial_state, raw_params)` accepts raw array
mappings and is the boundary intended for caller-owned JAX JIT, AOT, and
automatic differentiation. The provider does not hide compilation in either
path.

Prepared objects deliberately snapshot their numerical configuration and output
grid. If mutable system internals change, call `engine.clear_prepared_cache()`
and prepare again. The prepared path supports deterministic flat, PyTree, and block-state
Euler, Heun, RK4, and Dormand--Prince fixed stepping or replay of an already
discovered adaptive schedule. Structured prepared plans retain typed explicit
operators; the flat prepared path still rejects them because it has no
structured operator template. Stochastic/hybrid execution, implicit/IMEX
methods, and adaptive discovery remain on the ordinary provider path rather
than silently falling back. A `PreparedExecution` can be passed directly to
`jax.jit`; during tracing, its contract accounts for JAX canonicalizing dynamic
NumPy inputs and disabled-x64 dtypes into the effective JAX namespace.

### Structured and block state execution

The default `state_layout: flat` remains the compatibility path. An op_system
model that publishes `pytree_stepper_fn` and `template_shapes` can instead keep
each state template as its natural N-dimensional leaf throughout explicit
fixed-step integration:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    method: rk4
    fixed_max_step: 0.25
    state_layout: pytree
```

For a model whose op_system spec declares a separable `factorize_axes` entry,
`state_layout: block` consumes the published block stepper and axis positions.
With JAX arrays it applies `vmap` to the complete fixed-step solve while the
time integration remains one compact `lax.scan` shared across all blocks:

```yaml
config:
  method: rk4
  state_layout: block
  block_axis: loc
```

PyTree execution is array-namespace polymorphic; block execution currently
requires JAX because it relies on `vmap`. Both layouts use op_engine's own
validated Euler, Heun, RK4, and Dormand--Prince coefficients—Diffrax is not an
execution dependency. Explicit adaptive JAX discovery first builds one
conservative shared schedule from the full state; PyTree replay consumes that
schedule directly, while block replay vmaps it across the selected axis. Until
flepimop2 defines a structured engine-result contract, the provider flattens
only the completed history at its public `(time, state...)` result boundary.
Structured layouts intentionally reject IMEX/implicit, hybrid, stochastic, and
missing-metadata combinations instead of silently falling back to the flat path.
The public flat history is returned through Flepimop2's `Array` contract in
the originating NumPy or JAX namespace; the provider does not coerce it to
NumPy at that boundary.

Run `uv run python benchmarks/structured_blocks.py` to compare compile time,
median runtime, host allocation peaks, and device peak bytes (when reported by
the JAX backend) as the number of independent blocks grows. The script emits
CSV so scaling records can be retained alongside migration decisions. The
phase-separated fixed/adaptive, NumPy/JAX, gradient, SciPy, and optional
Diffrax matrix is documented in [`benchmarks/README.md`](benchmarks/README.md).

The provider advertises every canonical `CoreSolver` method. Fully nonlinear
`sdirk2` reads its distinct full flattened Jacobian from
`system.option("rhs_jacobian")`; a custom backend-neutral `NonlinearSolver` may
be supplied through `system.option("nonlinear_solver")`, otherwise the portable
dense Newton solver is used. This is deliberately separate from the
operator-axis `system.option("jacobian")` consumed by linearly implicit and
Rosenbrock methods.

### Adaptive schedule replay

Adaptive differentiation uses an explicit two-phase contract. First run the
controller and retain its accepted mesh. Explicit JAX inputs use a compiled,
bounded device-resident acceptance loop; other namespaces retain the portable
eager controller:

```python
discovered = engine.run_adaptive(
    system,
    times,
    initial_state,
    params,
)
schedule = discovered.schedule
```

Then replay the frozen mesh through the ordinary provider entry point:

```python
trajectory = engine.run(
    system,
    times,
    initial_state,
    params,
    adaptive_schedule=schedule,
)
```

The replay uses the same portable core kernels and keeps accept/reject decisions
frozen, so it can be enclosed by `jax.jit` and `jax.grad`. Compact replay also
recomputes each step's embedded or step-doubling error estimate to diagnose
whether the frozen mesh remains accurate for the active parameter values. With
the default `adaptive_replay: auto`, fixed-mesh replay for an explicit method
and JAX state is one compact `lax.scan`; other namespaces and methods retain the
portable core replay. `adaptive_replay: unrolled` preserves the prior explicit
JAX trace for comparison, while `adaptive_replay: compact` requires the
supported JAX explicit path rather than silently falling back.

Long reverse-mode computations can rematerialize each explicit step instead of
retaining its intermediates:

```yaml
config:
  method: dopri5
  adaptive: true
  adaptive_replay: compact
  replay_checkpoint: step
```

This checkpoint policy recomputes step operations during the backward pass;
`replay_checkpoint: chunk` plus `checkpoint_chunk_size` rematerializes bounded
accepted-step groups instead. Neither policy changes the accepted mesh or
numerical method. Fully nonlinear SDIRK2
replay intentionally remains on the existing path so its compiled-safe stage
diagnostics and post-execution `require_converged()` validation are preserved.

The schedule records and validates the solver method, output grid, adaptive
controller settings, state order/shape, parameter shapes, and structural system
metadata. op_system specs are included in that structural signature. Set an
explicit `schedule_tag` when a custom system has semantic model versions that
cannot be inferred from published metadata; a tag mismatch invalidates replay.
Parameter values are deliberately not hashed because replay must support
inference over dynamic values. Gradients remain conditional on the mesh, so
discover a fresh schedule after material parameter, tolerance, model, or
output-grid changes. A run launched through `Simulator` exposes its discovered
artifact as `engine.last_adaptive_schedule`.

Compact `run_adaptive(..., schedule=schedule)` results include array-valued
`replay_diagnostics`. By default, a replay is marked inaccurate when any scaled
local-error norm exceeds `replay_error_factor: 1.25`. Call
`result.require_accurate()` outside a JAX transform to raise
`AdaptiveScheduleAccuracyError` and trigger fresh discovery. Block execution
uses the maximum scaled cell error to discover its shared schedule and
aggregates validation across every block. Ordinary `engine.run()` remains
JIT/gradient-safe and returns only the trajectory; callers that need the
freshness decision should use `run_adaptive()` at an outer orchestration
boundary.

In addition to `rtol`, `atol`, and controller bounds, provider configuration
exposes `dt_init`, `max_reject`, and `max_steps`. These bound adaptive discovery
work per output interval and are recorded in the replay artifact, so changing
one requires a fresh schedule.

`run_adaptive()` also returns nonlinear replay diagnostics. Call
`result.require_converged()` after compiled execution before accepting a result
whose method uses nonlinear stages; this is separate from explicit-mesh
`require_accurate()` validation.

Run `uv run python benchmarks/adaptive_replay.py` to compare eager discovery,
legacy unrolled replay, compact replay, and step-checkpointed replay as accepted
step counts grow. Its CSV records trace size, compile/runtime, Python host peaks,
and the compiled value-and-gradient executable's temporary-memory estimate.

## Stochastic reaction networks

Set `mode: stochastic` to execute every named reaction artifact published by
`system.option("reactions")` as a discrete process. The provider does not parse
raw transition configuration. It expands each typed reaction's source cells
into flat channels and compiles the associated transition, source-only,
pinned-axis, and summed-axis bookkeeping into one stoichiometric matrix.

Three methods are available:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    mode: stochastic
    stochastic_method: tau-leaping  # also adaptive-tau-leaping, direct-ssa, thinning-ssa
    random_seed: 90210              # NumPy convenience path
    tau_max_step: 0.1               # fixed tau-leaping only
```

`tau-leaping` accepts `tau_max_step` and `stochastic_max_steps`.
All stochastic methods accept `forcing_breakpoints` in pure stochastic
mode. `direct-ssa` accepts `ssa_max_events`; its exact
interpretation requires propensities to remain constant in time between events
and declared forcing boundaries. All methods
preserve the usual `(time, state...)` provider trajectory and fail visibly on
invalid propensities, invalid random draws, or negative states. Populations are
never silently clipped.

For an explicitly piecewise-constant propensity producer, configure its forcing
changes independently of the requested observation times:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    mode: stochastic
    stochastic_method: direct-ssa
    random_seed: 90210
    forcing_breakpoints: [1.0, 2.0]
```

A pending direct-SSA or adaptive exact-fallback event at or beyond a breakpoint
is discarded and redrawn from the new rates; zero-rate intervals wait for a
future forcing change. Fixed and adaptive tau-leaps end at the next forcing
boundary and reevaluate the rates for the following leap. Adaptive retries
remain within the current forcing segment, and a critical event tied with its
end is discarded. These caps preserve the usual tau-leaping approximation;
state-dependent rates are still frozen during each leap. The producer
must return the new rates at each boundary and keep them constant between
boundaries while the state is unchanged for direct SSA and adaptive exact
fallback. Output times observe exact paths and also cap tau-leaps, so changing
the output grid may change a tau-leaping approximation.

Nonempty engine-config forcing schedules require `mode: stochastic`.
Forcing times must be finite real values in
strictly increasing order. JSON/YAML lists become immutable tuples; global
schedules may include times outside the run. An empty engine schedule adds no
extra boundaries. Forcing changes share the
whole-interval `stochastic_max_steps` guard for fixed and adaptive tau-leaping.

Pure stochastic runs automatically consume `system.option('forcing_breakpoints')`
when a producer publishes it. The engine validates that schedule and takes its
sorted union with explicit engine boundaries, removing overlaps. The combined
schedule is local to each run; the engine configuration stays unchanged when
the same engine is reused with another system. A producer without this option
retains the existing execution path. Observation times do not supply forcing
boundaries.

For a producer containing [op_system PR #241](https://github.com/ACCIDDA/op_system/pull/241),
select hold interpolation in the **system specification**. For example:

```yaml
kind: transitions
time_interpolation: previous
axes:
  - {name: group, coords: [a]}
  - {name: day, type: continuous, coords: [0.0, 0.5, 0.75]}
time_axis: day
state: ["A[group]", "B[group]"]
transitions:
  - name: transfer
    from: "A[group]"
    to: "B[group]"
    rate: "rate[day]"
    reactants: [{state: "A[group]", order: 1}]
```

The parameter producer supplies the full `rate[day]` table; its values are held
on each interval and clamp beyond the endpoints. The engine consumes the
declared changes at `0.5` and `0.75` without duplicating them in its config.
The development dependency pins include this producer change. Hold-table
support requires a producer containing it; the published minimum dependency
continues to support older producers for other execution paths.

Hybrid execution rejects nonempty producer schedules during validation and
execution because its split deterministic/stochastic steps do not yet support
forcing boundaries. Deterministic execution evaluates the producer's table
using the ordinary deterministic method and step controls.

`op_system` still defaults to linear interpolation. Linear tables publish no
forcing changes, and their time coordinates do not become engine boundaries.
Manually listing those coordinates does not make frozen-rate SSA exact for
smooth rates. Select bounded thinning for smooth time dependence with a valid
total-rate bound, as described below.

NumPy arrays use seeded `NumpyPoissonSampler`, `NumpySSASampler`, or
`NumpyThinningSampler` instances. Other namespaces inject a `poisson_sampler=`,
`ssa_sampler=`, or `thinning_sampler=` callable into
`run`. The callable must return arrays in the input namespace. Its integer
step/draw index is stable and run-global, including across hybrid subproblems,
so functional PRNG implementations can derive reproducible keys without
mutable state. This keeps JAX and other random libraries outside op_engine's
core dependency set.

`adaptive-tau-leaping` uses the bounded Cao--Gillespie--Petzold implementation
and accepts `stochastic_max_steps`, `tau_leap_tolerance`,
`tau_critical_threshold`, `tau_exact_fallback_multiplier`, and
`tau_max_retries`. It requires every selected op_system transition to declare
an explicit `reactants` list. This is intentionally stricter than fixed tau or
direct SSA: source/target changes cannot reveal catalytic reactants or
molecular multiplicity, and the provider never guesses them from a propensity
expression. A source-only zero-order reaction declares `reactants: []`.

Adaptive tau uses both a Poisson sampler and an SSA sampler because critical
events and low-count fallbacks are exact. NumPy supplies both from
`random_seed`; other namespaces inject both callables. Its eager stochastic
trajectory is not a pathwise-differentiable computation. JAX still preserves
array placement and remains available for deterministic differentiation, but a
stochastic gradient estimator must be supplied explicitly above this layer.

### Bounded thinning SSA

`thinning-ssa` is an exact method for time-dependent propensities in pure
stochastic mode. Use matching core/provider builds containing this feature.
The caller must supply a valid upper bound on the **total** propensity across
every expanded reaction channel and cell; the bound applies with the current
state held fixed. A zero instantaneous rate can become active later when the
bound is positive.

For example, a single source-only birth channel with rate `2 * t` over `[0, 1]`
has total bound `2`. Its system specification is:

```yaml
kind: transitions
axes: [{name: group, coords: [a]}]
state: ["X[group]"]
transitions:
  - name: birth
    from: null
    to: "X[group]"
    rate: "2 * t"
    reactants: []
```

Configure a constant bound independently of observation times:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    mode: stochastic
    stochastic_method: thinning-ssa
    thinning_rate_bound: 2.0
    thinning_max_candidates: 1000000
    random_seed: 173
```

For `N` such cells the constant bound is `2 * N`. A constant must bound every
state reached during the whole run, up to each forcing change or solve end.
It must be finite and non-negative. `thinning_max_candidates` counts all draws
throughout one run, including rejected candidates and candidates discarded
at forcing changes or bound expiry. Observations do not reset the guard.

For a bound that depends on time or state, leave `thinning_rate_bound` unset
and pass `rate_bound=` to `run`. It accepts a scalar or a callback with the
core `RateBoundFunction` contract. For the birth example above:

```python
from op_engine import TotalRateBound


def birth_bound(time, state, limit):
    # state has shape (n_flat_state_cells, 1) in the producer's state order.
    # This example has one source-only channel for each state cell.
    return TotalRateBound(2 * limit * state.shape[0], limit)


# engine is configured for thinning-ssa without thinning_rate_bound.
trajectory = engine.run(
    system, times, initial_state, params, rate_bound=birth_bound,
)
```

The callback receives the native state array and a `limit` equal to the next
combined producer/explicit forcing boundary or solve endpoint. It returns
`TotalRateBound(rate, valid_until)` with `time < valid_until <= limit`, bounding
all rates on `[time, valid_until)`. Treat inputs as read-only. Accepted events
refresh the bound at the updated state; rejected candidates retain it. Zero
bounds advance without drawing and can resume at the next bound interval.
Supplying both configuration and run-time bounds raises an error.

Candidates tied with expiry or a forcing change are discarded; forcing is
right-continuous. Pending candidates survive observations, so adding output
times with unchanged solve endpoints preserves a seeded exact path. The core
checks bounds at interval starts and candidate times and raises on encountered
violations. These checks cannot certify a bound at unsampled times. See the
[core bound guide](https://github.com/ACCIDDA/op_engine/blob/main/docs/guides/bounded-thinning.md)
for endpoint and floating-point details.

Non-NumPy runs inject `thinning_sampler(bound_rate, draw_index)`, returning
`ThinningSample(waiting_time, uniform)` with real floating scalar arrays in
the state namespace. The wait must be finite and positive, exponentially
distributed at `bound_rate`; the independent uniform lies in `[0, 1)`.
Draw indices increase globally, including rejection and expiry. Candidate-time
rates determine both acceptance and the selected flattened channel.
Hybrid thinning and thinning inputs supplied to other methods are rejected.

## Hybrid execution

Set `mode: hybrid` and list `stochastic_reactions` to execute those complete
named reaction families as jumps. The provider subtracts their compiled mean
drift from the bound op_system RHS, leaves every unselected contribution in
`CoreSolver`, and applies deterministic-then-stochastic first-order Lie
splitting on each requested output interval:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    mode: hybrid
    method: heun
    stochastic_method: tau-leaping
    stochastic_reactions: [expose, import_case]
    tau_max_step: 0.05
```

Reaction selection is by the typed artifact's parent name and includes all of
its expanded axis cells. Unknown or duplicate names fail validation. Methods
that require a full-system Jacobian are not allowed in hybrid mode: subtracting
selected mean drift changes that Jacobian, and op_system does not currently
publish the selected propensity derivatives needed to correct it. Use an
explicit or IMEX deterministic method. Exact-SSA waits are resampled at each
split boundary because the deterministic residual has just changed the state
and therefore the hazards.

The deterministic residual remains compatible with the ordinary JAX
differentiation path. A trajectory containing sampled Poisson or categorical
events is eager and discrete, however, and does not have ordinary pathwise
gradients. Gradient estimators for stochastic expectations belong above this
engine boundary rather than being implied by a JAX array result.
