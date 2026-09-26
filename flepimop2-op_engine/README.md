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

Diffrax is not required for JAX differentiation of fixed-step Euler, Heun, or
dense IMEX/implicit methods. Typed dense operator descriptors are compiled in
the evolving state's namespace, so descriptor parameters remain traceable too.
The compiler supports row-source axis-kernel generators and first-order upwind
advection plus centered diffusion on uniform axes, including dynamic signed
velocities and diffusion coefficients.
The NumPy path applies op_system's value-dependent generator validation and
eager scalar checks. A traced non-NumPy path can validate shapes and static
layout only; producers are responsible for maintaining generator, finiteness,
and non-negative diffusion-coefficient invariants in dynamic parameter values.

## Stochastic reaction networks

Set `mode: stochastic` to execute every named reaction artifact published by
`system.option("reactions")` as a discrete process. The provider does not parse
raw transition configuration. It expands each typed reaction's source cells
into flat channels and compiles the associated transition, source-only,
pinned-axis, and summed-axis bookkeeping into one stoichiometric matrix.

Two methods are available:

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    mode: stochastic
    stochastic_method: tau-leaping  # or direct-ssa
    random_seed: 90210              # NumPy convenience path
    tau_max_step: 0.1               # fixed tau-leaping only
```

`tau-leaping` accepts `tau_max_step` and `stochastic_max_steps`.
`direct-ssa` accepts `ssa_max_events`; its exact interpretation requires
propensities to remain time-homogeneous between events. Both methods preserve
the usual `(time, state...)` provider trajectory and fail visibly on invalid
propensities, invalid random draws, or negative states. Populations are never
silently clipped.

NumPy arrays use seeded `NumpyPoissonSampler` or `NumpySSASampler` instances.
Other namespaces inject a `poisson_sampler=` or `ssa_sampler=` callable into
`run`. The callable must return arrays in the input namespace. Its integer
step/draw index is stable and run-global, including across hybrid subproblems,
so functional PRNG implementations can derive reproducible keys without
mutable state. This keeps JAX and other random libraries outside op_engine's
core dependency set.

Adaptive tau-leaping is not exposed by this provider yet. Its pre-leap
selector needs the molecular reactant order, including catalytic reactants.
The current typed op_system artifact describes source consumption and target
scatter exactly, but an arbitrary propensity expression does not expose enough
information to infer that reactant-order matrix safely.

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
