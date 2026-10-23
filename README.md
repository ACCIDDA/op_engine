# op_engine

Public alpha software. Released solvers have executable examples and
method-specific numerical and backend tests; interfaces and coverage are still
evolving. The alpha classifier describes software maturity, separately from
the scientific assessment of a particular model or simulation.

Operator-Partitioned Engine (OP Engine) is a framework-independent numerical
core for structured dynamical systems, including spatial and trait-structured PDEs,
nonlocal integrals and mixing, compartmental models, ODEs and stochastic
reaction networks. It provides explicit and implicit time stepping, operator
splitting, exact SSA and approximate tau-leaping. PDE and nonlocal models
use specified discretizations and operators. The documentation lists the
tested methods and array backends.

## Methods and interfaces
- State, operator and observation interfaces for structured fields and populations.
- Explicit/IMEX/implicit integration and exact/approximate stochastic reaction methods.
- Typed interfaces and Array-API explicit and dense-implicit methods, including
  reusable higher-order Runge--Kutta tableaus.
- JAX autodiff and JIT support for the same portable fixed-step methods.
- Separates state/time management (`ModelCore`) from stepping logic (`CoreSolver`).
- Optional adapters (e.g., flepimop2) without affecting the core API.
- IMEX paths accept externally supplied operator tuples; defaults remain explicit-only.

## Core surface
- `ModelCore`: state/time manager; configure axes, dtype, and optional history.
- `CoreSolver`: portable explicit methods through Dormand--Prince 5(4), plus dense IMEX and linearly implicit methods; accepts `RunConfig` with `AdaptiveConfig`, `DtControllerConfig`, and `OperatorSpecs`.
- `DirectSSASolver`: exact Gillespie direct-method trajectories with injected, backend-specific exponential and categorical sampling.
- `BatchedDirectSSASolver`: independent-clock NumPy simulation banks with stable replicate streams and optional accepted firing counts; see the [replicate guide](https://accidda.github.io/op_engine/latest/guides/replicate-ssa/).
- `AdaptiveTauLeapingSolver`: bounded adaptive tau-leaping with leap-condition control, exact critical events, and explicit post-leap rejection. Non-mass-action propensities (frequency-dependent infection) are supported through per-reaction dependency incidence and elasticity orders.
- `TauLeapingSolver`: fixed-step stochastic reaction-network integration with injected, backend-specific Poisson sampling.
- `from_compiled_rhs` / `compile_reaction_network`: build the flat stochastic network (stoichiometry, reactant orders, propensity) from an op_system spec's reaction artifacts, with no flepimop2 dependency.
- `steady_state`: equilibria by pseudo-transient continuation, with absorbing states held fixed, declared conservation constraints, and a Newton-correction convergence test; runs under `jax.jit`/`vmap`.
- `local_stability`: eager dense NumPy CPU evidence for a residual-qualified
  equilibrium, including explicit Jacobian provenance, constraint-manifold
  spectra, boundary and failure statuses, and interpretable leading
  e-folding times. It does not certify nonlinear or global stability.
- `matrix_ops`: portable dense advection/diffusion, sparse Laplacian/Crank–Nicolson, implicit Euler/trapezoidal builders, predictor–corrector, implicit solve cache, Kronecker helpers, and grouped aggregations.
- Extras: `OperatorSpecs`, `RunConfig`, `AdaptiveConfig`, `DtControllerConfig`, `Operator`, `GridGeometry`, `DiffusionConfig`.

See the documentation guides for [solver selection](https://accidda.github.io/op_engine/latest/guides/solver-methods/),
[the biogeochemical splitting tutorial](https://accidda.github.io/op_engine/latest/guides/biogeochemical-network/),
[stochastic simulation from an op_system spec](https://accidda.github.io/op_engine/latest/guides/op-system-reactions/),
[steady states](https://accidda.github.io/op_engine/latest/guides/steady-states/),
and [backend boundaries](https://accidda.github.io/op_engine/latest/guides/backends/).

## Installation

```bash
pip install op-engine
```

With JAX as the array namespace:

```bash
pip install "op-engine[jax]"
```

For the other packaged Array-API backends:

```bash
pip install "op-engine[torch]"
pip install "op-engine[cupy]"
```

NumPy is installed with the core package. Backend support is method-specific;
consult the [backend guide](https://accidda.github.io/op_engine/latest/guides/backends/) before assuming that JIT,
automatic differentiation, adaptivity, or stochastic sampling have identical
semantics across namespaces.

The `flepimop2-op-engine` provider has an independent release lifecycle. Its
published 0.8.0 package requires `op-engine==0.8.0` and is not compatible with
this 0.9.0 core release. Compatible provider source is present in this
repository as the unpublished development version `0.9.0.dev0`; developers who
need to test the two together should clone the repository and run
`just provider-sync`. The same source environment supplies the optional
`op_system` integration used by provider development. Do not use the provider's
PyPI installation commands for an `op-engine==0.9.0` environment until a
compatible provider release is published.

## Quickstart

This SIR example is a compartmental ODE introducing the deterministic call surface. For other model classes,
see the [trait-structured PDE](https://accidda.github.io/op_engine/latest/guides/chemostat-pde/),
[spatial operator splitting](https://accidda.github.io/op_engine/latest/guides/biogeochemical-network/),
[nonlocal/operator factories](https://accidda.github.io/op_engine/latest/guides/solver-methods/), and
[stochastic reaction simulation](https://accidda.github.io/op_engine/latest/guides/op-system-reactions/).

```python
import numpy as np
from op_engine import ModelCore, CoreSolver


# Define RHS
def rhs(t, y):
    s, i, r = y
    beta, gamma = 0.3, 0.1
    return np.array([-beta * s * i, beta * s * i - gamma * i, gamma * i])


# Time grid and state
core = ModelCore(n_states=3, n_subgroups=1, time_grid=np.linspace(0, 10, 101))
core.set_initial_state(np.array([0.999, 0.001, 0.0])[..., None])

solver = CoreSolver(core)
solver.run(rhs)  # defaults to Heun/RK2

solution = core.state_array  # shape (n_timesteps, state, subgroup)
```

### Array namespaces

`ModelCore` infers its numerical namespace from the initial state through
`array_api_compat.array_namespace()`. The state, stored history, solver stages,
and dense solver operations stay in that namespace. There is no `xp=` or
backend option. This supports standard Array-API objects and native arrays such
as `torch.Tensor`. Fixed-step methods can be differentiated by JAX; the built-in
adaptive controller is eager Python control flow. See the
[backend guide](https://accidda.github.io/op_engine/latest/guides/backends/) for the precise contract and the
[solver guide](https://accidda.github.io/op_engine/latest/guides/solver-methods/) for method-specific examples.

### IMEX with operators (tuple form)

```python
import numpy as np
from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import OperatorSpecs, RunConfig

n = 4
times = np.linspace(0.0, 1.0, 11)
core = ModelCore(n_states=n, n_subgroups=1, time_grid=times)
core.set_initial_state(np.ones((n, 1)))

# Identity implicit operator along state axis
L = np.eye(n)
R = np.eye(n)
ops = OperatorSpecs(default=(L, R))


def rhs(t, y):
    return -0.1 * y


solver = CoreSolver(core, operator_axis="state")
solver.run(
    rhs,
    config=RunConfig(
        method="imex-heun-tr",
        operators=ops,
    ),
)
```

This compact example uses identity left/right operators to show the API. For a
nontrivial implicit term, construct timestep-aware operator factories as in the
[solver guide](https://accidda.github.io/op_engine/latest/guides/solver-methods/#imex-methods-and-operator-factories).

## Selected public API

This list is a practical entry point rather than an exhaustive root export
list. See the [generated API reference](https://accidda.github.io/op_engine/latest/api-reference/op-engine/)
for the documented module-level surface.

- `ModelCore`: state tensor + time grid manager; supports extra axes and optional history.
- `CoreSolver`: portable explicit and dense IMEX/implicit stepping selected by the state array namespace.
- `DirectSSASolver`: exact event-by-event stochastic reaction trajectories with NumPy, JAX, or another Array-API namespace supplying random draws.
- `AdaptiveTauLeapingSolver`: adaptive non-negative stochastic leaps using explicit reactant stoichiometry and injected Poisson plus exact-event sampling.
- `TauLeapingSolver`: portable stoichiometric updates with NumPy, JAX, or another Array-API namespace supplying Poisson samples.
- `op_engine.reactions`: `from_compiled_rhs(compiled, params)` and `compile_reaction_network(reactions, template_shapes=..., axis_sizes=..., params=...)` return a `CompiledReactionNetwork` whose `stoichiometry`, `reactant_stoichiometry`, and `propensity` feed the stochastic solvers; `mean_drift` checks the network against the deterministic RHS.
- Operator utilities (`matrix_ops`): portable upwind advection, Laplacian, Crank–Nicolson/implicit Euler/trapezoidal operators, predictor-corrector builders, implicit solve cache, Kronecker helpers, and grouped aggregation utilities.
- Configuration helpers: `RunConfig`, `OperatorSpecs`, `AdaptiveConfig`, and
  `DtControllerConfig` from `op_engine.core_solver` for method, IMEX, and
  adaptive control.
- Adapters: optional flepimop2 integration (extra dependency) via entrypoints in the adapter package. The adapter merges any `mixing_kernels` already computed by op_system (no automatic generation) and consumes config-supplied IMEX operator specs (dict or `OperatorSpecs`), forwarding the chosen `operator_axis` to `CoreSolver`.

## Development

```bash
uv sync --dev
just ci
```

## License

MIT License

Required CPU method qualification is documented in the
[capability matrix](https://accidda.github.io/op_engine/latest/guides/backend-qualification/). Run `just qualify`
alongside `just ci`, `just docs`, and `just build-all` before submitting a change.
NumPy/JAX/CPU Torch jobs reject missing/skipped selected contracts; GPU checks
and large benchmark timings remain explicit opt-in work.

See [CONTRIBUTING.md](https://github.com/ACCIDDA/op_engine/blob/main/CONTRIBUTING.md) for clean setup, focused submissions,
required checks and support. Report suspected vulnerabilities through the
[security policy](https://github.com/ACCIDDA/op_engine/blob/main/SECURITY.md). Use [CITATION.cff](https://github.com/ACCIDDA/op_engine/blob/main/CITATION.cff) for attribution and
[citation/archiving guidance](https://accidda.github.io/op_engine/latest/development/citation-and-archiving/) for
core and provider version-specific identifiers.

The [capability guide](https://accidda.github.io/op_engine/latest/guides/capabilities/) maps deterministic and
stochastic solvers, observations, replay, runnable examples and CI checks.
