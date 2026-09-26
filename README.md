# op_engine

Operator-Partitioned Engine (OP Engine) is a lightweight multiphysics solver core for time-dependent systems. It supports explicit ODE solvers, IMEX/operator-based schemes for PDE-like models, and exact or approximate stochastic reaction-network solvers while staying framework-agnostic.

## Why use it?
- Shared solver surface for ODEs and operator-split PDEs.
- Strong typing and Array-API explicit and dense-implicit paths, including
  reusable higher-order Runge--Kutta tableaus.
- JAX autodiff and JIT support for the same portable fixed-step methods.
- Separates state/time management (`ModelCore`) from stepping logic (`CoreSolver`).
- Optional adapters (e.g., flepimop2) without affecting the core API.
- IMEX paths accept externally supplied operator tuples; defaults remain explicit-only.

## Core surface
- `ModelCore`: state/time manager; configure axes, dtype, and optional history.
- `CoreSolver`: portable explicit methods through Dormand--Prince 5(4), plus dense IMEX and linearly implicit methods; accepts `RunConfig` with `AdaptiveConfig`, `DtControllerConfig`, and `OperatorSpecs`.
- `DirectSSASolver`: exact Gillespie direct-method trajectories with injected, backend-specific exponential and categorical sampling.
- `TauLeapingSolver`: fixed-step stochastic reaction-network integration with injected, backend-specific Poisson sampling.
- `matrix_ops`: portable dense advection/diffusion, sparse Laplacian/Crank–Nicolson, implicit Euler/trapezoidal builders, predictor–corrector, implicit solve cache, Kronecker helpers, and grouped aggregations.
- Extras: `OperatorSpecs`, `RunConfig`, `AdaptiveConfig`, `DtControllerConfig`, `Operator`, `GridGeometry`, `DiffusionConfig`.

See the documentation guides for [solver selection](docs/guides/solver-methods.md),
[the biogeochemical splitting tutorial](docs/guides/biogeochemical-network.md),
and [backend boundaries](docs/guides/backends.md).

## Installation

```bash
pip install op_engine
```

With JAX as the array namespace:

```bash
pip install "op_engine[jax]"
```

With flepimop2 adapter:

```bash
pip install "op_engine[flepimop2]"
```

## Quickstart

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
`initial_state.__array_namespace__()`. The state, stored history, solver stages,
and dense solver operations stay in that namespace. There is no `xp=` or
backend option. Fixed-step methods can be differentiated by JAX; the built-in
adaptive controller is eager Python control flow. See the
[backend guide](docs/guides/backends.md) for the precise contract and the
[solver guide](docs/guides/solver-methods.md) for method-specific examples.

### IMEX with operators (tuple form)

```python
import numpy as np
from op_engine import CoreSolver, ModelCore, OperatorSpecs

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


solver = CoreSolver(core, operators=ops.default, operator_axis="state")
solver.run(rhs, config=None)  # defaults: method="heun" (explicit)

# For IMEX methods set method and operators via RunConfig:
# from op_engine.core_solver import RunConfig, AdaptiveConfig, DtControllerConfig
```

## Public API
- `ModelCore`: state tensor + time grid manager; supports extra axes and optional history.
- `CoreSolver`: portable explicit and dense IMEX/implicit stepping selected by the state array namespace.
- `DirectSSASolver`: exact event-by-event stochastic reaction trajectories with NumPy, JAX, or another Array-API namespace supplying random draws.
- `TauLeapingSolver`: portable stoichiometric updates with NumPy, JAX, or another Array-API namespace supplying Poisson samples.
- Operator utilities (`matrix_ops`): portable upwind advection, Laplacian, Crank–Nicolson/implicit Euler/trapezoidal operators, predictor-corrector builders, implicit solve cache, Kronecker helpers, and grouped aggregation utilities.
- Configuration helpers: `RunConfig`, `OperatorSpecs`, `AdaptiveConfig`, `DtControllerConfig` for method/IMEX/adaptive control.
- Adapters: optional flepimop2 integration (extra dependency) via entrypoints in the adapter package. The adapter merges any `mixing_kernels` already computed by op_system (no automatic generation) and consumes config-supplied IMEX operator specs (dict or `OperatorSpecs`), forwarding the chosen `operator_axis` to `CoreSolver`.

## Development

```bash
uv sync --dev
just ci
```

## License

MIT License
