# op_engine

Operator-Partitioned Engine (OP Engine) is a lightweight multiphysics solver core for time-dependent systems. It supports explicit ODE solvers and IMEX/operator-based schemes for PDE-like models while staying framework-agnostic.

## Why use it?
- Shared solver surface for ODEs and operator-split PDEs.
- Strong typing and Array-API explicit and dense-implicit paths.
- JAX autodiff and JIT support for the same portable fixed-step methods.
- Separates state/time management (`ModelCore`) from stepping logic (`CoreSolver`).
- Optional adapters (e.g., flepimop2) without affecting the core API.
- IMEX paths accept externally supplied operator tuples; defaults remain explicit-only.

## Core surface
- `ModelCore`: state/time manager; configure axes, dtype, and optional history.
- `CoreSolver`: portable explicit and dense IMEX/implicit methods; accepts `RunConfig` with `AdaptiveConfig`, `DtControllerConfig`, and `OperatorSpecs`.
- `matrix_ops`: portable dense advection/diffusion, sparse Laplacian/Crank–Nicolson, implicit Euler/trapezoidal builders, predictor–corrector, implicit solve cache, Kronecker helpers, and grouped aggregations.
- Extras: `OperatorSpecs`, `RunConfig`, `AdaptiveConfig`, `DtControllerConfig`, `Operator`, `GridGeometry`, `DiffusionConfig`.

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
and adaptive error control stay in that namespace. There is no `xp=` or backend
option:

```python
import jax.numpy as jnp
import numpy as np

from op_engine import CoreSolver, ModelCore

core = ModelCore(1, 1, np.asarray([0.0, 0.5, 1.0]))
core.set_initial_state(jnp.asarray([[1.0]]))


def decay(_time, state):
    xp = state.__array_namespace__()
    return xp.multiply(state, -0.2)


CoreSolver(core).run(decay)
assert core.state_array.__array_namespace__() is jnp
```

The RHS must return an array in the input state's namespace. Dense implicit and
IMEX operators use that namespace's Array-API `linalg.solve`, so the same
methods work with NumPy and JAX arrays. Sparse operators use an acceleration
registry: SciPy is included for NumPy state, and CuPy sparse support is enabled
when the `cupy` extra is installed. A backend without a sparse adapter still has
the dense correctness path.

With JAX arrays, the fixed-step Euler, Heun, dense IMEX, and dense linearly
implicit methods can be used under `jax.jit` and differentiated with `jax.grad`.
This includes gradients through dynamic RHS parameters, initial state, and dense
operator or Jacobian values. No separate solver implementation is required.

The portable adaptive controller is eager-only under JAX: it keeps state arrays
in JAX, but its step acceptance uses Python scalar extraction and control flow.
Use fixed steps when compiling the portable methods. Compiled adaptive
controllers and checkpointed adjoints are outside the core Array-API method
contract and can be provided by specialized external integrations. See the
[backend guide](docs/guides/backends.md) for the precise boundary.

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
