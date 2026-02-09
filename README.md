# op_engine

Operator-Partitioned Engine (OP Engine) is a lightweight multiphysics solver core for time-dependent systems. It supports explicit ODE solvers and IMEX/operator-based schemes for PDE-like models while staying framework-agnostic.

## Why use it?
- Shared solver surface for ODEs and operator-split PDEs.
- Strong typing, minimal dependencies (NumPy + SciPy for implicit paths).
- Separates state/time management (`ModelCore`) from stepping logic (`CoreSolver`).
- Optional adapters (e.g., flepimop2) without affecting the core API.
- IMEX paths accept externally supplied operator tuples; defaults remain explicit-only.

## Core surface
- `ModelCore`: state/time manager; configure axes, dtype, and optional history.
- `CoreSolver`: explicit + IMEX methods (`euler`, `heun`, `imex-euler`, `imex-heun-tr`, `imex-trbdf2`); accepts `RunConfig` with `AdaptiveConfig`, `DtControllerConfig`, and `OperatorSpecs`.
- `matrix_ops`: Laplacian/Crank–Nicolson/implicit Euler/trapezoidal builders, predictor–corrector, implicit solve cache, Kronecker helpers, grouped aggregations.
- Extras: `OperatorSpecs`, `RunConfig`, `AdaptiveConfig`, `DtControllerConfig`, `Operator`, `GridGeometry`, `DiffusionConfig`.

## Installation

```bash
pip install op_engine
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
    return np.array([-beta*s*i, beta*s*i - gamma*i, gamma*i])

# Time grid and state
core = ModelCore(n_states=3, n_subgroups=1, time_grid=np.linspace(0, 10, 101))
core.set_initial_state(np.array([0.999, 0.001, 0.0])[..., None])

solver = CoreSolver(core)
solver.run(rhs)  # defaults to Heun/RK2

solution = core.state_array  # shape (n_timesteps, state, subgroup)
```

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
- `CoreSolver`: explicit and IMEX stepping; methods: `euler`, `heun`, `imex-euler`, `imex-heun-tr`, `imex-trbdf2`.
- Operator utilities (`matrix_ops`): Laplacian builders, Crank–Nicolson/implicit Euler/trapezoidal operators, predictor-corrector builders, implicit solve cache, Kronecker helpers, grouped aggregation utilities.
- Configuration helpers: `RunConfig`, `OperatorSpecs`, `AdaptiveConfig`, `DtControllerConfig` for method/IMEX/adaptive control.
- Adapters: optional flepimop2 integration (extra dependency) via entrypoints in the adapter package. Adapter merges `mixing_kernels` exposed by op_system and consumes config-supplied IMEX operator tuples when provided; operator metadata in op_system is not auto-translated yet.

## Development

```bash
uv sync --dev
just ci
```

## License

GPL-3.0
