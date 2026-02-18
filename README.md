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
- `CoreSolver`: explicit + IMEX + stiff ODE methods (`euler`, `heun`, `imex-euler`, `imex-heun-tr`, `imex-trbdf2`, `implicit-euler`, `trapezoidal`, `bdf2`, `ros2`); accepts `RunConfig` with `AdaptiveConfig`, `DtControllerConfig`, `OperatorSpecs`, and optional `jacobian`.
- `matrix_ops`: Laplacian/Crank–Nicolson/implicit Euler/trapezoidal builders, predictor–corrector, implicit solve cache, Kronecker helpers, grouped aggregations.
- Extras: `OperatorSpecs`, `RunConfig`, `AdaptiveConfig`, `DtControllerConfig`, `Operator`, `GridGeometry`, `DiffusionConfig`.

## Which method to pick?
- Explicit (`euler`, `heun`): non-stiff ODEs, small systems, cheap RHS; `heun` (default) is stable and second order.
- IMEX (`imex-euler`, `imex-heun-tr`): moderately stiff when a linear operator can be implicit; works best when `(L, R)` are constant across steps.
- IMEX TR-BDF2 (`imex-trbdf2`): higher stability/accuracy for mild/moderate stiffness; use stage-operator factories if dt changes.
- Fully implicit (`implicit-euler`, `trapezoidal`, `bdf2`): stiff ODEs without a split operator; `implicit-euler` for robustness, `trapezoidal` for A-stable order 2, `bdf2` for smoother stiff flows.
- Rosenbrock-W (`ros2`): linearly implicit stiff ODEs when you have a Jacobian and want a single linear solve per stage.

Operator guidance:
- Fixed dt and time-invariant operators → supply `(L, R)` tuples directly.
- Variable dt or operator depends on `t`/`y` → provide a `StageOperatorFactory`.
- Pick `operator_axis` to match the dimension your operators act on (default is `state`).

Adaptive stepping:
- Enable `adaptive=True` for smooth non-split RHS when you want automatic dt control.
- Prefer fixed-step for IMEX/operator paths when operators are pre-built for a specific dt.

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
from op_engine.core_solver import RunConfig

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
cfg = RunConfig(method="imex-heun-tr", operators=ops)
solver.run(rhs, config=cfg)
```

## Public API
- `ModelCore`: state tensor + time grid manager; supports extra axes and optional history.
- `CoreSolver`: explicit, IMEX, and stiff ODE stepping; methods: `euler`, `heun`, `imex-euler`, `imex-heun-tr`, `imex-trbdf2`, `implicit-euler`, `trapezoidal`, `bdf2`, `ros2`.
- Operator utilities (`matrix_ops`): Laplacian builders, Crank–Nicolson/implicit Euler/trapezoidal operators, predictor-corrector builders, implicit solve cache, Kronecker helpers, grouped aggregation utilities.
- Configuration helpers: `RunConfig`, `OperatorSpecs`, `AdaptiveConfig`, `DtControllerConfig` for method/IMEX/adaptive control.
- Adapters: optional flepimop2 integration (extra dependency) via entrypoints in the adapter package. The adapter merges any `mixing_kernels` already computed by op_system (no automatic generation) and consumes config-supplied IMEX operator specs (dict or `OperatorSpecs`), forwarding the chosen `operator_axis` to `CoreSolver`.

## Model shapes at a glance
- `n_states`: primary state dimension; `operator_axis` defaults here.
- `n_subgroups`: second axis often used for populations/ensembles.
- Extra axes: configure via `other_axes`/`axis_names` in `ModelCoreOptions`.
- `store_history`: keep full time history (True) or final slice only (False) for memory savings.

## Development

```bash
uv sync --dev
just ci
```

## License

GPL-3.0
