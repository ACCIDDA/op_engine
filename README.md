# op_engine

Operator-Partitioned Engine (OP Engine) is a lightweight multiphysics solver core for time-dependent systems. It supports explicit ODE solvers and IMEX/operator-based schemes for PDE-like models while staying framework-agnostic.

## Why use it?
- Shared solver surface for ODEs and operator-split PDEs.
- Strong typing, minimal dependencies (NumPy + SciPy for implicit paths).
- Separates state/time management (`ModelCore`) from stepping logic (`CoreSolver`).
- Optional adapters (e.g., flepimop2) without affecting the core API.
- IMEX paths accept externally supplied operator tuples; defaults remain explicit-only.

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

## Public API
- `ModelCore`: state tensor + time grid manager; supports extra axes and optional history.
- `CoreSolver`: explicit and IMEX stepping; methods: `euler`, `heun`, `imex-euler`, `imex-heun-tr`, `imex-trbdf2`.
- Operator utilities (`matrix_ops`): Laplacian builders, Crank–Nicolson/implicit Euler/trapezoidal operators, predictor-corrector builders, implicit solve cache, Kronecker helpers, grouped aggregation utilities.
- Adapters: optional flepimop2 integration (extra dependency) via entrypoints in the adapter package. Adapter merges any `mixing_kernels` exposed by op_system and will consume config-supplied IMEX operator tuples when provided.

## Development

```bash
uv sync --dev
just ci
```

## License

GPL-3.0
