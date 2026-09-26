# Backend and solve-strategy boundaries

`op_engine` selects the numerical namespace from the state array. Explicit and
dense implicit/IMEX methods use that namespace's Array-API operations, including
`linalg.solve`. The numerical method does not change when the array namespace
changes.

Namespace portability and automatic differentiation are related but distinct.
The Array API does not define `grad`, `jit`, or traced control flow. A namespace
such as JAX can nevertheless differentiate the ordinary `op_engine` methods
because their fixed-step numerical operations remain in that namespace.

## Portable fixed-step methods

With JAX state, fixed-step Euler, Heun, dense IMEX, and dense linearly implicit
methods can be used inside `jax.jit` and differentiated with `jax.grad`. Dynamic
values can include the initial state, RHS parameters, dense operator values, and
Jacobian values. The time grid, method selection, state shape, and Python solver
configuration are structural inputs and must remain static during a trace.

Sparse acceleration is backend-specific. SciPy sparse solves are not a JAX
differentiation path; use dense JAX operators when gradients through a solve are
required.

The built-in adaptive controller is an eager Python controller. It preserves a
JAX array namespace, but step acceptance extracts scalar error values and uses
Python loops and branches. Consequently, `adaptive=True` on the portable methods
is not currently a JAX-`jit`-compatible execution path. This limitation belongs
to the controller, not to the Euler, Heun, IMEX, or linearly implicit formulas.

## Optional Diffrax Tsit5 strategy

`diffrax-tsit5` is an optional JAX-specific adaptive strategy. It is not required
for differentiation of the portable fixed-step methods. It is useful when a run
specifically requires a compiled adaptive controller, checkpointed adjoints, or
other Diffrax capabilities.

Install the optional dependencies:

```bash
pip install "op_engine[jax]"
```

Initialize `ModelCore` with JAX state and select the adaptive method:

```python
import jax.numpy as jnp
import numpy as np

from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import AdaptiveConfig, RunConfig
from op_engine.model_core import ModelCoreOptions

times = np.linspace(0.0, 10.0, 101, dtype=np.float32)
core = ModelCore(
    1,
    1,
    times,
    options=ModelCoreOptions(dtype=np.float32),
)
core.set_initial_state(jnp.asarray([[1.0]], dtype=jnp.float32))


def decay(_time, state):
    return -0.2 * state


config = RunConfig(
    method="diffrax-tsit5",
    adaptive=True,
    adaptive_cfg=AdaptiveConfig(rtol=1e-6, atol=1e-8),
)
CoreSolver(core).run(decay, config=config)
```

The method uses Diffrax `Tsit5`, `PIDController`, and
`RecursiveCheckpointAdjoint`, saving at every `ModelCore.time_grid` entry.
The RHS and initial state must already be JAX-native; the solver rejects NumPy
state instead of silently moving it to another device.

The first Diffrax slice supports ordinary array state and a single deterministic
RHS. Shaped PyTree/provider execution, block and draw
batching, history/DDE terms, and hybrid CTMC execution are separate follow-up
capabilities. Existing Array-API and sparse-accelerated methods remain unchanged.
