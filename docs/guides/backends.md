# Backend and solve-strategy boundaries

`op_engine` selects ordinary numerical namespaces from the state array. Explicit
methods and dense implicit/IMEX methods use the state namespace's Array-API
operations, including `linalg.solve`. That path provides portable numerical
correctness without storing a backend module in solver configuration.

Differentiable adaptive integration is a different capability. The Array API
does not define adaptive ODE controllers, checkpointed adjoints, or traced solve
loops. For those operations, `CoreSolver` dispatches a method-specific strategy
while keeping the same `CoreSolver.run()` surface.

## Diffrax Tsit5

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
RHS. Operator splitting, shaped PyTree/provider execution, block and draw
batching, history/DDE terms, and hybrid CTMC execution are separate follow-up
capabilities. Existing Array-API and sparse-accelerated methods remain unchanged.
