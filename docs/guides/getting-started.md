# Getting Started

`ModelCore` owns a time grid and state history, while `CoreSolver` advances that
state. Explicit solves infer their Array-API namespace from the initial state.

```python
import numpy as np

from op_engine import CoreSolver, ModelCore

times = np.linspace(0.0, 2.0, 21)
core = ModelCore(n_states=1, n_subgroups=1, time_grid=times)
core.set_initial_state(np.asarray([[1.0]]))


def decay(_time, state):
    return -0.25 * state


CoreSolver(core).run(decay)
solution = core.state_array
```

## NumPy and JAX explicit solves

NumPy 2 and JAX arrays implement `__array_namespace__()`. Passing either one to
`set_initial_state` selects the namespace for state storage, history, Euler/Heun
stages, and adaptive error control. No backend argument is needed. An RHS should
derive operations from its state when it needs namespace functions:

```python
def portable_decay(_time, state):
    xp = state.__array_namespace__()
    return xp.multiply(state, -0.25)
```

The implicit and IMEX methods currently cross into SciPy sparse solves and
therefore require NumPy state. They fail early with a boundary-specific error
when given another namespace.
