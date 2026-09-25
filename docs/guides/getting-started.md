# Getting Started

`ModelCore` owns a time grid and state history, while `CoreSolver` advances that
state. Solves infer their Array-API namespace from the initial state.

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

## NumPy and JAX solves

NumPy 2 and JAX arrays implement `__array_namespace__()`. Passing either one to
`set_initial_state` selects the namespace for state storage, history, solver
stages, and adaptive error control. No backend argument is needed. An RHS should
derive operations from its state when it needs namespace functions:

```python
def portable_decay(_time, state):
    xp = state.__array_namespace__()
    return xp.multiply(state, -0.25)
```

Dense implicit and IMEX operators are converted into the state's namespace and
solved with its Array-API `linalg.solve` implementation. This provides the
portable correctness path for NumPy, JAX, and other conforming backends.

Sparse acceleration is an optional second tier. SciPy sparse operators retain
cached factorization for NumPy state. Installing `op-engine[cupy]` registers the
matching `cupyx.scipy.sparse` adapter; CuPy is never imported when it is not
installed. Each adapter detects only its own sparse objects. Backends without a
registered sparse implementation can use dense operators without engine changes.
