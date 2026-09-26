# Choosing and configuring a solver

`ModelCore` owns output times and state storage. `CoreSolver` advances that
state using a method selected by `RunConfig`. The state array selects the
Array-API namespace; choosing JAX instead of NumPy does not select a different
numerical method.

## Method guide

| Method | Order | Split/operator input | Typical use |
| --- | ---: | --- | --- |
| `euler` | 1 | None | Debugging and first-order reference runs |
| `heun` | 2 | None | Default non-stiff explicit integration |
| `imex-euler` | 1 | One implicit-Euler operator factory | Robust first-order split systems |
| `imex-heun-tr` | 2 | One trapezoidal operator factory | Second-order explicit/implicit splitting |
| `imex-trbdf2` | 2 | Trapezoidal and BDF2-stage factories | Split systems needing stronger damping |
| `implicit-euler` | 1 | Jacobian callable | Stiff systems where first-order damping is useful |
| `trapezoidal` | 2 | Jacobian callable | Second-order linearly implicit integration |
| `bdf2` | 2 | Jacobian callable | Uniform, fixed-step stiff integration |
| `ros2` | 2 | Jacobian callable | Second-order Rosenbrock-W integration |

`bdf2` currently requires a uniform output grid and `adaptive=False`. Its first
step uses linearly implicit Euler because no previous state exists yet. The
other methods support the built-in adaptive controller, subject to the
compiled-control-flow boundary described below.

## Explicit methods

Euler and Heun need only an RHS. The default is fixed-step Heun, with one step
per output interval:

```python
import numpy as np

from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import AdaptiveConfig, RunConfig

times = np.linspace(0.0, 4.0, 41)
core = ModelCore(n_states=1, n_subgroups=1, time_grid=times)
core.set_initial_state(np.asarray([[1.0]]))


def rhs(_time, state):
    return -0.5 * state


config = RunConfig(
    method="heun",
    adaptive=True,
    adaptive_cfg=AdaptiveConfig(rtol=1e-6, atol=1e-9),
)
CoreSolver(core).run(rhs, config=config)
```

With `adaptive=False`, every adjacent pair of output times defines one solver
step. With `adaptive=True`, the controller may take multiple internal steps but
still stores only the requested output times.

## IMEX methods and operator factories

IMEX methods solve a split system

\[
  y' = A(t, y)y + F(t, y),
\]

where the RHS callable supplies `F` and an operator factory supplies the
implicit part. A resolved operator tuple is either `(L, R)` or
`(predictor, L, R)`, with the core solving `L @ y_next = R @ x` along the
configured operator axis.

Use a `StageOperatorFactory` whenever the output spacing is non-uniform or
adaptive stepping is enabled. The factory receives the attempted full-step
size, a stage scale, and `StageOperatorContext`, so it can rebuild operators
for the actual stage time and state.

```python
import numpy as np

from op_engine import CoreSolver, ModelCore, build_diffusion_matrix
from op_engine.core_solver import OperatorSpecs, RunConfig
from op_engine.matrix_ops import (
    StageOperatorContext,
    make_stage_operator_factory,
)

n_cells = 32
dx = 1.0 / n_cells
diffusivity = 0.02
times = np.linspace(0.0, 1.0, 21)

core = ModelCore(n_states=n_cells, n_subgroups=1, time_grid=times)
core.set_initial_state(np.ones((n_cells, 1)))


def reaction(_time, state):
    return -0.1 * state


def diffusion(ctx: StageOperatorContext):
    return build_diffusion_matrix(
        n_cells,
        dx,
        diffusivity,
        bc="neumann",
        reference=ctx.y,
    )


trapezoidal = make_stage_operator_factory(diffusion, scheme="trapezoidal")
config = RunConfig(
    method="imex-heun-tr",
    operators=OperatorSpecs(default=trapezoidal),
)
CoreSolver(core, operator_axis="state").run(reaction, config=config)
```

For `imex-trbdf2`, provide both stage schemes:

```python
implicit_euler = make_stage_operator_factory(diffusion, scheme="implicit-euler")
config = RunConfig(
    method="imex-trbdf2",
    operators=OperatorSpecs(tr=trapezoidal, bdf2=implicit_euler),
)
```

The flepimop2 provider can compile typed `op_system` `axis_kernel`, `advection`,
and `diffusion` descriptors into these same factories. Descriptor parsing and
axis-label resolution belong to the provider; the numerical builders and
stepping methods remain in `op_engine`.

## Linearly implicit methods

The fully implicit and Rosenbrock methods take a Jacobian callable. The
callable returns an operator acting along `operator_axis`:

```python
def jacobian(_time, state):
    xp = state.__array_namespace__()
    return xp.multiply(xp.eye(state.shape[0], dtype=state.dtype), -0.5)


config = RunConfig(method="ros2", jacobian=jacobian)
CoreSolver(core, operator_axis="state").run(rhs, config=config)
```

Dense Jacobians stay in the state's namespace. SciPy sparse operators are an
optional NumPy acceleration path and are not a JAX differentiation path.

## Adaptivity and Array-API backends

`AdaptiveConfig` sets tolerances and attempt limits; `DtControllerConfig` sets
step-size bounds and growth factors. Their static scalar values are validated
when the configuration is constructed. Checks that depend on the time grid,
state shape, or resolved operators occur when `CoreSolver` builds its run plan.

Fixed-step explicit, dense IMEX, and dense linearly implicit methods use
ordinary Array-API operations. With JAX arrays, those same methods can run
under `jax.jit` and be differentiated with `jax.grad`; no JAX-specific solver
implementation is selected.

The built-in adaptive controller uses eager Python loops and scalar acceptance
decisions. It preserves JAX arrays but is not itself JIT-compatible. A compiled
adaptive controller or specialized adjoint implementation belongs in an
external provider and may return a trajectory through
`ModelCore.apply_trajectory`.

See [Backend and solve-strategy boundaries](backends.md) for the complete
portability contract and [the biogeochemical tutorial](biogeochemical-network.md)
for a larger IMEX splitting example.
