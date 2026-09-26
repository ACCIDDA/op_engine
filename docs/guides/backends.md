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

## Adaptive-controller boundary

The built-in adaptive controller preserves the active array namespace, but its
step acceptance is intentionally eager Python control flow. It is therefore not
a JAX-`jit`-compatible controller even though the numerical step formulas are
differentiable.

Projects that require a compiled adaptive controller, checkpointed adjoints, or
other solver-specific capabilities can provide those at an external plugin or
provider boundary. A specialized integration may return a complete trajectory
and adopt it through `ModelCore.apply_trajectory`; it does not need to add a
backend-specific method to `CoreSolver`.

This keeps optional packages such as Diffrax out of the core dependency and
method surfaces. The portable fixed-step methods and their differentiation
contract remain identical across conforming array namespaces.
