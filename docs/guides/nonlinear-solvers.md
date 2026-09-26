# Nonlinear solver contract

`op_engine` does not yet expose SDIRK or fully implicit Runge--Kutta methods.
Those methods require a genuine nonlinear solve for each implicit stage. The
existing `implicit-euler`, `trapezoidal`, and `bdf2` methods take one Jacobian
linearization per attempted step; despite their historical method names, they
are linearly implicit approximations rather than converged nonlinear methods.

The public nonlinear contract establishes the boundary needed before a fully
implicit method is added. `NonlinearProblem` supplies a residual plus an
optional dense Jacobian and/or Jacobian-vector product. Each callback must
preserve the iterate's array namespace and dtype. Residuals retain the iterate
shape; dense Jacobians act on the flattened state and therefore have shape
`(state_size, state_size)`.

```python
import numpy as np

from op_engine import DenseNewtonSolver, NonlinearProblem, require_converged


def residual(value):
    return value * value - 2.0


def jacobian(value):
    return np.reshape(2.0 * value, (1, 1))


result = DenseNewtonSolver().solve(
    NonlinearProblem(residual=residual, jacobian=jacobian),
    np.asarray([1.5]),
)
root = require_converged(result).value
```

The solver receives the initial guess on every call. It has no hidden warm-start
cache. A future implicit method is responsible for selecting a deterministic
stage predictor; a rejected attempt must not silently mutate solver state used
by its retry.

## Portable dense default

`DenseNewtonSolver` requires a floating-point initial guess and uses
[`xp.linalg.solve`](https://data-apis.org/array-api/2024.12/extensions/generated/array_api.linalg.solve.html)
from that array's Array-API namespace. It performs exactly
`NewtonConfig.max_iterations` updates. The fixed loop length is deliberate: it
can be statically unrolled by JAX and does not need a traced value to control a
Python loop.

The result always contains the candidate, final residual, and diagnostics.
`converged`, the initial/final residual norms, and the final step norm remain
zero-dimensional arrays in the active namespace. No host scalar conversion is
performed during the solve. The convergence test is

`final_rms <= atol + rtol * initial_rms`

and also requires every final residual element to be finite. A missing dense
Jacobian, callback shape/dtype/namespace violation, or exception raised by the
underlying linear solve is a fatal evaluation error. If a backend instead
represents a singular solve with non-finite arrays, the final diagnostic reports
nonconvergence.

The dense default is intentionally small. Sparse direct solvers, Krylov/JVP
methods, line searches, trust regions, and PETSc- or package-specific state
belong in optional implementations of the `NonlinearSolver` protocol. Their
objects are not part of `CoreSolver` or `RunConfig`.

## Failure and adaptive stepping

Nonconvergence is data until an integration boundary applies policy:

| Execution context | Required behavior |
| --- | --- |
| Eager fixed step | `require_converged` raises `NonlinearConvergenceError` with the failed result attached. |
| Future live adaptive step | Reject the attempted step, reduce `dt`, and charge the rejection against `AdaptiveConfig.max_reject`. Never advance state from a failed candidate. |
| Compiled accepted-schedule replay | Return stage diagnostics from compiled code and validate them afterward. A failed stage invalidates that replay; the recorded mesh cannot be changed inside the trace. |
| Callback or linear-algebra exception | Propagate as a fatal evaluation/configuration error; do not reinterpret arbitrary exceptions as a recoverable timestep rejection. |

These rules keep numerical nonconvergence distinct from malformed callbacks and
backend failures. They also prevent a compiled replay from silently accepting a
failed stage merely because its timestep sequence is fixed.

## JAX differentiation

With JAX arrays, the dense default is compatible with `jax.jit`, and automatic
differentiation follows the fixed sequence of Newton operations. This is
unrolled-iteration differentiation: computation and derivative cost grow with
`max_iterations`, and the derivative is that of the finite iteration, not an
implicit-function derivative of an exact root.

`require_converged` is an eager host boundary and must not be called inside a
JAX trace. Return the array-valued diagnostics from compiled code and inspect
them afterward.

A JAX-specific optional solver may instead implement implicit differentiation
or a custom solve behind the same protocol. For example, JAX provides
[`jax.lax.custom_root`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
for roots whose gradients are defined by the implicit function theorem. That
package-specific mechanism does not enter the core configuration or alter the
portable default's semantics.

An SDIRK prototype should be added only after its stage residuals, warm starts,
nonconvergence handling, and diagnostic propagation can be expressed entirely
through this contract.
