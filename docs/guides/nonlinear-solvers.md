# Nonlinear solver contract

`op_engine` exposes the two-stage Alexander SDIRK2 method for systems that need
a genuine nonlinear solve at each implicit stage. The existing
`implicit-euler`, `trapezoidal`, and `bdf2` methods still take only one Jacobian
linearization per attempted step; despite their historical method names, they
are linearly implicit approximations rather than converged nonlinear methods.

The public nonlinear contract used by SDIRK2 keeps the method independent of a
specific array package. `NonlinearProblem` supplies a residual plus an
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
cache. SDIRK2 selects a deterministic lower-triangular stage predictor; a
rejected attempt does not mutate solver state used by its retry.

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
belong in optional implementations of the `NonlinearSolver` protocol. Core
configuration depends only on that protocol; it has no SciPy-, JAX-, or
PETSc-specific field or backend selector.

## Failure and adaptive stepping

Nonconvergence is data until an integration boundary applies policy:

| Execution context | Required behavior |
| --- | --- |
| Eager fixed step | `CoreSolver.run` raises `NonlinearConvergenceError` with the failed stage result attached and does not advance state. |
| Live adaptive step | Reject the attempted step, reduce `dt`, and charge the rejection against `AdaptiveConfig.max_reject`. Never advance state from a failed candidate. |
| Compiled accepted-schedule replay | Return array-valued diagnostics and validate them afterward. A failed stage invalidates that replay; the recorded mesh cannot be changed inside the trace. |
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

## Configuring SDIRK2

The public `sdirk2` method uses the two-stage, second-order, L-stable formula
from [Alexander (1977)](https://epubs.siam.org/doi/10.1137/0714068), with
`gamma = 1 - 1/sqrt(2)`, coefficient rows `(gamma)` and
`(1 - gamma, gamma)`, and weights equal to the final row. It is stiffly
accurate: a converged final stage is the accepted state.

Every stage constructs a `NonlinearProblem`, starts from its explicit
lower-triangular predictor, and returns its `NonlinearSolveDiagnostics` without
a host decision. Adaptive step doubling evaluates one full step and two
half steps, scales their difference by `1 / (2**order - 1)`, and combines all
six stage-convergence flags for an adaptive controller.

Supply at least one full-system RHS derivative through
`NonlinearMethodConfig`: `rhs_jacobian(t, state)` or
`rhs_jvp(t, state, direction)`. The dense Jacobian has shape
`(state.size, state.size)`, even when the state has several axes. The JVP maps a
full-state direction to the same shape, dtype and namespace, including all
cross-axis coupling. Neither uses the operator-axis `RunConfig.jacobian` of
linearly implicit methods. Existing positional `(rhs_jacobian, solver)` calls
retain their meaning.

```python
import numpy as np

from op_engine import CoreSolver, ModelCore, NonlinearMethodConfig
from op_engine.core_solver import AdaptiveConfig, RunConfig

times = np.linspace(0.0, 1.0, 11)
core = ModelCore(1, 1, times)
core.set_initial_state(np.asarray([[1.0]]))


def rhs(_time, state):
    return -(state * state)


def full_rhs_jacobian(_time, state):
    return np.reshape(-2.0 * state, (1, 1))


config = RunConfig(
    method="sdirk2",
    adaptive=True,
    adaptive_cfg=AdaptiveConfig(rtol=1e-6, atol=1e-9),
    nonlinear=NonlinearMethodConfig(rhs_jacobian=full_rhs_jacobian),
)
diagnostics = CoreSolver(core).run(rhs, config=config)
assert diagnostics is not None
diagnostics.require_converged()
```

`NonlinearMethodConfig` defaults to the backend-neutral `DenseNewtonSolver` and
also accepts another `NonlinearSolver` implementation. It never stores an
array namespace or backend module. A JVP-only configuration needs a custom
solver that consumes `NonlinearProblem.jvp`; the default dense solver still
requires the dense callback and raises if it is absent.

### Matrix-free stage boundary

For stage time `t_i`, predictor `base` and diagonal step `d = dt * gamma`,
the stage residual is `R(Y) = Y - base - d * rhs(t_i, Y)`. Its action is exactly
`R'(Y) v = v - d * rhs_jvp(t_i, Y, v)`. The method supplies this action through
the existing `NonlinearProblem.jvp` field, checking both direction and result
shape, dtype and namespace before returning it to the solver. Callback errors
remain fatal model/derivative errors.

Provide `NonlinearMethodConfig(rhs_jvp=full_rhs_jvp, solver=jvp_solver)` for a
JVP-aware solver. When `rhs_jacobian` is omitted, the stage constructs neither a
dense identity nor a dense Jacobian, and never probes basis vectors to infer
one. Supplying both callbacks exposes both derivative forms; the provider must
ensure they describe the same RHS. Fixed integration, live adaptive step
doubling and frozen-mesh replay all use the same stage action.

The focused CPU contract in `tests/op_engine/test_sdirk_jvp.py` uses an
explicitly diagonal manufactured nonlinear problem and a test-only diagonal
Newton solver. It compares trajectories with dense Newton and the independent
continuous solution, checks a coupled time-dependent action against dense and
finite-difference references, and checks float32/64 native values in NumPy,
JAX and Torch. JAX fixed solves support JIT, first gradients and `vmap` in this
contract; Torch supports eager functional first gradients. The existing
public Torch SDIRK convergence check calls `.item()` and excludes `vmap`.
This change does not qualify Torch compilation or GPU execution.

These derivatives follow the **finite iterations performed by the supplied
solver**. They are not a converged-root derivative rule. A production
Newton--Krylov solver, its inner/outer tolerances, preconditioner, globalization
and linear-solve diagnostics remain separate work using the checked linear
operator interfaces. The stage boundary adds no operator registry, automatic
backend selection or implicit differentiation rule. Keep a custom solver's
configuration and derivative semantics fixed when replaying a recorded mesh;
custom configuration is caller-owned rather than automatically fingerprinted.

## Integration diagnostics and replay

`CoreSolver.run` and `replay_adaptive_schedule` return
`NonlinearIntegrationDiagnostics` for SDIRK2. All fields are native arrays.
`step_converged`, `step_accepted`, and `step_sizes` describe attempted steps;
rejected adaptive attempts remain present. `stages_per_step` maps the flattened
stage fields back to each attempt. The stage fields include convergence flags,
iteration/evaluation counts, and initial, final, and update RMS norms.

Live adaptive integration may therefore finish successfully even though a
rejected attempt has `step_converged=False`; the aggregate `converged` flag
requires every accepted step to converge. `last_nonlinear_diagnostics` retains
the most recent record, including after a fixed failure or exhausted rejection
budget.

Compiled replay cannot change its frozen mesh. Return its diagnostics from the
transformed function, then call `diagnostics.require_converged()` outside the
trace. A `NonlinearIntegrationConvergenceError` means the replayed state and
derivatives must be discarded. A valid replay remains differentiable through
the fixed Newton iterations with JAX.
