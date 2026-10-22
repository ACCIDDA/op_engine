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

## Eager NumPy Newton–Krylov

`NewtonKrylovSolver` is an optional real NumPy float32/float64 implementation.
It consumes the full residual JVP through the existing `ActionOperator`,
`LinearSolveRequest` and `solve_linear` contracts, with `ScipyGmresSolver` as
its default inner solver. It creates no full Jacobian, identity or basis-probed
matrix. Dense Newton remains the default. Krylov storage scales with state size
times restart length; supplying a preconditioner may require additional storage.

```python
import numpy as np
from op_engine import (
    NewtonKrylovConfig, NewtonKrylovSolver, NonlinearProblem, require_converged,
)

problem = NonlinearProblem(
    residual=lambda x: x * x - 2.0,
    jvp=lambda x, v: 2.0 * x * v,
)
result = NewtonKrylovSolver(
    NewtonKrylovConfig(max_iterations=20, rtol=0.0, atol=1e-12)
).solve(problem, np.array([0.01]))
require_converged(result)
```

The outer convergence test uses residual RMS and the initial residual scale:
`rms <= atol + rtol * initial_rms`. `max_iterations` limits **applied** updates;
an initially converged guess performs none. `linear_options` independently
controls the inner original-residual L2 test,
`||rhs - A*delta|| <= max(atol, rtol * ||rhs||)`, and its iteration budget.
The default GMRES budget counts inner iterations, with a configurable restart
on `ScipyGmresSolver`. Outer and inner tolerances have different norms and
scales; a passing residual gate alone does not certify solution accuracy.

Every checked Newton direction starts with the configured `damping`. Bounded
backtracking multiplies it by `backtrack_factor` until a finite trial reaches
the outer gate or reduces RMS by the Armijo fraction `armijo * step_length`.
`max_backtracks` counts reductions after the first trial. Exhaustion preserves
the last accepted iterate and reports nonconvergence. Callback exceptions and
invalid shapes/dtypes/namespaces propagate as fatal errors.
Validated residuals are copied before retention so callbacks may reuse output
buffers across trials. The SciPy adapter also owns action outputs before GMRES
orthogonalization; an action or inverse may return its input or a read-only view.
Matrix actions snapshot each column before another callback can reuse its
buffer, using native operations that retain the automatic differentiation graph.

`NewtonKrylovTermination` reports convergence (1), outer budget exhaustion (2),
failed checked inner solve (3), exhausted line search (4), or a nonfinite
residual (5). A failed inner candidate is never applied, even if the callback
reports a small residual; the original residual and solver success must both
pass. Existing fixed/adaptive/replay policies then apply. Solve diagnostics
retain each inner result, JVP count and step-length reduction count. Integration
diagnostics expose array-only `work`: per-stage termination/action counts,
`solves_per_stage`, `columns_per_solve`, and flattened inner convergence,
iterations, L2 residuals and thresholds. Rejected attempts remain visible.
Diagnostic norms use the integration dtype; inner acceptance is checked before
that conversion. Zero termination means unspecified for other solvers.

A `preconditioner_factory(request)` may return an inverse `ActionOperator` or
`None`. GMRES applies this inverse on the left. Each call receives a fresh
`NewtonLinearization` context with an owned read-only iterate and the original
problem metadata. SDIRK supplies `NonlinearStageContext`: stage time, full step
size, diagonal step scale and stage index. The factory runs once per current
linearization, including new stages and retries. The engine keeps no preparation
cache; callers that retain one must invalidate it when the context, coefficients,
mesh, state, scale or inverse version changes. Solver diagnostics keep no iterate
history.

Use `NonlinearMethodConfig(rhs_jvp=full_rhs_jvp,
solver=NewtonKrylovSolver(...))` for SDIRK. Schedule provenance records outer,
inner and line-search controls, the linear-solver class/restart and inverse
presence. A custom callback's hidden state and coefficient versions must be
represented by the caller's `RunConfig.replay_model_id`; discover a fresh mesh
when that contract changes.

This solver makes data-dependent host decisions and rejects native JAX/Torch
arrays before coercion. It supplies no JIT, vmap, automatic derivative or
converged-root implicit derivative. The portable dense solver retains its
separately qualified finite-iteration transformations; converged-root
differentiation remains a separate implementation task.

## Synthetic CPU measurements

The [raw worker evidence](../assets/newton-krylov-cpu/workers.jsonl) contains
54 isolated, single-threaded workers: three workers for each size, dtype and
solver case, with seven timed solves after warmup. Every timed, warmup and
separate memory run passed the same independently declared residual and
manufactured-root pointwise gates. Numerical and benchmark source hashes,
dependencies, thread settings, CPU model/affinity, setup time, individual solve
times, updates and process peak RSS are retained.

These are coupled synthetic nonlinear roots at a fixed spectral scale, rather
than a domain-refinement or application study. The dense implementation performs
its configured eight updates; the eager implementation stops at its outer gate.
The comparison includes both that policy difference and the derivative/linear
solve implementation. Times below are medians of the three worker medians;
allocation peaks come from a separate `tracemalloc` solve and exclude some native
allocator/BLAS work. Process RSS includes Python and dependency imports.

| dtype | state size | dense ms | GMRES ms | inverse-preconditioned GMRES ms | dense traced peak KiB | preconditioned traced peak KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 32 | 0.519 | 3.592 | 1.194 | 23.9 | 14.8 |
| float32 | 128 | 0.956 | 5.987 | 1.302 | 328.8 | 26.8 |
| float32 | 512 | 33.184 | 6.647 | 1.500 | 5148.3 | 78.1 |
| float64 | 32 | 0.272 | 13.173 | 2.859 | 24.0 | 21.5 |
| float64 | 128 | 0.913 | 23.526 | 4.521 | 268.5 | 45.7 |
| float64 | 512 | 42.075 | 40.784 | 9.847 | 4132.6 | 150.8 |

The small cases favor dense solves. At size 512, the inverse-preconditioned
path takes about 22.1× less time for float32 and 4.3× less for float64, with about
66× and 27× lower traced allocation peaks on this machine. These bounds support
an optional choice for qualified workloads; they do not select defaults for
other geometries, trajectories or hardware. The
[pre-fix candidate workers](../assets/newton-krylov-cpu/workers-a01366f.jsonl)
are retained with their original source hashes; this table uses refreshed runs
after the callback-buffer ownership corrections.

Reproduce on Linux with a suitable available CPU in place of `0`:

```bash
taskset -c 0 uv run --no-sync python -m scripts.benchmark_newton_krylov \
  --sizes 32 128 512 --repeats 7 --workers-per-case 3 --require-success \
  --output /tmp/newton-krylov-workers.jsonl
```

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
array namespace or backend module. A JVP-only configuration can use the eager
NumPy `NewtonKrylovSolver` or another implementation consuming
`NonlinearProblem.jvp`; the default dense solver still
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

These native derivatives follow the **finite iterations performed by the supplied
test solver**. The eager NumPy Newton–Krylov implementation described above has
its own value, failure and finite-difference sensitivity qualification; it
supplies no native transformation rule. The stage boundary adds no operator registry, automatic
backend selection or implicit differentiation rule. Keep a custom solver's
configuration and derivative semantics fixed when replaying a recorded mesh;
custom configuration is caller-owned rather than automatically fingerprinted.

## Integration diagnostics and replay

`CoreSolver.run` and `replay_adaptive_schedule` return
`NonlinearIntegrationDiagnostics` for SDIRK2. All arrays, including nested
`work` evidence, retain the native namespace.
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
