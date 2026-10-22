# Linear actions and external solves

`ActionOperator` describes a fixed real linear map using vector application,
shape, native dtype and an explicit Array-API namespace. A matrix RHS means
independent columns. Applications retain shape, float32/float64 precision and
namespace; concrete device metadata is checked when available. JAX tracers do
not always expose a device, so placement under transformations remains the
provider's responsibility. The wrapper performs no host/device transfer.

The optional `.T` capability is the real **Euclidean transpose**. A missing
transpose raises; it is never constructed by applying the forward callback to
basis vectors. The provider must supply a linear callback and a transpose of
that same fixed map. If a Jacobian action captures a state, time or parameters,
hold those fixed during a solve. Native differentiation follows the callback's
actual operations; the wrapper supplies no custom derivative rule.

## Eager checked solve contract

The existing `SparseAdapter` registry now also accepts an explicit
`linear_solver`. This route is eager NumPy float32/float64 and **uncached** by
the engine. It preserves the stage equation `L @ y = R @ x`. The callback
receives `LinearSolveRequest(L, R @ x, options, preconditioner, context)` and
returns a `LinearSolveCandidate`. Before adopting a result, the engine checks
shape and dtype and independently evaluates the original residual, per column:

\[
\|b-Ly\|_2 \le \max(\mathrm{atol},\mathrm{rtol}\|b\|_2).
\]

Nonfinite solutions/residuals and a false callback success status fail.
`solve_linear` returns evidence for successful and failed columns;
`require_linear_converged` raises `LinearSolveError` with the full result when
any column fails. Implicit stages apply that requirement automatically. A
callback must not mutate its RHS, operator or preconditioner during a solve.
Its reported iteration count must fit `max_iterations`; each concrete adapter
must document its iteration unit and stopping rule.

The following diagonal example exercises an actual action and an inverse
preconditioner without assembling a matrix. The callback knows the inverse
is exact for this example. A general solver must document how and on which side
it uses an approximate inverse, or reject an unsupported preconditioner.

```python
>>> import numpy as np
>>> from op_engine import (
...     ActionOperator, LinearSolveCandidate, LinearSolveRequest,
...     SparseAdapter, array_namespace, implicit_solve,
...     register_sparse_adapter, unregister_sparse_adapter,
... )
>>> diagonal = np.asarray([2.0, 3.0])
>>> xp = array_namespace(diagonal)
>>> left = ActionOperator((2, 2), diagonal.dtype, xp, lambda v: diagonal * v)
>>> inverse = ActionOperator((2, 2), diagonal.dtype, xp, lambda v: v / diagonal)
>>> def solve(request: LinearSolveRequest) -> LinearSolveCandidate:
...     return LinearSolveCandidate(
...         request.preconditioner @ request.rhs, success=True, iterations=1
...     )
>>> adapter = SparseAdapter(
...     "example-diagonal", lambda op: op is left, linear_solver=solve,
...     preconditioner=inverse, context=("mesh-v1", "coeff-v1", "stage-h=0.1"),
...     accepts_right=lambda op: isinstance(op, np.ndarray),
... )
>>> register_sparse_adapter(adapter)
>>> try:
...     result = implicit_solve(left, np.eye(2), np.asarray([4.0, 9.0]))
... finally:
...     unregister_sparse_adapter("example-diagonal")
>>> result.tolist()
[2.0, 3.0]
```

Use narrow, pure matching predicates. Registration refuses duplicate names
unless `replace=True`; replacement/removal clears cached factors for that
ecosystem. Multiple adapters matching the same pair are an error. Mixed pairs
(such as an action `L` with dense identity `R`) require an explicit
`accepts_right` predicate. Registered operators with an unsupported pair refuse
before dense conversion. Registry changes are setup/teardown operations, not
per-stage operations in concurrent solves.

## Ownership and stage support

Preparation, mesh/coefficient/stage identity and preconditioner lifecycle belong
to the provider. `context` is caller-owned metadata passed unchanged. A callback
with its own cache must include mesh, coefficient, stage-size/scale and
preconditioner generations in its key, and invalidate when any behavior changes.
The new callback route deliberately creates no implicit factorization cache.
Options, preconditioner and context supplied on `SparseAdapter` are adapter-wide
bindings. `CoreSolver` does not automatically replace them with stage context.
For changing stage data, a provider must derive the current identity from its
operator or otherwise manage its own context safely; do not update a shared
adapter per stage in concurrent solves. Direct `solve_linear` requests can carry
per-call options, preconditioner and context.
Existing SciPy/CuPy factorization adapters retain their historical object-identity
cache: keep the operators immutable while cached, and call
`clear_implicit_solver_cache()` after changing their entries.

Registered eager operators can flow through the existing `(L, R)` stage factory
and operator-axis layout. Implicit Euler, Heun--trapezoidal, TR-BDF2 and ARK3
have action-versus-assembled stage tests, including non-default operator axes.
Factories remain responsible for returning the numerical method's correct
stage matrices/actions. This extension does not change mass-matrix or explicit
load units; see the [assembly boundary](../development/fem-assembly-boundary.md).
Jacobian-driven methods that construct matrices still require an assembled
Jacobian and explicitly refuse an action there. Nonlinear stage JVPs and
Newton--Krylov integration are a separate extension.

The optional compiler's `state_actions(...).jvp` and `.vjp` can supply action
callbacks at fixed `(t, y, parameters)`. Use the compiler's exact state/output
layout; for a vector RHS these are state-Jacobian and transpose actions, while
an observable can have a different output shape. Unsupported operator, history
or gauge contributions must remain explicit. A compiler RHS action does not
automatically include separate engine operator terms, observation parameter
derivatives or posterior curvature. No compiler dependency is added here.

## Qualification boundary and next adapters

CPU tests cover float32/float64 native actions, rectangular forward/transpose
references, JAX JIT/gradient/vmap and eager Torch gradient/vmap. They do not
qualify native implicit solves or their differentiation. The eager solver
callback is deliberately separate from native action transforms. A provider
using SciPy/PETSc must supply and qualify any differentiation rule it needs.
Torch compilation and GPU execution are not claimed by this contract.

[Eager banded and matrix-free GMRES adapters](scipy-operators.md) now implement
this callback contract with independent references. On nonuniform grids,
diffusion can be self-adjoint in a weighted
inner product without being Euclidean symmetric. Do not infer plain CG
suitability; it requires justified positive definiteness/symmetry or an
explicit similarity transform. A general solver such as GMRES has separate
preconditioning and iteration semantics. The declared SciPy floor also needs
care: [SciPy 1.11 GMRES](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.sparse.linalg.gmres.html)
uses `tol`, whereas [current GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)
uses `rtol`. Adapter selection remains explicit and the dependency floor is
not silently changed.

Nonlocal callbacks may use tiled reductions to bound memory. That does not
make all-pairs arithmetic subquadratic; FFT/approximation paths need their own
mathematical assumptions, accuracy checks and measurements. Matrix RHS
application currently uses a Python column loop; no throughput or memory
improvement is claimed solely from providing this contract.
