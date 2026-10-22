# Eager banded and matrix-free SciPy solves

`scipy_banded_adapter()` and `scipy_gmres_adapter()` build opt-in adapters for
the existing registry. Both accept NumPy float32/float64 arrays and preserve
`L @ y = R @ x`, including matrix RHS columns and the core operator-axis layout.
They create no engine preparation, factorization, preconditioner or warm-start
cache. Every solve uses the [original-residual contract](linear-operators.md).
No native-device transfer, automatic differentiation or default selection is
provided by these eager adapters.

## Banded direct solve

`BandedOperator(bands, lower, upper)` copies SciPy diagonal-ordered storage:
`bands[upper + i - j, j]` is entry `(i, j)`. Valid entries must be finite; unused
padding is canonicalized to zero. The retained array is read-only. Construct a
new operator when geometry, coefficients or stage data change. Forward and
Euclidean-transpose application visit valid bands without assembling a dense
matrix. Storage is `O(n * (lower + upper + 1))`; a wide band can still require
quadratic storage. A transpose is not a weighted adjoint or a symmetry claim.

The adapter uses SciPy's banded direct solve and rejects a supplied
preconditioner. A zero-bandwidth diagonal uses direct division, including `n=1`
where older SciPy banded routines have a singleton indexing defect. Singular
systems return failure evidence and the stage refuses their candidate.
Direct solves report zero iterative updates. They remain subject to the
requested original-residual tolerance, including finite-precision roundoff.

## General matrix-free GMRES

`ScipyGmresSolver(restart=20)` is also usable directly with `solve_linear`.
Its input operator and optional approximate-inverse preconditioner remain
application callbacks. The optional preconditioner acts on the **left**;
SciPy minimizes a preconditioned residual while the engine independently checks
the original one. No Euclidean symmetry or positive definiteness is inferred.
This is appropriate for general nonsymmetric linear maps, including the tested
nonuniform finite-volume diffusion example, whose self-adjointness is weighted.

`max_iterations` counts **inner iterations per RHS column**. Each column starts
at zero and is solved separately. The reported `iterations` is the maximum
across columns; total work is their sum. The diagnostic message retains all
counts and SciPy status codes. Any unsuccessful column makes the overall
candidate unsuccessful; per-column original residual norms remain available.
`restart` bounds the Krylov cycle length; workspace scales with
`n * min(restart, n)` in addition to vectors. A matrix-free representation does
not eliminate this workspace or imply an arithmetic speedup.

The solver specifies `callback_type="legacy"` to keep that inner-iteration
budget consistent across supported SciPy APIs, and supplies explicit `atol`.
It bridges old `tol` and current `rtol` signatures. Required CI checks the oldest
NumPy 2-compatible SciPy series using SciPy 1.13.1, NumPy 2.0.2 and array-api-compat
1.12, plus current dependencies. SciPy introduced NumPy 2 support in its
[1.13 release](https://docs.scipy.org/doc/scipy-1.13.1/release/1.13.0-notes.html).
The older declared SciPy lower bound does not produce a compatible NumPy 2
environment by itself; dependency metadata still governs resolution. A
tol-only compatibility test forwards into the real current solver; it is not
a claim that the full engine is qualified on an incompatible dependency pair.

## Explicit registration example

Both adapters can coexist because their left predicates recognize different
operator types. Mixed right operands support NumPy arrays, SciPy CSR matrices,
banded operators and `ActionOperator`. Register during setup and unregister
during teardown; do not mutate registry bindings per stage in concurrent runs.

```python
import numpy as np
from op_engine import (
    ActionOperator, BandedOperator, LinearSolveOptions, array_namespace,
    implicit_solve, register_sparse_adapter, unregister_sparse_adapter,
    scipy_banded_adapter, scipy_gmres_adapter,
)

bands = np.asarray([[0.0, -1.0, -1.0], [3.0, 3.0, 3.0], [-0.5, -0.5, 0.0]])
left = BandedOperator(bands, lower=1, upper=1)
rhs = np.asarray([2.0, 1.5, 2.5])
xp = array_namespace(rhs)
action = ActionOperator(left.shape, left.dtype, xp, left.__matmul__, left.T.__matmul__)
inverse = ActionOperator(left.shape, left.dtype, xp, lambda v: v / 3.0)
options = LinearSolveOptions(rtol=1e-10, atol=1e-12)
register_sparse_adapter(scipy_banded_adapter(options=options))
register_sparse_adapter(scipy_gmres_adapter(options=options, preconditioner=inverse))
try:
    banded_result = implicit_solve(left, np.eye(3), rhs)
    matrix_free_result = implicit_solve(action, np.eye(3), rhs)
finally:
    unregister_sparse_adapter("scipy-banded")
    unregister_sparse_adapter("scipy-gmres")
np.testing.assert_allclose(banded_result, np.ones(3))
np.testing.assert_allclose(matrix_free_result, np.ones(3))
```

## Lifecycle and derivatives

There is no reuse keyed only by an operator's shape, object identity or stage
label in these adapters. New band objects own their copied coefficients; action
callbacks own their fixed linearization. An externally supplied preconditioner
must match that map. The provider owns mesh/coefficient/stage/preconditioner
generation tracking if it introduces any cache. Adapter options/context and
preconditioner bindings remain adapter-wide, as described in the contract guide.

Compiler-native JAX/Torch state actions may supply native application and
transpose callbacks, but these eager NumPy solvers deliberately refuse their
native inputs. Do not add a hidden host conversion to make them match. Native
implicit-stage derivatives and nonlinear Newton--Krylov solves
retain separate qualification. Nonlocal tiled callbacks remain possible without
claiming subquadratic all-pairs arithmetic.

## CPU measurements

The reproducible worker runner is `python -m scripts.benchmark_linear_operators`.
It compares dense, cached SciPy CSR, uncached banded direct, and matrix-free
GMRES with/without Jacobi preconditioning. Fixtures use a nonuniform
reaction-diffusion operator at a fixed spectral scale, with a known synthetic
solution and an independently computed cell-flux RHS. They are a solve/storage
experiment, not spatial-refinement or application evidence.

The [initial workers](../assets/eager-operators-cpu/initial.jsonl) retain all
180 attempts: 156 successes and 24 pointwise-accuracy failures. Those failed
GMRES candidates had passed the adapter's normwise residual gate. Thus residual
success alone did not establish the separately requested solution accuracy.
The initial runner and defaults are retained in commit
[`2d2094a`](https://github.com/ACCIDDA/op_engine/commit/2d2094a0132546d26624fd9487b3804892e37a56).
Those initial failures retain their worker stderr; they do not contain complete
worker-level timing/RSS diagnostics.

The [tolerance check](../assets/eager-operators-cpu/tolerance-check.jsonl) passed
180/180 with the same pointwise gates and tighter solve tolerances. The separate
[final timing run](../assets/eager-operators-cpu/refined.jsonl) repeated all 180
cases on CPU 0 after the other local qualification jobs finished;
all passed again. This shared workstation is an Intel Core Ultra 7 265U running
Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1. CPU affinity and source hashes were
checked for every final worker. Its command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0 python -m scripts.benchmark_linear_operators \
  --sizes 64 512 2048 --rhs 1 4 --dtypes float32 float64 \
  --workers-per-case 3 --repeats 7 --timeout 120 \
  --rtol-float32 1e-6 --rtol-float64 1e-12 \
  --require-success --output /tmp/eager-operators-refined.jsonl
```

Refined worker rows record CPU affinity/model, versions, thread settings, source
hashes, fixture/representation preparation, first solve, seven warm solves and
whole-worker peak RSS. Preparation includes synthetic fixture creation. CSR
warm timings reuse the existing LU cache; the new adapters are uncached and
include original-residual verification. Logical representation bytes exclude
cached factors, Krylov workspace, other working arrays and allocator overhead;
peak RSS includes the whole process. First-solve time is not process startup
time. These distinctions must accompany comparisons and any default decision.

The following values are medians across three fresh workers for `n=2048`,
one RHS, float64 (each warm value is itself the median of seven solves).
Full size/RHS/precision results and worker ranges are in the
[machine-readable summary](../assets/eager-operators-cpu/summary.json).

| Path | First solve, ms | Warm solve, ms | Whole-worker peak RSS, MiB | Logical operator buffers |
| --- | ---: | ---: | ---: | ---: |
| Dense | 198.38 | 191.62 | 140.24 | 64 MiB |
| Cached CSR | 0.931 | 0.0422 | 62.73 | 112 KiB |
| Banded direct | 0.464 | 0.166 | 61.81 | 48 KiB |
| Matrix-free GMRES | 4.368 | 4.030 | 61.90 | 48 KiB |
| GMRES + Jacobi | 2.267 | 1.776 | 61.92 | 64 KiB |

Cached CSR is the fastest warm path for this fixed tridiagonal fixture.
Banded direct avoids dense storage when narrow bands are available, and its
first solve is cheaper here than CSR factorization. GMRES supports an action-only
map but is slower than either structured direct path in this experiment.
The roughly one-MiB differences among the structured/action paths are not a
reliable memory ranking. Dense logical bytes include both `L` and dense identity
`R`; CSR includes both sparse operators, while the new paths use an identity
action for `R`. Factor/workspace and whole-process RSS must remain separate.
These scoped results do not establish an application speedup or justify changing
defaults. Choose a solver and tolerance against independently checked solution
or measured-output accuracy; a residual gate alone is insufficient.
