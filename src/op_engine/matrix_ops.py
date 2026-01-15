"""Matrix operations and linear solvers for multiphysics modeling.

This module provides small, performance-oriented numerical utilities used by
multiphysics engines:

- Construction of common 1D linear operators (e.g., Laplacian, Crank-Nicolson).
- Cached implicit solves for repeated linear systems with fixed operators.
- High-throughput aggregation utilities for large numbers of subpopulations.
- Optional Kronecker composition utilities for separable multi-axis operators.

Design notes:
    * CPU-first: dense paths rely on NumPy/SciPy BLAS/LAPACK; sparse paths rely
      on SciPy sparse factorizations.
    * Backend-friendly surface: public APIs operate on plain ndarrays or CSR
      matrices and avoid leaking SciPy-specific solver objects.
    * Cache semantics: implicit solver caching is keyed by (id(left_op),
      id(right_op)). For caching to be effective, operator objects must be
      constructed once and reused.

Future GPU backends:
    This module intentionally keeps a narrow public surface (operator builders,
    implicit_solve, and aggregation primitives). A future JAX/CuPy backend can
    mirror these functions and signatures while changing internal storage and
    solve strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, csr_matrix, diags, identity, issparse, kron
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import factorized as sparse_factorized

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike, NDArray


# =============================================================================
# Public operator types (backend-friendly)
# =============================================================================

DenseOperator: TypeAlias = "NDArray[np.floating]"
SparseOperator: TypeAlias = "csr_matrix"
Operator: TypeAlias = "DenseOperator | SparseOperator"


# =============================================================================
# Small data containers
# =============================================================================


@dataclass(frozen=True, slots=True)
class GridGeometry:
    """Geometry of a 1D spatial grid.

    Attributes:
        n: Number of grid points.
        dx: Grid spacing.
    """

    n: int
    dx: float


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """Configuration for diffusion-like linear operators.

    Attributes:
        coeff: Physical diffusion coefficient D (units length^2 / time).
        dtype: Floating dtype (e.g. np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".
    """

    coeff: float
    dtype: DTypeLike = np.float64
    bc: str = "neumann"


# === Internal autodispatch threshold (tuned empirically) ===
# Below this size, dense ops tend to be faster; above it, sparse is preferred.
_DISPATCH_THRESHOLD = 350

# =============================================================================
# Implicit solver cache
# =============================================================================

# Cache for implicit solvers (factorized L, prepped R)
# key = (id(L), id(R)) -> (meta, solver)
# meta is used to guard against unsafe id reuse in long-lived processes.
_SolverMeta = tuple[tuple[int, int], tuple[int, int], str, str, bool]
_IMPLICIT_SOLVER_CACHE: dict[
    tuple[int, int],
    tuple[_SolverMeta, Callable[[NDArray[np.floating]], NDArray[np.floating]]],
] = {}


# =============================================================================
# Error message constants
# =============================================================================

_UNKNOWN_BC_ERROR = "Unknown bc: {bc}"
_VALUES_2D_ERROR = "values must be 2D (N, K)"
_GROUP_IDS_LENGTH_ERROR = "group_ids and values must have the same length along axis 0"
_OPERATORS_SQUARE_ERROR = "Operator must be square; got shape {shape}"
_OPERATORS_DIM_ERROR = "Operator shape {shape} is incompatible with x shape {x_shape}"
_X_NDIM_ERROR = "x must be 1D or 2D; got ndim={ndim}"
_KRON_EMPTY_ERROR = "ops must contain at least one operator"
_KRON_INCOMPATIBLE_ERROR = "All operators must be square; got shapes: {shapes}"


# =============================================================================
# Core linear operators: Laplacian + Crank-Nicolson + Predictor-Corrector
# =============================================================================


def build_laplacian_tridiag(
    n: int,
    dx: float,
    coeff: float,
    dtype: DTypeLike = np.float64,
    bc: str = "neumann",
) -> csr_matrix:
    """Build a Laplacian tridiagonal matrix for a given boundary condition.

    The resulting operator corresponds to `coeff * Δ_h`, where `Δ_h` is the
    standard second-order central-difference Laplacian in 1D. No time-step
    scaling is applied here; `coeff` is interpreted as the physical diffusion
    coefficient `D` or a generic spatial scaling.

    For interior points, the stencil is:
        [1, -2, 1] / dx^2

    Boundary behavior depends on the `bc` argument.

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Physical diffusion coefficient D or a generic scaling factor.
        dtype: Floating dtype of the resulting matrix elements.
        bc: Boundary condition; either "neumann" or "absorbing".

    Returns:
        Scaled Laplacian matrix of shape (n, n) as a CSR sparse matrix.

    Raises:
        ValueError: If bc is not "neumann" or "absorbing".
    """
    dtype_obj = np.dtype(dtype)
    factor = coeff / dx**2

    main_diag = -2.0 * np.ones(n, dtype=dtype_obj)
    off_diag = np.ones(n - 1, dtype=dtype_obj)

    if bc == "neumann":
        main_diag[0] = -1.0
        main_diag[-1] = -1.0
    elif bc == "absorbing":
        main_diag[0] = -2.0
        main_diag[-1] = -2.0
    else:
        msg = _UNKNOWN_BC_ERROR.format(bc=bc)
        raise ValueError(msg)

    laplacian = diags(
        [off_diag.tolist(), main_diag.tolist(), off_diag.tolist()],
        [-1, 0, 1],
        shape=(n, n),
        dtype=dtype_obj,
    )

    scaled = laplacian * factor
    return cast("csr_matrix", scaled.tocsr())


def _build_crank_nicolson_sparse(
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[csr_matrix, csr_matrix]:
    """Build sparse Crank-Nicolson operator matrices (L, R).

    Args:
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration with coefficient D, dtype, and bc.
        dt: Time step size.

    Returns:
        A tuple (left_op, right_op) of CSR sparse matrices.
    """
    n = geom.n
    dx = geom.dx
    coeff = cfg.coeff
    dtype_obj = np.dtype(cfg.dtype)
    bc = cfg.bc

    laplacian = build_laplacian_tridiag(
        n=n,
        dx=dx,
        coeff=coeff,
        dtype=dtype_obj,
        bc=bc,
    )

    time_scaled_op = laplacian * dt
    identity_mat = identity(n, dtype=dtype_obj, format="csr")
    left_op = identity_mat - 0.5 * time_scaled_op
    right_op = identity_mat + 0.5 * time_scaled_op

    if bc == "absorbing":
        for mat in (left_op, right_op):
            mat[0, :] = 0.0
            mat[0, 0] = 1.0
            mat[-1, :] = 0.0
            mat[-1, -1] = 1.0

    return cast("csr_matrix", left_op.tocsr()), cast("csr_matrix", right_op.tocsr())


def _build_crank_nicolson_dense(
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[DenseOperator, DenseOperator]:
    """Build dense Crank-Nicolson operator matrices (L, R).

    Args:
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration with coefficient D, dtype, and bc.
        dt: Time step size.

    Returns:
        A tuple (left_op, right_op) of dense arrays.
    """
    n = geom.n
    dx = geom.dx
    coeff = cfg.coeff
    dtype_obj = np.dtype(cfg.dtype)
    bc = cfg.bc

    laplacian = build_laplacian_tridiag(
        n=n,
        dx=dx,
        coeff=coeff,
        dtype=dtype_obj,
        bc=bc,
    ).toarray()

    time_scaled_op = dt * laplacian
    identity_mat = np.eye(n, dtype=dtype_obj)
    left_op = identity_mat - 0.5 * time_scaled_op
    right_op = identity_mat + 0.5 * time_scaled_op

    if bc == "absorbing":
        for mat in (left_op, right_op):
            mat[0, :] = 0.0
            mat[0, 0] = 1.0
            mat[-1, :] = 0.0
            mat[-1, -1] = 1.0

    return cast("DenseOperator", left_op), cast("DenseOperator", right_op)


def build_crank_nicolson_operator(
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[Operator, Operator]:
    """Build Crank-Nicolson operators with dense/sparse autodispatch.

    Small systems are built as dense operators; large systems as sparse. The
    decision is based on the grid size `geom.n` and `_DISPATCH_THRESHOLD`.

    Args:
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration with coefficient D, dtype, and bc.
        dt: Time step size.

    Returns:
        A tuple (left_op, right_op) of operator matrices (dense or CSR).
    """
    if geom.n < _DISPATCH_THRESHOLD:
        return _build_crank_nicolson_dense(geom, cfg, dt)
    return _build_crank_nicolson_sparse(geom, cfg, dt)


def _build_predictor_corrector_dense(
    base_matrix: DenseOperator,
) -> tuple[DenseOperator, DenseOperator, DenseOperator]:
    """Build predictor-corrector matrices for a dense base matrix.

    Args:
        base_matrix: Time-scaled linear operator of shape (n, n), typically
            base_matrix = dt * A.

    Returns:
        (predictor, left_op, right_op) dense matrices.
    """
    n = base_matrix.shape[0]
    identity_mat = np.eye(n, dtype=base_matrix.dtype)
    predictor = identity_mat
    left_op = identity_mat - 0.5 * base_matrix
    right_op = identity_mat + 0.5 * base_matrix
    return (
        cast("DenseOperator", predictor),
        cast("DenseOperator", left_op),
        cast("DenseOperator", right_op),
    )


def _build_predictor_corrector_sparse(
    base_matrix: csr_matrix,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build predictor-corrector matrices for a sparse base matrix.

    Args:
        base_matrix: Time-scaled linear operator as CSR of shape (n, n).

    Returns:
        (predictor, left_op, right_op) CSR matrices.
    """
    n = base_matrix.shape[0]
    identity_mat = identity(n, format="csr", dtype=base_matrix.dtype)
    predictor = identity_mat
    left_op = identity_mat - 0.5 * base_matrix
    right_op = identity_mat + 0.5 * base_matrix
    return (
        cast("csr_matrix", predictor.tocsr()),
        cast("csr_matrix", left_op.tocsr()),
        cast("csr_matrix", right_op.tocsr()),
    )


def build_predictor_corrector(
    base_matrix: DenseOperator | csr_matrix,
) -> tuple[Operator, Operator, Operator]:
    """Build predictor-corrector matrices with dense/sparse autodispatch.

    Args:
        base_matrix: Time-scaled linear operator, either dense or CSR.

    Returns:
        A tuple (predictor, left_op, right_op), dense or CSR depending on input
        and problem size.
    """
    n = base_matrix.shape[0]
    if issparse(base_matrix) and n >= _DISPATCH_THRESHOLD:
        return _build_predictor_corrector_sparse(cast("csr_matrix", base_matrix))

    if issparse(base_matrix):
        dense_base = np.asarray(base_matrix.toarray())
    else:
        dense_base = np.asarray(base_matrix)

    return _build_predictor_corrector_dense(cast("DenseOperator", dense_base))


# =============================================================================
# Kronecker composition utilities (optional, but public)
# =============================================================================


def kron_prod(a: Operator, b: Operator) -> Operator:
    """Compute the Kronecker product of two operators.

    Args:
        a: Left operator (dense or CSR).
        b: Right operator (dense or CSR).

    Returns:
        Kronecker product operator. Returns CSR if either input is sparse;
        otherwise returns a dense ndarray.

    Notes:
        This materializes the Kronecker product. For large multi-axis problems,
        explicit materialization can be memory-heavy; consider matrix-free
        approaches in a future backend if needed.
    """
    if issparse(a) or issparse(b):
        a_csr = a if issparse(a) else csr_matrix(np.asarray(a))
        b_csr = b if issparse(b) else csr_matrix(np.asarray(b))
        return cast(
            "csr_matrix",
            kron(
                cast("csr_matrix", a_csr),
                cast("csr_matrix", b_csr),
                format="csr",
            ),
        )
    return cast("DenseOperator", np.kron(np.asarray(a), np.asarray(b)))


def kron_sum(ops: list[Operator]) -> Operator:
    """Compute a Kronecker sum of square operators.

    For operators A_1, ..., A_m with sizes n_1, ..., n_m, the Kronecker sum is:

        A = A_1 ⊕ A_2 ⊕ ... ⊕ A_m
          = A_1 ⊗ I ⊗ ... ⊗ I
          + I ⊗ A_2 ⊗ ... ⊗ I
          + ...
          + I ⊗ ... ⊗ I ⊗ A_m

    This is a standard construction for separable multi-axis diffusion/coupling.

    Args:
        ops: List of square operators (dense or CSR).

    Returns:
        Kronecker sum operator (dense or CSR).

    Raises:
        ValueError: If ops is empty or contains non-square operators.
    """
    if not ops:
        raise ValueError(_KRON_EMPTY_ERROR)

    shapes = [
        tuple(np.asarray(op).shape)
        if not issparse(op)
        else cast("csr_matrix", op).shape
        for op in ops
    ]

    if any(s[0] != s[1] for s in shapes):
        raise ValueError(_KRON_INCOMPATIBLE_ERROR.format(shapes=shapes))

    # Determine if we should stay sparse.
    any_sparse = any(issparse(op) for op in ops)

    sizes = [s[0] for s in shapes]
    dtype_obj = np.result_type(
        *[
            (cast("csr_matrix", op).dtype if issparse(op) else np.asarray(op).dtype)
            for op in ops
        ]
    )

    def _eye(n: int) -> Operator:
        if any_sparse:
            return cast("csr_matrix", identity(n, format="csr", dtype=dtype_obj))
        return cast("DenseOperator", np.eye(n, dtype=dtype_obj))

    total: Operator | None = None
    n_ops = len(ops)

    for i, op_i in enumerate(ops):
        term: Operator = op_i
        # Left kron with identities
        for j in range(i - 1, -1, -1):
            term = kron_prod(_eye(sizes[j]), term)
        # Right kron with identities
        for j in range(i + 1, n_ops):
            term = kron_prod(term, _eye(sizes[j]))
        total = term if total is None else cast("Operator", total + term)

    if total is None:
        # Unreachable, but keeps type-checkers happy.
        raise ValueError(_KRON_EMPTY_ERROR)
    return total


# =============================================================================
# Implicit solvers: factorized (cached) & backend-neutral wrappers
# =============================================================================


def clear_implicit_solver_cache() -> None:
    """Clear the internal implicit solver cache.

    This can be useful in long-running processes or test suites to avoid
    unbounded cache growth.
    """
    _IMPLICIT_SOLVER_CACHE.clear()


def _operator_meta(
    left_op: Operator,
    right_op: Operator,
) -> _SolverMeta:
    """Compute a metadata tuple used to validate cache hits."""
    if issparse(left_op):
        l_shape = cast("csr_matrix", left_op).shape
        l_dtype = str(cast("csr_matrix", left_op).dtype)
        l_sparse = True
    else:
        l_arr = np.asarray(left_op)
        l_shape = l_arr.shape
        l_dtype = str(l_arr.dtype)
        l_sparse = False

    if issparse(right_op):
        r_shape = cast("csr_matrix", right_op).shape
        r_dtype = str(cast("csr_matrix", right_op).dtype)
        r_sparse = True
    else:
        r_arr = np.asarray(right_op)
        r_shape = r_arr.shape
        r_dtype = str(r_arr.dtype)
        r_sparse = False

    # is_sparse indicates whether we treat this solve path as sparse.
    is_sparse = l_sparse and r_sparse
    return (
        cast("tuple[int, int]", l_shape),
        cast("tuple[int, int]", r_shape),
        l_dtype,
        r_dtype,
        is_sparse,
    )


def _validate_square_operator(op: Operator) -> tuple[int, int]:
    """Validate that an operator is square and return its shape.

    Args:
        op: Dense or sparse operator.

    Returns:
        Operator shape.

    Raises:
        ValueError: If operator is not square.
    """
    shape = cast("tuple[int, int]", op.shape)  # both ndarray and csr_matrix have .shape
    if shape[0] != shape[1]:
        raise ValueError(_OPERATORS_SQUARE_ERROR.format(shape=shape))
    return shape


def _validate_solve_dimensions(
    left_op: Operator,
    right_op: Operator,
    x: NDArray[np.floating],
) -> None:
    """Validate operator and RHS shapes for implicit solve."""
    l_shape = _validate_square_operator(left_op)
    r_shape = _validate_square_operator(right_op)
    if l_shape != r_shape:
        raise ValueError(_OPERATORS_SQUARE_ERROR.format(shape=(l_shape, r_shape)))

    if x.ndim not in (1, 2):
        raise ValueError(_X_NDIM_ERROR.format(ndim=x.ndim))

    n = l_shape[0]
    if x.shape[0] != n:
        raise ValueError(_OPERATORS_DIM_ERROR.format(shape=l_shape, x_shape=x.shape))


def _as_linear_operator(op: Operator) -> LinearOperator:
    """Convert an operator to a SciPy LinearOperator.

    This is currently used internally to keep a single conceptual pathway for
    "operator application" if/when iterative or matrix-free solves are added.

    Args:
        op: Dense or CSR operator.

    Returns:
        A SciPy LinearOperator wrapping op.
    """
    if issparse(op):
        op_csr = cast("csr_matrix", op)

        def sparse_matvec(v: NDArray[np.floating]) -> NDArray[np.floating]:
            return cast("NDArray[np.floating]", op_csr @ v)

        def sparse_matmat(m: NDArray[np.floating]) -> NDArray[np.floating]:
            return cast("NDArray[np.floating]", op_csr @ m)

        return LinearOperator(
            shape=op_csr.shape,
            dtype=op_csr.dtype,
            matvec=sparse_matvec,
            matmat=sparse_matmat,
        )

    op_arr = np.asarray(op)

    def dense_matvec(v: NDArray[np.floating]) -> NDArray[np.floating]:
        return cast("NDArray[np.floating]", op_arr @ v)

    def dense_matmat(m: NDArray[np.floating]) -> NDArray[np.floating]:
        return cast("NDArray[np.floating]", op_arr @ m)

    return LinearOperator(
        shape=op_arr.shape,
        dtype=op_arr.dtype,
        matvec=dense_matvec,
        matmat=dense_matmat,
    )


def _build_implicit_solver(
    left_op: Operator,
    right_op: Operator,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Build a reusable implicit solver for left_op @ y = right_op @ x.

    The returned callable solves the linear system:

        left_op @ y = right_op @ x

    for y given x.

    Dense path:
        Uses LU factorization (SciPy) and supports batched RHS (x is 2D) to
        exploit BLAS/LAPACK.

    Sparse path:
        Uses sparse factorization via scipy.sparse.linalg.factorized. For
        robustness, the solver supports both 1D and 2D x; the 2D case is
        solved column-wise.

    Args:
        left_op: Left-hand operator matrix L (dense or CSR).
        right_op: Right-hand operator matrix R (dense or CSR).

    Returns:
        A callable solver(x) that returns y with the same shape as x.
    """
    # Only treat as sparse if both operators are sparse.
    is_sparse = issparse(left_op) and issparse(right_op)

    if is_sparse:
        left_csr = cast("csr_matrix", left_op)
        right_csr = cast("csr_matrix", right_op)
        solve_left = sparse_factorized(left_csr)

        def sparse_solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
            """Solve left_op @ y = right_op @ x for sparse operators.

            Args:
                x: Right-hand side vector (n,) or matrix (n, k).

            Returns:
                Solution y with the same shape as x.
            """
            x_arr = np.asarray(x, dtype=left_csr.dtype)
            rhs = right_csr @ x_arr

            if rhs.ndim == 1:
                out_1d = solve_left(rhs)
                return np.asarray(out_1d, dtype=x_arr.dtype)

            # Column-wise for robust behavior across SciPy versions.
            n, k = rhs.shape
            out = np.empty((n, k), dtype=x_arr.dtype)
            for j in range(k):
                out[:, j] = np.asarray(solve_left(rhs[:, j]), dtype=x_arr.dtype)
            return out

        return sparse_solver

    left_dense = np.asarray(left_op)
    right_dense = np.asarray(right_op)
    lu, piv = lu_factor(left_dense)

    def dense_solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Solve left_op @ y = right_op @ x for dense operators.

        Args:
            x: Right-hand side vector (n,) or matrix (n, k).

        Returns:
            Solution y with the same shape as x.
        """
        x_arr = np.asarray(x, dtype=left_dense.dtype)
        rhs = right_dense @ x_arr
        out = lu_solve((lu, piv), rhs)
        return np.asarray(out, dtype=x_arr.dtype)

    return dense_solver


def implicit_solve(
    left_op: Operator,
    right_op: Operator,
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Perform an implicit solve with dense/sparse dispatch and caching.

    Solves:
        left_op @ y = right_op @ x

    This function supports x as either:
        * 1D array of shape (n,)
        * 2D array of shape (n, k) for batched solves (dense path exploits
          BLAS/LAPACK; sparse path solves column-wise).

    Args:
        left_op: Left-hand operator matrix L (dense or CSR).
        right_op: Right-hand operator matrix R (dense or CSR).
        x: Right-hand side vector or matrix.

    Returns:
        The solution array y with the same shape as x.

    Raises:
        ValueError: If operators are not square/compatible or x has invalid
            shape/dimension.
    """
    x_arr = np.asarray(x)
    _validate_solve_dimensions(left_op, right_op, cast("NDArray[np.floating]", x_arr))

    key = (id(left_op), id(right_op))
    meta = _operator_meta(left_op, right_op)

    cached = _IMPLICIT_SOLVER_CACHE.get(key)
    if cached is not None:
        cached_meta, solver = cached
        if cached_meta != meta:
            # Guard against unsafe id reuse or mutation.
            solver = _build_implicit_solver(left_op, right_op)
            _IMPLICIT_SOLVER_CACHE[key] = (meta, solver)
        return solver(cast("NDArray[np.floating]", x_arr))

    solver = _build_implicit_solver(left_op, right_op)
    _IMPLICIT_SOLVER_CACHE[key] = (meta, solver)
    return solver(cast("NDArray[np.floating]", x_arr))


# =============================================================================
# Grouped operations: sum, count, masked sum
# =============================================================================


def _matrix_grouped_sum_sparse(
    group_matrix: csr_matrix,
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped sums using a sparse membership matrix."""
    return np.asarray(group_matrix @ values)


def _matrix_grouped_sum_dense(
    group_matrix: DenseOperator,
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped sums using a dense membership matrix."""
    return np.asarray(group_matrix @ values)


def matrix_grouped_sum(
    group_matrix: csr_matrix | DenseOperator,
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped sums with dense/sparse autodispatch.

    Args:
        group_matrix: Group membership matrix of shape (G, N), dense or CSR.
        values: Values to sum, shape (N,) or (N, K).

    Returns:
        Grouped sums of shape (G,) or (G, K).
    """
    n_groups = group_matrix.shape[0]
    if issparse(group_matrix) and n_groups >= _DISPATCH_THRESHOLD:
        return _matrix_grouped_sum_sparse(cast("csr_matrix", group_matrix), values)
    if issparse(group_matrix):
        dense_matrix = np.asarray(cast("csr_matrix", group_matrix).toarray())
    else:
        dense_matrix = np.asarray(group_matrix)
    return _matrix_grouped_sum_dense(cast("DenseOperator", dense_matrix), values)


def _matrix_grouped_count_sparse(group_matrix: csr_matrix) -> NDArray[np.floating]:
    """Compute grouped counts using a sparse membership matrix."""
    counts = np.asarray(group_matrix.sum(axis=1)).ravel()
    return counts.astype(float)


def _matrix_grouped_count_dense(group_matrix: DenseOperator) -> NDArray[np.floating]:
    """Compute grouped counts using a dense membership matrix."""
    counts = np.asarray(group_matrix.sum(axis=1))
    return counts.astype(float)


def matrix_grouped_count(
    group_matrix: csr_matrix | DenseOperator,
) -> NDArray[np.floating]:
    """Compute grouped counts with dense/sparse autodispatch."""
    n_groups = group_matrix.shape[0]
    if issparse(group_matrix) and n_groups >= _DISPATCH_THRESHOLD:
        return _matrix_grouped_count_sparse(cast("csr_matrix", group_matrix))
    if issparse(group_matrix):
        dense_matrix = np.asarray(cast("csr_matrix", group_matrix).toarray())
    else:
        dense_matrix = np.asarray(group_matrix)
    return _matrix_grouped_count_dense(cast("DenseOperator", dense_matrix))


def _matrix_masked_sum_sparse(
    mask_matrix: csr_matrix,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums using a sparse mask matrix."""
    return np.asarray(mask_matrix @ data)


def _matrix_masked_sum_dense(
    mask_matrix: DenseOperator,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums using a dense mask matrix."""
    return np.asarray(mask_matrix @ data)


def matrix_masked_sum(
    mask_matrix: csr_matrix | DenseOperator,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums with dense/sparse autodispatch."""
    n_masks = mask_matrix.shape[0]
    if issparse(mask_matrix) and n_masks >= _DISPATCH_THRESHOLD:
        return _matrix_masked_sum_sparse(cast("csr_matrix", mask_matrix), data)
    if issparse(mask_matrix):
        dense_matrix = np.asarray(cast("csr_matrix", mask_matrix).toarray())
    else:
        dense_matrix = np.asarray(mask_matrix)
    return _matrix_masked_sum_dense(cast("DenseOperator", dense_matrix), data)


# =============================================================================
# Fast group-ID based paths (for age groups etc.)
# =============================================================================


def grouped_count_ids(
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Compute group sizes from integer group IDs."""
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    counts = np.bincount(group_ids_arr, minlength=n_groups)
    return counts.astype(float)


def grouped_sum_ids(
    values: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Compute grouped sums from integer group IDs for 1D values."""
    values_arr = np.asarray(values)
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    sums = np.bincount(group_ids_arr, weights=values_arr, minlength=n_groups)
    return sums.astype(float)


def grouped_sum_ids_2d(
    values: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Compute grouped sums from integer group IDs for 2D values."""
    values_arr = np.asarray(values)
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)

    if values_arr.ndim != 2:
        raise ValueError(_VALUES_2D_ERROR)

    n_items, n_features = values_arr.shape
    if group_ids_arr.shape[0] != n_items:
        raise ValueError(_GROUP_IDS_LENGTH_ERROR)

    out = np.zeros((n_groups, n_features), dtype=values_arr.dtype)
    for feature_idx in range(n_features):
        out[:, feature_idx] = np.bincount(
            group_ids_arr,
            weights=values_arr[:, feature_idx],
            minlength=n_groups,
        )
    return out


# =============================================================================
# Encoding helpers + smoothing
# =============================================================================


def _encode_sparse_groups(
    group_ids: NDArray[np.integer],
    n_groups: int,
    *,
    dtype: DTypeLike = np.float64,
) -> csr_matrix:
    """Encode group IDs as a sparse binary membership matrix."""
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    n_items = group_ids_arr.shape[0]
    row = group_ids_arr
    col = np.arange(n_items, dtype=np.int64)
    data = np.ones(n_items, dtype=np.dtype(dtype))
    return cast(
        "csr_matrix",
        coo_matrix((data, (row, col)), shape=(n_groups, n_items)).tocsr(),
    )


def _encode_dense_groups(
    group_ids: NDArray[np.integer],
    n_groups: int,
    *,
    dtype: DTypeLike = np.float64,
) -> DenseOperator:
    """Encode group IDs as a dense binary membership matrix."""
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    n_items = group_ids_arr.shape[0]
    group_matrix = np.zeros((n_groups, n_items), dtype=np.dtype(dtype))
    group_matrix[group_ids_arr, np.arange(n_items, dtype=np.int64)] = 1.0
    return cast("DenseOperator", group_matrix)


def encode_groups(
    group_ids: NDArray[np.integer],
    n_groups: int,
    *,
    prefer_sparse: bool | None = None,
    dtype: DTypeLike = np.float64,
) -> csr_matrix | DenseOperator:
    """Encode group IDs as a group membership matrix.

    This constructs a binary membership matrix of shape (n_groups, N) where each
    column corresponds to an item and contains a single 1.0 entry in the row for
    that item's group.

    Selection:
        * If prefer_sparse is True, always return CSR.
        * If prefer_sparse is False, always return dense.
        * If prefer_sparse is None, choose based on `_DISPATCH_THRESHOLD`.

    Args:
        group_ids: Integer group IDs of shape (N,) in the range [0, n_groups - 1].
        n_groups: Number of groups.
        prefer_sparse: Optional override for representation choice.
        dtype: Floating dtype of the membership values (default float64).

    Returns:
        A group membership matrix of shape (n_groups, N), returned as CSR or
        dense depending on selection rules.
    """
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)

    if prefer_sparse is True:
        return _encode_sparse_groups(group_ids_arr, n_groups, dtype=dtype)
    if prefer_sparse is False:
        return _encode_dense_groups(group_ids_arr, n_groups, dtype=dtype)

    if n_groups >= _DISPATCH_THRESHOLD:
        return _encode_sparse_groups(group_ids_arr, n_groups, dtype=dtype)
    return _encode_dense_groups(group_ids_arr, n_groups, dtype=dtype)


def smooth(
    x: NDArray[np.floating],
    alpha: float = 0.02,
    out: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Apply simple smoothing along the last axis.

    This applies an exponential-like smoothing that blends each value with the
    mean along the last axis:

        smoothed = (1 - alpha) * x + alpha * mean(x, axis=-1, keepdims=True)

    As alpha -> 0, the output approaches the original data; as alpha -> 1,
    the output approaches the mean along the last axis.

    Args:
        x: Input array; smoothing is applied along the last axis.
        alpha: Smoothing strength in [0, 1].
        out: Optional output array to write into.

    Returns:
        Smoothed array with the same shape as x.
    """
    x_arr = np.asarray(x)
    smoothed = (1.0 - alpha) * x_arr + alpha * x_arr.mean(axis=-1, keepdims=True)
    if out is not None:
        np.copyto(out, smoothed)
        return out
    return cast("NDArray[np.floating]", smoothed)
