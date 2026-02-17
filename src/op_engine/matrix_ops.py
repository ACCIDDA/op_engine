"""
Matrix operations and linear solvers for multiphysics modeling.

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

Stage operator factories (IMEX/TR-BDF2 support):
    TR-BDF2 and similar IMEX methods can require *stage-specific* implicit
    operators that depend on:
        - dt (time step)
        - scale (method stage scalar)
        - t (stage time)
        - y (stage state)
        - stage (a label, e.g. "tr" or "bdf2")

    This module supports dynamic base operators via a builder:
        base_builder(t, y, stage) -> Operator

    Then the stage-operator factory produces (L, R) to solve:
        L @ y_next = R @ y_in

    where (L, R) follow schemes like implicit Euler or trapezoidal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, csr_matrix, diags, identity, issparse, kron
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import factorized as sparse_factorized

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Public operator types (backend-friendly)
# =============================================================================

DenseOperator: TypeAlias = NDArray[np.floating]
SparseOperator: TypeAlias = csr_matrix
Operator: TypeAlias = DenseOperator | SparseOperator


# =============================================================================
# Stage operator factory types
# =============================================================================

StageName: TypeAlias = str | None


@dataclass(frozen=True, slots=True)
class StageOperatorContext:
    """Context passed to time/state-dependent operator builders.

    Attributes:
        t: Stage time.
        y: Stage state as a 1D float array (flattened along solver operator axis).
        stage: Optional stage label (e.g. "tr", "bdf2").
        extra: Optional extra payload for future use (kept generic).
    """

    t: float
    y: NDArray[np.floating]
    stage: StageName = None
    extra: Any | None = None


# Dynamic base operator builder:
#   base_builder(ctx) -> Operator
BaseOperatorBuilder: TypeAlias = "Callable[[StageOperatorContext], Operator]"

# Stage operator factory:
#   factory(dt, scale, ctx) -> (L, R)
StageOperatorFactory: TypeAlias = (
    "Callable[[float, float, StageOperatorContext], tuple[Operator, Operator]]"
)


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
_OPERATOR_SCALE_ERROR = "scale must be a finite float; got {scale}"
_UNKNOWN_SCHEME_ERROR = "Unknown scheme: {scheme}"
_BASE_BUILDER_ERROR = (
    "base_builder must return a dense ndarray or csr_matrix; got {typ}"
)


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

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Physical diffusion coefficient D (units length^2 / time).
        dtype: Floating dtype (e.g. np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".

    Raises:
        ValueError: If an unknown boundary condition is provided.

    Returns:
        Sparse CSR matrix representing the Laplacian operator.
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
    return scaled.tocsr()


def _build_crank_nicolson_sparse(
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Build sparse Crank-Nicolson operator matrices (L, R).

    Args:
        geom: Grid geometry.
        cfg: Diffusion configuration.
        dt: Time step.

    Returns:
        Tuple of (L, R) operators for Crank-Nicolson scheme.
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
    """
    Build dense Crank-Nicolson operator matrices (L, R).

    Args:
        geom: Grid geometry.
        cfg: Diffusion configuration.
        dt: Time step.

    Returns:
        Tuple of (L, R) operators for Crank-Nicolson scheme.
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
    """
    Build Crank-Nicolson operators with dense/sparse autodispatch.

    Args:
        geom: Grid geometry.
        cfg: Diffusion configuration.
        dt: Time step.

    Returns:
        Tuple of (L, R) operators for Crank-Nicolson scheme.
    """
    if geom.n < _DISPATCH_THRESHOLD:
        return _build_crank_nicolson_dense(geom, cfg, dt)
    return _build_crank_nicolson_sparse(geom, cfg, dt)


def _build_predictor_corrector_dense(
    base_matrix: DenseOperator,
) -> tuple[DenseOperator, DenseOperator, DenseOperator]:
    """
    Build predictor-corrector matrices for a dense base matrix.

    Args:
        base_matrix: Base linear operator A.

    Returns:
        Tuple of (predictor, L, R) operators for predictor-corrector scheme.
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
    """
    Build predictor-corrector matrices for a sparse base matrix.

    Args:
        base_matrix: Base linear operator A.

    Returns:
        Tuple of (predictor, L, R) operators for predictor-corrector scheme.
    """
    n = base_matrix.shape[0]
    identity_mat = identity(n, format="csr", dtype=base_matrix.dtype)
    predictor = identity_mat
    left_op = identity_mat - 0.5 * base_matrix
    right_op = identity_mat + 0.5 * base_matrix
    return (
        predictor.tocsr(),
        left_op.tocsr(),
        right_op.tocsr(),
    )


def build_predictor_corrector(
    base_matrix: DenseOperator | csr_matrix,
) -> tuple[Operator, Operator, Operator]:
    """
    Build predictor-corrector matrices with dense/sparse autodispatch.

    Args:
        base_matrix: Base linear operator A.

    Returns:
        Tuple of (predictor, L, R) operators for predictor-corrector scheme.
    """
    n = base_matrix.shape[0]
    if issparse(base_matrix) and n >= _DISPATCH_THRESHOLD:
        return _build_predictor_corrector_sparse(base_matrix)

    if issparse(base_matrix):
        dense_base = np.asarray(base_matrix.toarray())
    else:
        dense_base = np.asarray(base_matrix)

    return _build_predictor_corrector_dense(cast("DenseOperator", dense_base))


# =============================================================================
# IMEX stage operator builders (TR-BDF2 support)
# =============================================================================


def build_identity_operator(
    n: int,
    *,
    dtype: DTypeLike = np.float64,
    prefer_sparse: bool | None = None,
) -> Operator:
    """
    Build an identity operator with dense/sparse autodispatch.

    Args:
        n: Size of the identity operator (n x n).
        dtype: Floating dtype (e.g. np.float64).
        prefer_sparse: If True, always return a sparse operator; if False,
            always return a dense operator; if None, autodispatch based on n.

    Returns:
        Identity operator of shape (n, n) as either a dense ndarray or CSR matrix.
    """
    dtype_obj = np.dtype(dtype)

    if prefer_sparse is True:
        return identity(n, format="csr", dtype=dtype_obj)

    if prefer_sparse is False:
        return cast("DenseOperator", np.eye(n, dtype=dtype_obj))

    # Autodispatch
    if n >= _DISPATCH_THRESHOLD:
        return identity(n, format="csr", dtype=dtype_obj)

    return cast("DenseOperator", np.eye(n, dtype=dtype_obj))


def build_implicit_euler_operators(
    base_op: Operator,
    dt_scale: float,
) -> tuple[Operator, Operator]:
    """Build implicit Euler operators for a time-scaled linear operator.

    Args:
        base_op: Base linear operator A.
        dt_scale: Time-step scaling factor (dt * scale).

    Raises:
        ValueError: If dt_scale is not finite.

    Returns:
        Tuple of (L, R) operators for implicit Euler scheme.
    """
    if not np.isfinite(dt_scale):
        raise ValueError(_OPERATOR_SCALE_ERROR.format(scale=dt_scale))

    n = base_op.shape[0]

    if issparse(base_op):
        base_csr = base_op.tocsr()
        identity_csr = identity(n, format="csr", dtype=base_csr.dtype)
        left_csr = (identity_csr - (dt_scale * base_csr)).tocsr()
        right_csr = identity_csr.tocsr()
        return left_csr, right_csr

    base_arr = np.asarray(base_op)
    identity_arr = np.eye(n, dtype=base_arr.dtype)
    left_arr = identity_arr - (dt_scale * base_arr)
    right_arr = identity_arr
    return cast("DenseOperator", left_arr), cast("DenseOperator", right_arr)


def build_trapezoidal_operators(
    base_op: Operator,
    dt_scale: float,
) -> tuple[Operator, Operator]:
    """Build trapezoidal operators for a time-scaled linear operator.

    Args:
        base_op: Base linear operator A.
        dt_scale: Time-step scaling factor (dt * scale).

    Raises:
        ValueError: If dt_scale is not finite.

    Returns:
        Tuple of (L, R) operators for trapezoidal scheme.
    """
    if not np.isfinite(dt_scale):
        raise ValueError(_OPERATOR_SCALE_ERROR.format(scale=dt_scale))

    n = base_op.shape[0]
    half = 0.5 * dt_scale

    if issparse(base_op):
        base_csr = base_op.tocsr()
        identity_mat = identity(n, format="csr", dtype=base_csr.dtype)
        left_csr = (identity_mat - (half * base_csr)).tocsr()
        right_csr = (identity_mat + (half * base_csr)).tocsr()
        return cast("csr_matrix", left_csr), cast("csr_matrix", right_csr)

    base_arr = np.asarray(base_op)
    identity_arr = np.eye(n, dtype=base_arr.dtype)
    left_arr = identity_arr - (half * base_arr)
    right_arr = identity_arr + (half * base_arr)
    return cast("DenseOperator", left_arr), cast("DenseOperator", right_arr)


def _ensure_operator_type(op: np.ndarray | csr_matrix) -> Operator:
    if isinstance(op, np.ndarray):
        return cast("DenseOperator", op)
    if issparse(op):
        return op.tocsr()
    raise TypeError(_BASE_BUILDER_ERROR.format(typ=type(op)))


def make_stage_operator_factory(
    base_builder: BaseOperatorBuilder,
    *,
    scheme: str = "implicit-euler",
) -> StageOperatorFactory:
    """
    Create a stage operator factory supporting time/state dependent base ops.

    Args:
        base_builder: Function that builds a base operator given stage context.
        scheme: Implicit scheme; either "implicit-euler" or "trapezoidal".

    Raises:
        ValueError: If an unknown scheme is provided.

    Returns:
        A StageOperatorFactory that builds (L, R) operators for the given scheme.
    """
    scheme_norm = str(scheme).strip().lower()

    if scheme_norm == "implicit-euler":

        def _factory(
            dt: float, scale: float, ctx: StageOperatorContext
        ) -> tuple[Operator, Operator]:
            operator = _ensure_operator_type(base_builder(ctx))
            return build_implicit_euler_operators(operator, float(dt) * float(scale))

        return _factory

    if scheme_norm == "trapezoidal":

        def _factory(
            dt: float, scale: float, ctx: StageOperatorContext
        ) -> tuple[Operator, Operator]:
            operator = _ensure_operator_type(base_builder(ctx))
            return build_trapezoidal_operators(operator, float(dt) * float(scale))

        return _factory

    raise ValueError(_UNKNOWN_SCHEME_ERROR.format(scheme=scheme))


def make_constant_base_builder(operator: Operator) -> BaseOperatorBuilder:
    """
    Convenience: wrap a constant operator as a BaseOperatorBuilder.

    Args:
        operator: Constant operator to wrap.

    Returns:
        A BaseOperatorBuilder that always returns the given operator.
    """
    operator_0 = _ensure_operator_type(operator)

    def _builder(ctx: StageOperatorContext) -> Operator:  # noqa: ARG001
        return operator_0

    return _builder


# =============================================================================
# Kronecker composition utilities (optional, but public)
# =============================================================================


def kron_prod(a: Operator, b: Operator) -> Operator:
    """
    Compute the Kronecker product of two operators.

    Args:
        a: First operator.
        b: Second operator.

    Returns:
        The Kronecker product operator.
    """
    if issparse(a) or issparse(b):
        a_csr = a if issparse(a) else csr_matrix(np.asarray(a))
        b_csr = b if issparse(b) else csr_matrix(np.asarray(b))
        return kron(
            a_csr,
            b_csr,
            format="csr",
        )
    return cast("DenseOperator", np.kron(np.asarray(a), np.asarray(b)))


def kron_sum(ops: list[Operator]) -> Operator:
    """
    Compute a Kronecker sum of square operators.

    Args:
        ops: List of 2D square operators.

    Raises:
        ValueError: If ops is empty or if operators are not square or
            have incompatible shapes.

    Returns:
        The Kronecker sum operator.
    """
    if not ops:
        raise ValueError(_KRON_EMPTY_ERROR)

    shapes = [
        tuple(np.asarray(op).shape) if not issparse(op) else op.shape for op in ops
    ]

    if any(s[0] != s[1] for s in shapes):
        raise ValueError(_KRON_INCOMPATIBLE_ERROR.format(shapes=shapes))

    any_sparse = any(issparse(op) for op in ops)
    sizes = [s[0] for s in shapes]
    dtype_obj = np.result_type(*[
        (op.dtype if issparse(op) else np.asarray(op).dtype) for op in ops
    ])

    def _eye(n: int) -> Operator:
        if any_sparse:
            return identity(n, format="csr", dtype=dtype_obj)
        return cast("DenseOperator", np.eye(n, dtype=dtype_obj))

    total: Operator | None = None
    n_ops = len(ops)

    for i, op_i in enumerate(ops):
        term: Operator = op_i
        for j in range(i - 1, -1, -1):
            term = kron_prod(_eye(sizes[j]), term)
        for j in range(i + 1, n_ops):
            term = kron_prod(term, _eye(sizes[j]))
        total = term if total is None else cast("Operator", total + term)

    if total is None:
        raise ValueError(_KRON_EMPTY_ERROR)
    return total


# =============================================================================
# Implicit solvers: factorized (cached) & backend-neutral wrappers
# =============================================================================


def clear_implicit_solver_cache() -> None:
    """Clear the internal implicit solver cache."""
    _IMPLICIT_SOLVER_CACHE.clear()


def _operator_meta(
    left_op: Operator,
    right_op: Operator,
) -> _SolverMeta:
    """
    Compute a metadata tuple used to validate cache hits.

    Args:
        left_op: Left operator L in the equation L @ y = R @ x.
        right_op: Right operator R in the equation L @ y = R @ x.

    Returns:
        A metadata tuple describing the operators.
    """
    if issparse(left_op):
        l_shape = left_op.shape
        l_dtype = str(left_op.dtype)
        l_sparse = True
    else:
        l_arr = np.asarray(left_op)
        l_shape = l_arr.shape
        l_dtype = str(l_arr.dtype)
        l_sparse = False

    if issparse(right_op):
        r_shape = right_op.shape
        r_dtype = str(right_op.dtype)
        r_sparse = True
    else:
        r_arr = np.asarray(right_op)
        r_shape = r_arr.shape
        r_dtype = str(r_arr.dtype)
        r_sparse = False

    is_sparse = l_sparse and r_sparse
    return (
        l_shape,
        r_shape,
        l_dtype,
        r_dtype,
        is_sparse,
    )


def _validate_square_operator(op: Operator) -> tuple[int, int]:
    shape = cast("tuple[int, int]", op.shape)
    if shape[0] != shape[1]:
        raise ValueError(_OPERATORS_SQUARE_ERROR.format(shape=shape))
    return shape


def _validate_solve_dimensions(
    left_op: Operator,
    right_op: Operator,
    x: NDArray[np.floating],
) -> None:
    l_shape = _validate_square_operator(left_op)
    r_shape = _validate_square_operator(right_op)
    if l_shape != r_shape:
        raise ValueError(_OPERATORS_SQUARE_ERROR.format(shape=(l_shape, r_shape)))

    if x.ndim not in {1, 2}:
        raise ValueError(_X_NDIM_ERROR.format(ndim=x.ndim))

    n = l_shape[0]
    if x.shape[0] != n:
        raise ValueError(_OPERATORS_DIM_ERROR.format(shape=l_shape, x_shape=x.shape))


def _as_linear_operator(op: Operator) -> LinearOperator:
    """
    Convert an operator to a SciPy LinearOperator.

    Args:
        op: 2D operator (dense ndarray or csr_matrix).

    Returns:
        A SciPy LinearOperator wrapping the input operator.
    """
    if issparse(op):
        op_csr = op

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
        shape=op_arr.shape, dtype=op_arr.dtype, matvec=dense_matvec, matmat=dense_matmat
    )


def _build_implicit_solver(
    left_op: Operator,
    right_op: Operator,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """
    Build a reusable implicit solver for left_op @ y = right_op @ x.

    Args:
        left_op: Left operator L in the equation L @ y = R @ x.
        right_op: Right operator R in the equation L @ y = R @ x.

    Returns:
        A callable that takes x and returns the solution y.
    """
    is_sparse = issparse(left_op) and issparse(right_op)

    if is_sparse:
        left_csr = cast("csr_matrix", left_op)
        right_csr = cast("csr_matrix", right_op)
        solve_left = sparse_factorized(left_csr.tocsc())

        def sparse_solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
            x_arr = np.asarray(x, dtype=left_csr.dtype)
            rhs = right_csr @ x_arr

            if rhs.ndim == 1:
                out_1d = solve_left(rhs)
                return np.asarray(out_1d, dtype=x_arr.dtype)

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
        """
        Perform a dense implicit solve using precomputed LU factorization.

        Args:
            x: 1D or 2D array representing the input vector(s).

        Returns:
            A 1D or 2D array containing the solution vector(s).
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
    """
    Perform an implicit solve with dense/sparse dispatch and caching.

    Args:
        left_op: Left operator L in the equation L @ y = R @ x.
        right_op: Right operator R in the equation L @ y = R @ x.
        x: 1D or 2D array representing the input vector(s).

    Returns:
        A 1D or 2D array containing the solution vector(s) y.
    """
    x_arr = np.asarray(x)
    _validate_solve_dimensions(left_op, right_op, cast("NDArray[np.floating]", x_arr))

    key = (id(left_op), id(right_op))
    meta = _operator_meta(left_op, right_op)

    cached = _IMPLICIT_SOLVER_CACHE.get(key)
    if cached is not None:
        cached_meta, solver = cached
        if cached_meta != meta:
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
    return np.asarray(group_matrix @ values)


def _matrix_grouped_sum_dense(
    group_matrix: DenseOperator,
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    return np.asarray(group_matrix @ values)


def matrix_grouped_sum(
    group_matrix: csr_matrix | DenseOperator,
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Perform grouped sum using a group matrix.

    Args:
        group_matrix: 2D group matrix (csr_matrix or dense ndarray).
        values: 1D array of values to be summed.

    Returns:
        A 1D array containing the grouped sums.
    """
    n_groups = group_matrix.shape[0]
    if issparse(group_matrix) and n_groups >= _DISPATCH_THRESHOLD:
        return _matrix_grouped_sum_sparse(group_matrix, values)
    if issparse(group_matrix):
        dense_matrix = np.asarray(group_matrix.toarray())
    else:
        dense_matrix = np.asarray(group_matrix)
    return _matrix_grouped_sum_dense(cast("DenseOperator", dense_matrix), values)


def _matrix_grouped_count_sparse(group_matrix: csr_matrix) -> NDArray[np.floating]:
    counts = np.asarray(group_matrix.sum(axis=1)).ravel()
    return counts.astype(float)


def _matrix_grouped_count_dense(group_matrix: DenseOperator) -> NDArray[np.floating]:
    counts = np.asarray(group_matrix.sum(axis=1))
    return counts.astype(float)


def matrix_grouped_count(
    group_matrix: csr_matrix | DenseOperator,
) -> NDArray[np.floating]:
    """
    Perform grouped count using a group matrix.

    Args:
        group_matrix: 2D group matrix (csr_matrix or dense ndarray).

    Returns:
        A 1D array containing the counts for each group.
    """
    n_groups = group_matrix.shape[0]
    if issparse(group_matrix) and n_groups >= _DISPATCH_THRESHOLD:
        return _matrix_grouped_count_sparse(group_matrix)
    if issparse(group_matrix):
        dense_matrix = group_matrix.toarray()
    else:
        dense_matrix = np.asarray(group_matrix)
    return _matrix_grouped_count_dense(cast("DenseOperator", dense_matrix))


def _matrix_masked_sum_sparse(
    mask_matrix: csr_matrix,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    return np.asarray(mask_matrix @ data)


def _matrix_masked_sum_dense(
    mask_matrix: DenseOperator,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    return np.asarray(mask_matrix @ data)


def matrix_masked_sum(
    mask_matrix: csr_matrix | DenseOperator,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Perform masked sum using a mask matrix and data array.

    Args:
        mask_matrix: 2D mask matrix (csr_matrix or dense ndarray).
        data: 1D or 2D data array to be masked and summed.

    Returns:
        A 1D or 2D array containing the masked sums.
    """
    n_masks = mask_matrix.shape[0]
    if issparse(mask_matrix) and n_masks >= _DISPATCH_THRESHOLD:
        return _matrix_masked_sum_sparse(mask_matrix, data)
    if issparse(mask_matrix):
        dense_matrix = np.asarray(mask_matrix.toarray())
    else:
        dense_matrix = np.asarray(mask_matrix)
    return _matrix_masked_sum_dense(dense_matrix, data)


# =============================================================================
# Fast group-ID based paths (for age groups etc.)
# =============================================================================


def grouped_count_ids(
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """
    Perform grouped count using group IDs.

    Args:
        group_ids: 1D array of integer group IDs.
        n_groups: Total number of groups.

    Returns:
        A 1D array of length n_groups where each element contains the count of
        occurrences of the corresponding group ID.
    """
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    counts = np.bincount(group_ids_arr, minlength=n_groups)
    return counts.astype(float)


def grouped_sum_ids(
    values: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """
    Perform grouped sum over 1D values array using group IDs.

    Args:
        values: 1D array of values.
        group_ids: 1D array of integer group IDs.
        n_groups: Total number of groups.

    Returns:
        A 1D array of length n_groups where each element contains the sum of values
        for the corresponding group ID.
    """
    values_arr = np.asarray(values)
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    sums = np.bincount(group_ids_arr, weights=values_arr, minlength=n_groups)
    return sums.astype(float)


def grouped_sum_ids_2d(
    values: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """
    Perform grouped sum over 2D values array using group IDs.

    Args:
        values: 2D (N, K) array where N is num of items and K is num of features.
        group_ids: 1D array of integer group IDs of length N.
        n_groups: Total number of groups.

    Raises:
        ValueError: If values is not 2D or if group_ids length does not match
            the number of items in values.

    Returns:
        A 2D array of shape (n_groups, K) where each row contains the sum of values
        for the corresponding group ID.
    """
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
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    n_items = group_ids_arr.shape[0]
    row = group_ids_arr
    col = np.arange(n_items, dtype=np.int64)
    data = np.ones(n_items, dtype=np.dtype(dtype))
    return coo_matrix((data, (row, col)), shape=(n_groups, n_items)).tocsr()


def _encode_dense_groups(
    group_ids: NDArray[np.integer],
    n_groups: int,
    *,
    dtype: DTypeLike = np.float64,
) -> DenseOperator:
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
    """
    Encode group IDs into a one-hot group membership matrix.

    Args:
        group_ids: 1D array of integer group IDs for each item.
        n_groups: Total number of groups.
        prefer_sparse: If True, always return a sparse matrix; if False, always
            return a dense array; if None, autodispatch based on n_groups.
        dtype: Data type for the output matrix.

    Returns:
        A (n_groups, n_items) one-hot encoded group membership matrix.
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
    """
    Apply simple smoothing along the last axis.

    Args:
        x: Input array to smooth.
        alpha: Smoothing factor between 0 and 1.
        out: Optional output array to store the result.

    Returns:
        Smoothed array with the same shape as x.
    """
    x_arr = np.asarray(x)
    smoothed = (1.0 - alpha) * x_arr + alpha * x_arr.mean(axis=-1, keepdims=True)
    if out is not None:
        np.copyto(out, smoothed)
        return out
    return cast("NDArray[np.floating]", smoothed)
