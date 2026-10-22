"""Explicit eager SciPy adapters for banded and matrix-free linear solves."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from .linear_operator import (
    ActionOperator,
    LinearSolveCandidate,
    LinearSolveOptions,
    LinearSolveRequest,
)
from .matrix_ops import SparseAdapter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import Array
    from .linear_operator import ApplicableOperator


@dataclass(frozen=True, slots=True, eq=False)
class BandedOperator:
    """Real NumPy operator in SciPy's diagonal-ordered band storage.

    ``bands[upper + i - j, j]`` stores matrix entry ``(i, j)``. Construction
    copies the input and application touches only valid band entries. The
    retained storage is read-only; construct a new operator when coefficients,
    mesh or stage change. No dense matrix or factorization cache is created.
    """

    bands: NDArray[np.floating]
    lower: int
    upper: int

    def __post_init__(self) -> None:
        """Copy and validate storage and static bandwidths.

        Raises:
            ValueError: If dimensions, bandwidths or valid band entries are invalid.
            TypeError: If storage is not native NumPy float32/float64.
        """
        real_dtypes = (np.float32, np.float64)
        if (
            not isinstance(self.bands, np.ndarray)
            or self.bands.dtype not in real_dtypes
        ):
            msg = "band storage must be a NumPy float32/float64 array"
            raise TypeError(msg)
        if self.bands.ndim != 2 or self.bands.shape[1] < 1:
            msg = "band storage must have shape (lower + upper + 1, n), n >= 1"
            raise ValueError(msg)
        n = self.bands.shape[1]
        if (
            any(
                not isinstance(v, Integral) or isinstance(v, bool) or not 0 <= v < n
                for v in (self.lower, self.upper)
            )
            or self.bands.shape[0] != self.lower + self.upper + 1
        ):
            msg = "bandwidths must be integers in [0, n) matching storage rows"
            raise ValueError(msg)
        data = self.bands.copy()
        # Canonicalize unused padding; it is not part of the represented matrix.
        for offset in range(-self.upper, self.lower + 1):
            start, stop = max(0, -offset), min(n, n - offset)
            row = self.upper + offset
            data[row, :start] = 0
            data[row, stop:] = 0
        if not np.isfinite(data).all():
            msg = "valid band entries must be finite"
            raise ValueError(msg)
        data.flags.writeable = False
        object.__setattr__(self, "bands", data)

    @property
    def shape(self) -> tuple[int, int]:
        """Return square operator dimensions."""
        return (self.bands.shape[1], self.bands.shape[1])

    @property
    def dtype(self) -> object:
        """Return the native storage dtype."""
        return self.bands.dtype

    def __matmul__(self, other: Array, /) -> NDArray[np.floating]:
        """Apply valid bands to a vector or independent RHS columns.

        Returns:
            NumPy array with the same shape and dtype as the RHS.

        Raises:
            TypeError: If namespace or dtype differs.
            ValueError: If RHS dimensions disagree.
        """
        if not isinstance(other, np.ndarray) or other.dtype != self.dtype:
            msg = "banded application requires the same NumPy dtype"
            raise TypeError(msg)
        n = self.shape[0]
        if other.ndim not in {1, 2} or other.shape[0] != n:
            msg = "banded application requires vector/matrix RHS with matching rows"
            raise ValueError(msg)
        result = np.zeros_like(other)
        for offset in range(-self.upper, self.lower + 1):
            start, stop = max(0, -offset), min(n, n - offset)
            weights = self.bands[self.upper + offset, start:stop]
            if other.ndim == 2:
                weights = weights[:, None]
            result[start + offset : stop + offset] += weights * other[start:stop]
        return result

    @property
    def T(self) -> BandedOperator:  # noqa: N802
        """Return the real Euclidean transpose without dense assembly."""
        data = np.zeros(
            (self.lower + self.upper + 1, self.shape[0]), dtype=self.bands.dtype
        )
        n = self.shape[0]
        for offset in range(-self.upper, self.lower + 1):
            start, stop = max(0, -offset), min(n, n - offset)
            data[self.lower - offset, start + offset : stop + offset] = self.bands[
                self.upper + offset, start:stop
            ]
        return BandedOperator(data, lower=self.upper, upper=self.lower)


def _accepted_right(operator: object) -> bool:
    return isinstance(
        operator, (np.ndarray, csr_matrix, ActionOperator, BandedOperator)
    )


def _apply_numpy(
    operator: ApplicableOperator, value: NDArray[np.floating]
) -> NDArray[np.floating]:
    result = operator @ value
    if (
        not isinstance(result, np.ndarray)
        or result.dtype != value.dtype
        or result.shape != value.shape
    ):
        msg = "SciPy GMRES actions must preserve the NumPy vector shape and dtype"
        raise TypeError(msg)
    # GMRES overwrites action results while orthogonalizing. Providers may
    # return the input, a read-only view or reused scratch storage.
    return result.copy()


def _banded_solve(request: LinearSolveRequest) -> LinearSolveCandidate:
    if request.preconditioner is not None:
        msg = "the banded direct solver does not accept a preconditioner"
        raise ValueError(msg)
    operator = cast("BandedOperator", request.operator)
    if operator.lower == operator.upper == 0:
        diagonal = operator.bands[0]
        if np.any(diagonal == 0):
            return LinearSolveCandidate(
                np.zeros_like(request.rhs),
                success=False,
                iterations=0,
                message="singular diagonal operator",
            )
        divisor = diagonal if request.rhs.ndim == 1 else diagonal[:, None]
        # SciPy 1.13's singleton solve_banded indexes a nonexistent second row
        # for (0, 0) bandwidth. A true diagonal solve needs no LAPACK workspace.
        return LinearSolveCandidate(
            request.rhs / divisor,
            success=True,
            iterations=0,
            message="diagonal direct solve; no iterative updates",
        )
    try:
        value = solve_banded(
            (operator.lower, operator.upper),
            operator.bands,
            request.rhs,
            overwrite_ab=False,
            overwrite_b=False,
            check_finite=True,
        )
    except np.linalg.LinAlgError as error:
        return LinearSolveCandidate(
            np.zeros_like(request.rhs), success=False, iterations=0, message=str(error)
        )
    return LinearSolveCandidate(
        cast("NDArray[np.floating]", value),
        success=True,
        iterations=0,
        message="banded direct solve; no iterative updates",
    )


def scipy_banded_adapter(
    *,
    ecosystem_id: str = "scipy-banded",
    options: LinearSolveOptions | None = None,
) -> SparseAdapter:
    """Build an explicitly registered, uncached eager banded direct adapter.

    Returns:
        Existing-registry adapter recognizing a BandedOperator left operand.
    """
    return SparseAdapter(
        ecosystem_id,
        lambda op: isinstance(op, BandedOperator),
        linear_solver=_banded_solve,
        solve_options=options if options is not None else LinearSolveOptions(),
        accepts_right=_accepted_right,
    )


@dataclass(frozen=True, slots=True)
class ScipyGmresSolver:
    """Uncached eager GMRES with explicit left inverse preconditioning.

    The requested iteration budget is **inner iterations per RHS column**.
    ``iterations`` in the candidate/checked diagnostics is the maximum used by
    any column, not the total across columns. Each column starts at zero.
    There is no warm-start, basis, factorization or preconditioner cache.
    SciPy minimizes a preconditioned residual; solve_linear independently gates
    the original residual before stage adoption. No differentiation is supplied.
    """

    restart: int = 20

    def __post_init__(self) -> None:
        """Validate the static restart length.

        Raises:
            ValueError: If restart is not a positive integer.
        """
        if (
            not isinstance(self.restart, Integral)
            or isinstance(self.restart, bool)
            or self.restart < 1
        ):
            msg = "GMRES restart must be a positive integer"
            raise ValueError(msg)

    def __call__(self, request: LinearSolveRequest) -> LinearSolveCandidate:
        """Solve independent columns without forming the represented matrix.

        Returns:
            Candidate and solver termination evidence for original-residual checks.
        """
        n = request.rhs.shape[0]
        operator = LinearOperator(
            (n, n),
            dtype=request.rhs.dtype,
            matvec=lambda v: _apply_numpy(request.operator, v),
        )
        inverse = request.preconditioner
        preconditioner = (
            None
            if inverse is None
            else LinearOperator(
                (n, n),
                dtype=request.rhs.dtype,
                matvec=lambda v: _apply_numpy(inverse, v),
            )
        )
        columns = request.rhs[:, None] if request.rhs.ndim == 1 else request.rhs
        result = np.empty_like(columns)
        counts: list[int] = []
        statuses: list[int] = []
        # The legacy callback mode fixes maxiter to inner iterations on both
        # old (tol) and current (rtol) SciPy; explicit atol avoids legacy defaults.
        tolerance_name = "rtol" if "rtol" in signature(gmres).parameters else "tol"
        for column in range(columns.shape[1]):
            count = 0

            def update(_residual: object) -> None:
                nonlocal count
                count += 1

            value, status = cast("Any", gmres)(
                operator,
                columns[:, column],
                **{tolerance_name: request.options.rtol},
                atol=request.options.atol,
                restart=self.restart,
                maxiter=request.options.max_iterations,
                M=preconditioner,
                callback=update,
                callback_type="legacy",
            )
            result[:, column] = value
            counts.append(count)
            statuses.append(int(status))
        output = result[:, 0] if request.rhs.ndim == 1 else result
        return LinearSolveCandidate(
            output,
            success=all(status == 0 for status in statuses),
            iterations=max(counts, default=0),
            message=f"GMRES inner iterations per column={counts}; info={statuses}",
        )


def scipy_gmres_adapter(
    *,
    ecosystem_id: str = "scipy-gmres",
    restart: int = 20,
    options: LinearSolveOptions | None = None,
    preconditioner: ActionOperator | None = None,
    context: object | None = None,
) -> SparseAdapter:
    """Build an explicit eager matrix-free adapter in the existing registry.

    Returns:
        Adapter for ActionOperator left operands, with left inverse preconditioning.
    """
    return SparseAdapter(
        ecosystem_id,
        lambda op: isinstance(op, ActionOperator),
        linear_solver=ScipyGmresSolver(restart),
        solve_options=options if options is not None else LinearSolveOptions(),
        preconditioner=preconditioner,
        context=context,
        accepts_right=_accepted_right,
    )
