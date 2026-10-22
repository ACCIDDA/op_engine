"""Checked linear actions and explicit eager solve diagnostics.

Actions may use a native array namespace. The solver callback boundary in this
module is deliberately eager NumPy; it does not install an AD rule or transfer
arrays between devices. No action is materialized by probing basis vectors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from array_api_compat import device  # type: ignore[import-untyped]

from ._array import array_namespace

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import Array


class ApplicableOperator(Protocol):
    """Shape-bearing operator with vector and column-matrix application."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the two-dimensional operator shape."""
        ...

    @property
    def dtype(self) -> object:
        """Return the operator's native real floating dtype."""
        ...

    def __matmul__(self, other: Array, /) -> Array:
        """Apply the operator to a vector or matrix of independent columns."""
        ...


@dataclass(frozen=True, slots=True)
class ActionOperator:
    """Caller-owned, real linear action at a fixed linearization/context.

    ``matvec`` maps ``(shape[1],)`` to ``(shape[0],)`` without changing namespace,
    dtype or device. Matrix application applies that callback column by column.
    ``transpose_matvec`` represents the **Euclidean transpose** of the same
    fixed map, not an inverse or a weighted adjoint. Linearity and the correctness
    of the transpose are the provider's obligations; neither is inferred from
    a numerical probe. Only real float32/float64 actions are supported.

    Attributes:
        shape: Output and input dimensions.
        dtype: Native real floating dtype shared by inputs and outputs.
        namespace: Array-API namespace returned by :func:`array_namespace`.
        matvec: Linear vector application.
        transpose_matvec: Optional Euclidean transpose vector application.
    """

    shape: tuple[int, int]
    dtype: object
    namespace: Any
    matvec: Callable[[Array], Array]
    transpose_matvec: Callable[[Array], Array] | None = None

    def __post_init__(self) -> None:
        """Validate static metadata and callback availability.

        Raises:
            ValueError: If dimensions are invalid.
            TypeError: If dtype or callbacks are unsupported.
        """
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 2
            or any(
                not isinstance(size, Integral) or isinstance(size, bool) or size < 1
                for size in self.shape
            )
        ):
            msg = "action shape must contain two positive integer dimensions"
            raise ValueError(msg)
        if self.dtype not in (self.namespace.float32, self.namespace.float64):  # noqa: PLR6201 - equal NumPy dtype objects can have different hashes
            msg = "actions require native real float32 or float64 dtype"
            raise TypeError(msg)
        if not callable(self.matvec) or (
            self.transpose_matvec is not None and not callable(self.transpose_matvec)
        ):
            msg = "action callbacks must be callable"
            raise TypeError(msg)

    def _validate(self, value: Array, *, size: int) -> None:
        if array_namespace(value) is not self.namespace or value.dtype != self.dtype:
            msg = "action must preserve the declared namespace and dtype"
            raise TypeError(msg)
        if len(value.shape) not in {1, 2} or value.shape[0] != size:
            msg = f"action expected ({size},) or ({size}, columns); got {value.shape}"
            raise ValueError(msg)

    def _vector(self, value: Array) -> Array:
        result = self.matvec(value)
        self._validate(result, size=self.shape[0])
        if result.shape != (self.shape[0],):
            msg = "matvec must return a vector"
            raise ValueError(msg)
        result_device, input_device = device(result), device(value)
        # JAX tracers have no concrete device metadata. Concrete arrays and
        # Torch transforms do; the provider retains placement obligations in JIT.
        if (
            result_device is not None
            and input_device is not None
            and result_device != input_device
        ):
            msg = "action must preserve the input device"
            raise TypeError(msg)
        return result

    def __matmul__(self, other: Array, /) -> Array:
        """Apply the action without densifying or coercing the input.

        Returns:
            Vector or matrix of independent output columns.
        """
        self._validate(other, size=self.shape[1])
        if len(other.shape) == 1:
            return self._vector(other)
        xp = self.namespace
        if other.shape[1] == 0:
            return cast(
                "Array",
                xp.broadcast_to(
                    xp.sum(other, axis=0, keepdims=True), (self.shape[0], 0)
                ),
            )
        # Snapshot each output before the next call can reuse its storage.
        # A neutral native operation retains AD graphs in all supported namespaces.
        columns = [
            xp.multiply(self._vector(other[:, index]), 1)  # type: ignore[index]
            for index in range(other.shape[1])
        ]
        return cast("Array", xp.stack(columns, axis=1))

    @property
    def T(self) -> ActionOperator:  # noqa: N802
        """Return the explicitly supplied real Euclidean transpose.

        Raises:
            NotImplementedError: If no transpose callback was supplied.
        """
        if self.transpose_matvec is None:
            msg = "this action has no transpose capability"
            raise NotImplementedError(msg)
        return ActionOperator(
            (self.shape[1], self.shape[0]),
            self.dtype,
            self.namespace,
            self.transpose_matvec,
            self.matvec,
        )


@dataclass(frozen=True, slots=True)
class LinearSolveOptions:
    """Per-column true-residual tolerance and callback iteration budget.

    A column passes when ``||b - A x||_2 <= max(atol, rtol * ||b||_2)``.
    ``max_iterations`` counts the callback's documented iteration unit; a
    concrete adapter must declare whether this means inner steps or cycles.
    """

    rtol: float = 1e-8
    atol: float = 1e-10
    max_iterations: int = 100

    def __post_init__(self) -> None:
        """Validate static tolerances and iteration budget.

        Raises:
            ValueError: If a tolerance or iteration budget is invalid.
        """
        if any(not np.isfinite(v) or v < 0 for v in (self.rtol, self.atol)):
            msg = "linear solve tolerances must be finite and non-negative"
            raise ValueError(msg)
        if (
            not isinstance(self.max_iterations, Integral)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            msg = "max_iterations must be a positive integer"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LinearSolveRequest:
    """Explicit eager solve inputs; assembly and cache ownership stay external.

    ``preconditioner`` is an optional approximate inverse action with the same
    square shape, namespace and dtype as the operator. The concrete solver must
    document its side and use, or reject it. ``context`` is caller-owned metadata
    identifying mesh, coefficients, stage size/scale and preconditioner version.
    A callback that caches preparation must invalidate whenever any of these
    change. The engine creates no cache for this callback path.
    """

    operator: ApplicableOperator
    rhs: NDArray[np.floating]
    options: LinearSolveOptions = LinearSolveOptions()
    preconditioner: ActionOperator | None = None
    context: object | None = None


@dataclass(frozen=True, slots=True)
class LinearSolveCandidate:
    """Callback output before the engine verifies the original residual.

    ``success`` reports the solver's own termination status. A false status
    cannot be overridden by a small residual. Iterations use the concrete
    callback's declared unit and must fit its requested budget.
    """

    value: NDArray[np.floating]
    success: bool
    iterations: int
    message: str = ""


LinearSolver = Callable[[LinearSolveRequest], LinearSolveCandidate]


@dataclass(frozen=True, slots=True)
class LinearSolveDiagnostics:
    """Eager convergence evidence, one entry per RHS column (one for vectors)."""

    converged: NDArray[np.bool_]
    residual_norm: NDArray[np.floating]
    tolerance: NDArray[np.floating]
    iterations: int
    message: str


@dataclass(frozen=True, slots=True)
class LinearSolveResult:
    """Candidate solution with independently recomputed residual diagnostics."""

    value: NDArray[np.floating]
    diagnostics: LinearSolveDiagnostics


class LinearSolveError(RuntimeError):
    """A checked eager solve failed; ``result`` retains per-column evidence."""

    def __init__(self, result: LinearSolveResult) -> None:
        """Retain the failed result and identify failed RHS columns."""
        self.result = result
        failed = np.flatnonzero(~result.diagnostics.converged).tolist()
        super().__init__(
            f"linear solve failed for RHS columns {failed}: "
            f"{result.diagnostics.message}"
        )


def _numpy_like(value: object, reference: NDArray[np.floating], label: str) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != reference.dtype:
        msg = f"{label} must preserve the NumPy namespace and dtype"
        raise TypeError(msg)
    if value.shape != reference.shape:
        msg = f"{label} shape must be {reference.shape}; got {value.shape}"
        raise ValueError(msg)


def solve_linear(
    request: LinearSolveRequest, solver: LinearSolver
) -> LinearSolveResult:
    """Call an eager solver and verify every original, unpreconditioned residual.

    Returns:
        Candidate solution and per-column diagnostics, including failures.

    Raises:
        TypeError: If namespace, dtype or callback metadata is invalid.
        ValueError: If operator/RHS/result dimensions or iteration counts disagree.
    """
    rhs = request.rhs
    if not isinstance(rhs, np.ndarray) or rhs.dtype not in (np.float32, np.float64):  # noqa: PLR6201 - dtype equality is not hash equality
        msg = "checked linear solves require eager NumPy float32/float64 RHS arrays"
        raise TypeError(msg)
    shape = request.operator.shape
    if request.operator.dtype != rhs.dtype or (
        isinstance(request.operator, ActionOperator)
        and request.operator.namespace is not array_namespace(rhs)
    ):
        msg = "operator must match the NumPy RHS dtype and namespace"
        raise TypeError(msg)
    if len(shape) != 2 or shape[0] != shape[1] or len(rhs.shape) not in {1, 2}:
        msg = "checked linear solves require a square operator and vector/matrix RHS"
        raise ValueError(msg)
    if shape[0] != rhs.shape[0]:
        msg = "operator and RHS dimensions disagree"
        raise ValueError(msg)
    preconditioner = request.preconditioner
    if preconditioner is not None and (
        preconditioner.shape != shape
        or preconditioner.dtype != rhs.dtype
        or preconditioner.namespace is not array_namespace(rhs)
    ):
        msg = "inverse preconditioner must match operator shape and RHS dtype/namespace"
        raise ValueError(msg)
    candidate = solver(request)
    _numpy_like(candidate.value, rhs, "solve result")
    if not isinstance(candidate.success, bool):
        msg = "solver success must be a Python bool"
        raise TypeError(msg)
    if (
        not isinstance(candidate.iterations, Integral)
        or isinstance(candidate.iterations, bool)
        or not 0 <= candidate.iterations <= request.options.max_iterations
    ):
        msg = "solver iterations must be an integer within the requested budget"
        raise ValueError(msg)
    product = request.operator @ candidate.value
    _numpy_like(product, rhs, "operator result")
    # Accumulate float32 norms in float64 to avoid avoidable sum-of-squares overflow.
    residual = rhs.astype(np.float64) - cast("np.ndarray", product).astype(np.float64)
    residual_norm = np.atleast_1d(np.hypot.reduce(np.abs(residual), axis=0))
    rhs_norm = np.atleast_1d(np.hypot.reduce(np.abs(rhs.astype(np.float64)), axis=0))
    tolerance = np.maximum(request.options.atol, request.options.rtol * rhs_norm)
    converged = (
        np.isfinite(residual_norm)
        & np.isfinite(rhs_norm)
        & (residual_norm <= tolerance)
        & np.atleast_1d(np.all(np.isfinite(candidate.value), axis=0))
        & candidate.success
    )
    diagnostics = LinearSolveDiagnostics(
        converged, residual_norm, tolerance, candidate.iterations, candidate.message
    )
    return LinearSolveResult(candidate.value, diagnostics)


def require_linear_converged(result: LinearSolveResult) -> NDArray[np.floating]:
    """Return a successful eager solution or raise with per-column diagnostics.

    Raises:
        LinearSolveError: If any RHS column failed the checked solve.
    """
    if not np.all(result.diagnostics.converged):
        raise LinearSolveError(result)
    return result.value
