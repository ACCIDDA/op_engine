"""Eager local linear-stability evidence for candidate equilibria.

The diagnostic in this module is deliberately dense, host-only and local.  It
does not change :func:`op_engine.steady_state`, certify nonlinear or global
stability, or imply that every Array API namespace provides an eigensolver.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .steady_state import SteadyStateResult

NumpyArray = NDArray[Any]
StabilityRhs = Callable[[float, NumpyArray], NumpyArray]
StabilityJacobian = Callable[[float, NumpyArray], NumpyArray]


class LocalStabilityStatus(StrEnum):
    """Numerical status, independent of any spectral classification."""

    ANALYZED = "analyzed"
    SOLVE_FAILURE = "solve_failure"
    NOT_EQUILIBRIUM = "not_equilibrium"
    NONFINITE_STATE = "nonfinite_state"
    RESIDUAL_FAILURE = "residual_failure"
    NONFINITE_RESIDUAL = "nonfinite_residual"
    JACOBIAN_FAILURE = "jacobian_failure"
    NONFINITE_JACOBIAN = "nonfinite_jacobian"
    INCONSISTENT_INVARIANTS = "inconsistent_invariants"
    BOUNDARY = "boundary"
    INFEASIBLE = "infeasible"
    EIGENSOLVE_FAILED = "eigensolve_failed"
    NONFINITE_EIGENVALUES = "nonfinite_eigenvalues"
    NO_DYNAMIC_DIRECTIONS = "no_dynamic_directions"


class LocalSpectrum(StrEnum):
    """Strict classification of an assessed reduced linear spectrum."""

    ASYMPTOTICALLY_STABLE = "asymptotically_stable"
    UNSTABLE = "unstable"
    NEUTRAL_SPECTRUM = "neutral_spectrum"
    NONHYPERBOLIC = "nonhyperbolic"


class EquilibriumDomain(StrEnum):
    """Whether declared one-sided state constraints affect interpretation."""

    UNSPECIFIED = "unspecified"
    INTERIOR = "interior"
    BOUNDARY = "boundary"
    INFEASIBLE = "infeasible"


class StabilityJacobianSource(StrEnum):
    """Provenance of the dense full-state Jacobian."""

    FINITE_DIFFERENCE = "finite_difference"
    PROVIDED = "provided"
    OP_SYSTEM_SYMBOLIC = "op_system_symbolic"


class StabilityTimescaleKind(StrEnum):
    """Interpretation of an e-folding timescale."""

    DECAY = "decay"
    GROWTH = "growth"


class _DerivativeValues(Protocol):
    state: object


class SymbolicDerivativeArtifact(Protocol):
    """Structural surface consumed from ``op_system.SymbolicDerivatives``."""

    state_names: tuple[str, ...]
    model: object

    def evaluate(self, t: object, y: object, **parameters: object) -> _DerivativeValues:
        """Evaluate derivative matrices in the artifact's declared order."""
        ...


@dataclass(frozen=True, slots=True)
class LocalStabilityConfig:
    """Explicit numerical thresholds for :func:`local_stability`.

    ``residual_tolerance`` and ``boundary_tolerance`` may be scalars or one
    value per state coordinate.  Spectral tolerances have inverse-time units;
    their effective band is ``absolute + relative * max(abs(eigenvalue))``.
    """

    residual_tolerance: ArrayLike
    spectral_absolute_tolerance: float
    spectral_relative_tolerance: float
    boundary_tolerance: ArrayLike
    finite_difference_step: ArrayLike | None = None

    def __post_init__(self) -> None:
        """Reject invalid scalar controls before evaluating a model.

        Raises:
            ValueError: A spectral tolerance is negative, non-finite or not scalar.
        """
        for name in ("spectral_absolute_tolerance", "spectral_relative_tolerance"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                msg = f"{name} must be finite and non-negative"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LocalStabilityResult:
    """Evidence and interpretation from one local linearization."""

    status: LocalStabilityStatus
    classification: LocalSpectrum | None
    message: str
    state: NumpyArray
    time: float
    coordinates: tuple[str, ...]
    coordinate_system: str
    time_unit: str | None
    jacobian_source: StabilityJacobianSource
    jacobian_reference: str | None
    model_digest: str | None
    solver_converged: bool | None
    residual_tolerance: NumpyArray
    boundary_tolerance: NumpyArray
    fixed_mask: NumpyArray
    nonnegative_mask: NumpyArray
    domain: EquilibriumDomain = EquilibriumDomain.UNSPECIFIED
    raw_residual: NumpyArray | None = None
    active_residual: NumpyArray | None = None
    residual_scaled_rms: float | None = None
    boundary_coordinates: tuple[str, ...] = ()
    finite_difference_steps: NumpyArray | None = None
    jacobian: NumpyArray | None = None
    full_eigenvalues: NumpyArray | None = None
    active_jacobian: NumpyArray | None = None
    active_eigenvalues: NumpyArray | None = None
    invariant_rows: NumpyArray | None = None
    invariant_rank: int = 0
    constraint_rank: int = 0
    constraint_svd_tolerance: float | None = None
    tangent_basis: NumpyArray | None = None
    tangent_jacobian: NumpyArray | None = None
    tangent_eigenvalues: NumpyArray | None = None
    conservation_error: float | None = None
    conservation_tolerance: float | None = None
    invariant_flow_error: float | None = None
    invariant_flow_tolerance: float | None = None
    expected_conservation_neutral_modes: int = 0
    observed_active_neutral_modes: int | None = None
    effective_spectral_tolerance: float | None = None
    spectral_abscissa: float | None = None
    leading_eigenvalue: complex | None = None
    leading_timescale: float | None = None
    timescale_kind: StabilityTimescaleKind | None = None


def _vector_control(
    value: ArrayLike, n: int, name: str, *, positive: bool
) -> NDArray[np.float64]:
    """Return a finite length-``n`` host control vector.

    Raises:
        ValueError: The control has the wrong shape, sign or finite status.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        array = np.full(n, float(array), dtype=np.float64)
    if array.shape != (n,):
        msg = f"{name} must be scalar or have shape ({n},)"
        raise ValueError(msg)
    if not np.isfinite(array).all() or np.any(array <= 0 if positive else array < 0):
        bound = "positive" if positive else "non-negative"
        msg = f"{name} must be finite and {bound}"
        raise ValueError(msg)
    return array


def _coordinate_names(coordinates: Sequence[str] | None, n: int) -> tuple[str, ...]:
    """Validate or create ordered coordinate labels.

    Returns:
        One distinct, non-empty label for each coordinate.

    Raises:
        ValueError: Supplied labels are missing, duplicated or malformed.
    """
    names = (
        tuple(f"state[{index}]" for index in range(n))
        if coordinates is None
        else tuple(coordinates)
    )
    if (
        len(names) != n
        or len(set(names)) != n
        or any(not isinstance(name, str) or not name for name in names)
    ):
        msg = "coordinates must contain one distinct non-empty name per state"
        raise ValueError(msg)
    return names


def _selection_mask(value: object, n: int, name: str) -> NDArray[np.bool_]:
    """Parse indices or a length-``n`` boolean mask.

    Returns:
        A copied boolean selection mask.

    Raises:
        ValueError: The declaration is not a valid mask or index sequence.
    """
    if value is None:
        return np.zeros(n, dtype=np.bool_)
    array = np.asarray(value)
    if array.dtype == np.bool_:
        if array.shape != (n,):
            msg = f"{name} mask must have shape ({n},)"
            raise ValueError(msg)
        return cast("NDArray[np.bool_]", array.copy())
    if array.ndim > 1 or not np.issubdtype(array.dtype, np.integer):
        msg = f"{name} must contain integer indices or a boolean mask"
        raise ValueError(msg)
    indices = array.astype(np.int64, copy=False).ravel()
    if np.any((indices < 0) | (indices >= n)):
        msg = f"{name} indices must lie in [0, {n})"
        raise ValueError(msg)
    result = np.zeros(n, dtype=np.bool_)
    result[indices] = True
    return result


def _nonnegative_mask(value: object, n: int) -> NDArray[np.bool_]:
    """Parse the declared one-sided state domain.

    Returns:
        A copied mask of coordinates constrained to be non-negative.

    Raises:
        ValueError: A non-scalar declaration is not a length-``n`` boolean mask.
    """
    if isinstance(value, (bool, np.bool_)):
        return np.full(n, bool(value), dtype=np.bool_)
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != (n,):
        msg = f"nonnegative must be boolean or a boolean mask of shape ({n},)"
        raise ValueError(msg)
    return cast("NDArray[np.bool_]", array.copy())


def _invariant_rows(value: object, n: int) -> NDArray[np.float64]:
    """Return finite, normalized invariant rows without hiding zero rows.

    Raises:
        ValueError: A row has the wrong width, is non-finite or has zero norm.
    """
    if value is None:
        return np.empty((0, n), dtype=np.float64)
    rows = np.atleast_2d(np.asarray(value, dtype=np.float64))
    if rows.ndim != 2 or rows.shape[1] != n:
        msg = f"invariants must have shape (k, {n}); got {rows.shape}"
        raise ValueError(msg)
    if not np.isfinite(rows).all():
        msg = "invariants must be finite"
        raise ValueError(msg)
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms == 0):
        msg = "invariants must not contain zero rows"
        raise ValueError(msg)
    return rows / norms[:, None]


def _strict_numpy_vector(value: object, state: NumpyArray, name: str) -> NumpyArray:
    """Require a same-shape, same-dtype NumPy vector without hidden transfer.

    Returns:
        An owned copy of the validated array.

    Raises:
        TypeError: The callback did not return an eager NumPy array.
        ValueError: Its shape or dtype differs from the state.
    """
    if not isinstance(value, np.ndarray):
        msg = f"{name} must return a NumPy array; transfer device arrays explicitly"
        raise TypeError(msg)
    if value.shape != state.shape or value.dtype != state.dtype:
        msg = f"{name} must return shape {state.shape} and dtype {state.dtype}"
        raise ValueError(msg)
    return value.copy()


def _strict_numpy_matrix(value: object, state: NumpyArray, name: str) -> NumpyArray:
    """Require a same-dtype dense full-state NumPy Jacobian.

    Returns:
        An owned copy of the validated dense array.

    Raises:
        TypeError: The source did not return an eager NumPy array.
        ValueError: Its shape or dtype differs from the full-state contract.
    """
    expected = (state.size, state.size)
    if not isinstance(value, np.ndarray):
        msg = f"{name} must return a NumPy array; sparse/device actions are unsupported"
        raise TypeError(msg)
    if value.shape != expected or value.dtype != state.dtype:
        msg = f"{name} must return shape {expected} and dtype {state.dtype}"
        raise ValueError(msg)
    return value.copy()


def _finite_difference_jacobian(
    rhs: StabilityRhs,
    time: float,
    state: NumpyArray,
    configured_step: ArrayLike | None,
) -> tuple[NumpyArray, NumpyArray]:
    """Return a dense central-difference Jacobian and its perturbations."""
    n = state.size
    if configured_step is None:
        steps = np.cbrt(np.finfo(state.dtype).eps) * np.maximum(np.abs(state), 1)
    else:
        steps = _vector_control(
            configured_step, n, "finite_difference_step", positive=True
        ).astype(state.dtype)
    eye = np.eye(n, dtype=state.dtype)
    columns = []
    for index in range(n):
        offset = steps[index] * eye[index]
        plus = _strict_numpy_vector(rhs(time, state + offset), state, "rhs")
        minus = _strict_numpy_vector(rhs(time, state - offset), state, "rhs")
        columns.append((plus - minus) / (2 * steps[index]))
    return np.stack(columns, axis=1).astype(state.dtype, copy=False), steps


def _constraint_basis(
    fixed: NumpyArray, invariants: NDArray[np.float64]
) -> tuple[NumpyArray, int, float]:
    """Return an orthonormal basis for all declared equality constraints.

    Raises:
        ValueError: Fixed selectors and invariant rows are jointly dependent.
    """
    n = fixed.size
    rows = [np.eye(n, dtype=np.float64)[fixed]]
    if invariants.size:
        rows.append(invariants)
    constraint = np.concatenate(rows, axis=0)
    if constraint.shape[0] == 0:
        return np.eye(n, dtype=np.float64), 0, 0.0
    _, singular, vh = np.linalg.svd(constraint, full_matrices=True)
    largest = float(singular[0]) if singular.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(constraint.shape) * largest
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != constraint.shape[0]:
        msg = "fixed and invariant constraints must be jointly independent"
        raise ValueError(msg)
    return vh[rank:].T, rank, tolerance


def _source_and_coordinates(  # noqa: PLR0913,PLR0917
    derivatives: SymbolicDerivativeArtifact | None,
    jacobian: StabilityJacobian | None,
    parameters: Mapping[str, object] | None,
    coordinates: Sequence[str] | None,
    n: int,
    jacobian_reference: str | None,
) -> tuple[
    StabilityJacobianSource,
    tuple[str, ...],
    Mapping[str, object],
    str | None,
]:
    """Validate the mutually exclusive Jacobian routes and their provenance.

    Returns:
        Source kind, coordinate names, parameter bindings and model digest.

    Raises:
        ValueError: Sources, coordinate order or provenance are inconsistent.
    """
    if derivatives is not None and jacobian is not None:
        msg = "derivatives and jacobian are mutually exclusive"
        raise ValueError(msg)
    if derivatives is None and parameters:
        msg = "parameters are accepted only with a derivative artifact"
        raise ValueError(msg)
    if derivatives is not None and jacobian_reference is not None:
        msg = "derivative artifacts carry their own structural reference"
        raise ValueError(msg)
    if jacobian is not None and (
        not isinstance(jacobian_reference, str) or not jacobian_reference
    ):
        msg = "a supplied jacobian requires a non-empty jacobian_reference"
        raise ValueError(msg)
    if derivatives is not None:
        artifact_names = tuple(derivatives.state_names)
        if len(artifact_names) != n:
            msg = "derivative artifact state layout differs from the candidate"
            raise ValueError(msg)
        if coordinates is not None and tuple(coordinates) != artifact_names:
            msg = "coordinates must exactly match derivative artifact state_names"
            raise ValueError(msg)
        model_digest = getattr(derivatives.model, "digest", None)
        if not isinstance(model_digest, str) or not model_digest.startswith("sha256:"):
            msg = "derivative artifact lacks a structural model digest"
            raise ValueError(msg)
        return (
            StabilityJacobianSource.OP_SYSTEM_SYMBOLIC,
            _coordinate_names(artifact_names, n),
            {} if parameters is None else parameters,
            model_digest,
        )
    source = (
        StabilityJacobianSource.PROVIDED
        if jacobian is not None
        else StabilityJacobianSource.FINITE_DIFFERENCE
    )
    return source, _coordinate_names(coordinates, n), {}, None


def local_stability(  # noqa: C901,PLR0911,PLR0912,PLR0913,PLR0914,PLR0915
    rhs: StabilityRhs,
    candidate: NumpyArray | SteadyStateResult,
    *,
    config: LocalStabilityConfig,
    t: float = 0.0,
    coordinates: Sequence[str] | None = None,
    coordinate_system: str = "supplied state",
    fixed: Sequence[int] | NumpyArray | None = None,
    invariants: ArrayLike | None = None,
    nonnegative: bool | Sequence[bool] | NumpyArray = False,
    derivatives: SymbolicDerivativeArtifact | None = None,
    parameters: Mapping[str, object] | None = None,
    jacobian: StabilityJacobian | None = None,
    jacobian_reference: str | None = None,
    time_unit: str | None = None,
) -> LocalStabilityResult:
    """Assess a candidate equilibrium through an eager NumPy linearization.

    The classification applies only to an interior, residual-qualified point
    and, when constraints are declared, only to their tangent manifold.  A
    boundary status deliberately suppresses an interior stability conclusion.

    Args:
        rhs: Dense NumPy right-hand side in the supplied coordinates.
        candidate: NumPy state or NumPy-backed steady-state result.
        config: Explicit residual, spectral, boundary and difference controls.
        t: Frozen evaluation time. A time-dependent RHS is only diagnosed there.
        coordinates: Ordered state labels. An op_system artifact supplies these.
        coordinate_system: Human-readable coordinate-system declaration.
        fixed: Coordinates held fixed, as indices or a boolean mask.
        invariants: Declared invariant rows in the supplied coordinates.
        nonnegative: Coordinates with a one-sided non-negative domain.
        derivatives: Optional op_system symbolic derivative artifact.
        parameters: Exact runtime bindings for ``derivatives``.
        jacobian: Optional dense full-state Jacobian callback.
        jacobian_reference: Required provenance label for ``jacobian``.
        time_unit: Unit attached to any reported e-folding timescale.

    Returns:
        Partial or complete numerical evidence with an explicit status.

    Raises:
        TypeError: The state, solver flag or configuration is not eager NumPy.
        ValueError: Input declarations, shapes, dtypes or controls conflict.
    """
    if not isinstance(config, LocalStabilityConfig):
        msg = "config must be an explicit LocalStabilityConfig"
        raise TypeError(msg)
    solver_converged: bool | None = None
    state_value: object = candidate
    if isinstance(candidate, SteadyStateResult):
        state_value = candidate.state
    if not isinstance(state_value, np.ndarray):
        msg = "local_stability requires an eager NumPy state; transfer explicitly"
        raise TypeError(msg)
    if isinstance(candidate, SteadyStateResult):
        if not isinstance(candidate.converged, np.ndarray):
            msg = "local_stability requires an eager NumPy convergence flag"
            raise TypeError(msg)
        if candidate.converged.shape != ():
            msg = "SteadyStateResult.converged must be a scalar array"
            raise ValueError(msg)
        solver_converged = bool(candidate.converged.item())
    state = state_value
    if state.ndim != 1 or state.size < 1:
        msg = f"candidate must be a non-empty vector; got shape {state.shape}"
        raise ValueError(msg)
    if state.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        msg = "candidate must have NumPy float32 or float64 dtype"
        raise TypeError(msg)
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
        msg = "t must be a finite real scalar"
        raise ValueError(msg)
    if not isinstance(coordinate_system, str) or not coordinate_system:
        msg = "coordinate_system must be a non-empty string"
        raise ValueError(msg)
    if time_unit is not None and (not isinstance(time_unit, str) or not time_unit):
        msg = "time_unit must be None or a non-empty string"
        raise ValueError(msg)
    n = state.size
    residual_tolerance = _vector_control(
        config.residual_tolerance, n, "residual_tolerance", positive=True
    )
    boundary_tolerance = _vector_control(
        config.boundary_tolerance, n, "boundary_tolerance", positive=False
    )
    fixed_mask = _selection_mask(fixed, n, "fixed")
    nonnegative_mask = _nonnegative_mask(nonnegative, n)
    invariant_rows = _invariant_rows(invariants, n)
    source, names, bound_parameters, model_digest = _source_and_coordinates(
        derivatives,
        jacobian,
        parameters,
        coordinates,
        n,
        jacobian_reference,
    )

    evidence: dict[str, Any] = {
        "state": state.copy(),
        "time": float(t),
        "coordinates": names,
        "coordinate_system": coordinate_system,
        "time_unit": time_unit,
        "jacobian_source": source,
        "jacobian_reference": jacobian_reference,
        "model_digest": model_digest,
        "solver_converged": solver_converged,
        "residual_tolerance": residual_tolerance,
        "boundary_tolerance": boundary_tolerance,
        "fixed_mask": fixed_mask,
        "nonnegative_mask": nonnegative_mask,
        "invariant_rows": invariant_rows,
    }

    def result(
        status: LocalStabilityStatus,
        message: str,
        classification: LocalSpectrum | None = None,
        **updates: object,
    ) -> LocalStabilityResult:
        return LocalStabilityResult(
            status=status,
            classification=classification,
            message=message,
            **(evidence | updates),
        )

    if not np.isfinite(state).all():
        return result(LocalStabilityStatus.NONFINITE_STATE, "candidate is non-finite")
    infeasible = nonnegative_mask & (state < -boundary_tolerance)
    if np.any(infeasible):
        return result(
            LocalStabilityStatus.INFEASIBLE,
            "candidate violates a declared non-negative coordinate",
            domain=EquilibriumDomain.INFEASIBLE,
        )
    try:
        raw_residual = _strict_numpy_vector(rhs(float(t), state), state, "rhs")
    except (
        ArithmeticError,
        AttributeError,
        LookupError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return result(
            LocalStabilityStatus.RESIDUAL_FAILURE,
            f"RHS evaluation failed: {type(error).__name__}: {error}",
        )
    evidence["raw_residual"] = raw_residual
    active = ~fixed_mask
    active_residual = raw_residual[active]
    evidence["active_residual"] = active_residual
    if not np.isfinite(raw_residual).all():
        return result(
            LocalStabilityStatus.NONFINITE_RESIDUAL,
            "candidate residual is non-finite",
        )
    if active_residual.size:
        ratios = active_residual / residual_tolerance[active]
        residual_scaled_rms = float(np.sqrt(np.mean(ratios * ratios)))
    else:
        residual_scaled_rms = 0.0
    evidence["residual_scaled_rms"] = residual_scaled_rms
    if solver_converged is False:
        return result(
            LocalStabilityStatus.SOLVE_FAILURE,
            "input SteadyStateResult did not converge",
        )
    if residual_scaled_rms > 1:
        return result(
            LocalStabilityStatus.NOT_EQUILIBRIUM,
            "active residual exceeds its declared tolerance",
        )

    constrained = nonnegative_mask & active
    boundary = constrained & (np.abs(state) <= boundary_tolerance)
    boundary_coordinates = tuple(
        name for name, hit in zip(names, boundary, strict=True) if hit
    )
    evidence["boundary_coordinates"] = boundary_coordinates
    domain = (
        EquilibriumDomain.BOUNDARY
        if np.any(boundary)
        else EquilibriumDomain.INTERIOR
        if np.any(constrained)
        else EquilibriumDomain.UNSPECIFIED
    )
    evidence["domain"] = domain
    if (
        domain is EquilibriumDomain.BOUNDARY
        and source is StabilityJacobianSource.FINITE_DIFFERENCE
    ):
        return result(
            LocalStabilityStatus.BOUNDARY,
            "central finite differences are not evaluated across a declared boundary",
        )

    try:
        finite_difference_steps = None
        if derivatives is not None:
            values = derivatives.evaluate(float(t), state, **bound_parameters)
            matrix = _strict_numpy_matrix(
                values.state, state, "symbolic derivative artifact"
            )
        elif jacobian is not None:
            matrix = _strict_numpy_matrix(jacobian(float(t), state), state, "jacobian")
        else:
            matrix, finite_difference_steps = _finite_difference_jacobian(
                rhs, float(t), state, config.finite_difference_step
            )
    except (
        ArithmeticError,
        AttributeError,
        LookupError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return result(
            LocalStabilityStatus.JACOBIAN_FAILURE,
            f"Jacobian evaluation failed: {type(error).__name__}: {error}",
        )
    evidence["finite_difference_steps"] = finite_difference_steps
    evidence["jacobian"] = matrix
    if not np.isfinite(matrix).all():
        return result(
            LocalStabilityStatus.NONFINITE_JACOBIAN,
            "Jacobian contains non-finite entries",
        )

    free = np.flatnonzero(active)
    active_jacobian = matrix[np.ix_(free, free)]
    evidence["active_jacobian"] = active_jacobian
    try:
        full_eigenvalues = np.linalg.eigvals(matrix)
        active_eigenvalues = np.linalg.eigvals(active_jacobian)
    except np.linalg.LinAlgError as error:
        return result(
            LocalStabilityStatus.EIGENSOLVE_FAILED,
            f"full or active eigensolve failed: {error}",
        )
    evidence["full_eigenvalues"] = full_eigenvalues
    evidence["active_eigenvalues"] = active_eigenvalues
    if (
        not np.isfinite(full_eigenvalues).all()
        or not np.isfinite(active_eigenvalues).all()
    ):
        return result(
            LocalStabilityStatus.NONFINITE_EIGENVALUES,
            "full or active eigenspectrum is non-finite",
        )

    try:
        tangent_basis, constraint_rank, constraint_svd_tolerance = _constraint_basis(
            fixed_mask, invariant_rows
        )
    except (np.linalg.LinAlgError, ValueError) as error:
        return result(
            LocalStabilityStatus.INCONSISTENT_INVARIANTS,
            f"constraint reduction failed: {error}",
        )
    evidence["constraint_rank"] = constraint_rank
    evidence["constraint_svd_tolerance"] = constraint_svd_tolerance
    evidence["tangent_basis"] = tangent_basis

    invariant_rank = invariant_rows.shape[0]
    evidence["invariant_rank"] = invariant_rank
    evidence["expected_conservation_neutral_modes"] = invariant_rank
    if tangent_basis.shape[1] == 0:
        evidence["observed_active_neutral_modes"] = 0
        return result(
            LocalStabilityStatus.NO_DYNAMIC_DIRECTIONS,
            "declared constraints leave no dynamic tangent directions",
        )
    active_spectral_tolerance = (
        config.spectral_absolute_tolerance
        + config.spectral_relative_tolerance * float(np.max(np.abs(active_eigenvalues)))
    )
    evidence["observed_active_neutral_modes"] = int(
        np.count_nonzero(np.abs(active_eigenvalues) <= active_spectral_tolerance)
    )
    if invariant_rank:
        invariant_free = invariant_rows[:, active]
        try:
            active_jacobian_scale = float(np.linalg.norm(active_jacobian, ord=2))
            conservation_error = float(
                np.linalg.norm(invariant_free @ active_jacobian, ord=2)
            )
        except np.linalg.LinAlgError as error:
            return result(
                LocalStabilityStatus.INCONSISTENT_INVARIANTS,
                f"invariant norm evaluation failed: {error}",
            )
        conservation_tolerance = (
            config.spectral_absolute_tolerance
            + config.spectral_relative_tolerance * active_jacobian_scale
        )
        invariant_flow_error = float(np.max(np.abs(invariant_free @ active_residual)))
        invariant_flow_tolerance = float(
            np.max(np.abs(invariant_free) @ residual_tolerance[active])
        )
        evidence.update({
            "conservation_error": conservation_error,
            "conservation_tolerance": conservation_tolerance,
            "invariant_flow_error": invariant_flow_error,
            "invariant_flow_tolerance": invariant_flow_tolerance,
        })
        if conservation_error > conservation_tolerance:
            return result(
                LocalStabilityStatus.INCONSISTENT_INVARIANTS,
                "declared invariant rows do not annihilate the local Jacobian",
            )

    tangent_jacobian = tangent_basis.T @ matrix.astype(np.float64) @ tangent_basis
    evidence["tangent_jacobian"] = tangent_jacobian
    try:
        tangent_eigenvalues = np.linalg.eigvals(tangent_jacobian)
    except np.linalg.LinAlgError as error:
        return result(
            LocalStabilityStatus.EIGENSOLVE_FAILED,
            f"tangent eigensolve failed: {error}",
        )
    evidence["tangent_eigenvalues"] = tangent_eigenvalues
    if not np.isfinite(tangent_eigenvalues).all():
        return result(
            LocalStabilityStatus.NONFINITE_EIGENVALUES,
            "tangent eigenspectrum is non-finite",
        )
    effective_tolerance = (
        config.spectral_absolute_tolerance
        + config.spectral_relative_tolerance
        * float(np.max(np.abs(tangent_eigenvalues)))
    )
    real_parts = np.real(tangent_eigenvalues)
    spectral_abscissa = float(np.max(real_parts))
    evidence["effective_spectral_tolerance"] = effective_tolerance
    evidence["spectral_abscissa"] = spectral_abscissa
    if domain is EquilibriumDomain.BOUNDARY:
        return result(
            LocalStabilityStatus.BOUNDARY,
            "active boundary requires a one-sided tangent-cone analysis",
        )
    if np.any(real_parts > effective_tolerance):
        classification = LocalSpectrum.UNSTABLE
        leading_index = int(np.argmax(real_parts))
        leading_eigenvalue = complex(tangent_eigenvalues[leading_index])
        evidence.update({
            "leading_eigenvalue": leading_eigenvalue,
            "leading_timescale": 1 / spectral_abscissa,
            "timescale_kind": StabilityTimescaleKind.GROWTH,
        })
    elif np.all(real_parts < -effective_tolerance):
        classification = LocalSpectrum.ASYMPTOTICALLY_STABLE
        leading_index = int(np.argmax(real_parts))
        leading_eigenvalue = complex(tangent_eigenvalues[leading_index])
        evidence.update({
            "leading_eigenvalue": leading_eigenvalue,
            "leading_timescale": -1 / spectral_abscissa,
            "timescale_kind": StabilityTimescaleKind.DECAY,
        })
    elif np.all(np.abs(real_parts) <= effective_tolerance):
        classification = LocalSpectrum.NEUTRAL_SPECTRUM
    else:
        classification = LocalSpectrum.NONHYPERBOLIC
    return result(
        LocalStabilityStatus.ANALYZED,
        (
            "classification is local to the declared constraint manifold"
            if invariant_rank or np.any(fixed_mask)
            else "classification is a local interior linearization"
        ),
        classification,
    )


__all__ = [
    "EquilibriumDomain",
    "LocalSpectrum",
    "LocalStabilityConfig",
    "LocalStabilityResult",
    "LocalStabilityStatus",
    "StabilityJacobianSource",
    "StabilityTimescaleKind",
    "SymbolicDerivativeArtifact",
    "local_stability",
]
