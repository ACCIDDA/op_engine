"""Coefficient-driven prototype kernels for singly diagonal implicit RK methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import cast

from ._typing import Array
from .nonlinear_solver import (
    NonlinearProblem,
    NonlinearSolveDiagnostics,
    NonlinearSolver,
    NonlinearSolveResult,
    _namespace_of,
)

ImplicitRhs = Callable[[float, Array], Array]
ImplicitRhsJacobian = Callable[[float, Array], Array]


@dataclass(slots=True, frozen=True)
class SdirkTableau:
    """Validated coefficients for a singly diagonal implicit RK method.

    Attributes:
        name: Human-readable method name.
        a: Lower-triangular Butcher matrix in compact row form, including the
            shared nonzero diagonal coefficient.
        b: Accepted-state weights.
        c: Stage-time coefficients.
        order: Formal solution order.
        stiffly_accurate: Whether the accepted state equals the last stage.
    """

    name: str
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    stiffly_accurate: bool = False

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate dimensions, row consistency, and SDIRK structure.

        Raises:
            TypeError: If a boolean flag has the wrong type.
            ValueError: If coefficients or formal order are inconsistent.
        """
        n_stages = len(self.c)
        if not self.name.strip():
            msg = "SDIRK tableau name must not be empty"
            raise ValueError(msg)
        if n_stages < 1:
            msg = "SDIRK tableau must contain at least one stage"
            raise ValueError(msg)
        if len(self.a) != n_stages or len(self.b) != n_stages:
            msg = "SDIRK tableau a, b, and c dimensions must agree"
            raise ValueError(msg)
        if (
            not isinstance(self.order, Integral)
            or isinstance(self.order, bool)
            or self.order < 1
        ):
            msg = "SDIRK order must be a positive integer"
            raise ValueError(msg)
        if not isinstance(self.stiffly_accurate, bool):
            msg = "SDIRK stiffly_accurate flag must be boolean"
            raise TypeError(msg)

        coefficients = (*self.b, *self.c, *(value for row in self.a for value in row))
        if any(not math.isfinite(value) for value in coefficients):
            msg = "SDIRK coefficients must be finite"
            raise ValueError(msg)

        gamma = self.a[0][0] if self.a[0] else 0.0
        if gamma == 0.0:
            msg = "SDIRK diagonal coefficient must be nonzero"
            raise ValueError(msg)
        for stage, (row, stage_time) in enumerate(zip(self.a, self.c, strict=True)):
            if len(row) != stage + 1:
                msg = "SDIRK row i must contain exactly i + 1 coefficients"
                raise ValueError(msg)
            if not math.isclose(row[-1], gamma, rel_tol=1e-13, abs_tol=1e-15):
                msg = "SDIRK diagonal coefficients must be equal"
                raise ValueError(msg)
            if not math.isclose(
                math.fsum(row),
                stage_time,
                rel_tol=1e-13,
                abs_tol=1e-15,
            ):
                msg = "SDIRK stage coefficients must sum to c"
                raise ValueError(msg)
        if not math.isclose(math.fsum(self.b), 1.0, rel_tol=1e-13, abs_tol=1e-15):
            msg = "SDIRK solution weights must sum to one"
            raise ValueError(msg)
        if self.stiffly_accurate and (
            not math.isclose(self.c[-1], 1.0, rel_tol=0.0, abs_tol=1e-15)
            or any(
                not math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-15)
                for left, right in zip(self.a[-1], self.b, strict=True)
            )
        ):
            msg = "Stiffly accurate SDIRK tableau must end at c=1 with last row=b"
            raise ValueError(msg)

    @property
    def gamma(self) -> float:
        """Return the shared diagonal coefficient."""
        return self.a[0][0]

    @property
    def n_stages(self) -> int:
        """Return the number of implicit stages."""
        return len(self.c)


@dataclass(slots=True, frozen=True)
class SdirkStepResult:
    """One SDIRK step with all nonlinear-stage diagnostics."""

    state: Array
    converged: Array
    stage_results: tuple[NonlinearSolveResult, ...]

    @property
    def stage_diagnostics(self) -> tuple[NonlinearSolveDiagnostics, ...]:
        """Return diagnostics in deterministic stage order."""
        return tuple(result.diagnostics for result in self.stage_results)


@dataclass(slots=True, frozen=True)
class SdirkStepAttempt:
    """Step-doubling candidate, local error, and nonlinear diagnostics."""

    state: Array
    error: Array
    converged: Array
    controller_order: int
    stage_results: tuple[NonlinearSolveResult, ...]

    @property
    def stage_diagnostics(self) -> tuple[NonlinearSolveDiagnostics, ...]:
        """Return full, first-half, then second-half stage diagnostics."""
        return tuple(result.diagnostics for result in self.stage_results)


def _weighted_state(
    base: Array,
    dt: float,
    weights: Sequence[float],
    derivatives: Sequence[Array],
) -> Array:
    """Return ``base + dt * sum(weights[i] * derivatives[i])``."""
    xp = _namespace_of(base)
    result = base
    for weight, derivative in zip(weights, derivatives, strict=True):
        if weight != 0.0:
            result = cast(
                "Array",
                xp.add(result, xp.multiply(derivative, dt * weight)),
            )
    return result


def _evaluate_rhs(
    rhs: ImplicitRhs,
    t: float,
    state: Array,
) -> Array:
    """Evaluate and validate an SDIRK right-hand side.

    Returns:
        Derivative in ``state``'s namespace.

    Raises:
        TypeError: If namespace or dtype changes.
        ValueError: If the derivative shape is incorrect.
    """
    derivative = rhs(float(t), state)
    if _namespace_of(derivative) is not _namespace_of(state):
        msg = "SDIRK rhs must preserve the state array namespace"
        raise TypeError(msg)
    if derivative.shape != state.shape:
        msg = f"SDIRK rhs shape {derivative.shape} does not match state {state.shape}"
        raise ValueError(msg)
    if derivative.dtype != state.dtype:
        msg = "SDIRK rhs must preserve the state dtype"
        raise TypeError(msg)
    return derivative


def _make_stage_problem(  # noqa: PLR0913
    rhs: ImplicitRhs,
    rhs_jacobian: ImplicitRhsJacobian,
    *,
    base: Array,
    stage_t: float,
    dt_diagonal: float,
    identity: Array,
) -> NonlinearProblem:
    """Bind one stage's residual and Jacobian callbacks.

    Returns:
        Nonlinear problem for one implicit stage.
    """
    xp = _namespace_of(base)

    def residual(stage_state: Array) -> Array:
        stage_rhs = _evaluate_rhs(rhs, stage_t, stage_state)
        return cast(
            "Array",
            xp.subtract(
                xp.subtract(stage_state, base),
                xp.multiply(stage_rhs, dt_diagonal),
            ),
        )

    def jacobian(stage_state: Array) -> Array:
        rhs_jacobian_value = rhs_jacobian(float(stage_t), stage_state)
        if _namespace_of(rhs_jacobian_value) is not xp:
            msg = "SDIRK rhs_jacobian must preserve the state array namespace"
            raise TypeError(msg)
        if rhs_jacobian_value.shape != identity.shape:
            msg = (
                f"SDIRK rhs_jacobian shape {rhs_jacobian_value.shape} does not "
                f"match flattened system shape {identity.shape}"
            )
            raise ValueError(msg)
        if rhs_jacobian_value.dtype != base.dtype:
            msg = "SDIRK rhs_jacobian must preserve the state dtype"
            raise TypeError(msg)
        return cast(
            "Array",
            xp.subtract(
                identity,
                xp.multiply(rhs_jacobian_value, dt_diagonal),
            ),
        )

    return NonlinearProblem(residual=residual, jacobian=jacobian)


def evaluate_sdirk_step(  # noqa: PLR0913
    *,
    tableau: SdirkTableau,
    nonlinear_solver: NonlinearSolver,
    rhs: ImplicitRhs,
    rhs_jacobian: ImplicitRhsJacobian,
    t: float,
    dt: float,
    y: Array,
) -> SdirkStepResult:
    """Evaluate one SDIRK step without host convergence decisions.

    Each stage starts from its explicit lower-triangular predictor. The solver
    is stateless, so rejected attempts cannot leak a hidden warm start.

    Returns:
        Candidate state and per-stage nonlinear diagnostics.

    Raises:
        RuntimeError: If a validated tableau unexpectedly has no stages.
    """
    xp = _namespace_of(y)
    n_unknowns = math.prod(y.shape)
    identity = xp.eye(n_unknowns, dtype=y.dtype)
    derivatives: list[Array] = []
    stage_results: list[NonlinearSolveResult] = []
    converged: Array | None = None

    for stage_index, (row, stage_time) in enumerate(
        zip(tableau.a, tableau.c, strict=True)
    ):
        prior_weights = row[:stage_index]
        base = _weighted_state(y, dt, prior_weights, derivatives)
        stage_t = t + stage_time * dt
        diagonal = row[-1]

        stage_result = nonlinear_solver.solve(
            _make_stage_problem(
                rhs,
                rhs_jacobian,
                base=base,
                stage_t=stage_t,
                dt_diagonal=dt * diagonal,
                identity=identity,
            ),
            base,
        )
        stage_results.append(stage_result)
        converged = (
            stage_result.diagnostics.converged
            if converged is None
            else cast(
                "Array",
                xp.logical_and(converged, stage_result.diagnostics.converged),
            )
        )
        derivatives.append(_evaluate_rhs(rhs, stage_t, stage_result.value))

    if converged is None:
        msg = "SDIRK tableau unexpectedly contained no stages"
        raise RuntimeError(msg)
    state = _weighted_state(y, dt, tableau.b, derivatives)
    return SdirkStepResult(state, converged, tuple(stage_results))


def attempt_sdirk_step(  # noqa: PLR0913
    *,
    tableau: SdirkTableau,
    nonlinear_solver: NonlinearSolver,
    rhs: ImplicitRhs,
    rhs_jacobian: ImplicitRhsJacobian,
    t: float,
    dt: float,
    y: Array,
) -> SdirkStepAttempt:
    """Return a step-doubling SDIRK candidate and local-error estimate."""
    xp = _namespace_of(y)
    full = evaluate_sdirk_step(
        tableau=tableau,
        nonlinear_solver=nonlinear_solver,
        rhs=rhs,
        rhs_jacobian=rhs_jacobian,
        t=t,
        dt=dt,
        y=y,
    )
    half_dt = 0.5 * dt
    first_half = evaluate_sdirk_step(
        tableau=tableau,
        nonlinear_solver=nonlinear_solver,
        rhs=rhs,
        rhs_jacobian=rhs_jacobian,
        t=t,
        dt=half_dt,
        y=y,
    )
    second_half = evaluate_sdirk_step(
        tableau=tableau,
        nonlinear_solver=nonlinear_solver,
        rhs=rhs,
        rhs_jacobian=rhs_jacobian,
        t=t + half_dt,
        dt=half_dt,
        y=first_half.state,
    )
    error_scale = 1.0 / (2.0**tableau.order - 1.0)
    error = cast(
        "Array",
        xp.multiply(xp.subtract(second_half.state, full.state), error_scale),
    )
    converged = cast(
        "Array",
        xp.logical_and(
            full.converged,
            xp.logical_and(first_half.converged, second_half.converged),
        ),
    )
    all_stage_results = (
        *full.stage_results,
        *first_half.stage_results,
        *second_half.stage_results,
    )
    return SdirkStepAttempt(
        state=second_half.state,
        error=error,
        converged=converged,
        controller_order=tableau.order,
        stage_results=all_stage_results,
    )


_GAMMA = 1.0 - 1.0 / math.sqrt(2.0)
SDIRK2_ALEXANDER = SdirkTableau(
    name="Alexander SDIRK2",
    a=((_GAMMA,), (1.0 - _GAMMA, _GAMMA)),
    b=(1.0 - _GAMMA, _GAMMA),
    c=(_GAMMA, 1.0),
    order=2,
    stiffly_accurate=True,
)
