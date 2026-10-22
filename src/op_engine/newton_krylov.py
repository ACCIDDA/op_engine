"""Explicit eager NumPy Newton-GMRES with checked residuals and backtracking.

This implementation has data-dependent host decisions and supplies no automatic
differentiation rule. It never constructs a dense Jacobian or probes a basis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum
from numbers import Integral
from typing import TYPE_CHECKING, cast

import numpy as np

from ._array import array_namespace
from .linear_operator import (
    ActionOperator,
    LinearSolveOptions,
    LinearSolver,
    LinearSolveRequest,
    solve_linear,
)
from .nonlinear_solver import (
    NewtonConfig,
    NonlinearProblem,
    NonlinearSolveDiagnostics,
    NonlinearSolveResult,
    NonlinearStageContext,
    _validate_like,
)
from .scipy_operator import ScipyGmresSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._typing import Array
    from .linear_operator import LinearSolveDiagnostics


class NewtonKrylovTermination(IntEnum):
    """Eager termination codes, retained in stage/integration diagnostics."""

    CONVERGED = 1
    MAX_ITERATIONS = 2
    LINEAR_FAILURE = 3
    LINE_SEARCH_FAILURE = 4
    NONFINITE_RESIDUAL = 5


@dataclass(frozen=True, slots=True, eq=False)
class NewtonLinearization:
    """Unique per-linearization preconditioner context, with an owned iterate.

    ``iterate`` is a read-only snapshot. Its full shape, dtype and coefficients,
    ``problem_context`` (including SDIRK scale/time when applicable) and this
    record's identity define the preparation lifetime. A factory is invoked
    afresh for every linearization; a caller retaining a cache owns invalidation.
    No iterate history is retained in solve diagnostics.
    """

    iteration: int
    iterate: NDArray[np.floating]
    problem_context: object | None


InversePreconditionerFactory = Callable[[LinearSolveRequest], ActionOperator | None]


@dataclass(frozen=True, slots=True)
class NewtonKrylovConfig(NewtonConfig):
    """Outer RMS tolerances, inner true-residual gate and Armijo backtracking.

    Inherited ``max_iterations`` is a maximum number of applied Newton updates,
    not an exact unroll count. The outer gate is ``atol + rtol * initial_rms``.
    Inner options independently use the checked linear solver's L2 gate
    ``max(atol, rtol * ||rhs||)`` and its documented iteration budget.

    Attributes:
        linear_options: Explicit inner tolerance and iteration budget.
        max_backtracks: Maximum reductions after the initial damped trial.
        backtrack_factor: Multiplicative step-length reduction, strictly in (0,1).
        armijo: Required fractional RMS decrease per unit step, strictly in (0,1).
    """

    linear_options: LinearSolveOptions = field(
        default_factory=lambda: LinearSolveOptions(rtol=1e-6, atol=0.0)
    )
    max_backtracks: int = 12
    backtrack_factor: float = 0.5
    armijo: float = 1e-4

    def __post_init__(self) -> None:
        """Validate outer, inner and globalization settings.

        Raises:
            TypeError: If inner options do not use the existing checked contract.
            ValueError: If backtracking settings are invalid.
        """
        NewtonConfig.__post_init__(self)
        if not isinstance(self.linear_options, LinearSolveOptions):
            msg = "linear_options must be LinearSolveOptions"
            raise TypeError(msg)
        if (
            not isinstance(self.max_backtracks, Integral)
            or isinstance(self.max_backtracks, bool)
            or self.max_backtracks < 0
        ):
            msg = "max_backtracks must be a nonnegative integer"
            raise ValueError(msg)
        if any(
            not np.isfinite(value) or not 0.0 < value < 1.0
            for value in (self.backtrack_factor, self.armijo)
        ):
            msg = "backtrack_factor and armijo must be finite and strictly in (0,1)"
            raise ValueError(msg)


def _rms(value: NDArray[np.floating]) -> float:
    """Compute a scaled RMS without overflowing a sum of squares.

    Returns:
        RMS in host double precision.
    """
    magnitude = np.abs(value.astype(np.float64)).ravel()
    scale = float(np.max(magnitude))
    if scale == 0.0 or not np.isfinite(scale):
        return scale
    return scale * float(np.sqrt(np.mean(np.square(magnitude / scale))))


@dataclass(slots=True)
class _Work:
    """One solve's local counters; no state escapes through the solver object."""

    residuals: int = 0
    actions: int = 0
    iterations: int = 0
    backtracks: int = 0
    linear: list[LinearSolveDiagnostics] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class NewtonKrylovSolver:
    """Stateless eager NumPy Newton-GMRES with a per-linearization inverse factory.

    The default inner solver is :class:`ScipyGmresSolver` (left inverse
    preconditioning). A failed checked inner solve returns nonlinear
    nonconvergence without applying its candidate; malformed callbacks and
    arbitrary linear-solver exceptions remain fatal. The integration boundary
    applies its existing fixed/adaptive/replay failure policy.

    Attributes:
        config: Outer/inner tolerances and bounded line search.
        linear_solver: Existing checked eager linear-solver callback.
        preconditioner_factory: Optional factory receiving the exact current
            request, operator and :class:`NewtonLinearization` context.
    """

    config: NewtonKrylovConfig = NewtonKrylovConfig()
    linear_solver: LinearSolver = field(default_factory=ScipyGmresSolver)
    preconditioner_factory: InversePreconditionerFactory | None = None

    def __post_init__(self) -> None:
        """Validate explicit configuration and callbacks.

        Raises:
            TypeError: If configuration or callbacks are unsupported.
        """
        if not isinstance(self.config, NewtonKrylovConfig):
            msg = "config must be NewtonKrylovConfig"
            raise TypeError(msg)
        if not callable(self.linear_solver) or (
            self.preconditioner_factory is not None
            and not callable(self.preconditioner_factory)
        ):
            msg = "linear solver and preconditioner factory must be callable"
            raise TypeError(msg)

    @staticmethod
    def _residual(
        problem: NonlinearProblem, iterate: NDArray[np.floating], work: _Work
    ) -> NDArray[np.floating]:
        result = problem.residual(iterate)
        _validate_like(result, iterate, label="residual")
        work.residuals += 1
        # A callback may reuse its output while rejected trials are evaluated.
        owned = np.array(cast("NDArray[np.floating]", result), copy=True)
        owned.flags.writeable = False
        return owned

    def _direction(
        self,
        problem: NonlinearProblem,
        iterate: NDArray[np.floating],
        residual: NDArray[np.floating],
        work: _Work,
    ) -> NDArray[np.floating] | None:
        """Bind the exact current action and verify the inner true residual.

        Returns:
            Checked Newton direction, or None on numerical nonconvergence.

        Raises:
            TypeError: If the inverse factory returns an unsupported object.
        """
        snapshot = iterate.copy()
        snapshot.flags.writeable = False
        jvp = problem.jvp
        assert jvp is not None  # noqa: S101 - validated before any iteration

        def matvec(vector: Array) -> Array:
            direction = cast("NDArray[np.floating]", vector).reshape(snapshot.shape)
            result = jvp(snapshot, direction)
            _validate_like(result, snapshot, label="jvp result")
            work.actions += 1
            return cast("NDArray[np.floating]", result).reshape((snapshot.size,))

        operator = ActionOperator(
            (snapshot.size, snapshot.size),
            snapshot.dtype,
            array_namespace(snapshot),
            matvec,
        )
        request = LinearSolveRequest(
            operator,
            -residual.ravel(),
            self.config.linear_options,
            context=NewtonLinearization(work.iterations, snapshot, problem.context),
        )
        if self.preconditioner_factory is not None:
            inverse = self.preconditioner_factory(request)
            if inverse is not None and not isinstance(inverse, ActionOperator):
                msg = "preconditioner factory must return ActionOperator or None"
                raise TypeError(msg)
            request = replace(request, preconditioner=inverse)
        linear = solve_linear(request, self.linear_solver)
        work.linear.append(linear.diagnostics)
        if not np.all(linear.diagnostics.converged):
            return None
        return linear.value.reshape(iterate.shape)

    def _trial(  # noqa: PLR0913 - explicit current solve and globalization state
        self,
        problem: NonlinearProblem,
        iterate: NDArray[np.floating],
        direction: NDArray[np.floating],
        *,
        norm: float,
        tolerance: float,
        work: _Work,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """Accept a finite Armijo-decreasing trial or leave the iterate unchanged.

        Returns:
            Accepted iterate/residual pair, or None after exhausting the budget.
        """
        step = self.config.damping
        for trial in range(self.config.max_backtracks + 1):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = iterate + step * direction
            candidate.flags.writeable = False
            if np.isfinite(candidate).all():
                residual = self._residual(problem, candidate, work)
                candidate_norm = _rms(residual)
                if np.isfinite(candidate_norm) and (
                    candidate_norm <= tolerance
                    or candidate_norm <= (1.0 - self.config.armijo * step) * norm
                ):
                    return candidate, residual
            if trial < self.config.max_backtracks:
                step *= self.config.backtrack_factor
                work.backtracks += 1
        return None

    def solve(
        self, problem: NonlinearProblem, initial_guess: Array, /
    ) -> NonlinearSolveResult:
        """Return a root candidate and explicit eager termination evidence.

        Returns:
            Owned candidate and residual with checked inner/outer diagnostics.

        Raises:
            TypeError: If input or callback arrays are not native real NumPy
                float32/float64 with matching shape/dtype/namespace.
            ValueError: If the state is empty or no residual JVP is supplied.
        """
        real_dtypes = (np.float32, np.float64)
        if (
            not isinstance(initial_guess, np.ndarray)
            or initial_guess.dtype not in real_dtypes
        ):
            msg = "NewtonKrylovSolver requires eager NumPy float32/float64 arrays"
            raise TypeError(msg)
        if initial_guess.size < 1 or problem.jvp is None:
            msg = (
                "NewtonKrylovSolver requires a nonempty state and NonlinearProblem.jvp"
            )
            raise ValueError(msg)
        if not np.isfinite(initial_guess).all():
            msg = "initial_guess must be finite"
            raise ValueError(msg)
        iterate = initial_guess.copy()
        iterate.flags.writeable = False
        work = _Work()
        residual = self._residual(problem, iterate, work)
        initial_norm = _rms(residual)
        tolerance = self.config.atol + self.config.rtol * initial_norm
        step_norm = 0.0
        termination = NewtonKrylovTermination.MAX_ITERATIONS
        for _ in range(self.config.max_iterations):
            norm = _rms(residual)
            if not np.isfinite(norm):
                termination = NewtonKrylovTermination.NONFINITE_RESIDUAL
                break
            if norm <= tolerance:
                termination = NewtonKrylovTermination.CONVERGED
                break
            direction = self._direction(problem, iterate, residual, work)
            if direction is None:
                termination = NewtonKrylovTermination.LINEAR_FAILURE
                break
            trial = self._trial(
                problem, iterate, direction, norm=norm, tolerance=tolerance, work=work
            )
            if trial is None:
                termination = NewtonKrylovTermination.LINE_SEARCH_FAILURE
                break
            candidate, residual = trial
            step_norm = _rms(candidate - iterate)
            iterate = candidate
            work.iterations += 1
        norm = _rms(residual)
        converged = np.isfinite(norm) and norm <= tolerance
        if converged:
            termination = NewtonKrylovTermination.CONVERGED
        diagnostics = NonlinearSolveDiagnostics(
            converged=np.asarray(converged),
            iterations=work.iterations,
            residual_evaluations=work.residuals,
            jacobian_evaluations=0,
            initial_residual_norm=np.asarray(initial_norm, dtype=iterate.dtype),
            residual_norm=np.asarray(norm, dtype=iterate.dtype),
            step_norm=np.asarray(step_norm, dtype=iterate.dtype),
            linear_solves=tuple(work.linear),
            termination=int(termination),
            jvp_evaluations=work.actions,
            backtracks=work.backtracks,
        )
        return NonlinearSolveResult(iterate, residual, diagnostics)


__all__ = [
    "InversePreconditionerFactory",
    "NewtonKrylovConfig",
    "NewtonKrylovSolver",
    "NewtonKrylovTermination",
    "NewtonLinearization",
    "NonlinearStageContext",
]
