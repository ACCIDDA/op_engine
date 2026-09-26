"""Backend-neutral contracts for nonlinear stage solves.

The portable dense implementation intentionally performs a fixed number of
Newton iterations. Static iteration counts keep the numerical loop traceable
by array systems such as JAX; convergence is reported as array-valued
diagnostics and is never converted to a host boolean inside the solve.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from ._typing import Array

NonlinearResidual = Callable[[Array], Array]
NonlinearJacobian = Callable[[Array], Array]
JacobianVectorProduct = Callable[[Array, Array], Array]


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace advertised by ``value``.

    Raises:
        TypeError: If ``value`` is not an Array-API array.
    """
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        msg = f"Nonlinear solves require Array-API arrays; got {type(value).__name__}."
        raise TypeError(msg)
    return namespace()


@dataclass(slots=True, frozen=True)
class NonlinearProblem:
    """Residual and optional derivative actions for a nonlinear equation.

    ``residual(x)`` must return an array with the same shape, dtype, and array
    namespace as ``x``. ``jacobian(x)`` returns a dense square array acting on
    the flattened residual. ``jvp(x, vector)`` returns ``J(x) @ vector`` with
    the same structure as ``x``. Solvers may require either derivative form;
    :class:`DenseNewtonSolver` specifically requires ``jacobian``.

    Attributes:
        residual: Residual function whose root is sought.
        jacobian: Optional dense Jacobian function.
        jvp: Optional matrix-free Jacobian-vector product.
    """

    residual: NonlinearResidual
    jacobian: NonlinearJacobian | None = None
    jvp: JacobianVectorProduct | None = None

    def __post_init__(self) -> None:
        """Validate callback presence.

        Raises:
            TypeError: If a supplied callback is not callable.
        """
        for name, callback in (
            ("residual", self.residual),
            ("jacobian", self.jacobian),
            ("jvp", self.jvp),
        ):
            if callback is not None and not callable(callback):
                msg = f"Nonlinear problem {name} must be callable"
                raise TypeError(msg)


@dataclass(slots=True, frozen=True)
class NewtonConfig:
    """Static configuration for the portable dense Newton solver.

    Attributes:
        max_iterations: Exact number of Newton updates to unroll.
        rtol: Relative convergence tolerance against the initial residual norm.
        atol: Absolute convergence tolerance.
        damping: Fixed multiplier applied to every Newton update.
    """

    max_iterations: int = 8
    rtol: float = 1e-8
    atol: float = 1e-10
    damping: float = 1.0

    def __post_init__(self) -> None:
        """Validate static solver parameters.

        Raises:
            ValueError: If an iteration count or tolerance is invalid.
        """
        if (
            not isinstance(self.max_iterations, Integral)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            msg = "max_iterations must be a positive integer"
            raise ValueError(msg)
        if not np.isfinite(self.rtol) or self.rtol < 0.0:
            msg = "rtol must be finite and non-negative"
            raise ValueError(msg)
        if not np.isfinite(self.atol) or self.atol < 0.0:
            msg = "atol must be finite and non-negative"
            raise ValueError(msg)
        if not np.isfinite(self.damping) or not (0.0 < self.damping <= 1.0):
            msg = "damping must be finite and in (0, 1]"
            raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class NonlinearSolveDiagnostics:
    """Array-safe diagnostics from a nonlinear solve.

    ``converged`` and the norm fields are zero-dimensional arrays in the
    initial guess's namespace. Keeping them on-device avoids an accidental
    host synchronization or tracer conversion in compiled code.

    Attributes:
        converged: Boolean array indicating finite residual convergence.
        iterations: Number of nonlinear updates performed.
        residual_evaluations: Number of residual evaluations performed.
        jacobian_evaluations: Number of dense Jacobian evaluations performed.
        initial_residual_norm: RMS norm before the first update.
        residual_norm: RMS norm after the final update.
        step_norm: RMS norm of the final damped update.
    """

    converged: Array
    iterations: int
    residual_evaluations: int
    jacobian_evaluations: int
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array


@dataclass(slots=True, frozen=True)
class NonlinearSolveResult:
    """Candidate root, final residual, and convergence diagnostics."""

    value: Array
    residual: Array
    diagnostics: NonlinearSolveDiagnostics


@runtime_checkable
class NonlinearSolver(Protocol):
    """Backend-neutral interface implemented by nonlinear solvers."""

    def solve(
        self,
        problem: NonlinearProblem,
        initial_guess: Array,
        /,
    ) -> NonlinearSolveResult:
        """Return a candidate root and explicit convergence diagnostics."""


def _validate_like(value: Array, reference: Array, *, label: str) -> None:
    """Validate a callback result against its iterate.

    Raises:
        TypeError: If array namespace or dtype changes.
        ValueError: If array shape changes.
    """
    expected_namespace = _namespace_of(reference)
    actual_namespace = _namespace_of(value)
    if actual_namespace is not expected_namespace:
        msg = f"{label} must preserve the iterate array namespace"
        raise TypeError(msg)
    if value.shape != reference.shape:
        msg = (
            f"{label} shape {value.shape} does not match iterate shape "
            f"{reference.shape}"
        )
        raise ValueError(msg)
    if value.dtype != reference.dtype:
        msg = f"{label} must preserve the iterate dtype"
        raise TypeError(msg)


def _rms_norm(value: Array) -> Array:
    """Return the RMS norm as a scalar in ``value``'s namespace."""
    xp = _namespace_of(value)
    magnitude = xp.abs(value)
    squared = xp.multiply(magnitude, magnitude)
    return cast("Array", xp.sqrt(xp.mean(squared)))


@dataclass(slots=True, frozen=True)
class DenseNewtonSolver:
    """Portable dense Newton solver with statically unrolled iterations.

    The solver has no hidden warm-start state. Every call begins from the
    explicit ``initial_guess`` supplied by the numerical method. Automatic
    differentiation follows the performed Newton iterations; this class does
    not install an implicit-function or custom derivative rule.

    Attributes:
        config: Static iteration, tolerance, and damping configuration.
    """

    config: NewtonConfig = NewtonConfig()

    def solve(  # noqa: PLR0914
        self,
        problem: NonlinearProblem,
        initial_guess: Array,
        /,
    ) -> NonlinearSolveResult:
        """Run a fixed number of dense Newton updates.

        Args:
            problem: Residual and dense Jacobian callbacks.
            initial_guess: Explicit warm start in the desired array namespace.

        Returns:
            Candidate root with array-valued convergence diagnostics.

        Raises:
            TypeError: If callbacks change namespace or dtype.
            ValueError: If no dense Jacobian is supplied or shapes are invalid.
        """
        jacobian = problem.jacobian
        if jacobian is None:
            msg = "DenseNewtonSolver requires NonlinearProblem.jacobian"
            raise ValueError(msg)

        xp = _namespace_of(initial_guess)
        n_unknowns = math.prod(initial_guess.shape)
        if n_unknowns < 1:
            msg = "initial_guess must contain at least one value"
            raise ValueError(msg)
        if not (
            xp.isdtype(initial_guess.dtype, "real floating")
            or xp.isdtype(initial_guess.dtype, "complex floating")
        ):
            msg = "initial_guess must have a floating-point dtype"
            raise TypeError(msg)

        iterate = cast(
            "Array",
            xp.asarray(initial_guess, dtype=initial_guess.dtype),
        )
        residual = problem.residual(iterate)
        _validate_like(residual, iterate, label="residual")
        initial_residual_norm = _rms_norm(residual)
        applied_step = cast("Array", xp.zeros_like(iterate))

        for _ in range(self.config.max_iterations):
            jacobian_value = jacobian(iterate)
            if _namespace_of(jacobian_value) is not xp:
                msg = "jacobian must preserve the iterate array namespace"
                raise TypeError(msg)
            expected_jacobian_shape = (n_unknowns, n_unknowns)
            if jacobian_value.shape != expected_jacobian_shape:
                msg = (
                    f"jacobian shape {jacobian_value.shape} does not match "
                    f"flattened system shape {expected_jacobian_shape}"
                )
                raise ValueError(msg)
            if jacobian_value.dtype != iterate.dtype:
                msg = "jacobian must preserve the iterate dtype"
                raise TypeError(msg)

            residual_flat = xp.reshape(residual, (n_unknowns,))
            delta_flat = xp.linalg.solve(
                jacobian_value,
                xp.negative(residual_flat),
            )
            delta = xp.reshape(delta_flat, iterate.shape)
            applied_step = cast(
                "Array",
                xp.multiply(delta, self.config.damping),
            )
            iterate = cast("Array", xp.add(iterate, applied_step))
            residual = problem.residual(iterate)
            _validate_like(residual, iterate, label="residual")

        residual_norm = _rms_norm(residual)
        step_norm = _rms_norm(applied_step)
        tolerance = xp.add(
            xp.asarray(self.config.atol, dtype=residual_norm.dtype),
            xp.multiply(initial_residual_norm, self.config.rtol),
        )
        finite = xp.all(xp.isfinite(residual))
        converged = cast(
            "Array",
            xp.logical_and(finite, xp.less_equal(residual_norm, tolerance)),
        )
        diagnostics = NonlinearSolveDiagnostics(
            converged=converged,
            iterations=self.config.max_iterations,
            residual_evaluations=self.config.max_iterations + 1,
            jacobian_evaluations=self.config.max_iterations,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            step_norm=step_norm,
        )
        return NonlinearSolveResult(iterate, residual, diagnostics)


class NonlinearConvergenceError(RuntimeError):
    """Raised by an eager boundary when a nonlinear solve did not converge."""

    def __init__(self, result: NonlinearSolveResult) -> None:
        """Store the failed result and its diagnostics."""
        self.result = result
        diagnostics = result.diagnostics
        super().__init__(
            "Nonlinear solve did not converge after "
            f"{diagnostics.iterations} iterations; "
            f"residual RMS={diagnostics.residual_norm.item()!r}"
        )


def require_converged(result: NonlinearSolveResult) -> NonlinearSolveResult:
    """Return a converged result or raise at an eager host boundary.

    This function intentionally converts the scalar convergence array to a
    Python boolean. Do not call it from ``jax.jit`` or another traced region;
    compiled integrations must return diagnostics and validate them afterward.

    Args:
        result: Result to validate.

    Returns:
        The unchanged converged result.

    Raises:
        NonlinearConvergenceError: If the result did not converge.
    """
    if not bool(result.diagnostics.converged.item()):
        raise NonlinearConvergenceError(result)
    return result


__all__ = [
    "DenseNewtonSolver",
    "JacobianVectorProduct",
    "NewtonConfig",
    "NonlinearConvergenceError",
    "NonlinearJacobian",
    "NonlinearProblem",
    "NonlinearResidual",
    "NonlinearSolveDiagnostics",
    "NonlinearSolveResult",
    "NonlinearSolver",
    "require_converged",
]
