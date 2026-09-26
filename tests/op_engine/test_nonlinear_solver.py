"""Tests for the backend-neutral nonlinear solver contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    DenseNewtonSolver,
    NewtonConfig,
    NonlinearConvergenceError,
    NonlinearProblem,
    NonlinearSolver,
    require_converged,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_iterations": 0}, "positive integer"),
        ({"max_iterations": True}, "positive integer"),
        ({"rtol": -1.0}, "non-negative"),
        ({"atol": float("nan")}, "non-negative"),
        ({"damping": 0.0}, r"in \(0, 1]"),
        ({"damping": 1.1}, r"in \(0, 1]"),
    ],
)
def test_newton_config_rejects_invalid_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Static iteration policy is validated before numerical execution."""
    with pytest.raises(ValueError, match=match):
        NewtonConfig(**kwargs)  # type: ignore[arg-type]


def test_dense_newton_solves_scalar_root_and_exposes_diagnostics() -> None:
    """A scalar root converges with complete, array-valued diagnostics."""

    def residual(value: Array) -> Array:
        xp = value.__array_namespace__()
        return cast("Array", xp.subtract(xp.multiply(value, value), 2.0))

    def jacobian(value: Array) -> Array:
        xp = value.__array_namespace__()
        return cast("Array", xp.reshape(xp.multiply(value, 2.0), (1, 1)))

    solver = DenseNewtonSolver(NewtonConfig(max_iterations=6, atol=1e-14))
    result = solver.solve(
        NonlinearProblem(residual=residual, jacobian=jacobian),
        np.asarray([1.5], dtype=np.float64),
    )

    assert isinstance(solver, NonlinearSolver)
    assert result.value.__array_namespace__() is np
    np.testing.assert_allclose(result.value, np.sqrt(2.0), rtol=1e-13)
    assert bool(result.diagnostics.converged.item())
    assert result.diagnostics.iterations == 6
    assert result.diagnostics.residual_evaluations == 7
    assert result.diagnostics.jacobian_evaluations == 6
    assert float(result.diagnostics.residual_norm.item()) < 1e-13
    assert require_converged(result) is result


def test_dense_newton_solves_vector_root() -> None:
    """The dense default flattens a vector residual into one Newton system."""
    target = np.asarray([2.0, 3.0], dtype=np.float64)

    def residual(value: Array) -> Array:
        value_numpy = cast("FloatArray", value)
        return value_numpy * value_numpy - target

    def jacobian(value: Array) -> Array:
        return np.diag(2.0 * cast("FloatArray", value))

    result = DenseNewtonSolver().solve(
        NonlinearProblem(residual=residual, jacobian=jacobian),
        np.asarray([1.5, 2.0], dtype=np.float64),
    )

    np.testing.assert_allclose(result.value, np.sqrt(target), rtol=1e-13)
    assert bool(result.diagnostics.converged.item())


def test_nonconvergence_is_data_until_checked_at_eager_boundary() -> None:
    """Iteration exhaustion is inspectable and only raises when explicitly checked."""

    def residual(value: Array) -> Array:
        xp = value.__array_namespace__()
        return cast("Array", xp.subtract(xp.multiply(value, value), 2.0))

    def jacobian(value: Array) -> Array:
        xp = value.__array_namespace__()
        return cast("Array", xp.reshape(xp.multiply(value, 2.0), (1, 1)))

    result = DenseNewtonSolver(
        NewtonConfig(max_iterations=1, rtol=0.0, atol=1e-14)
    ).solve(
        NonlinearProblem(residual=residual, jacobian=jacobian),
        np.asarray([10.0], dtype=np.float64),
    )

    assert not bool(result.diagnostics.converged.item())
    with pytest.raises(NonlinearConvergenceError) as caught:
        require_converged(result)
    assert caught.value.result is result


def test_dense_newton_requires_dense_jacobian() -> None:
    """A JVP-only problem remains valid for custom solvers, not the dense default."""
    problem = NonlinearProblem(
        residual=lambda value: value,
        jvp=lambda _value, vector: vector,
    )

    with pytest.raises(ValueError, match=r"requires NonlinearProblem.jacobian"):
        DenseNewtonSolver().solve(problem, np.asarray([1.0]))


def test_dense_newton_requires_floating_initial_guess() -> None:
    """Dense Array-API linear solves reject integer nonlinear iterates early."""
    problem = NonlinearProblem(
        residual=lambda value: value,
        jacobian=lambda _value: np.asarray([[1]]),
    )

    with pytest.raises(TypeError, match="floating-point dtype"):
        DenseNewtonSolver().solve(problem, np.asarray([1], dtype=np.int64))


def test_dense_newton_validates_callback_shape_and_dtype() -> None:
    """Callbacks cannot silently change the numerical problem boundary."""
    initial = np.asarray([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="residual shape"):
        DenseNewtonSolver().solve(
            NonlinearProblem(
                residual=lambda _value: np.asarray([1.0]),
                jacobian=lambda _value: np.eye(2),
            ),
            initial,
        )

    with pytest.raises(TypeError, match=r"residual must preserve.*dtype"):
        DenseNewtonSolver().solve(
            NonlinearProblem(
                residual=lambda value: np.ones_like(value, dtype=np.float32),
                jacobian=lambda _value: np.eye(2),
            ),
            initial,
        )


def test_dense_newton_jax_jit_and_grad_unroll_iterations() -> None:
    """JAX traces and differentiates the fixed Newton iterations without host reads."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    solver = DenseNewtonSolver(NewtonConfig(max_iterations=6, rtol=1e-6, atol=1e-7))

    def solve(parameter: Array) -> tuple[Array, Array, Array]:
        def residual(value: Array) -> Array:
            return cast("Array", jnp.subtract(jnp.multiply(value, value), parameter))

        def jacobian(value: Array) -> Array:
            return cast("Array", jnp.reshape(jnp.multiply(value, 2.0), (1, 1)))

        result = solver.solve(
            NonlinearProblem(residual=residual, jacobian=jacobian),
            cast("Array", jnp.asarray([1.5], dtype=parameter.dtype)),
        )
        return (
            cast("Array", result.value[0]),
            result.diagnostics.converged,
            result.diagnostics.residual_norm,
        )

    parameter = jnp.asarray(2.0, dtype=jnp.float32)
    eager_value, eager_converged, _eager_norm = solve(parameter)
    compiled = jax.jit(solve)
    compiled_value, compiled_converged, compiled_norm = compiled(parameter)
    differentiated = jax.jit(jax.value_and_grad(lambda value: solve(value)[0]))
    value, gradient = differentiated(parameter)

    assert eager_value.__array_namespace__() is jnp
    assert bool(eager_converged.item())
    assert bool(compiled_converged.item())
    assert float(compiled_norm.item()) < 1e-6
    assert np.allclose(compiled_value, eager_value, rtol=1e-6)
    assert np.allclose(value, np.sqrt(2.0), rtol=1e-6)
    assert np.allclose(gradient, 1.0 / (2.0 * np.sqrt(2.0)), rtol=1e-5)
