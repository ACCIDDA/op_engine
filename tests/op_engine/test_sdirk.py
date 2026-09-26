"""Tests for the coefficient-driven SDIRK prototype."""

# ruff: noqa: PLC2701

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import Array, DenseNewtonSolver, NewtonConfig
from op_engine._sdirk import (
    SDIRK2_ALEXANDER,
    SdirkTableau,
    attempt_sdirk_step,
    evaluate_sdirk_step,
)


def test_alexander_sdirk2_exposes_validated_metadata() -> None:
    """The prototype declares formal order and stiff accuracy explicitly."""
    tableau = SDIRK2_ALEXANDER
    assert tableau.order == 2
    assert tableau.n_stages == 2
    assert tableau.stiffly_accurate
    assert np.isclose(tableau.gamma, 1.0 - 1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(tableau.a[-1], tableau.b)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"a": ((0.25,),)}, "dimensions"),
        ({"a": ((0.25,), (0.5,))}, "i \\+ 1"),
        ({"a": ((0.25,), (0.5, 0.3))}, "diagonal"),
        ({"a": ((0.25,), (0.5, 0.25)), "c": (0.25, 0.8)}, "sum to c"),
        ({"b": (0.5, 0.4)}, "sum to one"),
        ({"stiffly_accurate": True}, "Stiffly accurate"),
    ],
)
def test_sdirk_tableau_rejects_inconsistent_coefficients(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Malformed implicit coefficient sets fail when declared."""
    defaults: dict[str, object] = {
        "name": "test",
        "a": ((0.25,), (0.75, 0.25)),
        "b": (0.5, 0.5),
        "c": (0.25, 1.0),
        "order": 2,
    }
    with pytest.raises((TypeError, ValueError), match=match):
        SdirkTableau(**(defaults | kwargs))  # type: ignore[arg-type]


def _integrate_nonlinear_decay(dt: float) -> float:
    solver = DenseNewtonSolver(NewtonConfig(max_iterations=6, atol=1e-13))
    state = np.asarray([1.0], dtype=np.float64)

    def rhs(_time: float, value: Array) -> Array:
        return cast("Array", np.negative(np.multiply(value, value)))

    def rhs_jacobian(_time: float, value: Array) -> Array:
        return cast("Array", np.reshape(np.multiply(value, -2.0), (1, 1)))

    time = 0.0
    while time < 1.0 - 0.5 * dt:
        result = evaluate_sdirk_step(
            tableau=SDIRK2_ALEXANDER,
            nonlinear_solver=solver,
            rhs=rhs,
            rhs_jacobian=rhs_jacobian,
            t=time,
            dt=dt,
            y=state,
        )
        assert all(
            bool(diagnostics.converged.item())
            for diagnostics in result.stage_diagnostics
        )
        state = cast("np.ndarray", result.state)
        time += dt
    return float(state[0])


def test_alexander_sdirk2_has_second_order_on_nonlinear_decay() -> None:
    """The prototype reaches its formal order on a nonlinear ODE."""
    errors = [abs(_integrate_nonlinear_decay(dt) - 0.5) for dt in (0.1, 0.05, 0.025)]
    orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(2)
    ]
    assert min(orders) > 1.8


def test_alexander_sdirk2_damps_a_very_stiff_mode() -> None:
    """The L-stable prototype damps a large negative scalar mode."""
    rate = -1000.0
    result = evaluate_sdirk_step(
        tableau=SDIRK2_ALEXANDER,
        nonlinear_solver=DenseNewtonSolver(),
        rhs=lambda _time, value: cast("Array", np.multiply(value, rate)),
        rhs_jacobian=lambda _time, _value: cast(
            "Array", np.asarray([[rate]], dtype=np.float64)
        ),
        t=0.0,
        dt=1.0,
        y=np.asarray([1.0]),
    )

    assert bool(result.converged.item())
    assert abs(float(result.state[0])) < 0.01


def test_step_doubling_exposes_all_stage_failures_without_host_conversion() -> None:
    """Adaptive policy receives one convergence scalar and all stage diagnostics."""
    attempt = attempt_sdirk_step(
        tableau=SDIRK2_ALEXANDER,
        nonlinear_solver=DenseNewtonSolver(
            NewtonConfig(max_iterations=1, rtol=0.0, atol=1e-14)
        ),
        rhs=lambda _time, value: cast("Array", np.negative(np.multiply(value, value))),
        rhs_jacobian=lambda _time, value: cast(
            "Array", np.reshape(np.multiply(value, -2.0), (1, 1))
        ),
        t=0.0,
        dt=1.0,
        y=np.asarray([10.0]),
    )

    assert not bool(attempt.converged.item())
    assert attempt.controller_order == 2
    assert len(attempt.stage_diagnostics) == 6
    assert attempt.error.shape == (1,)


def test_alexander_sdirk2_jax_jit_and_grad() -> None:
    """The staged nonlinear solves compile and differentiate by unrolling Newton."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    solver = DenseNewtonSolver(NewtonConfig(max_iterations=4, atol=1e-7, rtol=1e-6))

    def advance(rate: Array, initial: Array) -> tuple[Array, Array]:
        def rhs(_time: float, value: Array) -> Array:
            return cast("Array", jnp.multiply(value, rate))

        def rhs_jacobian(_time: float, _value: Array) -> Array:
            return cast("Array", jnp.reshape(rate, (1, 1)))

        result = evaluate_sdirk_step(
            tableau=SDIRK2_ALEXANDER,
            nonlinear_solver=solver,
            rhs=rhs,
            rhs_jacobian=rhs_jacobian,
            t=0.0,
            dt=0.2,
            y=cast("Array", jnp.reshape(initial, (1,))),
        )
        return cast("Array", result.state[0]), result.converged

    rate = jnp.asarray(-0.4, dtype=jnp.float32)
    initial = jnp.asarray(1.2, dtype=jnp.float32)
    compiled = jax.jit(advance)
    value, converged = compiled(rate, initial)
    differentiated = jax.jit(
        jax.value_and_grad(lambda r, y: advance(r, y)[0], argnums=(0, 1))
    )
    differentiated_value, gradients = differentiated(rate, initial)

    epsilon = 1e-3
    finite_rate = (
        advance(rate + epsilon, initial)[0] - advance(rate - epsilon, initial)[0]
    ) / (2.0 * epsilon)
    finite_initial = (
        advance(rate, initial + epsilon)[0] - advance(rate, initial - epsilon)[0]
    ) / (2.0 * epsilon)

    assert value.__array_namespace__() is jnp
    assert bool(converged.item())
    assert np.allclose(differentiated_value, value, rtol=1e-6)
    assert np.allclose(gradients[0], finite_rate, rtol=2e-3, atol=2e-4)
    assert np.allclose(gradients[1], finite_initial, rtol=2e-3, atol=2e-4)
