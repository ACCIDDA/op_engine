"""Independent state, initial-condition and measured-output sensitivity checks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, NamedTuple

import numpy as np
import pytest

from op_engine.core_solver import RunConfig
from op_engine.sensitivities import (
    SensitivityInitial,
    SensitivityOptions,
    SensitivitySystem,
    solve_sensitivities,
)


class _Identity(NamedTuple):
    digest: str


class _Matrices(NamedTuple):
    state: object
    parameters: object
    observable_state: object
    observable_parameters: object


class _DecayDerivatives:
    """Independent two-local-state reference with one shared rate and gain."""

    state_names = ("x", "z")
    parameter_names = ("rate", "gain")
    observable_names = ("measurement",)
    model = _Identity("synthetic-shared-linear-decay-v1")

    @staticmethod
    def evaluate(_t: object, y: Any, **parameters: Any) -> _Matrices:  # noqa: ANN401
        """Return independently specified derivative matrices.

        Returns:
            Native analytic derivatives for f=-rate*y and g=gain*(x+2*z).
        """
        from op_engine import array_namespace  # noqa: PLC0415

        xp = array_namespace(y)
        rate, gain = parameters["rate"], parameters["gain"]
        zero = xp.zeros_like(rate)
        return _Matrices(
            xp.stack((xp.stack((-rate, zero)), xp.stack((zero, -rate)))),
            xp.stack((-y, xp.zeros_like(y)), axis=1),
            xp.reshape(xp.stack((gain, 2 * gain)), (1, 2)),
            xp.reshape(xp.stack((zero, y[0] + 2 * y[1])), (1, 2)),
        )


def _system() -> SensitivitySystem:
    def rhs(_t: object, y: Any, **parameters: Any) -> Any:  # noqa: ANN401
        return -parameters["rate"] * y

    return SensitivitySystem(rhs, _DecayDerivatives(), ("amplitude",))


def _backend(name: str) -> tuple[Any, Any]:
    if name == "numpy":
        return np, nullcontext()
    module = pytest.importorskip(name)
    if name == "torch":
        return module, nullcontext()
    enable = getattr(module, "enable_x64", None)
    if enable is None:
        enable = module.experimental.enable_x64
    return pytest.importorskip("jax.numpy"), enable()


def _solve(vector: Any, xp: Any, options: SensitivityOptions) -> Any:  # noqa: ANN401
    zero = xp.zeros_like(vector[0])
    one = xp.ones_like(vector[0])
    initial = SensitivityInitial(
        xp.stack((vector[2], 2 * vector[2])),
        xp.stack((xp.stack((zero, zero, one)), xp.stack((zero, zero, 2 * one)))),
    )
    return solve_sensitivities(
        _system(),
        dict(zip(("rate", "gain", "amplitude"), vector, strict=True)),
        initial,
        np.asarray([0.0, 0.25, 0.5, 1.0]),
        options=options,
    )


def _host(value: Any) -> Any:  # noqa: ANN401
    return value.detach().numpy() if hasattr(value, "detach") else np.asarray(value)


def _discrete_reference(method: str, times: Any) -> tuple[Any, Any, Any]:  # noqa: ANN401
    coefficients = {
        "euler": [1, 1],
        "heun": [1, 1, 1 / 2],
        "rk4": [1, 1, 1 / 2, 1 / 6, 1 / 24],
        "dopri5": [1, 1, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 600],
    }[method]
    factor = np.polynomial.polynomial.polyval(-0.7 * 0.0625, coefficients)
    rate_factor = (
        np.polynomial.polynomial.polyval(
            -0.7 * 0.0625, np.polynomial.polynomial.polyder(coefficients)
        )
        * -0.0625
    )
    steps = np.rint(times / 0.0625)
    states = factor ** steps[:, None] * np.array([2.0, 4.0])
    reference = np.stack(
        (
            steps[:, None] * rate_factor / factor * states,
            np.zeros_like(states),
            states / 2.0,
        ),
        axis=2,
    )
    total = states @ np.array([1.0, 2.0])
    measured = np.stack(
        (steps * rate_factor / factor * 1.5 * total, total, 1.5 * total / 2.0), axis=1
    )
    return states, reference, measured


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
def test_native_sensitivities_include_shared_initial_and_measurement_effects(
    backend: str,
    dtype: str,
    method: str,
) -> None:
    """Independent stability polynomials catch stage, initial and output errors."""
    xp, precision = _backend(backend)
    with precision:
        vector = xp.asarray([0.7, 1.5, 2.0], dtype=getattr(xp, dtype))
        solution = _solve(
            vector,
            xp,
            SensitivityOptions(
                solver=RunConfig(method=method, fixed_max_step=0.0625),
                store_history=True,
            ),
        )
        times = np.asarray(solution.times)
        states, reference, measured = _discrete_reference(method, times)
        tolerance = 2e-6
        np.testing.assert_allclose(_host(solution.states), states, rtol=tolerance)
        np.testing.assert_allclose(
            _host(solution.parameter_derivatives), reference, rtol=tolerance, atol=1e-7
        )
        np.testing.assert_allclose(
            _host(solution.observable_derivatives)[:, 0],
            measured,
            rtol=tolerance,
            atol=1e-7,
        )
        assert solution.augmented_dimension == 8
        assert solution.parameter_names == ("rate", "gain", "amplitude")
        assert solution.states.dtype == vector.dtype


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_final_only_sensitivities_and_same_mesh_finite_differences(
    backend: str,
) -> None:
    """Final-only storage agrees with history; discrete finite differences agree."""
    xp, precision = _backend(backend)
    with precision:
        vector = xp.asarray([0.7, 1.5, 2.0], dtype=xp.float64)
        options = SensitivityOptions(
            solver=RunConfig(method="rk4", fixed_max_step=0.125)
        )
        solution = _solve(vector, xp, options)
        assert solution.times == (1.0,)
        assert solution.states.shape == (1, 2)
        for i in range(3):
            delta = np.zeros(3)
            delta[i] = 1e-5
            plus = _solve(vector + xp.asarray(delta), xp, options)
            minus = _solve(vector - xp.asarray(delta), xp, options)
            difference = (_host(plus.states) - _host(minus.states)) / 2e-5
            np.testing.assert_allclose(
                _host(solution.parameter_derivatives)[:, :, i],
                difference,
                rtol=1e-9,
                atol=1e-9,
            )


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_ad_and_jit_compare_the_same_discretized_sensitivity(
    backend: str,
) -> None:
    """Reverse AD and forward equations agree on the fixed RK4 mesh."""
    xp, precision = _backend(backend)
    with precision:
        options = SensitivityOptions(
            solver=RunConfig(
                method="rk4",
                fixed_max_step=0.125,
                fixed_loop="scan" if backend == "jax" else "unroll",
                fixed_checkpoint=backend == "jax",
            )
        )

        def evaluate(vector: Any) -> Any:  # noqa: ANN401
            solution = _solve(vector, xp, options)
            return vector[1] * (solution.states[-1, 0] + 2 * solution.states[-1, 1])

        vector = xp.asarray([0.7, 1.5, 2.0], dtype=xp.float64)
        if backend == "jax":
            jax = pytest.importorskip("jax")
            derivative = jax.jit(jax.grad(evaluate))(vector)
            batched = jax.jit(jax.vmap(evaluate))(xp.stack((vector, vector * 1.1)))
            assert batched.shape == (2,)
        else:
            vector.requires_grad_()
            evaluate(vector).backward()
            derivative = vector.grad
        solution = _solve(vector, xp, options)
        np.testing.assert_allclose(
            _host(derivative),
            _host(solution.observable_derivatives)[-1, 0],
            rtol=1e-11,
            atol=1e-11,
        )


def test_mesh_refinement_reduces_continuous_sensitivity_error() -> None:
    """RK4 sensitivity error decreases at its expected order away from roundoff."""
    reference = -2 * np.exp(-0.7)
    errors = [
        abs(
            _solve(
                np.array([0.7, 1.5, 2.0]),
                np,
                SensitivityOptions(
                    solver=RunConfig(method="rk4", fixed_max_step=step),
                ),
            ).parameter_derivatives[0, 0, 0]
            - reference
        )
        for step in (0.25, 0.125)
    ]
    assert errors[0] > 12 * errors[1] > 0


def test_unsupported_solver_and_storage_contracts_fail_explicitly() -> None:
    """Storage budgets and unavailable dynamical cases are not silently ignored."""
    for config in (
        RunConfig(adaptive=True),
        RunConfig(method="sdirk2"),
        RunConfig(forcing_breakpoints=(0.2,)),
    ):
        with pytest.raises(ValueError, match="smooth fixed explicit"):
            SensitivityOptions(solver=config)
    vector = np.array([0.7, 1.5, 2.0])
    for options in (
        SensitivityOptions(max_augmented_states=7),
        SensitivityOptions(max_output_bytes=1),
    ):
        with pytest.raises(ValueError, match="storage limit"):
            _solve(vector, np, options)
    with pytest.raises(ValueError, match="unique"):
        SensitivitySystem(_system().rhs, _DecayDerivatives(), ("rate",))
