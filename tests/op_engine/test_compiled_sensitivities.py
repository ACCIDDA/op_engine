"""Optional compiler boundary; mandatory under the pinned compiler CI lane."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from op_engine import array_namespace
from op_engine.core_solver import RunConfig
from op_engine.sensitivities import (
    SensitivityInitial,
    SensitivityOptions,
    SensitivitySystem,
    solve_sensitivities,
)


def _measured_reference() -> np.ndarray:
    """Differentiate a triangular two-rate transfer analytically.

    Returns:
        Measurement derivatives in a,b,gain,initial-amplitude order at t=1.
    """
    a, b, gain, amplitude = 0.7, 1.1, 1.5, 2.0
    left, right = np.exp(-a), np.exp(-b)
    difference = left - right
    factor = a / (b - a)
    derivative_a = amplitude * (b / (b - a) ** 2 * difference - factor * left)
    derivative_b = amplitude * (-a / (b - a) ** 2 * difference + factor * right)
    total = amplitude * (left + 2 * factor * difference)
    return np.asarray([
        [
            [
                -gain * amplitude * left + 2 * gain * derivative_a,
                2 * gain * derivative_b,
                total,
                gain * total / amplitude,
            ]
        ]
    ])


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compiled_sensitivity_artifact_preserves_measured_coordinates(
    backend: str, dtype: str
) -> None:
    """Coupled source IR agrees with a separate two-rate analytic reference."""
    compiler = pytest.importorskip("op_system")
    pytest.importorskip("op_system.symbolic")
    module = np
    precision = nullcontext()
    if backend != "numpy":
        module = pytest.importorskip(backend)
    if backend == "jax":
        enable = getattr(module, "enable_x64", None)
        if enable is None:
            enable = module.experimental.enable_x64
        precision = enable()
        module = pytest.importorskip("jax.numpy")
    compiled = compiler.compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"transfer": "a*x-b*z"},
        "equations": {"x": "-a*x", "z": "transfer"},
        "observables": {"measurement": {"expr": "gain*(x+2*z)"}},
    })
    system = SensitivitySystem(
        compiled.eval_fn, compiler.symbolic_derivatives(compiled), ("amplitude",)
    )
    assert system.parameter_names == ("a", "b", "gain", "amplitude")
    with precision:
        y = module.asarray([2.0, 0.0], dtype=getattr(module, dtype))
        xp = array_namespace(y)
        initial = SensitivityInitial(
            y,
            module.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=y.dtype),
        )
        solution = solve_sensitivities(
            system,
            {
                name: module.asarray(value, dtype=y.dtype)
                for name, value in zip(
                    system.parameter_names, (0.7, 1.1, 1.5, 2.0), strict=True
                )
            },
            initial,
            np.asarray([0.0, 0.5, 1.0]),
            options=SensitivityOptions(
                solver=RunConfig(method="rk4", fixed_max_step=0.0625)
            ),
        )
        result = solution.observable_derivatives
        if backend == "torch":
            result = result.detach().numpy()
        np.testing.assert_allclose(
            result,
            _measured_reference(),
            rtol=2e-6,
            atol=2e-6,
        )
        assert array_namespace(solution.states) is xp
