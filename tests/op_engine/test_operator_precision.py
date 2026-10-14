"""Reference-controlled precision for namespace-aware finite-volume builders."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from op_engine import array_namespace, build_advection_matrix, build_diffusion_matrix

BACKENDS = ("numpy", "jax", "torch")
STENCILS = {
    "advection": np.asarray([[-1, 0, 0], [1, -1, 0], [0, 1, 0]]),
    "diffusion": np.asarray([[-1, 1, 0], [1, -2, 1], [0, 1, -1]]),
}
BUILDERS = {"advection": build_advection_matrix, "diffusion": build_diffusion_matrix}


def _host(value: Any) -> np.ndarray:  # noqa: ANN401
    """Return eager values for independent assertions.

    Returns:
        Host NumPy values.
    """
    return np.asarray(value.detach().numpy() if hasattr(value, "detach") else value)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["advection", "diffusion"])
def test_floating_reference_selects_operator_precision(
    backend: str, dtype: str, kind: str
) -> None:
    """Known reflecting stencils retain the reference's precision and namespace."""
    module = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    precision = (
        pytest.importorskip("jax").enable_x64() if backend == "jax" else nullcontext()
    )
    with precision:
        reference = module.asarray(np.ones((2, 3, 3), dtype=dtype))
        result = BUILDERS[kind](3, 1.0, 0.4, bc="reflecting", reference=reference)
        assert result.dtype == reference.dtype
        assert result.shape == (3, 3)
        assert array_namespace(result) is array_namespace(reference)
        np.testing.assert_allclose(_host(result), 0.4 * STENCILS[kind], rtol=2e-7)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["advection", "diffusion"])
def test_reference_precision_preserves_dynamic_coefficient_derivatives(
    backend: str, dtype: str, kind: str
) -> None:
    """Coefficient derivatives match the known stencil's quadratic objective."""
    module = pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    precision = (
        pytest.importorskip("jax").enable_x64() if backend == "jax" else nullcontext()
    )
    with precision:
        reference = module.asarray(np.ones(3, dtype=dtype))
        coefficient = module.asarray(np.asarray(0.4, dtype=dtype))

        def objective(value: Any) -> Any:  # noqa: ANN401
            operator = BUILDERS[kind](
                3, 1.0, value, bc="reflecting", reference=reference
            )
            return module.sum(operator * operator)

        if backend == "jax":
            jax = pytest.importorskip("jax")
            result, gradient = jax.jit(jax.value_and_grad(objective))(coefficient)
        else:
            result = objective(coefficient)
            gradient = module.func.jacrev(objective)(coefficient)
        assert result.dtype == reference.dtype
        assert gradient.dtype == reference.dtype
        expected = 0.8 * sum(float(value) ** 2 for value in STENCILS[kind].ravel())
        np.testing.assert_allclose(_host(gradient), expected, rtol=2e-7)


@pytest.mark.parametrize("kind", ["advection", "diffusion"])
def test_integer_reference_still_constructs_floating_operator(kind: str) -> None:
    """Integer geometry/state does not truncate coefficients or stencil values."""
    result = BUILDERS[kind](
        3, 1.0, 0.4, bc="reflecting", reference=np.ones(3, dtype=np.int64)
    )
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, 0.4 * STENCILS[kind])
