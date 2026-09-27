"""Tests for the external finite-element assembly proof of concept."""

from __future__ import annotations

import numpy as np
import pytest

from op_engine.matrix_ops import StageOperatorContext

from .fem_diffusion import (
    assemble_p1_diffusion,
    make_mass_trapezoidal_factory,
    run_fem_diffusion,
)


def test_p1_assembly_has_expected_energy_signs() -> None:
    """Consistent mass is positive and signed diffusion is negative."""
    nodes = np.asarray([0.0, 0.1, 0.35, 0.7, 1.0])
    mass, stiffness = assemble_p1_diffusion(nodes, diffusivity=0.2)

    np.testing.assert_allclose(mass.toarray(), mass.toarray().T)
    np.testing.assert_allclose(stiffness.toarray(), stiffness.toarray().T)
    assert np.all(np.linalg.eigvalsh(mass.toarray()) > 0.0)  # noqa: S101
    assert np.all(np.linalg.eigvalsh(stiffness.toarray()) < 0.0)  # noqa: S101


def test_mass_factory_reuses_operators_for_sparse_factor_cache() -> None:
    """Repeated stage sizes preserve identity for op_engine's sparse cache."""
    mass, stiffness = assemble_p1_diffusion(
        np.linspace(0.0, 1.0, 8),
        diffusivity=0.1,
    )
    factory = make_mass_trapezoidal_factory(mass, stiffness)

    context = StageOperatorContext(t=0.0, y=np.ones((6, 1)), stage="test")
    first = factory(0.1, 0.5, context)
    second = factory(0.05, 1.0, context)

    assert first[0] is second[0]  # noqa: S101
    assert first[1] is second[1]  # noqa: S101


def test_fem_diffusion_matches_heat_equation_mode() -> None:
    """The external P1 system converges to the analytic heat solution."""
    diffusivity = 0.1
    nodes = np.linspace(0.0, 1.0, 41)
    times = np.linspace(0.0, 0.2, 101)

    result = run_fem_diffusion(
        nodes=nodes,
        times=times,
        diffusivity=diffusivity,
    )
    expected = np.sin(np.pi * nodes[1:-1]) * np.exp(-diffusivity * np.pi**2 * times[-1])

    np.testing.assert_equal(
        result.states.shape,
        (times.size, nodes.size - 2, 1),
    )
    np.testing.assert_allclose(result.states[-1, :, 0], expected, atol=1.2e-4)


@pytest.mark.parametrize(
    ("nodes", "diffusivity", "message"),
    [
        (np.asarray([0.0, 1.0]), 0.1, "at least three"),
        (np.asarray([0.0, 0.5, 0.4, 1.0]), 0.1, "strictly increasing"),
        (np.asarray([0.0, 0.5, 1.0]), -0.1, "non-negative"),
    ],
)
def test_p1_assembly_rejects_invalid_inputs(
    nodes: np.ndarray,
    diffusivity: float,
    message: str,
) -> None:
    """The illustrative assembler rejects invalid static geometry."""
    with pytest.raises(ValueError, match=message):
        assemble_p1_diffusion(nodes, diffusivity)
