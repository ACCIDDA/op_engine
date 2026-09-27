"""Regression tests for the trait-structured chemostat PDE example."""
# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest

from .chemostat_pde import (
    ChemostatParameters,
    ChemostatResult,
    build_trait_grid,
    cell_volume_weights,
    compute_observables,
    diffusion_operator,
    initial_state,
    run_chemostat,
    save_summary_figure,
)

if TYPE_CHECKING:
    from pathlib import Path

mpl.use("Agg")


@pytest.fixture(scope="module")
def comparison() -> ChemostatResult:
    """Run one compact IMEX/RK45 comparison shared by integration tests.

    Returns:
        Comparison result reused by the accuracy and figure tests.
    """
    return run_chemostat(
        n_cells=21,
        times=np.linspace(0.0, 2.0, 101, dtype=np.float64),
    )


def test_adaptive_grid_and_diffusion_preserve_weighted_population() -> None:
    """The non-uniform no-flux stencil conserves FV-weighted biomass."""
    parameters = ChemostatParameters()
    trait = build_trait_grid(25, parameters=parameters)
    weights = cell_volume_weights(trait)
    operator = diffusion_operator(trait, parameters)[:-1, :-1]

    assert trait[0] == pytest.approx(-2.0)
    assert trait[-1] == pytest.approx(2.0)
    assert np.ptp(np.diff(trait)) > 1e-3
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(operator @ np.ones(trait.size), 0.0, atol=1e-13)
    np.testing.assert_allclose(weights @ operator, 0.0, atol=1e-13)

    state = initial_state(trait, weights, parameters)
    assert np.dot(weights, state[:-1]) == pytest.approx(
        parameters.initial_biomass,
    )


def test_observables_use_finite_volume_weights() -> None:
    """Discrete moments match a hand-computed symmetric distribution."""
    trait = np.asarray([-1.0, 0.0, 1.0])
    weights = np.ones(3)
    states = np.asarray([[1.0, 2.0, 1.0, 1.5]])

    observables = compute_observables(states, trait, weights)

    np.testing.assert_allclose(observables.biomass, [4.0])
    np.testing.assert_allclose(observables.mean, [0.0])
    np.testing.assert_allclose(observables.variance, [0.5])
    np.testing.assert_allclose(observables.skewness, [0.0])
    np.testing.assert_allclose(observables.kurtosis, [2.0])
    np.testing.assert_allclose(observables.resource, [1.5])


def test_imex_trajectory_matches_rk45_reference(
    comparison: ChemostatResult,
) -> None:
    """The canonical split stays close to the full semi-discrete RK45 solve."""
    state_error = np.max(
        np.abs(comparison.imex_states - comparison.rk45_states),
    )
    biomass_error = np.max(
        np.abs(
            comparison.imex_observables.biomass - comparison.rk45_observables.biomass,
        ),
    )

    assert state_error < 2e-5
    assert biomass_error < 2e-5
    assert np.all(comparison.imex_states > 0.0)
    assert np.all(comparison.rk45_states > 0.0)


def test_summary_figure_is_written(
    comparison: ChemostatResult,
    tmp_path: Path,
) -> None:
    """The documented three-panel diagnostic is a non-empty image."""
    output = tmp_path / "chemostat-summary.png"

    save_summary_figure(comparison, output)

    assert output.stat().st_size > 10_000
