"""Tests for curvature-weighted spatial grid generation."""

from __future__ import annotations

import numpy as np
import pytest

from op_engine import generate_adaptive_grid, generate_adaptive_grid_from_data


def test_constant_profile_produces_uniform_grid() -> None:
    """The epsilon floor equidistributes a zero-curvature profile uniformly."""
    grid = generate_adaptive_grid(
        np.ones_like,
        (-2.0, 3.0),
        21,
        sampling_points=201,
    )

    np.testing.assert_array_equal(grid, np.linspace(-2.0, 3.0, 21))


def test_gaussian_profile_concentrates_points_near_peak() -> None:
    """A narrow Gaussian receives finer coordinates around its curved peak."""
    grid = generate_adaptive_grid(
        lambda x: np.exp(-10.0 * x * x),
        (-1.0, 1.0),
        81,
        epsilon=0.05,
    )
    spacing = np.diff(grid)
    midpoints = 0.5 * (grid[:-1] + grid[1:])

    center_spacing = np.median(spacing[np.abs(midpoints) < 0.2])
    outer_spacing = np.median(spacing[np.abs(midpoints) > 0.7])
    assert center_spacing < 0.1 * outer_spacing
    assert grid[0] == -1.0
    assert grid[-1] == 1.0
    assert np.all(spacing > 0.0)


def test_callable_and_sampled_data_paths_match() -> None:
    """Callable generation delegates exactly to the sampled-data algorithm."""
    coordinates = np.linspace(-1.0, 1.0, 1001)
    values = np.exp(-6.0 * coordinates * coordinates)

    from_callable = generate_adaptive_grid(
        lambda x: np.exp(-6.0 * x * x),
        (-1.0, 1.0),
        51,
        sampling_points=coordinates.size,
        smoothing_window=5,
    )
    from_data = generate_adaptive_grid_from_data(
        coordinates,
        values,
        51,
        smoothing_window=5,
    )

    np.testing.assert_array_equal(from_callable, from_data)


def test_uniform_refinement_halves_maximum_spacing() -> None:
    """Doubling uniform intervals halves the maximum output spacing."""
    coarse = generate_adaptive_grid(lambda x: 0.0 * x, (0.0, 1.0), 33)
    fine = generate_adaptive_grid(lambda x: 0.0 * x, (0.0, 1.0), 65)

    assert np.max(np.diff(coarse)) / np.max(np.diff(fine)) == pytest.approx(2.0)


def test_minimum_spacing_is_enforced_with_fixed_endpoints() -> None:
    """The optional spacing projection limits an aggressive concentration."""
    grid = generate_adaptive_grid(
        lambda x: np.exp(-40.0 * x * x),
        (-1.0, 1.0),
        81,
        minimum_spacing=0.01,
    )

    assert grid[0] == -1.0
    assert grid[-1] == 1.0
    assert np.min(np.diff(grid)) >= 0.01 - 1e-14


@pytest.mark.parametrize(
    ("coordinates", "values", "kwargs", "match"),
    [
        ([0.0, 1.0], [0.0, 1.0], {}, "at least three"),
        ([0.0, 1.0, 2.0], [0.0, 1.0], {}, "values must have shape"),
        ([0.0, 1.0, 0.5], [0.0, 1.0, 2.0], {}, "strictly increasing"),
        ([0.0, 1.0, 2.0], [0.0, np.nan, 2.0], {}, "must be finite"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], {"epsilon": 0.0}, "epsilon"),
        (
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            {"smoothing_window": 2},
            "must be odd",
        ),
        (
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            {"minimum_spacing": 1.1},
            "infeasible",
        ),
    ],
)
def test_sampled_grid_rejects_invalid_inputs(
    coordinates: list[float],
    values: list[float],
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Sample geometry and optional controls fail closed."""
    with pytest.raises(ValueError, match=match):
        generate_adaptive_grid_from_data(
            coordinates,
            values,
            3,
            **kwargs,  # type: ignore[arg-type]
        )


def test_callable_grid_rejects_invalid_controls() -> None:
    """Callable generation validates domain, counts, and returned shape."""
    with pytest.raises(TypeError, match="profile must be callable"):
        generate_adaptive_grid(1.0, (0.0, 1.0), 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_points"):
        generate_adaptive_grid(lambda x: x, (0.0, 1.0), 1)
    with pytest.raises(ValueError, match="domain"):
        generate_adaptive_grid(lambda x: x, (1.0, 0.0), 5)
    with pytest.raises(ValueError, match="greater than or equal"):
        generate_adaptive_grid(
            lambda x: x,
            (0.0, 1.0),
            5,
            sampling_points=4,
        )
    with pytest.raises(ValueError, match="values must have shape"):
        generate_adaptive_grid(lambda _x: np.asarray([1.0]), (0.0, 1.0), 5)
