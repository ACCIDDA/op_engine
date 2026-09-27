"""Curvature-weighted one-dimensional spatial grid generation."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


def _positive_integer(value: int, *, name: str, minimum: int) -> int:
    """Validate an integer control value.

    Returns:
        The validated integer.

    Raises:
        ValueError: If the value is not an integer meeting the lower bound.
    """
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        msg = f"{name} must be an integer greater than or equal to {minimum}."
        raise ValueError(msg)
    return int(value)


def _smooth_curvature(
    curvature: NDArray[np.float64],
    smoothing_window: int | None,
) -> NDArray[np.float64]:
    """Apply an optional centered moving average to sampled curvature.

    Returns:
        The original or smoothed curvature samples.

    Raises:
        ValueError: If the smoothing window is invalid.
    """
    if smoothing_window is None:
        return curvature
    window = _positive_integer(
        smoothing_window,
        name="smoothing_window",
        minimum=1,
    )
    if window % 2 == 0:
        msg = "smoothing_window must be odd."
        raise ValueError(msg)
    if window > curvature.size:
        msg = "smoothing_window cannot exceed the number of profile samples."
        raise ValueError(msg)
    if window == 1:
        return curvature
    padding = window // 2
    padded = np.pad(curvature, padding, mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.asarray(np.convolve(padded, kernel, mode="valid"), dtype=np.float64)


def _enforce_minimum_spacing(
    grid: NDArray[np.float64],
    minimum_spacing: float | None,
) -> NDArray[np.float64]:
    """Project an increasing endpoint-fixed grid onto a spacing lower bound.

    Returns:
        The original grid or a spacing-constrained copy.

    Raises:
        ValueError: If the spacing is invalid or infeasible.
    """
    if minimum_spacing is None:
        return grid
    if not np.isfinite(minimum_spacing) or minimum_spacing <= 0.0:
        msg = "minimum_spacing must be finite and positive."
        raise ValueError(msg)
    span = float(grid[-1] - grid[0])
    required_span = minimum_spacing * (grid.size - 1)
    if required_span > span * (1.0 + 1e-12):
        msg = (
            "minimum_spacing is infeasible for the domain and requested "
            "number of points."
        )
        raise ValueError(msg)

    adjusted = grid.copy()
    for index in range(1, adjusted.size):
        adjusted[index] = max(
            adjusted[index],
            adjusted[index - 1] + minimum_spacing,
        )
    adjusted[-1] = grid[-1]
    for index in range(adjusted.size - 2, -1, -1):
        adjusted[index] = min(
            adjusted[index],
            adjusted[index + 1] - minimum_spacing,
        )
    adjusted[0] = grid[0]
    return adjusted


def generate_adaptive_grid_from_data(  # noqa: PLR0913, PLR0914
    coordinates: ArrayLike,
    values: ArrayLike,
    n_points: int,
    *,
    epsilon: float = 1e-3,
    smoothing_window: int | None = None,
    minimum_spacing: float | None = None,
) -> NDArray[np.float64]:
    """Equidistribute points according to sampled profile curvature.

    The curvature monitor is
    ``abs(f'') / (1 + f'**2)**(3/2) + epsilon``. Its trapezoidal cumulative
    integral defines a monotone distribution whose evenly spaced quantiles
    are inverted with linear interpolation. The result contains exactly
    ``n_points`` coordinates, including both input endpoints.

    This is an eager geometry-preprocessing utility: it always returns a NumPy
    array. The resulting static coordinates can be passed to numerical
    operators running in any supported Array-API namespace.

    Args:
        coordinates: Strictly increasing one-dimensional sample coordinates.
        values: Finite profile values at ``coordinates``.
        n_points: Number of output points, including both endpoints.
        epsilon: Positive curvature-density floor.
        smoothing_window: Optional odd moving-average width applied to the
            sampled curvature before integration.
        minimum_spacing: Optional positive lower bound on adjacent output
            spacing. It must be feasible within the sampled domain.

    Returns:
        Strictly increasing curvature-weighted coordinates with shape
        ``(n_points,)``.

    Raises:
        ValueError: If samples or controls are invalid.
    """
    count = _positive_integer(n_points, name="n_points", minimum=2)
    x = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 3:
        msg = "coordinates must be one-dimensional with at least three samples."
        raise ValueError(msg)
    if y.shape != x.shape:
        msg = f"values must have shape {x.shape}; got {y.shape}."
        raise ValueError(msg)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        msg = "coordinates and values must be finite."
        raise ValueError(msg)
    if not np.all(np.diff(x) > 0.0):
        msg = "coordinates must be strictly increasing."
        raise ValueError(msg)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        msg = "epsilon must be finite and positive."
        raise ValueError(msg)

    first_derivative = np.gradient(y, x, edge_order=2)
    second_derivative = np.gradient(first_derivative, x, edge_order=2)
    curvature = np.abs(second_derivative) / np.power(
        1.0 + first_derivative * first_derivative,
        1.5,
    )
    sample_spacing = float(np.min(np.diff(x)))
    value_scale = max(1.0, float(np.max(np.abs(y))))
    curvature_noise = 64.0 * np.finfo(np.float64).eps * value_scale / sample_spacing**2
    curvature = np.where(curvature <= curvature_noise, 0.0, curvature)
    curvature = _smooth_curvature(
        np.asarray(curvature, dtype=np.float64),
        smoothing_window,
    )
    if not np.any(curvature):
        uniform = np.linspace(x[0], x[-1], count, dtype=np.float64)
        return _enforce_minimum_spacing(uniform, minimum_spacing)

    density = curvature + epsilon
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(x)
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(increments)))
    cumulative /= cumulative[-1]
    quantiles = np.linspace(0.0, 1.0, count, dtype=np.float64)
    grid = np.asarray(np.interp(quantiles, cumulative, x), dtype=np.float64)
    grid[0] = x[0]
    grid[-1] = x[-1]
    return _enforce_minimum_spacing(grid, minimum_spacing)


def generate_adaptive_grid(  # noqa: PLR0913
    profile: Callable[[NDArray[np.float64]], ArrayLike],
    domain: tuple[float, float],
    n_points: int,
    *,
    epsilon: float = 1e-3,
    sampling_points: int = 4097,
    smoothing_window: int | None = None,
    minimum_spacing: float | None = None,
) -> NDArray[np.float64]:
    """Generate a curvature-weighted grid from a vectorized profile callable.

    Args:
        profile: Callable evaluated on a one-dimensional NumPy sample grid. It
            must return one finite value per sample.
        domain: Finite increasing ``(start, end)`` coordinate interval.
        n_points: Number of output points, including both domain endpoints.
        epsilon: Positive curvature-density floor.
        sampling_points: Number of uniform samples used to estimate curvature.
            This must be at least ``n_points`` and at least three.
        smoothing_window: Optional odd moving-average width for curvature.
        minimum_spacing: Optional positive lower bound on output spacing.

    Returns:
        Strictly increasing curvature-weighted coordinates with shape
        ``(n_points,)``.

    Raises:
        TypeError: If ``profile`` is not callable.
        ValueError: If the domain, sampled profile, or controls are invalid.
    """
    if not callable(profile):
        msg = "profile must be callable."
        raise TypeError(msg)
    count = _positive_integer(n_points, name="n_points", minimum=2)
    sample_count = _positive_integer(
        sampling_points,
        name="sampling_points",
        minimum=3,
    )
    if sample_count < count:
        msg = "sampling_points must be greater than or equal to n_points."
        raise ValueError(msg)
    start, end = domain
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        msg = "domain must contain two finite, strictly increasing bounds."
        raise ValueError(msg)

    coordinates = np.linspace(start, end, sample_count, dtype=np.float64)
    values = np.asarray(profile(coordinates), dtype=np.float64)
    return generate_adaptive_grid_from_data(
        coordinates,
        values,
        count,
        epsilon=epsilon,
        smoothing_window=smoothing_window,
        minimum_spacing=minimum_spacing,
    )


__all__ = ["generate_adaptive_grid", "generate_adaptive_grid_from_data"]
