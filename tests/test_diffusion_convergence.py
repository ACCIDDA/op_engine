# tests/test_diffusion_convergence.py
"""Convergence tests for Crank-Nicolson diffusion operators.

These tests validate that the 1D diffusion implementation exhibits decreasing
error under temporal and spatial refinement.

Key test design choices (robust to boundary/discretization details):
    - Temporal convergence uses a numerical "reference" solution computed on the
      same spatial grid with a much finer timestep. This isolates temporal error
      and avoids conflating boundary/spatial discretization mismatch with time
      integration accuracy.
    - Spatial convergence uses nested grids and compares successive refinements
      by restricting the finer-grid solution to the coarser grid. This validates
      spatial convergence of the implemented discretization without relying on a
      continuous analytic solution (which can be sensitive to discrete boundary
      handling details).

Notes:
    - We use bc="absorbing" for Dirichlet-like boundary behavior.
    - Convergence tests can be more expensive than unit tests; parameters are
      chosen to remain CI-friendly.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import norm
from scipy.sparse import SparseEfficiencyWarning

from op_engine.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_operator,
    implicit_solve,
)

# Also apply a runtime filter as a backstop (covers environments where pytest
# mark filtering is not applied as expected).
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth multi-mode initial condition on [0, 1] with zero boundary values.

    Args:
        x: 1D array of spatial points.

    Returns:
        Initial condition evaluated at x.
    """
    return np.sin(np.pi * x) + 0.3 * np.sin(3 * np.pi * x) + 0.1 * np.sin(5 * np.pi * x)


def _run_cn_fixed_steps(
    *,
    n_x: int,
    n_steps: int,
    diffusivity: float,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run CN diffusion using a fixed number of time points.

    This ensures total_time is hit exactly by defining dt := total_time/(n_steps-1).

    Args:
        n_x: Number of spatial grid points.
        n_steps: Number of time points (including the initial time point).
        diffusivity: Diffusion coefficient.
        total_time: Total simulation time.

    Returns:
        Tuple of (final solution array, spatial grid points, dt).
    """
    x = np.linspace(0.0, 1.0, n_x)
    dx = float(x[1] - x[0])
    dt = float(total_time / (n_steps - 1))

    geom = GridGeometry(n=n_x, dx=dx)
    cfg = DiffusionConfig(coeff=diffusivity, bc="absorbing")

    left_op, right_op = build_crank_nicolson_operator(geom, cfg, dt)

    state = _initial_condition(x)
    for _ in range(n_steps - 1):
        state = implicit_solve(left_op, right_op, state)

    return state, x, dt


def _run_cn_fixed_dt(
    *,
    n_x: int,
    dt: float,
    diffusivity: float,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run CN diffusion using a target dt, adjusted to hit total_time exactly.

    We choose n_steps = round(total_time/dt) + 1 and then redefine
    dt_eff = total_time/(n_steps-1) to land exactly on total_time.

    Args:
        n_x: Number of spatial grid points.
        dt: Target time step size (will be slightly adjusted).
        diffusivity: Diffusion coefficient.
        total_time: Total simulation time.

    Returns:
        Tuple of (final solution array, spatial grid points, dt_eff, n_steps).
    """
    n_steps = round(total_time / dt) + 1
    n_steps = max(n_steps, 2)
    dt_eff = float(total_time / (n_steps - 1))

    state, x, _ = _run_cn_fixed_steps(
        n_x=n_x,
        n_steps=n_steps,
        diffusivity=diffusivity,
        total_time=total_time,
    )
    return state, x, dt_eff, n_steps


def _restrict_fine_to_coarse(fine: np.ndarray) -> np.ndarray:
    """Restrict a fine-grid solution to a nested coarse grid.

    Assumes nested grids with:
        fine_n = 2*(coarse_n - 1) + 1

    In that case, coarse nodes correspond to every other fine node.

    Args:
        fine: Fine-grid solution of shape (fine_n,).
        coarse_n: Coarse grid size.

    Returns:
        Restricted solution of shape (coarse_n,).

    """
    return fine[::2]


def _robust_order(
    errors: np.ndarray,
    steps: np.ndarray,
) -> float:
    """Estimate convergence order from a sequence of errors.

    Computes pairwise orders and returns the median. This is more robust than
    using only the last two points when roundoff or an error floor appears.

    Args:
        errors: Error values, shape (m,).
        steps: Step sizes (dt or dx), shape (m,).

    Returns:
        Median estimated order (float).
    """
    pair_orders: list[float] = []
    for i in range(errors.size - 1):
        e0 = float(errors[i])
        e1 = float(errors[i + 1])
        h0 = float(steps[i])
        h1 = float(steps[i + 1])

        if e0 <= 0.0 or e1 <= 0.0:
            continue
        if h0 <= 0.0 or h1 <= 0.0:
            continue

        denom = np.log(h0 / h1)
        if denom == 0.0:
            continue
        pair_orders.append(float(np.log(e0 / e1) / denom))

    if not pair_orders:
        return float("nan")
    return float(np.median(np.asarray(pair_orders)))


# ---------------------------------------------------------------------
# CN Tests
# ---------------------------------------------------------------------


def test_diffusion_cn_temporal_convergence() -> None:
    """Temporal convergence: error decreases as dt is refined (same spatial grid).

    We measure temporal error against a numerical reference solution computed on
    the same spatial grid with a substantially finer timestep. This isolates
    temporal accuracy from spatial/boundary discretization effects.
    """
    diffusivity = 0.1
    total_time = 0.25

    # Keep moderate for CI. Dense dispatch (n_x < 350) keeps behavior consistent.
    n_x = 257

    # dt halves each refinement (via n_steps).
    n_steps_list = [26, 51, 101]  # dt: 0.01, 0.005, 0.0025

    refine_factor = 8  # reference uses dt/refine_factor

    temporal_errors: list[float] = []
    dts: list[float] = []

    for n_steps in n_steps_list:
        sol, _x, dt = _run_cn_fixed_steps(
            n_x=n_x,
            n_steps=n_steps,
            diffusivity=diffusivity,
            total_time=total_time,
        )

        # Numerical reference: same grid, finer dt achieved by increasing steps.
        n_steps_ref = (n_steps - 1) * refine_factor + 1
        ref, _x_ref, _dt_ref = _run_cn_fixed_steps(
            n_x=n_x,
            n_steps=n_steps_ref,
            diffusivity=diffusivity,
            total_time=total_time,
        )

        err = float(norm(sol - ref, ord=np.inf))
        temporal_errors.append(err)
        dts.append(float(dt))

    errors_arr = np.asarray(temporal_errors, dtype=float)
    dts_arr = np.asarray(dts, dtype=float)

    # Must be decreasing overall (coarsest should be worse than finest).
    assert errors_arr[0] > errors_arr[-1], f"Temporal errors not dec.: {errors_arr}"

    # Estimate order robustly across points; CN should be ~2 in time.
    order = _robust_order(errors_arr, dts_arr)

    # Require clearly better than first order, but keep some slack for CI noise.
    assert np.isfinite(order), f"Non-finite time ord.: {order} (errors={errors_arr})"
    assert order > 1.0, f"Temporal order too low: got {order} (errors={errors_arr})"


def test_diffusion_cn_spatial_convergence() -> None:
    """Spatial convergence: error decreases as dx is refined (dt sufficiently small).

    We use nested grids and measure error between successive refinements by
    restricting the fine-grid solution to the coarse grid. This validates
    spatial convergence of the implemented discretization without relying on an
    external analytic solution.
    """
    diffusivity = 0.1
    total_time = 0.25

    # Choose dt small enough to suppress temporal error, but CI-friendly.
    # dt will be slightly adjusted internally to land exactly on total_time.
    dt_target = 5.0e-4

    # Nested grids: n_fine = 2*(n_coarse-1) + 1
    n_x_list = [65, 129, 257]  # dx halves each refinement

    sols: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    dxs: list[float] = []

    for n_x in n_x_list:
        sol, x, _dt_eff, _n_steps = _run_cn_fixed_dt(
            n_x=n_x,
            dt=dt_target,
            diffusivity=diffusivity,
            total_time=total_time,
        )
        sols.append(np.asarray(sol, dtype=float))
        grids.append(np.asarray(x, dtype=float))
        dxs.append(float(x[1] - x[0]))

    # Compute errors between successive refinements on the coarse grid.
    spatial_errors: list[float] = []
    for i in range(len(n_x_list) - 1):
        coarse_n = n_x_list[i]
        fine_n = n_x_list[i + 1]
        coarse_sol = sols[i]
        fine_sol = sols[i + 1]

        # Restrict fine solution to coarse grid for comparison.
        fine_on_coarse = _restrict_fine_to_coarse(fine_sol)

        # Sanity: matching shapes.
        assert fine_on_coarse.shape == coarse_sol.shape, (
            f"Restriction shape mismatch: fine_on_coarse={fine_on_coarse.shape}, "
            f"coarse={coarse_sol.shape}, fine_n={fine_n}, coarse_n={coarse_n}"
        )

        spatial_errors.append(float(norm(coarse_sol - fine_on_coarse, ord=np.inf)))

    errors_arr = np.asarray(spatial_errors, dtype=float)
    dxs_arr = np.asarray(dxs[:-1], dtype=float)  # associated with the coarse grids

    assert errors_arr[0] > errors_arr[-1], f"Spatial errors not dec.: {errors_arr}"

    # Second-order spatial discretization is expected; require >1 for robustness.
    order = _robust_order(errors_arr, dxs_arr)

    assert np.isfinite(order), f"Non-finite spatial ord.: {order} (errors={errors_arr})"
    assert order > 1.0, f"Spatial order too low: got {order} (errors={errors_arr})"
