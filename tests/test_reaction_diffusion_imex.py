# tests/test_reaction_diffusion_imex.py
"""Tests for IMEX predictor-corrector reaction-diffusion stepping.

This module validates:
1) CoreSolver.run_imex reduces to a pure implicit diffusion step when the
   reaction term is identically zero (and matches the corresponding CN path).
2) CoreSolver.run_imex exhibits temporal convergence for a linear
   reaction-diffusion system y' = A y + λ y, comparing against an exact matrix
   exponential reference.

Notes:
    - The solver treats the linear operator A implicitly via configured
      operators and treats the reaction term explicitly using a Heun-style
      predictor-corrector.
    - For operator construction we rely on matrix_ops builders; if there is any
      mismatch between dense/sparse paths, the tests choose parameters that keep
      operator sizes in the dense-dispatch regime for consistency.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.linalg import expm

from op_engine.core_solver import CoreSolver, ReactionRHSFunction, RHSFunction
from op_engine.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_operator,
    build_laplacian_tridiag,
    build_predictor_corrector,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

FloatArray = NDArray[np.floating]

# ---------------------------------------------------------------------
# Module-level constants (avoid extra locals in tests)
# ---------------------------------------------------------------------

DIFFUSIVITY = 0.1

# Keep < 350 to stay in dense dispatch for build_crank_nicolson_operator.
N_POINTS = 32

TOTAL_TIME_CN = 0.5
N_STEPS_CN = 51

TOTAL_TIME_RD = 0.2
LAMBDA_REACT = -0.3
N_STEPS_LIST = (41, 81, 161)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _initial_condition(x: FloatArray) -> FloatArray:
    """Smooth multi-mode initial condition on [0, 1] with zero boundary values.

    Args:
        x: 1D array of spatial points.

    Returns:
        Initial condition evaluated at x.
    """
    result = (
        np.sin(np.pi * x)
        + 0.3 * np.sin(3.0 * np.pi * x)
        + 0.1 * np.sin(5.0 * np.pi * x)
    )
    return cast("FloatArray", np.asarray(result, dtype=float))


def _core_no_history(time_grid: FloatArray, n_states: int) -> ModelCore:
    """Construct a ModelCore with store_history disabled (via ModelCoreOptions)."""
    opts = ModelCoreOptions(store_history=False)
    return ModelCore(
        n_states=n_states,
        n_subgroups=1,
        time_grid=time_grid,
        options=opts,
    )


def _run_cn_diffusion(
    y0: FloatArray,
    time_grid: FloatArray,
    geom: GridGeometry,
    cfg: DiffusionConfig,
) -> FloatArray:
    """Run pure diffusion using Crank-Nicolson via CoreSolver.run.

    Args:
        y0: Initial condition array of shape (n_points, 1).
        time_grid: 1D array of time points.
        geom: GridGeometry object defining spatial grid.
        cfg: DiffusionConfig object defining diffusion parameters.

    Returns:
        Final solution array of shape (n_points,).
    """
    dt = float(time_grid[1] - time_grid[0])
    left_cn, right_cn = build_crank_nicolson_operator(geom, cfg, dt)

    core = _core_no_history(time_grid=time_grid, n_states=int(y0.shape[0]))
    core.set_initial_state(y0)

    def rhs_cn(
        _t: float,
        state: FloatArray,
    ) -> FloatArray:
        # For pure diffusion with CN, rhs is just the current state.
        return state

    solver = CoreSolver(core, operators=(left_cn, right_cn))
    solver.run(cast("RHSFunction", rhs_cn))

    return cast("FloatArray", core.current_state[:, 0].copy())


def _run_imex_diffusion_zero_reaction(
    y0: FloatArray,
    time_grid: FloatArray,
    geom: GridGeometry,
    cfg: DiffusionConfig,
) -> FloatArray:
    """Run diffusion using IMEX with a reaction term identically zero.

    Args:
        y0: Initial condition array of shape (n_points, 1).
        time_grid: 1D array of time points.
        geom: GridGeometry object defining spatial grid.
        cfg: DiffusionConfig object defining diffusion parameters.

    Returns:
        Final solution array of shape (n_points,).
    """
    dt = float(time_grid[1] - time_grid[0])

    # A is the discrete diffusion operator (already includes coeff scaling).
    lap = build_laplacian_tridiag(
        n=geom.n,
        dx=geom.dx,
        coeff=cfg.coeff,
        dtype=np.float64,
        bc=cfg.bc,
    ).toarray()

    time_scaled_op = dt * np.asarray(lap, dtype=float)
    predictor, left_imex, right_imex = build_predictor_corrector(time_scaled_op)

    core = _core_no_history(time_grid=time_grid, n_states=int(y0.shape[0]))
    core.set_initial_state(y0)

    def reaction_zero(
        _t: float,
        state: FloatArray,
    ) -> FloatArray:
        # No reaction: F(t, y) = 0
        return cast("FloatArray", np.zeros_like(state))

    solver = CoreSolver(core, operators=(predictor, left_imex, right_imex))
    solver.run_imex(cast("ReactionRHSFunction", reaction_zero))

    return cast("FloatArray", core.current_state[:, 0].copy())


def _build_linear_rxn_diffusion_setup() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build diffusion operator, initial condition, and exact solution.

    We consider the semi-discrete system:

        y' = A y + λ y

    where:
        - A is the discrete diffusion operator (D * Δ_h) with absorbing BCs
          (Dirichlet-like behavior).
        - λ is a scalar reaction rate.

    The exact solution at time T is:

        y(T) = exp((A + λ I) * T) @ y0

    Returns:
        Tuple of (diffusion operator A as dense array, initial condition y0,
        exact solution y_exact).
    """
    x = np.linspace(0.0, 1.0, N_POINTS, dtype=float)
    geom = GridGeometry(n=N_POINTS, dx=float(x[1] - x[0]))
    cfg = DiffusionConfig(coeff=DIFFUSIVITY, bc="absorbing")

    # A = D * Δ_h (discrete diffusion operator)
    lap = build_laplacian_tridiag(
        n=geom.n,
        dx=geom.dx,
        coeff=cfg.coeff,
        dtype=np.float64,
        bc=cfg.bc,
    ).toarray()

    a = np.asarray(lap, dtype=float)

    # Full linear operator B = A + λ I
    identity_mat = np.eye(N_POINTS, dtype=float)
    operator_full = a + LAMBDA_REACT * identity_mat

    y0 = _initial_condition(cast("FloatArray", x)).astype(float)
    y_exact = expm(TOTAL_TIME_RD * operator_full) @ y0

    return cast("FloatArray", a), cast("FloatArray", y0), cast("FloatArray", y_exact)


def _imex_temporal_error(
    a: FloatArray,
    y0: FloatArray,
    y_exact: FloatArray,
    n_steps: int,
) -> tuple[float, float]:
    """Run IMEX for n_steps and return (dt, error) at final time."""
    time_grid = np.linspace(0.0, TOTAL_TIME_RD, n_steps, dtype=float)
    dt = float(time_grid[1] - time_grid[0])

    time_scaled_op = dt * np.asarray(a, dtype=float)
    predictor, left_op, right_op = build_predictor_corrector(time_scaled_op)

    core = _core_no_history(time_grid=cast("FloatArray", time_grid), n_states=N_POINTS)
    core.set_initial_state(y0.reshape(N_POINTS, 1))

    def reaction_rhs(
        _t: float,
        state: FloatArray,
    ) -> FloatArray:
        # Linear reaction: F(t, y) = λ y
        return cast("FloatArray", LAMBDA_REACT * state)

    solver = CoreSolver(core, operators=(predictor, left_op, right_op))
    solver.run_imex(cast("ReactionRHSFunction", reaction_rhs))

    y_num = core.current_state[:, 0].copy()
    err = float(norm(y_num - y_exact, ord=np.inf))
    return float(dt), err


# ---------------------------------------------------------------------
# Test 1: IMEX reduces to CN when reaction term is zero
# ---------------------------------------------------------------------


def test_imex_reduces_to_cn_when_reaction_zero() -> None:
    """IMEX run_imex with zero reaction matches pure CN diffusion.

    We run the same pure diffusion problem in two ways:

    1) Direct Crank-Nicolson using CoreSolver.run with rhs = state.
    2) IMEX predictor-corrector using CoreSolver.run_imex with reaction RHS = 0.

    Because both paths apply the same implicit operator step to the same input
    tensors, the final states should agree to tight tolerances.
    """
    time_grid = np.linspace(0.0, TOTAL_TIME_CN, N_STEPS_CN, dtype=float)

    x = np.linspace(0.0, 1.0, N_POINTS, dtype=float)
    geom = GridGeometry(n=N_POINTS, dx=float(x[1] - x[0]))
    cfg = DiffusionConfig(coeff=DIFFUSIVITY, bc="neumann")

    y0 = _initial_condition(cast("FloatArray", x)).reshape(N_POINTS, 1)

    y_cn = _run_cn_diffusion(y0, cast("FloatArray", time_grid), geom, cfg)
    y_imex = _run_imex_diffusion_zero_reaction(
        y0, cast("FloatArray", time_grid), geom, cfg
    )

    assert np.allclose(y_imex, y_cn, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------
# Test 2: IMEX reaction-diffusion temporal convergence (linear reaction)
# ---------------------------------------------------------------------


def test_imex_reaction_diffusion_temporal_convergence() -> None:
    """IMEX predictor-corrector shows temporal convergence for linear rxn-diffusion.

    We consider the semi-discrete linear system:

        y' = A y + λ y

    and compare to the exact reference:

        y(T) = exp((A + λ I) T) @ y0

    We expect the IMEX method (implicit CN for A + explicit Heun for λ y) to be
    second-order in time for this linear problem; we require > 1.0 to remain
    robust to floating-point noise in small problems.
    """
    a, y0, y_exact = _build_linear_rxn_diffusion_setup()

    errors: list[float] = []
    dts: list[float] = []

    for n_steps in N_STEPS_LIST:
        dt, err = _imex_temporal_error(a, y0, y_exact, n_steps)
        dts.append(dt)
        errors.append(err)

    errors_arr = np.asarray(errors, dtype=float)
    dts_arr = np.asarray(dts, dtype=float)

    assert errors_arr[0] > errors_arr[-1], f"IMEX time error not dec.: {errors_arr}"

    order = np.log(errors_arr[-2] / errors_arr[-1]) / np.log(dts_arr[-2] / dts_arr[-1])
    assert float(order) > 1.0, f"IMEX time order too low: got {order}"
