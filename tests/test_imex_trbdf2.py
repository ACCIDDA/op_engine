# tests/test_imex_trbdf2.py
"""Unit tests for op_engine.core_solver TR-BDF2 IMEX integration.

This module verifies:
- TR-BDF2 stiff damping vs trapezoidal/CN (via run_imex) on a stiff linear system.
- TR-BDF2 stability and basic accuracy on a split stiff linear + mild forcing system.
- TR-BDF2 demonstrates ~2nd order convergence on a linear test problem.
- TR-BDF2 fallback behavior when stage operators are omitted (uses solver defaults).
- TR-BDF2 handles non-uniform time grids without NaNs/Infs and matches a fine reference.
- TR-BDF2 supports time-varying (dt-dependent) stage operator factories (callable form).

These tests match the *updated* op_engine.matrix_ops API:
    - make_stage_operator_factory(base_builder, scheme=...)
    - make_constant_base_builder(A)
    - Stage factories have signature: factory(dt, scale, ctx) -> (L, R)

And the updated CoreSolver TR-BDF2 contract:
    - CoreSolver.run_imex_trbdf2(..., operators_tr=..., operators_bdf2=...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_engine.core_solver import CoreSolver
from op_engine.matrix_ops import (
    StageOperatorContext,
    build_implicit_euler_operators,
    build_trapezoidal_operators,
    make_constant_base_builder,
    make_stage_operator_factory,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_scalar_core(time_grid: FloatArray) -> ModelCore:
    """Create a scalar ModelCore with state tensor shape (1, 1)."""
    opts = ModelCoreOptions(
        other_axes=(),
        axis_names=("state", "subgroup"),
        store_history=True,
        dtype=np.float64,
    )
    return ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid, options=opts)


def _scalar_state(y: float) -> FloatArray:
    """Pack scalar into state tensor shape (1, 1)."""
    return np.array([[y]], dtype=np.float64)


def _unpack_scalar(state: FloatArray) -> float:
    """Unpack scalar from state tensor shape (1, 1)."""
    return float(state[0, 0])


# -------------------------------------------------------------------
# 1) Stiff linear: TR-BDF2 damps strongly vs CN does not (L-stability check)
# -------------------------------------------------------------------


def test_trbdf2_stiff_linear_damps_strongly_vs_cn() -> None:
    """TR-BDF2 should strongly damp stiff modes; trapezoidal/CN typically does not.

    System: y' = A y, with A = -lambda, F=0.
    With large lambda*dt, trapezoidal/CN amplification tends to ~ -1 (weak damping,
    oscillatory), whereas TR-BDF2 is L-stable and damps toward 0.
    """
    lam = 1000.0
    y0 = 1.0

    dt = 0.1
    t_end = 1.0
    time_grid = np.arange(0.0, t_end + 0.5 * dt, dt, dtype=np.float64)

    A = np.array([[-lam]], dtype=np.float64)

    def reaction_zero(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.zeros_like(y)

    # --- CN reference using run_imex with F=0 -> y_{n+1} = L^{-1} R y_n
    core_cn = _make_scalar_core(time_grid)
    core_cn.set_initial_state(_scalar_state(y0))

    L_cn, R_cn = build_trapezoidal_operators(A, dt_scale=dt)
    solver_cn = CoreSolver(core_cn, operators=(L_cn, R_cn), operator_axis="state")
    solver_cn.run_imex(reaction_zero)

    # --- TR-BDF2 with stage factories (constant base operator)
    core_tr = _make_scalar_core(time_grid)
    core_tr.set_initial_state(_scalar_state(y0))

    base = make_constant_base_builder(A)
    tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

    solver_tr = CoreSolver(core_tr, operators=None, operator_axis="state")
    solver_tr.run_imex_trbdf2(
        reaction_zero,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    assert core_cn.state_array is not None
    assert core_tr.state_array is not None

    y_cn_final = _unpack_scalar(core_cn.state_array[-1])
    y_tr_final = _unpack_scalar(core_tr.state_array[-1])

    # CN/trapezoidal is not L-stable: with z=-lam*dt very negative, |amp| ~ 1
    assert abs(y_cn_final) > 1e-3

    # TR-BDF2 should significantly damp stiff decay
    assert abs(y_tr_final) < 1e-6


# -------------------------------------------------------------------
# 2) Split system: stiff linear + mild forcing, TR-BDF2 stays stable
# -------------------------------------------------------------------


def test_trbdf2_split_stiff_forced_system_reasonable_accuracy() -> None:
    """TR-BDF2 should remain stable on a stiff linear + mild forcing split system.

    System:
        y' = A y + F(t, y)
        A = -lambda
        F(t, y) = sin(t)

    We check:
        - No NaNs/Infs
        - Rough agreement with a reference computed on a much finer TR-BDF2 grid
    """
    lam = 50.0
    y0 = 0.25

    t_end = 4.0
    dt_coarse = 0.05
    dt_fine = 0.005

    tg_coarse = np.arange(0.0, t_end + 0.5 * dt_coarse, dt_coarse, dtype=np.float64)
    tg_fine = np.arange(0.0, t_end + 0.5 * dt_fine, dt_fine, dtype=np.float64)

    A = np.array([[-lam]], dtype=np.float64)

    def reaction_sin(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
        out = np.empty_like(y)
        out[0, 0] = np.sin(t)
        return out

    base = make_constant_base_builder(A)
    tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

    # Coarse solve
    core_c = _make_scalar_core(tg_coarse)
    core_c.set_initial_state(_scalar_state(y0))
    solver_c = CoreSolver(core_c, operators=None, operator_axis="state")
    solver_c.run_imex_trbdf2(
        reaction_sin,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    # Fine reference solve
    core_f = _make_scalar_core(tg_fine)
    core_f.set_initial_state(_scalar_state(y0))
    solver_f = CoreSolver(core_f, operators=None, operator_axis="state")
    solver_f.run_imex_trbdf2(
        reaction_sin,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    assert core_c.state_array is not None
    assert core_f.state_array is not None

    y_c = core_c.state_array[:, 0, 0]
    y_f = core_f.state_array[:, 0, 0]

    assert np.all(np.isfinite(y_c))
    assert np.all(np.isfinite(y_f))

    y_c_final = float(y_c[-1])
    y_f_final = float(y_f[-1])

    assert abs(y_c_final - y_f_final) < 6e-3


# -------------------------------------------------------------------
# 3) Order sanity: linear decay should show ~2nd order convergence
# -------------------------------------------------------------------


def test_trbdf2_second_order_convergence_linear_decay() -> None:
    """TR-BDF2 should demonstrate ~second-order convergence on linear decay.

    System:
        y' = A y, A=-1, F=0
    Compare final-time error for dt and dt/2; expect error ratio ~4.
    """
    lam = 1.0
    y0 = 1.0
    t_end = 2.0

    A = np.array([[-lam]], dtype=np.float64)

    def reaction_zero(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.zeros_like(y)

    base = make_constant_base_builder(A)
    tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

    def run(dt: float) -> float:
        tg = np.arange(0.0, t_end + 0.5 * dt, dt, dtype=np.float64)
        core = _make_scalar_core(tg)
        core.set_initial_state(_scalar_state(y0))
        solver = CoreSolver(core, operators=None, operator_axis="state")
        solver.run_imex_trbdf2(
            reaction_zero,
            operators_tr=tr_factory,
            operators_bdf2=be_factory,
        )
        assert core.state_array is not None
        y_num = _unpack_scalar(core.state_array[-1])
        y_true = float(y0 * np.exp(-lam * tg[-1]))
        return abs(y_num - y_true)

    # Keep these small but not too small to avoid cancellation to zero error.
    err_dt = run(5e-4)
    err_dt2 = run(2.5e-4)

    if err_dt2 == 0.0:
        pytest.skip("Reference error is numerically zero; cannot assess convergence ratio.")

    ratio = err_dt / err_dt2

    # Expect ~4 in asymptotic regime; allow slack.
    assert ratio > 2.5


# -------------------------------------------------------------------
# 4) Fallback operators: when operators_tr/bdf2 are omitted, solver defaults are used
# -------------------------------------------------------------------


def test_trbdf2_uses_default_operators_when_stage_operators_omitted() -> None:
    """If operators_tr and operators_bdf2 are None, CoreSolver uses its default operators.

    We configure the solver with a simple implicit Euler operator for a stiff decay and
    verify it runs and damps (finite and small).
    """
    lam = 200.0
    y0 = 1.0
    dt = 0.05
    t_end = 1.0
    tg = np.arange(0.0, t_end + 0.5 * dt, dt, dtype=np.float64)

    A = np.array([[-lam]], dtype=np.float64)

    def reaction_zero(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.zeros_like(y)

    # Default operators: BE-style operators, built with dt_scale=dt.
    L_be, R_be = build_implicit_euler_operators(A, dt_scale=dt)

    core = _make_scalar_core(tg)
    core.set_initial_state(_scalar_state(y0))
    solver = CoreSolver(core, operators=(L_be, R_be), operator_axis="state")

    solver.run_imex_trbdf2(
        reaction_zero,
        operators_tr=None,
        operators_bdf2=None,
    )

    assert core.state_array is not None
    y_final = _unpack_scalar(core.state_array[-1])

    assert np.isfinite(y_final)
    assert abs(y_final) < 1e-4


# -------------------------------------------------------------------
# 5) Non-uniform time grid: should run stably and match a fine reference
# -------------------------------------------------------------------


def test_trbdf2_nonuniform_time_grid_matches_fine_reference() -> None:
    """TR-BDF2 should handle non-uniform dt without NaNs/Infs and be reasonably accurate.

    System:
        y' = -lambda y + sin(t)
    Compare a non-uniform coarse grid to a uniform fine reference grid (same method).
    """
    lam = 40.0
    y0 = 0.1

    tg = np.array(
        [
            0.0,
            0.013,
            0.029,
            0.041,
            0.060,
            0.082,
            0.107,
            0.135,
            0.166,
            0.200,
            0.238,
            0.279,
            0.323,
            0.370,
            0.420,
            0.473,
            0.529,
            0.588,
            0.650,
            0.715,
            0.783,
            0.854,
            0.928,
            1.0,
        ],
        dtype=np.float64,
    )

    dt_fine = 0.001
    tg_f = np.arange(0.0, float(tg[-1]) + 0.5 * dt_fine, dt_fine, dtype=np.float64)

    A = np.array([[-lam]], dtype=np.float64)

    def reaction_sin(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
        out = np.empty_like(y)
        out[0, 0] = np.sin(t)
        return out

    base = make_constant_base_builder(A)
    tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

    # Non-uniform coarse solve
    core = _make_scalar_core(tg)
    core.set_initial_state(_scalar_state(y0))
    solver = CoreSolver(core, operators=None, operator_axis="state")
    solver.run_imex_trbdf2(
        reaction_sin,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    # Fine reference solve
    core_f = _make_scalar_core(tg_f)
    core_f.set_initial_state(_scalar_state(y0))
    solver_f = CoreSolver(core_f, operators=None, operator_axis="state")
    solver_f.run_imex_trbdf2(
        reaction_sin,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    assert core.state_array is not None
    assert core_f.state_array is not None

    y_coarse = core.state_array[:, 0, 0]
    y_fine = core_f.state_array[:, 0, 0]

    assert np.all(np.isfinite(y_coarse))
    assert np.all(np.isfinite(y_fine))

    y_c_final = float(y_coarse[-1])
    y_f_final = float(y_fine[-1])

    assert abs(y_c_final - y_f_final) < 0.01


# -------------------------------------------------------------------
# 6) Time-varying stage operators: callable factories that depend on dt/ctx
# -------------------------------------------------------------------


def test_trbdf2_time_varying_stage_operators_callable_dt_dependent() -> None:
    """TR-BDF2 should accept callable stage-operator factories with signature (dt, scale, ctx).

    We construct dt-dependent operators for a scalar linear decay y' = -lam(t)*y,
    where lam(t) varies with stage time:
        lam(t) = lam0 * (1 + 0.5*sin(t))

    We only check that:
        - it runs without error
        - results are finite
        - decay occurs (final magnitude smaller than initial for positive lam0)
    """
    lam0 = 30.0
    y0 = 1.0

    # Moderate dt; nonuniform to ensure dt plumbing is used.
    tg = np.array([0.0, 0.03, 0.07, 0.10, 0.16, 0.23, 0.31, 0.40], dtype=np.float64)

    def reaction_zero(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.zeros_like(y)

    # Base builder uses ctx.t to vary lambda.
    def base_builder(ctx: StageOperatorContext):
        lam_t = lam0 * (1.0 + 0.5 * np.sin(ctx.t))
        return np.array([[-lam_t]], dtype=np.float64)

    tr_factory = make_stage_operator_factory(base_builder, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base_builder, scheme="implicit-euler")

    core = _make_scalar_core(tg)
    core.set_initial_state(_scalar_state(y0))
    solver = CoreSolver(core, operators=None, operator_axis="state")
    solver.run_imex_trbdf2(
        reaction_zero,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    assert core.state_array is not None
    y = core.state_array[:, 0, 0]

    assert np.all(np.isfinite(y))
    assert abs(float(y[-1])) < abs(float(y[0]))
