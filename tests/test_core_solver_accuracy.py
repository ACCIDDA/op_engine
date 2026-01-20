# tests/test_core_solver_accuracy.py
"""Accuracy and convergence tests for op_engine.core_solver.CoreSolver.run().

Design principle:
- Convergence/order is assessed by *halving dt* and comparing both runs against a
  *numerical reference* computed on a much finer grid (same method), i.e.
      err(dt)   = |y(dt)   - y_ref|
      err(dt/2) = |y(dt/2) - y_ref|
      p ~= log2(err(dt)/err(dt/2))
This avoids false failures when the method is a *split* scheme (IMEX), where the
analytic solution of the unsplit system may not align cleanly with the split map.

Coverage in this file:
1) Explicit ODE methods: euler, heun (order checks vs numerical reference)
2) IMEX split linear: imex-euler, imex-heun-tr, imex-trbdf2 (order checks)
3) Non-uniform output grids (ODE + IMEX representative) vs fine numerical reference
4) Adaptive stepping sanity: adaptive should not be worse than fixed-step for same grid
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from op_engine.core_solver import (
    AdaptiveConfig,
    CoreOperators,
    CoreSolver,
    DtControllerConfig,
    OperatorSpecs,
    StageOperatorFactory,
)
from op_engine.core_solver import (
    RunConfig as SolverRunConfig,
)
from op_engine.matrix_ops import (
    build_implicit_euler_operators,
    build_trapezoidal_operators,
    make_constant_base_builder,
    make_stage_operator_factory,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]

RHSFunction = Callable[[float, "FloatArray"], "FloatArray"]
MethodName = Literal["euler", "heun", "imex-euler", "imex-heun-tr", "imex-trbdf2"]


@dataclass(slots=True, frozen=True)
class ScalarRunCase:
    """Parameters for running a scalar CoreSolver case."""

    method: MethodName
    time_grid: FloatArray
    y0: float
    rhs: RHSFunction
    operators: CoreOperators | StageOperatorFactory | None = None
    operators_tr: CoreOperators | StageOperatorFactory | None = None
    operators_bdf2: CoreOperators | StageOperatorFactory | None = None
    adaptive: bool = False
    dt_init: float | None = None
    rtol: float = 1e-7
    atol: float = 1e-10


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_scalar_core(time_grid: FloatArray) -> ModelCore:
    """
    Scalar ModelCore with state tensor shape (1, 1).

    Args:
        time_grid: 1D array of output times.

    Returns:
        ModelCore instance.
    """
    opts = ModelCoreOptions(
        other_axes=(),
        axis_names=("state", "subgroup"),
        store_history=True,
        dtype=np.float64,
    )
    return ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid, options=opts)


def _scalar_state(y: float) -> FloatArray:
    return np.array([[y]], dtype=np.float64)


def _unpack_scalar(state: FloatArray) -> float:
    return float(state[0, 0])


def _time_grid_uniform(t_end: float, dt: float) -> FloatArray:
    return np.arange(0.0, t_end + 0.5 * dt, dt, dtype=np.float64)


def _run_scalar(case: ScalarRunCase) -> float:
    """
    Run a scalar problem and return the final value y(t_end).

    Args:
        case: ScalarRunCase with run parameters.

    Returns:
        y(t_end): Final solution value.
    """
    core = _make_scalar_core(case.time_grid)
    core.set_initial_state(_scalar_state(case.y0))

    # Keep the solver default operators empty; drive everything via RunConfig.
    solver = CoreSolver(core, operators=None, operator_axis="state")

    cfg = SolverRunConfig(
        method=str(case.method),
        adaptive=bool(case.adaptive),
        strict=True,
        operators=OperatorSpecs(
            default=case.operators,
            tr=case.operators_tr,
            bdf2=case.operators_bdf2,
        ),
        adaptive_cfg=AdaptiveConfig(
            rtol=float(case.rtol),
            atol=case.atol,
            dt_init=case.dt_init,
        ),
        dt_controller=DtControllerConfig(),
    )

    solver.run(case.rhs, config=cfg)

    assert core.state_array is not None
    return _unpack_scalar(core.state_array[-1])


def _order_from_two_errors(err_dt: float, err_dt2: float) -> float:
    """
    Obtain order estimate p from two errors at dt and dt/2.

    Args:
        err_dt: Error at step size dt.
        err_dt2: Error at step size dt/2.

    Returns:
        p: Estimated order.
    """
    if err_dt <= 0.0 or err_dt2 <= 0.0:
        return float("nan")
    return float(np.log(err_dt / err_dt2) / np.log(2.0))


def _reference_solution(case: ScalarRunCase, *, t_end: float, dt_ref: float) -> float:
    """
    Compute y_ref on a much finer uniform grid (same method).

    Args:
        case: Base ScalarRunCase.
        t_end: End time.
        dt_ref: Fine grid step size.

    Returns:
        y_ref: Reference solution at t_end.
    """
    tg_ref = _time_grid_uniform(t_end, dt_ref)
    ref_case = replace(case, time_grid=tg_ref, adaptive=False, dt_init=None)
    return _run_scalar(ref_case)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ode_decay_setup() -> tuple[float, float]:
    """
    Setup (k, y0) for y' = -k y.

    Args:
        None

    Returns:
        k: Decay rate.
        y0: Initial condition.
    """
    return 2.3, 1.0


@pytest.fixture(scope="module")
def imex_linear_split_setup() -> tuple[FloatArray, float, float]:
    """
    Setup for y' = A y + lam*y with constant split.

    Args:
        None

    Returns:
        operator: Linear operator A as (1, 1) ndarray.
        lam_react: Reaction rate lambda.
        y0: Initial condition.
    """
    operator = np.array([[-3.0]], dtype=np.float64)
    lam_react = -0.7
    y0 = 1.0
    return operator, lam_react, y0


# -----------------------------------------------------------------------------
# 1) Explicit methods: order checks vs numerical reference
# -----------------------------------------------------------------------------


EXPLICIT_CASES: list[tuple[MethodName, float]] = [
    ("euler", 1.0),
    ("heun", 2.0),
]


@pytest.mark.parametrize(("method", "expected_order"), EXPLICIT_CASES)
def test_explicit_methods_convergence_order_against_numerical_reference(
    method: MethodName,
    expected_order: float,
    ode_decay_setup: tuple[float, float],
) -> None:
    """Convergence order on y'=-k y, using numerical reference on dt_ref = dt/16."""
    k, y0 = ode_decay_setup
    t_end = 1.0

    def rhs_decay(_t: float, y: FloatArray) -> FloatArray:
        out = np.empty_like(y)
        out[0, 0] = -k * float(y[0, 0])
        return out

    dt = 2e-2
    dt2 = dt / 2.0
    dt_ref = dt / 16.0

    base_case = ScalarRunCase(
        method=method,
        time_grid=_time_grid_uniform(t_end, dt),
        y0=y0,
        rhs=rhs_decay,
    )
    y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)

    y_dt = _run_scalar(base_case)
    y_dt2 = _run_scalar(
        replace(base_case, time_grid=_time_grid_uniform(t_end, dt2)),
    )

    err_dt = abs(y_dt - y_ref)
    err_dt2 = abs(y_dt2 - y_ref)
    p = _order_from_two_errors(err_dt, err_dt2)

    assert np.isfinite(p), (
        f"Non-finite order estimate p={p} (err_dt={err_dt}, err_dt/2={err_dt2})"
    )
    msg_1 = f"Expected ~1st order; got p={p} (err_dt={err_dt}, err_dt/2={err_dt2})"
    msg_2 = f"Expected ~2nd order; got p={p} (err_dt={err_dt}, err_dt/2={err_dt2})"
    if expected_order < 1.5:
        assert p > 0.7, msg_1
    else:
        assert p > 1.3, msg_2


# -----------------------------------------------------------------------------
# 2) IMEX methods: order checks on linear split vs numerical reference
# -----------------------------------------------------------------------------


IMEX_CASES: list[tuple[MethodName, float]] = [
    ("imex-euler", 1.0),
    ("imex-heun-tr", 2.0),
    ("imex-trbdf2", 2.0),
]


@pytest.mark.parametrize(("method", "expected_order"), IMEX_CASES)
def test_imex_methods_convergence_order_on_linear_split_against_numerical_reference(  # noqa: PLR0914
    method: MethodName,
    expected_order: float,
    imex_linear_split_setup: tuple[FloatArray, float, float],
) -> None:
    """Order check on y' = A y + lam*y (constant split), using numerical reference."""
    operator, lam_react, y0 = imex_linear_split_setup
    t_end = 1.0

    def reaction_rhs(_t: float, y: FloatArray) -> FloatArray:
        out = np.empty_like(y)
        out[0, 0] = lam_react * float(y[0, 0])
        return out

    dt = 1e-2
    dt2 = dt / 2.0
    dt_ref = dt / 16.0

    if method == "imex-euler":

        def solve_at(d: float) -> float:
            left_d, right_d = build_implicit_euler_operators(operator, dt_scale=d)
            return _run_scalar(
                ScalarRunCase(
                    method=method,
                    time_grid=_time_grid_uniform(t_end, d),
                    y0=y0,
                    rhs=reaction_rhs,
                    operators=(left_d, right_d),
                )
            )

        left_ref, right_ref = build_implicit_euler_operators(operator, dt_scale=dt_ref)
        y_ref = _run_scalar(
            ScalarRunCase(
                method=method,
                time_grid=_time_grid_uniform(t_end, dt_ref),
                y0=y0,
                rhs=reaction_rhs,
                operators=(left_ref, right_ref),
            )
        )
        y_dt = solve_at(dt)
        y_dt2 = solve_at(dt2)

    elif method == "imex-heun-tr":

        def solve_at(d: float) -> float:
            left_d, right_d = build_trapezoidal_operators(operator, dt_scale=d)
            return _run_scalar(
                ScalarRunCase(
                    method=method,
                    time_grid=_time_grid_uniform(t_end, d),
                    y0=y0,
                    rhs=reaction_rhs,
                    operators=(left_d, right_d),
                )
            )

        left_ref, right_ref = build_trapezoidal_operators(operator, dt_scale=dt_ref)
        y_ref = _run_scalar(
            ScalarRunCase(
                method=method,
                time_grid=_time_grid_uniform(t_end, dt_ref),
                y0=y0,
                rhs=reaction_rhs,
                operators=(left_ref, right_ref),
            )
        )
        y_dt = solve_at(dt)
        y_dt2 = solve_at(dt2)

    else:
        assert method == "imex-trbdf2"
        base = make_constant_base_builder(operator)
        tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
        be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

        base_case = ScalarRunCase(
            method=method,
            time_grid=_time_grid_uniform(t_end, dt),
            y0=y0,
            rhs=reaction_rhs,
            operators_tr=tr_factory,
            operators_bdf2=be_factory,
        )
        y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)
        y_dt = _run_scalar(base_case)
        y_dt2 = _run_scalar(
            replace(base_case, time_grid=_time_grid_uniform(t_end, dt2))
        )

    err_dt = abs(y_dt - y_ref)
    err_dt2 = abs(y_dt2 - y_ref)
    p = _order_from_two_errors(err_dt, err_dt2)

    assert np.isfinite(p), (
        f"Non-finite order estimate p={p} (err_dt={err_dt}, err_dt/2={err_dt2})"
    )
    msg = f"Expected ~1st order; got p={p} (err_dt={err_dt}, err_dt/2={err_dt2})"
    if expected_order < 1.5:
        assert p > 0.7, msg
    else:
        assert p > 1.3, msg


# -----------------------------------------------------------------------------
# 3) Non-uniform output grid checks (representative ODE + IMEX)
# -----------------------------------------------------------------------------


def test_nonuniform_output_grid_heun_matches_fine_reference() -> None:
    """Heun on a nonuniform output grid should be stable and close to fine reference."""
    y0 = 0.0
    t_end = 1.0

    def rhs_sin(t: float, y: FloatArray) -> FloatArray:
        out = np.empty_like(y)
        out[0, 0] = np.sin(t)
        return out

    tg = np.array(
        [
            0.0,
            0.03,
            0.05,
            0.09,
            0.15,
            0.21,
            0.25,
            0.32,
            0.4,
            0.44,
            0.5,
            0.57,
            0.6,
            0.68,
            0.72,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
        ],
        dtype=np.float64,
    )
    assert float(tg[-1]) == pytest.approx(t_end)

    dt_ref = 1e-4
    base_case = ScalarRunCase(method="heun", time_grid=tg, y0=y0, rhs=rhs_sin)
    y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)
    y_non = _run_scalar(base_case)

    assert np.isfinite(y_non)
    assert abs(y_non - y_ref) < 5e-4


def test_nonuniform_output_grid_imex_trbdf2_matches_fine_reference() -> None:
    """IMEX TR-BDF2 should handle nonuniform output grids with dt-aware factories."""
    operator = np.array([[-40.0]], dtype=np.float64)
    y0 = 0.1
    t_end = 1.0

    def reaction_sin(t: float, y: FloatArray) -> FloatArray:
        out = np.empty_like(y)
        out[0, 0] = np.sin(t)
        return out

    tg = np.array(
        [
            0.0,
            0.03,
            0.05,
            0.09,
            0.12,
            0.15,
            0.21,
            0.25,
            0.3,
            0.32,
            0.36,
            0.4,
            0.44,
            0.5,
            0.52,
            0.57,
            0.6,
            0.64,
            0.68,
            0.72,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
        ],
        dtype=np.float64,
    )
    assert float(tg[-1]) == pytest.approx(t_end)

    base = make_constant_base_builder(operator)
    tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

    dt_ref = 5e-4
    base_case = ScalarRunCase(
        method="imex-trbdf2",
        time_grid=tg,
        y0=y0,
        rhs=reaction_sin,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )
    y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)
    y_non = _run_scalar(base_case)

    assert np.isfinite(y_non)
    assert abs(y_non - y_ref) < 5e-3


# -----------------------------------------------------------------------------
# 4) Adaptive stepping accuracy sanity
# -----------------------------------------------------------------------------


ADAPTIVE_CASES: list[MethodName] = ["heun", "imex-heun-tr", "imex-trbdf2"]


@pytest.mark.parametrize("method", ADAPTIVE_CASES)
def test_adaptive_not_worse_than_fixed_against_reference(  # noqa: PLR0914
    method: MethodName,
    imex_linear_split_setup: tuple[FloatArray, float, float],
) -> None:
    """Adaptive stepping should not be worse than fixed-step at the same output grid."""
    t_end = 1.0
    dt_out = 2e-2
    tg = _time_grid_uniform(t_end, dt_out)

    operator, lam_react, y0 = imex_linear_split_setup

    if method == "heun":

        def rhs(t: float, y: FloatArray) -> FloatArray:
            out = np.empty_like(y)
            out[0, 0] = np.cos(t)
            return out

        dt_ref = 1e-4
        base_case = ScalarRunCase(method="heun", time_grid=tg, y0=0.0, rhs=rhs)
        y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)

        y_fixed = _run_scalar(base_case)
        y_adapt = _run_scalar(
            replace(
                base_case,
                adaptive=True,
                dt_init=dt_out,
                rtol=1e-7,
                atol=1e-10,
            )
        )

    else:

        def rhs(t: float, y: FloatArray) -> FloatArray:  # noqa: ARG001
            out = np.empty_like(y)
            out[0, 0] = lam_react * float(y[0, 0])
            return out

        base = make_constant_base_builder(operator)

        if method == "imex-heun-tr":
            op_default = make_stage_operator_factory(base, scheme="trapezoidal")

            dt_ref = 2e-3
            base_case = ScalarRunCase(
                method="imex-heun-tr",
                time_grid=tg,
                y0=y0,
                rhs=rhs,
                operators=op_default,
            )
            y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)

            y_fixed = _run_scalar(base_case)
            y_adapt = _run_scalar(
                replace(
                    base_case,
                    adaptive=True,
                    dt_init=dt_out,
                    rtol=1e-7,
                    atol=1e-10,
                )
            )

        else:
            assert method == "imex-trbdf2"
            tr_factory = make_stage_operator_factory(base, scheme="trapezoidal")
            be_factory = make_stage_operator_factory(base, scheme="implicit-euler")

            dt_ref = 2e-3
            base_case = ScalarRunCase(
                method="imex-trbdf2",
                time_grid=tg,
                y0=y0,
                rhs=rhs,
                operators_tr=tr_factory,
                operators_bdf2=be_factory,
            )
            y_ref = _reference_solution(base_case, t_end=t_end, dt_ref=dt_ref)

            y_fixed = _run_scalar(base_case)
            y_adapt = _run_scalar(
                replace(
                    base_case,
                    adaptive=True,
                    dt_init=dt_out,
                    rtol=1e-7,
                    atol=1e-10,
                )
            )

    err_fixed = abs(y_fixed - y_ref)
    err_adapt = abs(y_adapt - y_ref)

    assert np.isfinite(err_fixed)
    assert np.isfinite(err_adapt)

    if err_fixed > 1e-12:
        assert err_adapt <= 1.25 * err_fixed, (
            f"Adaptive worse than fixed: method={method} "
            f"err_adapt={err_adapt} err_fixed={err_fixed}"
        )
