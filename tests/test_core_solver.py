# tests/test_core_solver.py
"""Unit tests for the CoreSolver class.

This module contains tests that verify:
- CoreSolver operates correctly in Euler mode without operators.
- CoreSolver applies (L, R) operators along the default "state" axis and batches
  over remaining axes.
- CoreSolver can apply operators along a non-default axis (e.g., "subgroup").
- CoreSolver supports predictor-corrector operator tuples (P, L, R).
- CoreSolver.run_imex implements an explicit Heun step when A = 0 (no operators),
  including on non-uniform time grids.
- CoreSolver validates operator sizes against the configured operator axis.
- CoreSolver.run_imex_trbdf2:
    * works with no operators (purely explicit, second order on smooth problems),
    * supports stage-operator factories that depend on time (ctx.t),
    * supports stage-operator factories that depend on stage state (ctx.y),
    * passes correct ctx.stage labels ("tr", "bdf2") to factories.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_engine.core_solver import CoreSolver, ReactionRHSFunction, RHSFunction
from op_engine.matrix_ops import StageOperatorContext, implicit_solve
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


def _make_core(
    *,
    n_states: int,
    n_subgroups: int,
    time_grid: NDArray[np.floating],
    options: ModelCoreOptions | None = None,
) -> ModelCore:
    """Create a ModelCore with a fixed store_history default used in tests."""
    opts = options or ModelCoreOptions(store_history=True)
    # Ensure history on unless a test explicitly overrides it.
    if options is None:
        opts = ModelCoreOptions(store_history=True)
    return ModelCore(
        n_states=n_states,
        n_subgroups=n_subgroups,
        time_grid=np.asarray(time_grid, dtype=float),
        options=opts,
    )


# -------------------------------------------------------------------
# Legacy / base behavior
# -------------------------------------------------------------------


def test_core_solver_euler_mode_uses_rhs_as_next_state() -> None:
    """Test CoreSolver in Euler-like mode (no operators)."""
    time_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=time_grid)

    init = np.array([[0.0]], dtype=float)
    core.set_initial_state(init)

    def rhs_func(t: float, state: FloatArray) -> FloatArray:  # noqa: ARG001
        return state + 1.0

    rhs: RHSFunction = rhs_func
    solver = CoreSolver(core, operators=None)
    solver.run(rhs)

    assert core.state_array is not None
    # state_array shape: (n_timesteps, *state_shape) == (4, 1, 1)
    assert np.allclose(core.state_array[:, 0, 0], np.array([0.0, 1.0, 2.0, 3.0]))


def test_core_solver_cn_matches_direct_implicit_solve_batched_state_axis() -> None:
    """CN path matches direct implicit_solve when batching over other axes."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(3,), store_history=True)
    core = _make_core(n_states=5, n_subgroups=2, time_grid=time_grid, options=opts)

    init = np.arange(np.prod(core.state_shape), dtype=float).reshape(core.state_shape)
    core.set_initial_state(init)

    n = core.state_shape[core.axis_index("state")]
    left = 2.0 * np.eye(n, dtype=float)
    right = np.eye(n, dtype=float)

    def rhs_func(_t: float, state: FloatArray) -> FloatArray:
        return state

    rhs: RHSFunction = rhs_func
    solver = CoreSolver(core, operators=(left, right))

    # Manual batched solve via ModelCore reshaping helpers.
    rhs2d, original_shape, _ = core.reshape_for_axis_solve(init, "state")
    out2d = np.asarray(implicit_solve(left, right, rhs2d), dtype=float)
    manual = core.unreshape_from_axis_solve(out2d, original_shape, "state")

    solver.run(rhs)

    assert core.current_step == 1
    x1 = core.get_state_at(1)
    assert np.allclose(x1, manual)


def test_core_solver_operator_axis_subgroup_applies_along_subgroup() -> None:
    """Operators can act along a non-default axis (e.g., subgroup)."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=2, n_subgroups=3, time_grid=time_grid)

    init = np.arange(6, dtype=float).reshape(core.state_shape)  # (2, 3)
    core.set_initial_state(init)

    m = core.state_shape[core.axis_index("subgroup")]
    left = 2.0 * np.eye(m, dtype=float)
    right = np.eye(m, dtype=float)

    def rhs_func(_t: float, state: FloatArray) -> FloatArray:
        return state

    rhs: RHSFunction = rhs_func
    solver = CoreSolver(core, operators=(left, right), operator_axis="subgroup")
    solver.run(rhs)

    # Along subgroup axis: y = 0.5 * x
    expected = 0.5 * init
    x1 = core.get_state_at(1)
    assert np.allclose(x1, expected)


def test_core_solver_predictor_corrector_applies_predictor_then_solve() -> None:
    """Predictor-corrector tuple (P, L, R) is applied as documented."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=4, n_subgroups=1, time_grid=time_grid)

    init = np.arange(4, dtype=float).reshape(core.state_shape)  # (4, 1)
    core.set_initial_state(init)

    n = core.state_shape[core.axis_index("state")]
    predictor = 3.0 * np.eye(n, dtype=float)
    left = 2.0 * np.eye(n, dtype=float)
    right = np.eye(n, dtype=float)

    def rhs_func(_t: float, state: FloatArray) -> FloatArray:
        return state

    rhs: RHSFunction = rhs_func
    solver = CoreSolver(core, operators=(predictor, left, right))
    solver.run(rhs)

    # rhs_tilde = 3 * rhs; solve(2I y = I rhs_tilde) => y = 0.5 * rhs_tilde = 1.5 * rhs
    expected = 1.5 * init
    x1 = core.get_state_at(1)
    assert np.allclose(x1, expected)


def test_core_solver_run_imex_no_operators_constant_reaction_nonuniform_dt() -> None:
    """run_imex reduces to explicit Heun when A = 0 (operators=None)."""
    time_grid = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)  # non-uniform
    core = _make_core(n_states=1, n_subgroups=1, time_grid=time_grid)

    init = np.array([[0.0]], dtype=float)
    core.set_initial_state(init)

    def reaction_rhs(t: float, state: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.ones_like(state)

    reaction: ReactionRHSFunction = reaction_rhs
    solver = CoreSolver(core, operators=None)
    solver.run_imex(reaction)

    # For y' = 1, Heun is exact => y(T) = T = 1.0
    assert np.allclose(core.current_state[0, 0], 1.0, atol=1e-12, rtol=0.0)

    assert core.state_array is not None
    # Should increase by each dt step: [0, 0.1, 0.4, 1.0]
    assert np.allclose(core.state_array[:, 0, 0], time_grid)


def test_core_solver_run_imex_no_reaction_with_operators_is_pure_linear_step() -> None:
    """If F=0 and operators are present, run_imex applies only the implicit map."""
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=time_grid)

    init = np.array([[8.0]], dtype=float)
    core.set_initial_state(init)

    n = 1
    left = 2.0 * np.eye(n, dtype=float)
    right = np.eye(n, dtype=float)

    def reaction_rhs(_t: float, state: FloatArray) -> FloatArray:
        return np.zeros_like(state)

    reaction: ReactionRHSFunction = reaction_rhs
    solver = CoreSolver(core, operators=(left, right))
    solver.run_imex(reaction)

    # Each step applies y <- 0.5 * y; two steps => y_final = 8 * (0.5^2) = 2
    assert np.allclose(core.current_state[0, 0], 2.0, atol=1e-12, rtol=0.0)


def test_core_solver_operator_size_mismatch_raises() -> None:
    """Operator dimension must match the operator_axis length."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=2, n_subgroups=3, time_grid=time_grid)

    # operator_axis="subgroup" => length 3, but provide 2x2 operator.
    left = np.eye(2, dtype=float)
    right = np.eye(2, dtype=float)

    with pytest.raises(ValueError, match="Operator axis length"):
        _ = CoreSolver(core, operators=(left, right), operator_axis="subgroup")


# -------------------------------------------------------------------
# New capability: TR-BDF2 explicit fallback with no operators
# -------------------------------------------------------------------


def test_core_solver_run_imex_trbdf2_no_operators_constant_reaction_exact() -> None:
    """run_imex_trbdf2 runs without operators and is exact on y' = 1."""
    time_grid = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)  # non-uniform
    core = _make_core(n_states=1, n_subgroups=1, time_grid=time_grid)

    init = np.array([[0.0]], dtype=float)
    core.set_initial_state(init)

    def reaction_rhs(_t: float, state: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.ones_like(state)

    solver = CoreSolver(core, operators=None)
    solver.run_imex_trbdf2(reaction_rhs, operators_tr=None, operators_bdf2=None)

    assert np.allclose(core.current_state[0, 0], 1.0, atol=1e-12, rtol=0.0)
    assert core.state_array is not None
    assert np.allclose(core.state_array[:, 0, 0], time_grid, atol=1e-12, rtol=0.0)


# -------------------------------------------------------------------
# New capability: TR-BDF2 stage-operator factories with ctx (time/state/stage)
# -------------------------------------------------------------------


def test_core_solver_trbdf2_stage_operator_factory_receives_time_and_stage_labels() -> None:
    """TR-BDF2 passes ctx.t and ctx.stage to operator factories.

    We use a single-step run with F=0 so the method is purely the linear map.
    The factories build 1x1 operators whose coefficients depend on ctx.t, and we
    also assert the ctx.stage labels are as expected.
    """
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=time_grid)
    core.set_initial_state(np.array([[1.0]], dtype=float))

    # F(t,y) = 0
    def reaction_rhs(_t: float, state: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.zeros_like(state)

    seen: list[StageOperatorContext] = []

    # TR stage: trapezoidal-like operators with a(t) = ctx.t
    def tr_factory(dt: float, scale: float, ctx: StageOperatorContext):
        seen.append(ctx)
        a = float(ctx.t)  # time-varying "A"
        s = float(dt) * float(scale)
        left = np.array([[1.0 - 0.5 * s * a]], dtype=float)
        right = np.array([[1.0 + 0.5 * s * a]], dtype=float)
        return left, right

    # BDF2 stage: implicit-euler-like operators with a(t) = ctx.t
    def bdf_factory(dt: float, scale: float, ctx: StageOperatorContext):
        seen.append(ctx)
        a = float(ctx.t)
        s = float(dt) * float(scale)
        left = np.array([[1.0 - s * a]], dtype=float)
        right = np.array([[1.0]], dtype=float)
        return left, right

    solver = CoreSolver(core, operators=None)
    solver.run_imex_trbdf2(
        reaction_rhs,
        operators_tr=tr_factory,
        operators_bdf2=bdf_factory,
    )

    # We expect exactly two factory calls: TR stage, then BDF2 stage.
    assert len(seen) == 2

    ctx_tr, ctx_bdf = seen
    assert ctx_tr.stage == "tr"
    assert ctx_bdf.stage == "bdf2"

    # Time points:
    gamma = float(2.0 - np.sqrt(2.0))
    assert np.isclose(ctx_tr.t, gamma, atol=0.0, rtol=0.0)
    assert np.isclose(ctx_bdf.t, 1.0, atol=0.0, rtol=0.0)

    # Both contexts should include a stage state tensor shaped like the model state.
    assert ctx_tr.y.shape == core.state_shape
    assert ctx_bdf.y.shape == core.state_shape

    # Sanity: purely linear map with stable operators should keep the solution finite.
    assert np.isfinite(core.current_state[0, 0])


def test_core_solver_trbdf2_stage_operator_factory_can_depend_on_stage_state() -> None:
    """Factories can depend on ctx.y (stage state) to produce different operators.

    This is a plumbing test: we make the BDF2 left operator depend on ctx.y in a
    way that (a) is stable and (b) produces a detectable effect compared to a
    time-only operator.
    """
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=time_grid)
    core.set_initial_state(np.array([[2.0]], dtype=float))

    def reaction_rhs(_t: float, state: FloatArray) -> FloatArray:  # noqa: ARG001
        return np.zeros_like(state)

    # TR stage: choose constant A=0 so y1 == y0 deterministically.
    def tr_factory(_dt: float, _scale: float, ctx: StageOperatorContext):
        assert ctx.stage == "tr"
        left = np.array([[1.0]], dtype=float)
        right = np.array([[1.0]], dtype=float)
        return left, right

    # BDF2 stage: set A = alpha * |y_stage1| so the left operator changes with state.
    alpha = 0.1

    def bdf_factory(dt: float, scale: float, ctx: StageOperatorContext):
        assert ctx.stage == "bdf2"
        ymag = float(np.abs(np.asarray(ctx.y, dtype=float)[0, 0]))
        a = alpha * ymag
        s = float(dt) * float(scale)
        left = np.array([[1.0 - s * a]], dtype=float)
        right = np.array([[1.0]], dtype=float)
        return left, right

    solver = CoreSolver(core, operators=None)
    solver.run_imex_trbdf2(
        reaction_rhs,
        operators_tr=tr_factory,
        operators_bdf2=bdf_factory,
    )

    # With F=0 and TR stage identity, ctx.y_stage1 == y0 == 2, so a = 0.2.
    # That yields a non-identity left op and therefore y_{n+1} != y0 in general.
    # We only assert it changes and remains finite (plumbing + stability).
    y_final = float(core.current_state[0, 0])
    assert np.isfinite(y_final)
    assert not np.isclose(y_final, 2.0)
