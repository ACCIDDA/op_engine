# tests/test_core_solver_plumbing.py
"""Plumbing/contract tests for op_engine.core_solver.CoreSolver.

These tests focus on CoreSolver.run() semantics and operator plumbing, *not*
high-order accuracy (handled in a separate accuracy/convergence file).

They are written to match the current CoreSolver behavior:

- ModelCore.time_grid is treated as *output times*.
- If adaptive=False, the solver takes exactly one attempted step per output interval.
  (Some methods internally use step-doubling kernels even in the non-adaptive path.)
- Explicit methods ("euler", "heun") ignore operators and emit RuntimeWarning
  when strict=True and operators are provided.
- IMEX methods require operators (or factories) and require factories when dt varies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine.core_solver import AdaptiveConfig, CoreSolver, RunConfig
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_core(
    *,
    n_states: int,
    n_subgroups: int,
    time_grid: NDArray[np.floating],
    options: ModelCoreOptions | None = None,
) -> ModelCore:
    """
    Create a ModelCore with store_history=True by default for inspection.

    Args:
        n_states: Number of states.
        n_subgroups: Number of subgroups.
        time_grid: 1D array of output times.
        options: Optional ModelCoreOptions to override defaults.

    Returns:
        Configured ModelCore instance.
    """
    opts = options or ModelCoreOptions(store_history=True)
    if options is None:
        opts = ModelCoreOptions(store_history=True)
    return ModelCore(
        n_states=n_states,
        n_subgroups=n_subgroups,
        time_grid=np.asarray(time_grid, dtype=float),
        options=opts,
    )


def _reaction_zero(_t: float, y: FloatArray) -> FloatArray:
    return np.zeros_like(y)


# -----------------------------------------------------------------------------
# A) Output-time semantics
# -----------------------------------------------------------------------------


def test_run_stores_exactly_output_times_history_on() -> None:
    """CoreSolver.run advances exactly n_timesteps-1 times and stores history."""
    tg = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)  # non-uniform output times
    core = _make_core(n_states=2, n_subgroups=3, time_grid=tg)
    core.set_initial_state(np.ones(core.state_shape, dtype=float))

    solver = CoreSolver(core, operators=None)
    solver.run(
        _reaction_zero,
        config=RunConfig(method="heun", adaptive=False),
    )

    assert core.current_step == core.n_timesteps - 1
    assert core.state_array is not None
    assert core.state_array.shape[0] == tg.size


def test_adaptive_true_lands_exactly_on_next_output_time() -> None:
    """With adaptive=True, dt_init < dt_out, solver substeps but keeps output times."""
    tg = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=tg)
    core.set_initial_state(np.array([[0.0]], dtype=float))

    def rhs_const(_t: float, y: FloatArray) -> FloatArray:
        return np.ones_like(y)

    solver = CoreSolver(core, operators=None)
    solver.run(
        rhs_const,
        config=RunConfig(
            method="heun",
            adaptive=True,
            adaptive_cfg=AdaptiveConfig(
                dt_init=0.1,  # force multiple internal steps
                rtol=1e-9,
                atol=1e-12,
            ),
        ),
    )

    # For y' = 1, solution at t=1 is exactly 1.0; adaptive should land at t=1.
    assert core.current_step == 1
    assert np.isclose(core.current_time, 1.0)
    assert np.all(np.isfinite(core.get_current_state()))
    assert np.isclose(float(core.get_current_state()[0, 0]), 1.0, atol=1e-12, rtol=0.0)


# -----------------------------------------------------------------------------
# B) Axis + batching plumbing for operator application
# -----------------------------------------------------------------------------


def test_imex_euler_operator_solve_applies_along_state_axis_with_batch_axes() -> None:
    """Operator application batches all non-operator axes.

    Important: imex-euler uses a step-doubling kernel even when adaptive=False and
    returns y_two_half. With static (L,R) tuples, both half-attempts use the same
    operator mapping, so the mapping is applied twice per output step.

    Here we use F=0 so the update is purely operator mapping.
    For L=2I, R=I, the mapping S(x)=0.5x, so the returned update is S(S(x))=0.25x.
    """
    tg = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(3,), store_history=True)
    core = _make_core(n_states=5, n_subgroups=2, time_grid=tg, options=opts)

    init = np.arange(np.prod(core.state_shape), dtype=float).reshape(core.state_shape)
    core.set_initial_state(init)

    n = int(core.state_shape[core.axis_index("state")])
    left = 2.0 * np.eye(n, dtype=float)
    right = np.eye(n, dtype=float)

    solver = CoreSolver(core, operators=(left, right), operator_axis="state")
    solver.run(
        _reaction_zero,
        config=RunConfig(method="imex-euler", adaptive=False),
    )

    x1 = core.get_state_at(1)
    expected = 0.25 * init
    assert np.allclose(x1, expected)


def test_imex_euler_operator_axis_subgroup_applies_along_subgroup() -> None:
    """Operators can act along a non-default axis, and batching still works.

    With imex-euler doubling behavior and static (L,R), expected scaling is again 0.25x.
    """
    tg = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=2, n_subgroups=3, time_grid=tg)

    init = np.arange(6, dtype=float).reshape(core.state_shape)  # (2, 3)
    core.set_initial_state(init)

    m = int(core.state_shape[core.axis_index("subgroup")])
    left = 2.0 * np.eye(m, dtype=float)
    right = np.eye(m, dtype=float)

    solver = CoreSolver(core, operators=(left, right), operator_axis="subgroup")
    solver.run(
        _reaction_zero,
        config=RunConfig(method="imex-euler", adaptive=False),
    )

    x1 = core.get_state_at(1)
    expected = 0.25 * init
    assert np.allclose(x1, expected)


def test_imex_euler_predictor_then_solve_applied_when_ops_is_3tuple() -> None:
    """3-tuple operators apply predictor then implicit solve.

    With P=3I, L=2I, R=I, one mapping is S(x)=0.5*(3x)=1.5x.
    imex-euler non-adaptive returns y_two_half = S(S(x)) = (1.5^2) x = 2.25 x.
    """
    tg = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=4, n_subgroups=1, time_grid=tg)

    init = np.arange(4, dtype=float).reshape(core.state_shape)  # (4, 1)
    core.set_initial_state(init)

    n = int(core.state_shape[core.axis_index("state")])
    predictor = 3.0 * np.eye(n, dtype=float)
    left = 2.0 * np.eye(n, dtype=float)
    right = np.eye(n, dtype=float)

    solver = CoreSolver(core, operators=(predictor, left, right), operator_axis="state")
    solver.run(
        _reaction_zero,
        config=RunConfig(method="imex-euler", adaptive=False),
    )

    x1 = core.get_state_at(1)
    expected = 2.25 * init
    assert np.allclose(x1, expected)


# -----------------------------------------------------------------------------
# C) dt variability rules (factory required) and missing operators behavior
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["imex-euler", "imex-heun-tr", "imex-trbdf2"])
def test_imex_methods_require_factory_when_output_dt_nonuniform_strict(
    method: str,
) -> None:
    """Non-uniform output dt requires factory operators when strict=True (default)."""
    tg = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)  # variable dt
    core = _make_core(n_states=1, n_subgroups=1, time_grid=tg)
    core.set_initial_state(np.array([[1.0]], dtype=float))

    # Static tuple (dt-dependent in principle) should be rejected when dt varies.
    left = np.array([[2.0]], dtype=float)
    right = np.array([[1.0]], dtype=float)
    solver = CoreSolver(core, operators=(left, right), operator_axis="state")

    with pytest.raises(ValueError, match="variable dt requires a StageOperatorFactory"):
        solver.run(
            _reaction_zero,
            config=RunConfig(method=method, adaptive=False, strict=True),
        )


@pytest.mark.parametrize(
    ("method", "expected_downshift"),
    [
        ("imex-euler", "euler"),
        ("imex-heun-tr", "heun"),
        ("imex-trbdf2", "heun"),
    ],
)
def test_imex_methods_warn_and_downshift_when_operators_missing_nonstrict(
    method: str,
    expected_downshift: str,  # noqa: ARG001
) -> None:
    """When strict=False and operators are missing, CoreSolver warns and downshifts."""
    tg = np.array([0.0, 0.25], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=tg)
    core.set_initial_state(np.array([[0.0]], dtype=float))

    def rhs_const(_t: float, y: FloatArray) -> FloatArray:
        return np.ones_like(y)

    solver = CoreSolver(core, operators=None)

    with pytest.warns(RuntimeWarning, match="falling back to explicit"):
        solver.run(
            rhs_const,
            config=RunConfig(method=method, adaptive=False, strict=False),
        )

    # For y'=1, both Euler-doubling and Heun are exact over one interval.
    assert np.isclose(float(core.get_current_state()[0, 0]), 0.25, atol=1e-12, rtol=0.0)


# -----------------------------------------------------------------------------
# D) Explicit methods ignore operators (warning + no scaling)
# -----------------------------------------------------------------------------


def test_explicit_method_ignores_operators_and_emits_runtimewarning() -> None:
    """Explicit methods ignore operators and emit RuntimeWarning when strict=True."""
    tg = np.array([0.0, 0.25], dtype=float)
    core = _make_core(n_states=1, n_subgroups=1, time_grid=tg)
    core.set_initial_state(np.array([[0.0]], dtype=float))

    # Nontrivial operator that would scale if applied.
    left = np.array([[2.0]], dtype=float)
    right = np.array([[1.0]], dtype=float)
    solver = CoreSolver(core, operators=(left, right), operator_axis="state")

    def rhs_const(_t: float, y: FloatArray) -> FloatArray:
        return np.ones_like(y)

    # Message in CoreSolver is "Method 'heun' is explicit; provided operators ignored."
    with pytest.warns(RuntimeWarning, match="explicit; provided operators ignored"):
        solver.run(
            rhs_const,
            config=RunConfig(method="heun", adaptive=False, strict=True),
        )

    # Heun is exact for constant derivative: y(0.25)=0.25
    assert np.isclose(float(core.get_current_state()[0, 0]), 0.25, atol=1e-12, rtol=0.0)


# -----------------------------------------------------------------------------
# E) RHS shape enforcement
# -----------------------------------------------------------------------------


def test_rhs_shape_mismatch_raises() -> None:
    """RHS returning wrong shape raises ValueError."""
    tg = np.array([0.0, 1.0], dtype=float)
    core = _make_core(n_states=2, n_subgroups=2, time_grid=tg)
    core.set_initial_state(np.zeros(core.state_shape, dtype=float))

    def bad_rhs(_t: float, _y: FloatArray) -> FloatArray:
        return cast("FloatArray", np.zeros((1, 1), dtype=float))  # wrong shape

    solver = CoreSolver(core, operators=None)
    with pytest.raises(ValueError, match="rhs shape"):
        solver.run(
            bad_rhs,
            config=RunConfig(method="heun", adaptive=False),
        )
