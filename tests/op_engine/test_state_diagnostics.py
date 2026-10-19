"""Output-only state flags, independent counts and nonintrusive solver wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_engine import (
    Array,
    CoreSolver,
    ModelCore,
    OutputStateCheckConfig,
    array_namespace,
    diagnose_trajectory,
)
from op_engine.core_solver import AdaptiveStepSchedule, NonlinearMethodConfig, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Iterator


PATH = np.asarray([
    [[1.0, 2.0], [5.0, 4.0]],
    [[-1.0, 1.0], [4.0, 5.0]],
    [[0.5, 0.25], [3.0, 4.0]],
])


@pytest.fixture
def jax_x64() -> Iterator[None]:
    """Scope double precision for diagnostic and derivative comparisons."""
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        yield


def test_selected_states_and_monotone_increments() -> None:
    """Counts identify both negative states and decreasing positive counters."""
    immutable = PATH.copy()
    immutable.flags.writeable = False
    result = diagnose_trajectory(
        immutable,
        OutputStateCheckConfig(nonnegative_indices=(0,), monotone_indices=(1,)),
    )
    assert result.n_outputs == 3
    assert result.complete
    np.testing.assert_array_equal(result.min_state, [-1, 0.25])
    np.testing.assert_array_equal(result.n_negative_outputs, [1, 0])
    np.testing.assert_array_equal(result.n_nonfinite_outputs, [0, 0])
    np.testing.assert_array_equal(result.min_increment, [-1, -1])
    np.testing.assert_array_equal(result.n_decreasing_intervals, [2, 1])
    np.testing.assert_array_equal(result.valid, [False, False])
    np.testing.assert_array_equal(immutable, PATH)


def test_tolerances_change_flags_without_clamping_values() -> None:
    """Absolute diagnostic thresholds do not alter values or boundary equality."""
    result = diagnose_trajectory(
        PATH,
        OutputStateCheckConfig(
            nonnegative_indices=(0,),
            monotone_indices=(1,),
            negative_tolerance=1.0,
            monotone_tolerance=1.0,
        ),
    )
    np.testing.assert_array_equal(result.min_state, [-1, 0.25])
    np.testing.assert_array_equal(result.min_increment, [-1, -1])
    np.testing.assert_array_equal(result.valid, [True, True])


def test_nonfinite_components_are_checked_outside_selected_states() -> None:
    """A nonfinite unconstrained component still invalidates its output record."""
    path = PATH.copy()
    path[1, 1, 0] = np.nan
    path[2, 1, 0] = np.inf
    result = diagnose_trajectory(path, OutputStateCheckConfig(nonnegative_indices=(0,)))
    np.testing.assert_array_equal(result.n_nonfinite_outputs, [2, 0])
    assert result.min_increment is None
    assert result.n_decreasing_intervals is None
    np.testing.assert_array_equal(result.min_state, [-1, 0.25])


def test_field_axes_remain_separate() -> None:
    """Diagnostics preserve subgroup and spatial axes rather than pooling them."""
    field = np.broadcast_to(PATH[..., None], (*PATH.shape, 4))
    result = diagnose_trajectory(
        field, OutputStateCheckConfig(nonnegative_indices=(0,))
    )
    assert result.min_state.shape == (2, 4)
    np.testing.assert_array_equal(result.n_negative_outputs, [[1] * 4, [0] * 4])


def test_initial_only_trajectory_has_no_monotone_intervals() -> None:
    """An initial-state record is valid without inventing a zero increment."""
    result = diagnose_trajectory(
        PATH[:1], OutputStateCheckConfig(monotone_indices=(1,))
    )
    assert result.n_outputs == 1
    np.testing.assert_array_equal(result.n_decreasing_intervals, [0, 0])
    np.testing.assert_array_equal(result.min_increment, [np.inf, np.inf])
    np.testing.assert_array_equal(result.valid, [True, True])


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_namespace_diagnostic_counts(backend: str) -> None:
    """Output reductions and flags remain in each qualified native namespace."""
    if backend == "jax":
        xp = pytest.importorskip("jax.numpy")
        native = xp.asarray(PATH, dtype=xp.float32)
    else:
        xp = pytest.importorskip(backend)
        native = xp.asarray(PATH, dtype=xp.float64)
    result = diagnose_trajectory(
        native, OutputStateCheckConfig(nonnegative_indices=(0,), monotone_indices=(1,))
    )
    assert array_namespace(result.min_state) is array_namespace(native)
    for actual, expected in [
        (result.min_state, [-1, 0.25]),
        (result.n_negative_outputs, [1, 0]),
        (result.n_decreasing_intervals, [2, 1]),
    ]:
        host = xp.asarray(actual).detach().numpy() if backend == "torch" else actual
        np.testing.assert_array_equal(host, expected)


@pytest.mark.parametrize("store_history", [False, True])
def test_euler_output_flags_preserve_bank_members_and_trajectory(
    *, store_history: bool
) -> None:
    """The unstable member is flagged without changing either numerical path."""
    trajectories = []
    for enabled in (False, True):
        core = ModelCore(
            1,
            2,
            np.asarray([0.0, 1.0, 2.0]),
            options=ModelCoreOptions(store_history=store_history),
        )
        core.set_initial_state(np.ones((1, 2)))
        solver = CoreSolver(core)
        returned = solver.run(
            lambda _t, state: state * np.asarray([[-0.1, -2.0]]),
            config=RunConfig(
                method="euler",
                state_checks=OutputStateCheckConfig() if enabled else None,
            ),
        )
        assert returned is None
        trajectories.append(np.asarray(core.get_current_state()).copy())
        result = solver.last_state_diagnostics
        if enabled:
            assert result is not None
            assert result.complete
            assert result.n_outputs == 3
            np.testing.assert_allclose(result.min_state, [0.81, -1.0])
            np.testing.assert_array_equal(result.n_negative_outputs, [0, 1])
            np.testing.assert_array_equal(result.valid, [True, False])
        else:
            assert result is None
        assert (core.state_array is not None) == store_history
    assert trajectories[0].tobytes() == trajectories[1].tobytes()


@pytest.mark.parametrize("method", ["rk4", "implicit-euler", "sdirk2"])
def test_monotone_counters_across_method_families(method: str) -> None:
    """Positive cumulative outputs can be invalid even when no state is negative."""
    core = ModelCore(
        1, 1, np.asarray([0.0, 1.0, 2.0]), options=ModelCoreOptions(store_history=False)
    )
    core.set_initial_state(np.asarray([[5.0]]))
    solver = CoreSolver(core)
    returned = solver.run(
        lambda _t, state: -np.ones_like(state),
        config=RunConfig(
            method=method,
            jacobian=lambda _t, _y: np.zeros((1, 1)),
            nonlinear=NonlinearMethodConfig(
                rhs_jacobian=lambda _t, _y: np.zeros((1, 1))
            )
            if method == "sdirk2"
            else None,
            state_checks=OutputStateCheckConfig(monotone_indices=(0,)),
        ),
    )
    assert (returned is not None) == (method == "sdirk2")
    result = solver.last_state_diagnostics
    assert result is not None
    np.testing.assert_allclose(result.min_state, [3.0])
    np.testing.assert_allclose(result.min_increment, [-1.0])
    np.testing.assert_array_equal(result.n_negative_outputs, [0])
    np.testing.assert_array_equal(result.n_decreasing_intervals, [2])
    np.testing.assert_array_equal(result.valid, [False])


def test_substeps_are_outside_output_diagnostic_scope() -> None:
    """A negative internal state can be missed by a sparse output grid."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))
    solver = CoreSolver(core)
    solver.run(
        lambda _t, state: -4 * state,
        config=RunConfig(
            method="euler", fixed_max_step=0.5, state_checks=OutputStateCheckConfig()
        ),
    )
    # Full steps visit 1 -> -1 -> 1, but only 1 and 1 are requested outputs.
    result = solver.last_state_diagnostics
    assert result is not None
    assert result.n_outputs == 2
    np.testing.assert_array_equal(result.min_state, [1.0])
    np.testing.assert_array_equal(result.valid, [True])


@pytest.mark.parametrize("loop", ["unroll", "auto"])
def test_replayed_outputs_use_same_check_contract(loop: str) -> None:
    """Frozen replay checks the known output values and increments."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0, 2.0]))
    core.set_initial_state(np.asarray([[5.0]]))
    solver = CoreSolver(core)
    solver.replay_adaptive_schedule(
        lambda _t, state: -np.ones_like(state),
        AdaptiveStepSchedule((0.0, 1.0, 2.0), ((0.5, 0.5), (0.25, 0.75))),
        config=RunConfig(
            method="rk4",
            adaptive=True,
            replay_loop=loop,
            state_checks=OutputStateCheckConfig(monotone_indices=(0,)),
        ),
    )
    result = solver.last_state_diagnostics
    assert result is not None
    assert result.complete
    np.testing.assert_allclose(result.min_state, [3.0])
    np.testing.assert_array_equal(result.n_decreasing_intervals, [2])


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("loop", ["unroll", "scan"])
def test_jax_jit_vmap_and_unchanged_state_gradient(loop: str) -> None:
    """JIT and batching return native flags without changing Euler's gradient."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def solve(rate: Array) -> tuple[Array, Array, Array]:
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 1.0, 2.0]),
            options=ModelCoreOptions(store_history=False),
        )
        core.set_initial_state(jnp.asarray([[1.0]]))
        solver = CoreSolver(core)
        solver.run(
            lambda _t, state: rate * state,
            config=RunConfig(
                method="euler", fixed_loop=loop, state_checks=OutputStateCheckConfig()
            ),
        )
        result = solver.last_state_diagnostics
        assert result is not None
        return core.get_current_state()[0, 0], result.min_state, result.valid

    rates = jnp.asarray([-0.1, -2.0])
    final, minimum, valid = jax.jit(jax.vmap(solve))(rates)
    np.testing.assert_allclose(final, [0.81, 1.0])
    np.testing.assert_allclose(minimum[:, 0], [0.81, -1.0])
    np.testing.assert_array_equal(valid[:, 0], [True, False])
    gradient = jax.jit(jax.vmap(jax.grad(lambda rate: solve(rate)[0])))(rates)
    np.testing.assert_allclose(gradient, [1.8, -2.0])


def test_failed_run_retains_partial_diagnostics_and_reset_clears_them() -> None:
    """A record from a failed run cannot appear complete or survive a new run."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0, 2.0]))
    core.set_initial_state(np.asarray([[5.0]]))
    solver = CoreSolver(core)

    def rhs(time: float, state: Array) -> Array:
        if time >= 1:
            msg = "Example solve failed after its first output"
            raise RuntimeError(msg)
        return -np.ones_like(state)

    with pytest.raises(RuntimeError, match="Example solve failed"):
        solver.run(
            rhs, config=RunConfig(method="euler", state_checks=OutputStateCheckConfig())
        )
    result = solver.last_state_diagnostics
    assert result is not None
    assert not result.complete
    assert result.n_outputs == 2
    np.testing.assert_array_equal(result.valid, [False])
    core.set_initial_state(np.asarray([[5.0]]))
    solver.run(lambda _t, state: -np.ones_like(state), config=RunConfig(method="euler"))
    assert solver.last_state_diagnostics is None


def test_config_and_trajectory_validation() -> None:
    """Invalid selectors, tolerances and shapes fail before being misreported."""
    with pytest.raises(TypeError, match="state_checks"):
        RunConfig(state_checks=True)  # type: ignore[arg-type]
    for indices in [(-1,), (0, 0), ()]:
        with pytest.raises(ValueError, match="nonnegative_indices"):
            OutputStateCheckConfig(nonnegative_indices=indices)
    with pytest.raises(TypeError, match="integer"):
        OutputStateCheckConfig(monotone_indices=(True,))
    for tolerance in [-1.0, np.inf, np.nan]:
        with pytest.raises(ValueError, match="tolerance"):
            OutputStateCheckConfig(negative_tolerance=tolerance)
    with pytest.raises(ValueError, match="index exceeds"):
        diagnose_trajectory(PATH, OutputStateCheckConfig(monotone_indices=(2,)))
    with pytest.raises(ValueError, match="output and state axes"):
        diagnose_trajectory(np.ones(2))
    with pytest.raises(TypeError, match="real floating"):
        diagnose_trajectory(np.ones((2, 1), dtype=int))
    with pytest.raises(TypeError, match="config"):
        diagnose_trajectory(PATH, config=False)  # type: ignore[arg-type]


def test_public_output_check_example() -> None:
    """The complete example asserts both independent failure cases."""
    import runpy  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    example = Path(__file__).resolve().parents[2] / "examples/output_state_checks.py"
    runpy.run_path(str(example))["main"]()
