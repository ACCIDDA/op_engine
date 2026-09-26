"""Tests for reusable linear-multistep coefficients and state history."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore
from op_engine._multistep import (
    BDF1,
    BDF2,
    BDF3,
    LinearMultistepTableau,
    MultistepHistory,
    select_multistep_tableau,
)
from op_engine.core_solver import RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from typing import Any


def test_bdf_tableaus_declare_coefficients_order_and_startup() -> None:
    """BDF family metadata contains every fact needed by orchestration."""
    assert BDF1.alpha == (1.0, -1.0)
    assert BDF2.alpha == (1.5, -2.0, 0.5)
    assert BDF3.alpha == (11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0)
    assert (BDF1.order, BDF2.order, BDF3.order) == (1, 2, 3)
    assert BDF1.startup_methods == ()
    assert BDF2.startup_methods == ("bdf1",)
    assert BDF3.startup_methods == ("bdf1", "bdf2")
    assert (BDF1.stored_state_history, BDF2.stored_state_history) == (0, 1)
    assert BDF3.stored_state_history == 2
    assert not BDF1.requires_uniform_step
    assert BDF2.requires_uniform_step
    assert BDF3.requires_uniform_step


def test_startup_selection_follows_available_history() -> None:
    """A higher-step formula advances through its declared startup sequence."""
    assert select_multistep_tableau(BDF2, 0) is BDF1
    assert select_multistep_tableau(BDF2, 1) is BDF2
    assert select_multistep_tableau(BDF3, 0) is BDF1
    assert select_multistep_tableau(BDF3, 1) is BDF2
    assert select_multistep_tableau(BDF3, 2) is BDF3


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"name": ""}, ValueError, "name"),
        ({"beta": (1.0, 0.0)}, ValueError, "dimensions"),
        ({"alpha": (0.0, -2.0, 0.5)}, ValueError, r"alpha\[0\]"),
        ({"alpha": (np.nan, -2.0, 0.5)}, ValueError, "finite"),
        ({"order": 0}, ValueError, "positive integer"),
        ({"order": 3}, ValueError, "declared order"),
        ({"startup_methods": ()}, ValueError, "Startup sequence"),
        ({"requires_uniform_step": 1}, TypeError, "boolean"),
    ],
)
def test_tableau_rejects_inconsistent_declarations(
    kwargs: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    """Malformed multistep families fail where they are declared."""
    defaults: dict[str, object] = {
        "name": "test BDF2",
        "alpha": (1.5, -2.0, 0.5),
        "beta": (1.0, 0.0, 0.0),
        "order": 2,
        "startup_methods": ("bdf1",),
        "requires_uniform_step": True,
    }
    with pytest.raises(exception, match=match):
        LinearMultistepTableau(**(defaults | kwargs))  # type: ignore[arg-type]


def test_numpy_history_snapshots_trim_and_restart() -> None:
    """History owns snapshots, orders them newest first, and resets explicitly."""
    first = np.asarray([1.0, 2.0])
    history = MultistepHistory[Array](capacity=2).push(first)
    first[0] = 99.0
    history = history.push(np.asarray([3.0, 4.0]))
    history = history.push(np.asarray([5.0, 6.0]))

    assert len(history.states) == 2
    assert np.array_equal(history.states[0], [5.0, 6.0])
    assert np.array_equal(history.states[1], [3.0, 4.0])
    assert history.ready_for(BDF3)
    assert history.restart().states == ()


def test_jax_history_preserves_namespace() -> None:
    """History storage follows the state value instead of importing a backend."""
    jnp = pytest.importorskip("jax.numpy")
    state = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    history = MultistepHistory[Array](capacity=1).push(cast("Array", state))

    assert history.states[0].__array_namespace__() is jnp
    assert np.array_equal(np.asarray(history.states[0]), [1.0, 2.0])


def _expected_bdf2(rate: float, initial: float, dt: float, n_steps: int) -> np.ndarray:
    """Return the exact scalar recurrence implemented by fixed-step BDF2."""
    values = [initial, initial / (1.0 - dt * rate)]
    for _ in range(1, n_steps):
        values.append((2.0 * values[-1] - 0.5 * values[-2]) / (1.5 - dt * rate))
    return np.asarray(values)


@pytest.mark.parametrize("use_jax", [False, True])
def test_core_bdf2_matches_declared_recurrence_and_startup(*, use_jax: bool) -> None:
    """CoreSolver uses BDF1 once, then the declared BDF2 coefficients."""
    xp: Any = pytest.importorskip("jax.numpy") if use_jax else np
    rate = -0.3
    initial = 1.2
    dt = 0.2
    dtype = np.float32 if use_jax else np.float64
    times = np.arange(0.0, 0.8, dt)
    core = ModelCore(
        1,
        1,
        times,
        options=ModelCoreOptions(dtype=dtype),
    )
    core.set_initial_state(xp.asarray([[initial]], dtype=dtype))

    def rhs(_time: float, state: Array) -> Array:
        namespace = cast("Any", state.__array_namespace__())
        return cast("Array", namespace.multiply(state, rate))

    def jacobian(_time: float, state: Array) -> Array:
        namespace = cast("Any", state.__array_namespace__())
        return cast(
            "Array",
            namespace.asarray([[rate]], dtype=state.dtype),
        )

    CoreSolver(core).run(
        rhs,
        config=RunConfig(method="bdf2", jacobian=jacobian),
    )

    assert core.state_array is not None
    actual = np.asarray(core.state_array)[:, 0, 0]
    assert np.allclose(actual, _expected_bdf2(rate, initial, dt, len(times) - 1))


def test_reusing_solver_after_external_reset_restarts_bdf2() -> None:
    """A new run cannot consume stale history from a prior trajectory."""
    rate = -0.4
    dt = 0.25
    times = np.asarray([0.0, dt, 2.0 * dt])
    core = ModelCore(1, 1, times)
    solver = CoreSolver(core)

    def rhs(_time: float, state: Array) -> Array:
        return cast("Array", np.multiply(state, rate))

    def jacobian(_time: float, _state: Array) -> Array:
        return cast("Array", np.asarray([[rate]]))

    config = RunConfig(method="bdf2", jacobian=jacobian)
    core.set_initial_state(np.asarray([[1.0]]))
    solver.run(rhs, config=config)
    core.set_initial_state(np.asarray([[2.0]]))
    solver.run(rhs, config=config)

    assert core.state_array is not None
    expected = _expected_bdf2(rate, 2.0, dt, len(times) - 1)
    assert np.allclose(np.asarray(core.state_array)[:, 0, 0], expected)


def _amplification_radius(
    tableau: LinearMultistepTableau,
    scaled_eigenvalue: complex,
) -> float:
    """Return the largest recurrence-root magnitude for a scalar test mode."""
    coefficients = np.asarray(tableau.alpha, dtype=np.complex128)
    coefficients[0] -= scaled_eigenvalue * tableau.beta[0]
    return float(np.max(np.abs(np.roots(coefficients))))


def test_bdf3_loses_stability_on_stiff_oscillatory_benchmark() -> None:
    """BDF3 grows a mode that remains inside BDF2's stability region.

    ``z = -0.001 + i`` corresponds, for example, to an eigenvalue
    ``-1 + 1000 i`` at ``dt=0.001``. Adding a fast real mode such as ``-10000``
    makes this a concrete stiff, weakly damped oscillatory system without
    changing the recurrence root tested here.
    """
    scaled_eigenvalue = complex(-0.001, 1.0)

    assert _amplification_radius(BDF2, scaled_eigenvalue) < 1.0
    assert _amplification_radius(BDF3, scaled_eigenvalue) > 1.04
