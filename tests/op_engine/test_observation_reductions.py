"""Independent sampled-window values, streaming storage and native gradients."""

from __future__ import annotations

import weakref
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    CoreSolver,
    ModelCore,
    ObservationMetadata,
    ObservationSpec,
    array_namespace,
    reduce_observations,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Iterator


def _total(_t: object, y: object, **params: object) -> object:
    """Return a native scalar with an optional observation-only parameter."""
    return array_namespace(y).sum(y) * params.get("gain", 1.0)


def _field(_t: object, y: object, **_params: object) -> object:
    """Keep the native field without changing its scientific layout.

    Returns:
        The supplied native state.
    """
    return y


def test_numpy_nonuniform_linear_fields_and_counter_changes() -> None:
    """Analytic linear integrals differ from endpoint changes and final values."""
    times = [0.0, 0.1, 0.7, 2.0]
    field = ObservationSpec(
        "field",
        _field,
        ObservationMetadata(
            output_shape=(2,), coordinates=(("x", ("a", "b")),), units="items"
        ),
    )
    samples = ((time, np.array([2 + 3 * time, 5 - time])) for time in times)
    result = reduce_observations(samples, [field, ObservationSpec("counter", _total)])
    assert result.window == (0.0, 2.0)
    assert result.sample_count == 4
    assert result.quadrature == "trapezoid"
    assert result.diagnostics.evaluation_count == 8
    assert result.diagnostics.temporal_interpolation == "none"
    assert result.diagnostics.error_estimate is None
    np.testing.assert_allclose(result.values["field"].initial, [2, 5])
    np.testing.assert_allclose(result.values["field"].final, [8, 3])
    np.testing.assert_allclose(result.values["field"].integral, [10, 8])
    np.testing.assert_allclose(result.values["field"].change, [6, -2])
    np.testing.assert_allclose(result.values["counter"].change, 4)
    np.testing.assert_allclose(result.values["counter"].integral, 18)
    assert result.values["field"].spec == field
    with pytest.raises(TypeError):
        result.values["new"] = result.values["field"]  # type: ignore[index]


def test_numpy_reused_producer_buffer_is_snapshotted() -> None:
    """A producer's later mutation cannot change retained endpoints or integrals."""
    buffer = np.empty(2)

    def samples() -> Iterator[tuple[float, Array]]:
        for time in [0.0, 0.5, 2.0]:
            buffer[:] = [2 + 3 * time, 5 - time]
            yield time, buffer
        buffer[:] = -100

    result = reduce_observations(samples(), [ObservationSpec("total", _total)])
    np.testing.assert_allclose(result.values["total"].initial, 7)
    np.testing.assert_allclose(result.values["total"].final, 11)
    np.testing.assert_allclose(result.values["total"].change, 4)
    np.testing.assert_allclose(result.values["total"].integral, 18)


def test_numpy_input_states_are_not_retained_as_a_trajectory() -> None:
    """Old generator-produced states become collectible during forward reduction."""
    references: list[weakref.ReferenceType[np.ndarray]] = []

    def samples() -> Iterator[tuple[float, Array]]:
        for index in range(128):
            state = np.full(2, float(index))
            references.append(weakref.ref(state))
            yield float(index), state
            assert sum(reference() is not None for reference in references) <= 2

    result = reduce_observations(samples(), [ObservationSpec("total", _total)])
    assert result.sample_count == 128
    np.testing.assert_allclose(result.values["total"].integral, 127**2)
    assert all(reference() is None for reference in references)


def _decay_summary(n_intervals: int) -> tuple[float, float]:
    """Drive the public RK4 step kernel without ModelCore history.

    Returns:
        Final state error and time-integral error against an analytic decay.
    """
    times = np.linspace(0, 1, n_intervals + 1) ** 1.3
    core = ModelCore(1, 1, times, options=ModelCoreOptions(store_history=False))
    core.set_initial_state(np.array([[3.0]]))
    solver = CoreSolver(core)

    def rhs(_time: float, y: Array) -> Array:
        return cast("Array", np.multiply(-0.7, y))

    def samples() -> Iterator[tuple[float, Array]]:
        yield float(times[0]), core.get_current_state()
        for start, end in pairwise(times):
            state, _ = solver.fixed_explicit_step(
                rhs,
                method="rk4",
                t=float(start),
                dt=float(end - start),
                y=core.get_current_state(),
            )
            core.advance_timestep(state)
            yield float(end), state

    result = reduce_observations(samples(), [ObservationSpec("total", _total)])
    assert core.state_array is None
    assert core.current_step == n_intervals
    value = result.values["total"]
    return (
        abs(float(value.final.item()) - 3 * np.exp(-0.7)),
        abs(float(value.integral.item()) - 3 * (1 - np.exp(-0.7)) / 0.7),
    )


def test_numpy_native_step_driver_refines_state_and_sampled_integral() -> None:
    """Sampled trapezoid accuracy is separate from fourth-order RK4 accuracy."""
    coarse_state, coarse_integral = _decay_summary(16)
    fine_state, fine_integral = _decay_summary(32)
    assert 14 < coarse_state / fine_state < 18
    assert 3.5 < coarse_integral / fine_integral < 4.5


def test_numpy_single_sample_zero_duration_and_empty_selection() -> None:
    """A single endpoint has no elapsed integral or counter change."""
    result = reduce_observations(
        [(2.0, np.ones(2))], [ObservationSpec("total", _total)]
    )
    assert result.window == (2.0, 2.0)
    np.testing.assert_allclose(result.values["total"].initial, 2)
    np.testing.assert_allclose(result.values["total"].integral, 0)
    np.testing.assert_allclose(result.values["total"].change, 0)
    assert reduce_observations([(0.0, np.ones(2))], []).values == {}
    with pytest.raises(ValueError, match="at least one sample"):
        reduce_observations([], [])


@pytest.mark.parametrize("time", [np.nan, np.inf, True, "1", [1], 1j])
def test_numpy_rejects_invalid_static_sample_time(time: object) -> None:
    """Invalid real scalar times fail even with no evaluator selection."""
    with pytest.raises(ValueError, match="finite real static"):
        reduce_observations([(cast("float", time), np.ones(2))], [])


@pytest.mark.parametrize("times", [[0.0, 0.0], [1.0, 0.0], [-1e308, 1e308]])
def test_numpy_rejects_nonincreasing_or_overflowed_intervals(
    times: list[float],
) -> None:
    """An invalid interval cannot manufacture a window integral."""
    with pytest.raises(ValueError, match="increase with finite intervals"):
        reduce_observations(((time, np.ones(2)) for time in times), [])


def test_numpy_rejects_state_shape_or_dtype_changes() -> None:
    """One summary cannot silently mix layouts or floating precisions."""
    with pytest.raises(ValueError, match="preserve state shape"):
        reduce_observations([(0.0, np.ones(2)), (1.0, np.ones(3))], [])
    with pytest.raises(TypeError, match="namespace and dtype"):
        reduce_observations(
            [(0.0, np.ones(2)), (1.0, np.ones(2, dtype=np.float32))], []
        )
    with pytest.raises(TypeError, match="real floating"):
        reduce_observations([(0.0, np.ones(2, dtype=int))], [])


def test_jax_jit_vmap_state_and_parameter_gradients() -> None:
    """Independent weighted-linear references qualify native integral/change AD."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        states = jnp.arange(6.0).reshape(3, 2)

        def objective(y: Array, gain: Array) -> Array:
            samples = ((time, y[index]) for index, time in enumerate([0.0, 0.25, 1.0]))
            summary = reduce_observations(
                samples, [ObservationSpec("total", _total)], params={"gain": gain}
            )
            value = summary.values["total"]
            return cast("Array", jnp.add(value.integral, value.change))

        np.testing.assert_allclose(jax.jit(objective)(states, jnp.asarray(2.0)), 28)
        state_grad, gain_grad = jax.jit(jax.grad(objective, argnums=(0, 1)))(
            states, jnp.asarray(2.0)
        )
        np.testing.assert_allclose(state_grad, [[-1.75, -1.75], [1, 1], [2.75, 2.75]])
        np.testing.assert_allclose(gain_grad, 14)
        np.testing.assert_allclose(
            jax.vmap(lambda gain: objective(states, gain))(jnp.asarray([1.0, 2.0])),
            [14, 28],
        )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_state_parameter_gradients_and_namespace(dtype_name: str) -> None:
    """CPU Torch retains native precision and endpoint/integral gradients."""
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    states = torch.arange(6, dtype=dtype).reshape(3, 2).requires_grad_()
    gain = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    result = reduce_observations(
        ((time, states[index]) for index, time in enumerate([0.0, 0.25, 1.0])),
        [ObservationSpec("total", _total)],
        params={"gain": gain},
    )
    value = result.values["total"]
    assert value.integral.dtype == dtype
    assert array_namespace(value.integral) is array_namespace(states)
    state_grad, gain_grad = torch.autograd.grad(
        value.integral + value.change, (states, gain)
    )
    np.testing.assert_allclose(state_grad, [[-1.75, -1.75], [1, 1], [2.75, 2.75]])
    np.testing.assert_allclose(gain_grad, 14)
    with pytest.raises(TypeError, match="namespace and dtype"):
        reduce_observations(
            [(0.0, np.ones(2)), (1.0, torch.ones(2, dtype=torch.float64))], []
        )
