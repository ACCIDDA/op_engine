"""Independent stored-output observation values, error contracts and derivatives."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import (
    Array,
    ModelCore,
    ObservationMetadata,
    ObservationSpec,
    array_namespace,
    observe_model,
    observe_solution,
)


def _evaluate(t: object, y: object, **params: object) -> object:
    xp = array_namespace(y)
    return xp.sum(y) * params["gain"] + t


def test_numpy_values_metadata_and_inputs_are_preserved() -> None:
    """Native scalar/field outputs retain separate times and scientific axes."""
    times = np.array([0.0, 0.2, 0.9])
    states = np.arange(6.0).reshape(3, 2)
    before = states.copy()
    scalar = ObservationSpec("total", _evaluate, ObservationMetadata(units="items"))

    def field(_t: object, y: object, **_params: object) -> object:
        return y

    field_spec = ObservationSpec(
        "field",
        field,
        ObservationMetadata(output_shape=(2,), coordinates=(("x", ("0.25", "0.75")),)),
    )
    result = observe_solution(times, states, [scalar, field_spec], params={"gain": 2.0})
    np.testing.assert_allclose(result.series["total"].values, [2.0, 10.2, 18.9])
    np.testing.assert_array_equal(result.series["field"].values, before)
    assert result.series["total"].spec.metadata.units == "items"
    assert result.series["field"].spec.metadata.coordinates[0][0] == "x"
    assert result.times == (0.0, 0.2, 0.9)
    assert result.diagnostics.evaluation_count == 6
    assert result.diagnostics.temporal_interpolation == "none"
    assert result.diagnostics.dense_output_order is None
    assert result.diagnostics.error_estimate is None
    times[:] = -1
    assert result.times == (0.0, 0.2, 0.9)
    np.testing.assert_array_equal(states, before)
    with pytest.raises(TypeError):
        result.series["other"] = result.series["total"]  # type: ignore[index]


@pytest.mark.parametrize(
    "times",
    [[], [[0.0]], [0.0, 0.0], [1.0, 0.0], [np.nan], [np.inf], [1j], [True], ["0"]],
)
def test_numpy_invalid_output_times(times: object) -> None:
    """Invalid static grids fail before any evaluator can execute."""
    with pytest.raises(ValueError, match="observation times"):
        observe_solution(cast("np.ndarray", times), np.ones((1, 2)), [])


def test_numpy_single_time_empty_request_and_shape_contracts() -> None:
    """A single stored state and empty output selection are valid."""
    assert observe_solution([2.0], np.ones((1, 2)), []).series == {}
    spec = ObservationSpec("total", _evaluate)
    result = observe_solution([2.0], np.ones((1, 2)), [spec], params={"gain": 3.0})
    np.testing.assert_array_equal(result.series["total"].values, [8.0])
    with pytest.raises(ValueError, match="leading state row"):
        observe_solution([0.0, 1.0], np.ones((1, 2)), [spec])
    with pytest.raises(TypeError, match="real floating"):
        observe_solution([0.0], np.ones((1, 2), dtype=int), [spec])
    with pytest.raises(ValueError, match="nonempty and unique"):
        observe_solution([0.0], np.ones((1, 2)), [spec, spec])
    with pytest.raises(ValueError, match="must have shape"):
        observe_solution(
            [0.0],
            np.ones((1, 2)),
            [spec._replace(metadata=ObservationMetadata(output_shape=(2,)))],
            params={"gain": 1.0},
        )
    with pytest.raises(ValueError, match="coordinates must match"):
        observe_solution(
            [0.0],
            np.ones((1, 2)),
            [
                spec._replace(
                    metadata=ObservationMetadata(
                        output_shape=(2,), coordinates=(("x", ("one",)),)
                    )
                )
            ],
        )


def test_numpy_model_history_requires_completion_and_storage() -> None:
    """Unfilled history rows and disabled history cannot become observed data."""
    core = ModelCore(1, 1, np.array([0.0, 0.2]))
    core.set_initial_state(np.array([[1.0]]))
    with pytest.raises(ValueError, match="completed"):
        observe_model(core, [])
    core.advance_timestep(np.array([[0.5]]))
    result = observe_model(
        core, [ObservationSpec("total", _evaluate)], params={"gain": 2.0}
    )
    np.testing.assert_allclose(result.series["total"].values, [2.0, 1.2])
    core.store_history = False
    with pytest.raises(ValueError, match="stored history"):
        observe_model(core, [])


def test_jax_jit_vmap_state_and_parameter_gradients() -> None:
    """Static-time native observations retain JAX transformations."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        states = jnp.arange(6.0).reshape(3, 2)
        spec = ObservationSpec("total", _evaluate)

        def objective(y: Array, gain: Array) -> Array:
            result = observe_solution([0.0, 0.2, 0.9], y, [spec], params={"gain": gain})
            return cast("Array", jnp.sum(result.series["total"].values))

        np.testing.assert_allclose(jax.jit(objective)(states, jnp.asarray(2.0)), 31.1)
        state_grad, gain_grad = jax.jit(jax.grad(objective, argnums=(0, 1)))(
            states, jnp.asarray(2.0)
        )
        np.testing.assert_array_equal(state_grad, np.full((3, 2), 2.0))
        np.testing.assert_allclose(gain_grad, 15.0)
        np.testing.assert_allclose(
            jax.vmap(lambda gain: objective(states, gain))(jnp.asarray([1.0, 2.0])),
            [16.1, 31.1],
        )


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test_torch_state_parameter_gradients_and_namespace(dtype_name: str) -> None:
    """Observation stacking preserves CPU Torch precision and autograd."""
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    states = torch.arange(6.0, dtype=dtype).reshape(3, 2).requires_grad_()
    gain = torch.tensor(2.0, dtype=dtype, requires_grad=True)
    result = observe_solution(
        [0.0, 0.2, 0.9],
        states,
        [ObservationSpec("total", _evaluate)],
        params={"gain": gain},
    )
    values = result.series["total"].values
    assert values.dtype == dtype
    assert array_namespace(values) is array_namespace(states)
    state_grad, gain_grad = torch.autograd.grad(torch.sum(values), (states, gain))
    np.testing.assert_allclose(state_grad.detach().numpy(), np.full((3, 2), 2.0))
    np.testing.assert_allclose(gain_grad.detach().numpy(), 15.0)


def test_torch_wrong_namespace_and_dtype_are_rejected() -> None:
    """Callbacks cannot silently convert or change numerical precision."""
    torch = pytest.importorskip("torch")

    def foreign(_t: object, _y: object, **_params: object) -> object:
        return np.asarray(1.0)

    def wrong_dtype(_t: object, _y: object, **_params: object) -> object:
        return torch.tensor(1.0, dtype=torch.float32)

    for evaluator in (foreign, wrong_dtype):
        with pytest.raises(TypeError, match="preserve trajectory namespace and dtype"):
            observe_solution(
                [0.0],
                torch.ones((1, 2), dtype=torch.float64),
                [ObservationSpec("bad", evaluator)],
            )
