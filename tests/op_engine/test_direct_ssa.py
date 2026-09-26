"""Conformance tests for the exact direct-SSA reaction solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    DirectSSAConfig,
    DirectSSASolver,
    ModelCore,
    NumpySSASampler,
    SSASample,
    SSASampler,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def _accept_sampler_protocol(_sampler: SSASampler) -> None:
    """Type-check the public sampler protocol."""


class _RecordingSSASampler:
    """Return prescribed SSA samples while recording normalized inputs."""

    def __init__(self, *samples: SSASample) -> None:
        self._samples = list(samples)
        self.total_rates: list[NDArray[np.floating]] = []
        self.probabilities: list[NDArray[np.floating]] = []
        self.draw_indices: list[int] = []

    def __call__(
        self,
        total_rate: Array,
        probabilities: Array,
        draw_index: int,
        /,
    ) -> SSASample:
        """Record and return the next prescribed sample.

        Returns:
            Next sample supplied at construction.
        """
        self.total_rates.append(np.asarray(total_rate).copy())
        self.probabilities.append(np.asarray(probabilities).copy())
        self.draw_indices.append(draw_index)
        return self._samples.pop(0)


class _ConstantSSASampler:
    """Return a fixed positive wait and the first event."""

    def __init__(self, waiting_time: float) -> None:
        self.waiting_time = waiting_time

    def __call__(
        self,
        total_rate: Array,
        probabilities: Array,
        draw_index: int,
        /,
    ) -> SSASample:
        """Return one sample in the inputs' NumPy namespace.

        Returns:
            Fixed waiting time and event zero.
        """
        del total_rate, probabilities, draw_index
        return SSASample(
            cast("Array", np.asarray(self.waiting_time)),
            cast("Array", np.asarray(0)),
        )


def _sample(waiting_time: object, event_index: object) -> SSASample:
    """Create one NumPy SSA sample.

    Returns:
        Sample containing scalar or deliberately invalid test arrays.
    """
    return SSASample(
        cast("Array", np.asarray(waiting_time)),
        cast("Array", np.asarray(event_index)),
    )


def _make_core(
    n_species: int,
    n_batch: int,
    times: NDArray[np.floating],
    *,
    dtype: object = np.float64,
) -> ModelCore:
    """Create a history-storing reaction-network core.

    Returns:
        Configured core.
    """
    return ModelCore(
        n_species,
        n_batch,
        times,
        options=ModelCoreOptions(dtype=dtype),
    )


@pytest.mark.parametrize("max_events", [0, -1, 1.5, True])
def test_direct_ssa_config_rejects_invalid_event_limit(max_events: object) -> None:
    """The per-interval event guard is a positive integer."""
    with pytest.raises(ValueError, match="max_events"):
        DirectSSAConfig(max_events=max_events)  # type: ignore[arg-type]


def test_direct_ssa_retains_event_across_output_boundaries() -> None:
    """Observation times neither discard nor resample a pending event."""
    core = _make_core(1, 1, np.asarray([0.0, 0.5, 1.0]))
    core.set_initial_state(np.asarray([[3.0]]))
    sampler = _RecordingSSASampler(_sample(0.75, 0), _sample(0.5, 0))
    propensity_times: list[float] = []

    def propensity(time: float, _state: Array) -> Array:
        propensity_times.append(time)
        return cast("Array", np.asarray([[2.0]]))

    solver = DirectSSASolver(core, np.asarray([[1]]))
    solver.run(propensity, sampler)

    assert solver.n_reactions == 1
    assert core.state_array is not None
    np.testing.assert_array_equal(core.state_array[:, 0, 0], [3.0, 3.0, 4.0])
    np.testing.assert_allclose(propensity_times, [0.0, 0.75])
    np.testing.assert_allclose(sampler.total_rates, [2.0, 2.0])
    np.testing.assert_allclose(sampler.probabilities, [[[1.0]], [[1.0]]])
    assert sampler.draw_indices == [0, 1]


def test_direct_ssa_uses_normalized_channel_probabilities() -> None:
    """Direct-method category weights equal each propensity over total rate."""
    core = _make_core(1, 1, np.asarray([0.0, 0.1]))
    core.set_initial_state(np.asarray([[4.0]]))
    sampler = _RecordingSSASampler(_sample(1.0, 0))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.asarray([[2.0], [6.0]]))

    DirectSSASolver(core, np.asarray([[1, -1]])).run(propensity, sampler)

    np.testing.assert_allclose(sampler.total_rates[0], 8.0)
    np.testing.assert_allclose(sampler.probabilities[0], [[0.25], [0.75]])


def test_direct_ssa_flattens_reaction_and_batch_events() -> None:
    """One category selects one reaction and batch on a nonleading axis."""
    core = _make_core(2, 3, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    sampler = _RecordingSSASampler(_sample(0.5, 2), _sample(1.0, 0))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((2, 2)))

    DirectSSASolver(
        core,
        np.asarray([[-1, 0], [1, -1], [0, 1]]),
        reaction_axis="subgroup",
    ).run(propensity, sampler)

    np.testing.assert_array_equal(
        core.get_current_state(),
        np.asarray([[10.0, 0.0, 0.0], [19.0, 1.0, 0.0]]),
    )


def test_zero_total_propensity_is_absorbing() -> None:
    """An absorbing network stores all outputs without requesting randomness."""
    core = _make_core(1, 1, np.asarray([0.0, 0.5, 1.0, 2.0]))
    core.set_initial_state(np.asarray([[7.0]]))
    propensity_calls = 0

    def propensity(_time: float, _state: Array) -> Array:
        nonlocal propensity_calls
        propensity_calls += 1
        return cast("Array", np.zeros((1, 1)))

    def unexpected_sampler(
        _total_rate: Array,
        _probabilities: Array,
        _draw_index: int,
    ) -> SSASample:
        pytest.fail("absorbing state must not invoke the sampler")

    DirectSSASolver(core, np.asarray([[1]])).run(propensity, unexpected_sampler)

    assert propensity_calls == 1
    assert core.state_array is not None
    np.testing.assert_array_equal(core.state_array[:, 0, 0], 7.0)


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (_sample(0.0, 0), "waiting_time"),
        (_sample(np.inf, 0), "waiting_time"),
        (_sample([0.1], 0), "waiting_time"),
        (_sample(0.1, -1), "flat_event_index"),
        (_sample(0.1, 0.5), "flat_event_index"),
        (_sample(0.1, 1), "flat_event_index"),
        (_sample(0.1, [0]), "flat_event_index"),
    ],
)
def test_direct_ssa_rejects_invalid_sampler_values(
    sample: SSASample,
    message: str,
) -> None:
    """Injected waiting times and categorical indices are checked eagerly."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((1, 1)))

    with pytest.raises(ValueError, match=message):
        DirectSSASolver(core, np.asarray([[1]])).run(
            propensity,
            _RecordingSSASampler(sample),
        )


def test_direct_ssa_rejects_negative_population() -> None:
    """A positive impossible-channel propensity fails instead of clipping."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((1, 1)))

    with pytest.raises(RuntimeError, match="negative population"):
        DirectSSASolver(core, np.asarray([[-1]])).run(
            propensity,
            _RecordingSSASampler(_sample(0.1, 0)),
        )


def test_direct_ssa_enforces_per_interval_event_limit() -> None:
    """Explosive paths stop at their explicit event-count guard."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((1, 1)))

    with pytest.raises(RuntimeError, match="max_events"):
        DirectSSASolver(core, np.asarray([[1]])).run(
            propensity,
            _ConstantSSASampler(0.1),
            config=DirectSSAConfig(max_events=2),
        )


def test_numpy_ssa_sampler_matches_analytic_wait_and_channel_laws() -> None:
    """NumPy draws recover exponential means and categorical probabilities."""
    sampler = NumpySSASampler(90210)
    _accept_sampler_protocol(sampler)
    total_rate = cast("Array", np.asarray(4.0))
    probabilities = cast("Array", np.asarray([[0.25], [0.75]]))
    draws = [sampler(total_rate, probabilities, index) for index in range(20_000)]
    waiting_times = np.asarray([float(draw.waiting_time) for draw in draws])
    event_indices = np.asarray([int(draw.flat_event_index) for draw in draws])

    assert float(np.mean(waiting_times)) == pytest.approx(0.25, rel=0.03)
    assert float(np.mean(event_indices == 1)) == pytest.approx(0.75, rel=0.02)


def _run_seeded_birth_process(seed: int) -> NDArray[np.floating]:
    """Run a small reproducibility fixture.

    Returns:
        Stored state history.
    """
    core = _make_core(1, 1, np.asarray([0.0, 0.5, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.asarray([[3.0]]))

    DirectSSASolver(core, np.asarray([[1]])).run(
        propensity,
        NumpySSASampler(seed),
    )
    assert core.state_array is not None
    return np.asarray(core.state_array)


def test_numpy_ssa_sampler_is_reproducible() -> None:
    """Equal seeds produce equal exact stochastic trajectories."""
    np.testing.assert_array_equal(
        _run_seeded_birth_process(1234),
        _run_seeded_birth_process(1234),
    )


def test_direct_ssa_matches_poisson_birth_distribution() -> None:
    """Independent constant-rate births recover exact Poisson moments."""
    n_replicates = 800
    rate = 2.0
    core = _make_core(1, n_replicates, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.zeros((1, n_replicates)))

    def propensity(_time: float, state: Array) -> Array:
        xp = state.__array_namespace__()
        return cast("Array", xp.full(state.shape, rate, dtype=state.dtype))

    DirectSSASolver(core, np.asarray([[1]])).run(
        propensity,
        NumpySSASampler(2026),
    )
    samples = np.asarray(core.get_current_state())[0]

    assert float(np.mean(samples)) == pytest.approx(rate, rel=0.08)
    assert float(np.var(samples)) == pytest.approx(rate, rel=0.12)


def test_direct_ssa_matches_binomial_death_distribution() -> None:
    """Independent linear deaths recover the analytic binomial moments."""
    n_replicates = 800
    initial_count = 4
    rate = 0.7
    core = _make_core(1, n_replicates, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.full((1, n_replicates), initial_count))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", rate * state)

    DirectSSASolver(core, np.asarray([[-1]])).run(
        propensity,
        NumpySSASampler(2027),
    )
    samples = np.asarray(core.get_current_state())[0]
    survival_probability = np.exp(-rate)
    expected_mean = initial_count * survival_probability
    expected_variance = (
        initial_count * survival_probability * (1.0 - survival_probability)
    )

    assert float(np.mean(samples)) == pytest.approx(expected_mean, rel=0.06)
    assert float(np.var(samples)) == pytest.approx(expected_variance, rel=0.12)


def test_direct_ssa_preserves_jax_namespace_with_explicit_keys() -> None:
    """JAX supplies eager randomness without becoming a solver dependency."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(17)
    core = _make_core(
        2,
        1,
        np.asarray([0.0, 0.2, 0.4]),
        dtype=np.float32,
    )
    core.set_initial_state(jnp.asarray([[20.0], [0.0]], dtype=jnp.float32))
    draw_indices: list[int] = []

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", jnp.reshape(0.4 * state[0, 0], (1, 1)))

    def sample(
        total_rate: Array,
        probabilities: Array,
        draw_index: int,
    ) -> SSASample:
        draw_indices.append(draw_index)
        wait_key, event_key = jax.random.split(jax.random.fold_in(key, draw_index))
        wait = (
            jax.random.exponential(
                wait_key,
                shape=(),
                dtype=total_rate.dtype,
            )
            / total_rate
        )
        event = jax.random.categorical(
            event_key,
            jnp.log(jnp.reshape(probabilities, (-1,))),
        )
        return SSASample(cast("Array", wait), cast("Array", event))

    DirectSSASolver(core, np.asarray([[-1], [1]])).run(propensity, sample)

    state = core.get_current_state()
    assert state.__array_namespace__() is jnp
    assert core.state_array is not None
    assert core.state_array.__array_namespace__() is jnp
    assert float(jnp.sum(state)) == pytest.approx(20.0)
    assert bool(jnp.all(state >= 0.0))
    assert draw_indices == list(range(len(draw_indices)))


@pytest.mark.parametrize(
    "propensity",
    [
        lambda _time, _state: np.asarray([[-1.0]]),
        lambda _time, _state: np.asarray([[np.inf]]),
        lambda _time, _state: np.ones((2, 1)),
    ],
)
def test_direct_ssa_uses_shared_propensity_validation(
    propensity: Callable[[float, Array], NDArray[np.floating]],
) -> None:
    """SSA and tau-leaping share propensity value and shape validation."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    with pytest.raises(ValueError, match="propensit"):
        DirectSSASolver(core, np.asarray([[1]])).run(
            propensity,
            NumpySSASampler(1),
        )
