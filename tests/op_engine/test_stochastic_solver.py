"""Conformance tests for stochastic reaction-network solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    ModelCore,
    NumpyPoissonSampler,
    PoissonSampler,
    TauLeapingConfig,
    TauLeapingSolver,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _accept_sampler_protocol(_sampler: PoissonSampler) -> None:
    """Type-check the public sampler protocol."""


class _RecordingSampler:
    """Return prescribed firing arrays while recording means and step indices."""

    def __init__(self, *samples: NDArray[np.number]) -> None:
        self._samples = list(samples)
        self.means: list[NDArray[np.floating]] = []
        self.step_indices: list[int] = []

    def __call__(self, mean: Array, step_index: int, /) -> Array:
        """Record one call and return its prescribed sample.

        Returns:
            The next prescribed sample, or zeros if none remain.
        """
        self.means.append(np.asarray(mean).copy())
        self.step_indices.append(step_index)
        if self._samples:
            return cast("Array", self._samples.pop(0))
        return cast("Array", np.zeros(mean.shape, dtype=np.int64))


def _make_core(
    n_species: int,
    n_batch: int,
    times: NDArray[np.floating],
    *,
    dtype: object = np.float64,
) -> ModelCore:
    """Create a history-storing reaction-network core.

    Returns:
        Configured ``ModelCore``.
    """
    return ModelCore(
        n_species,
        n_batch,
        times,
        options=ModelCoreOptions(dtype=dtype),
    )


@pytest.mark.parametrize("max_step", [0.0, -1.0, np.inf, np.nan])
def test_tau_config_rejects_invalid_max_step(max_step: float) -> None:
    """Internal tau must be finite and positive when configured."""
    with pytest.raises(ValueError, match="max_step"):
        TauLeapingConfig(max_step=max_step)


@pytest.mark.parametrize("max_steps", [0, -1, 1.5, True])
def test_tau_config_rejects_invalid_max_steps(max_steps: object) -> None:
    """The per-interval leap limit is a positive integer."""
    with pytest.raises(ValueError, match="max_steps"):
        TauLeapingConfig(max_steps=max_steps)  # type: ignore[arg-type]


def test_tau_solver_validates_stoichiometry() -> None:
    """Stoichiometry must be a compatible integer-valued matrix."""
    core = _make_core(2, 1, np.asarray([0.0, 1.0]))

    with pytest.raises(ValueError, match="2D"):
        TauLeapingSolver(core, np.asarray([1.0, -1.0]))
    with pytest.raises(ValueError, match="species dimension"):
        TauLeapingSolver(core, np.ones((3, 1)))
    with pytest.raises(ValueError, match="integers"):
        TauLeapingSolver(core, np.asarray([[-0.5], [0.5]]))


def test_tau_leap_applies_stoichiometry_and_conserves_population() -> None:
    """One A-to-B firing channel moves the sampled integer count exactly."""
    core = _make_core(2, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[10.0], [0.0]]))
    sampler = _RecordingSampler(np.asarray([[3]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.asarray([[2.5]]))

    solver = TauLeapingSolver(core, np.asarray([[-1], [1]]))
    solver.run(propensity, sampler)

    assert solver.n_reactions == 1
    assert core.state_array is not None
    np.testing.assert_array_equal(core.state_array[-1, :, 0], [7.0, 3.0])
    np.testing.assert_array_equal(np.sum(core.state_array, axis=1), 10.0)
    np.testing.assert_allclose(sampler.means[0], [[2.5]])
    assert sampler.step_indices == [0]


def test_tau_leap_supports_a_nonleading_reaction_axis() -> None:
    """Stoichiometry acts along the configured axis and batches the others."""
    core = _make_core(2, 3, np.asarray([0.0, 1.0]))
    initial = np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    core.set_initial_state(initial)
    samples = np.asarray([[2, 1], [3, 2]])
    sampler = _RecordingSampler(samples)

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((2, 2)))

    stoichiometry = np.asarray([[-1, 0], [1, -1], [0, 1]])
    TauLeapingSolver(
        core,
        stoichiometry,
        reaction_axis="subgroup",
    ).run(propensity, sampler)

    expected = np.asarray([[8.0, 1.0, 1.0], [17.0, 1.0, 2.0]])
    np.testing.assert_array_equal(core.get_current_state(), expected)


def test_tau_leap_substeps_land_on_each_output_time() -> None:
    """A maximum tau creates stable global step indices and clipped final steps."""
    core = _make_core(1, 1, np.asarray([0.0, 0.5, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))
    sampler = _RecordingSampler()
    call_times: list[float] = []

    def propensity(time: float, _state: Array) -> Array:
        call_times.append(time)
        return cast("Array", np.ones((1, 1)))

    TauLeapingSolver(core, np.asarray([[1]])).run(
        propensity,
        sampler,
        config=TauLeapingConfig(max_step=0.2),
    )

    np.testing.assert_allclose(call_times, [0.0, 0.2, 0.4, 0.5, 0.7, 0.9])
    np.testing.assert_allclose(
        [float(mean[0, 0]) for mean in sampler.means],
        [0.2, 0.2, 0.1, 0.2, 0.2, 0.1],
    )
    assert sampler.step_indices == list(range(6))
    assert core.current_step == 2


def test_tau_leap_enforces_per_interval_step_limit() -> None:
    """Pathological fixed-tau configurations stop at their explicit guard."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.zeros((1, 1)))

    with pytest.raises(RuntimeError, match="max_steps"):
        TauLeapingSolver(core, np.asarray([[1]])).run(
            propensity,
            _RecordingSampler(),
            config=TauLeapingConfig(max_step=0.2, max_steps=2),
        )


@pytest.mark.parametrize(
    "propensity",
    [np.asarray([[-1.0]]), np.asarray([[np.inf]])],
)
def test_tau_leap_rejects_invalid_propensities(propensity: np.ndarray) -> None:
    """Poisson means cannot be negative or non-finite."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))

    with pytest.raises(ValueError, match="propensities"):
        TauLeapingSolver(core, np.asarray([[1]])).run(
            lambda _time, _state: cast("Array", propensity),
            _RecordingSampler(),
        )


@pytest.mark.parametrize(
    "sample",
    [np.asarray([[-1.0]]), np.asarray([[0.5]]), np.asarray([[np.inf]])],
)
def test_tau_leap_rejects_invalid_firing_counts(sample: np.ndarray) -> None:
    """Injected samplers must return finite non-negative integer counts."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((1, 1)))

    with pytest.raises(ValueError, match="firing counts"):
        TauLeapingSolver(core, np.asarray([[1]])).run(
            propensity,
            _RecordingSampler(sample),
        )


def test_tau_leap_rejects_negative_proposed_population() -> None:
    """Unsafe Poisson overshoot fails instead of silently clipping or biasing."""
    core = _make_core(2, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0], [0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((1, 1)))

    with pytest.raises(RuntimeError, match="negative population"):
        TauLeapingSolver(core, np.asarray([[-1], [1]])).run(
            propensity,
            _RecordingSampler(np.asarray([[2]])),
        )


def _run_seeded_birth_process(seed: int) -> np.ndarray:
    """Run a small reproducibility fixture.

    Returns:
        The stored state history.
    """
    core = _make_core(1, 1, np.asarray([0.0, 0.5, 1.0]))
    core.set_initial_state(np.asarray([[0.0]]))

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.full((1, 1), 4.0))

    TauLeapingSolver(core, np.asarray([[1]])).run(
        propensity,
        NumpyPoissonSampler(seed),
        config=TauLeapingConfig(max_step=0.1),
    )
    assert core.state_array is not None
    return np.asarray(core.state_array)


def test_numpy_poisson_sampler_is_reproducible() -> None:
    """Equal seeds and leap schedules produce equal stochastic trajectories."""
    _accept_sampler_protocol(NumpyPoissonSampler(1234))
    np.testing.assert_array_equal(
        _run_seeded_birth_process(1234),
        _run_seeded_birth_process(1234),
    )


def test_tau_leap_matches_poisson_mean_and_variance() -> None:
    """Independent batched birth channels recover analytic Poisson moments."""
    n_replicates = 50_000
    rate = 4.0
    core = _make_core(1, n_replicates, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.zeros((1, n_replicates)))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", np.full(state.shape, rate))

    TauLeapingSolver(core, np.asarray([[1]])).run(
        propensity,
        NumpyPoissonSampler(90210),
    )
    samples = np.asarray(core.get_current_state())[0]

    assert float(np.mean(samples)) == pytest.approx(rate, rel=0.02)
    assert float(np.var(samples)) == pytest.approx(rate, rel=0.04)


def test_tau_leap_preserves_jax_namespace_with_explicit_key_sampler() -> None:
    """JAX supplies randomness without becoming a core solver dependency."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(17)
    core = _make_core(
        2,
        1,
        np.asarray([0.0, 0.1, 0.2]),
        dtype=np.float32,
    )
    core.set_initial_state(jnp.asarray([[100.0], [0.0]], dtype=jnp.float32))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", jnp.reshape(0.2 * state[0, 0], (1, 1)))

    def poisson(mean: Array, step_index: int) -> Array:
        step_key = jax.random.fold_in(key, step_index)
        return cast("Array", jax.random.poisson(step_key, mean))

    TauLeapingSolver(core, np.asarray([[-1], [1]])).run(propensity, poisson)

    state = core.get_current_state()
    assert state.__array_namespace__() is jnp
    assert core.state_array is not None
    assert core.state_array.__array_namespace__() is jnp
    assert float(jnp.sum(state)) == pytest.approx(100.0)
    assert bool(jnp.all(state >= 0.0))
