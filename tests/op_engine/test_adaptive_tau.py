"""Conformance tests for bounded adaptive tau-leaping."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    AdaptiveTauLeapingConfig,
    AdaptiveTauLeapingSolver,
    Array,
    DirectSSASolver,
    ModelCore,
    NumpyPoissonSampler,
    NumpySSASampler,
    SSASample,
    TauLeapingConfig,
    TauLeapingSolver,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _RecordingPoissonSampler:
    """Return prescribed Poisson arrays and record means and draw indices."""

    def __init__(self, *samples: NDArray[np.number]) -> None:
        self._samples = list(samples)
        self.means: list[NDArray[np.floating]] = []
        self.draw_indices: list[int] = []

    def __call__(self, mean: Array, draw_index: int, /) -> Array:
        """Return the next sample or zeros.

        Returns:
            Firing counts with the shape of the supplied mean.
        """
        self.means.append(np.asarray(mean).copy())
        self.draw_indices.append(draw_index)
        if self._samples:
            return cast("Array", self._samples.pop(0))
        return cast("Array", np.zeros(mean.shape, dtype=np.int64))


class _AlwaysOvershootSampler:
    """Return two firings for every channel on every retry."""

    def __call__(self, mean: Array, draw_index: int, /) -> Array:
        """Return deliberate overshoots.

        Returns:
            Integer array filled with two.
        """
        del draw_index
        return cast("Array", np.full(mean.shape, 2, dtype=np.int64))


class _RecordingSSASampler:
    """Return prescribed exact-event samples and record draw indices."""

    def __init__(self, *samples: SSASample) -> None:
        self._samples = list(samples)
        self.draw_indices: list[int] = []
        self.probabilities: list[NDArray[np.floating]] = []

    def __call__(
        self,
        total_rate: Array,
        probabilities: Array,
        draw_index: int,
        /,
    ) -> SSASample:
        """Return the next exact-event sample.

        Returns:
            Sample supplied at construction.
        """
        del total_rate
        self.draw_indices.append(draw_index)
        self.probabilities.append(np.asarray(probabilities).copy())
        return self._samples.pop(0)


def _sample(waiting_time: float, event_index: int) -> SSASample:
    """Create one NumPy exact-event sample.

    Returns:
        Scalar waiting time and event index.
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
    """Create a history-storing core.

    Returns:
        Configured model core.
    """
    return ModelCore(
        n_species,
        n_batch,
        times,
        options=ModelCoreOptions(dtype=dtype),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("leap_tolerance", 0.0),
        ("leap_tolerance", 1.0),
        ("leap_tolerance", np.inf),
        ("critical_threshold", -1),
        ("critical_threshold", 1.5),
        ("exact_fallback_multiplier", -1.0),
        ("max_steps", 0),
        ("max_retries", -1),
    ],
)
def test_adaptive_tau_config_rejects_invalid_controls(
    field: str,
    value: object,
) -> None:
    """Adaptive, critical, fallback, and retry controls are validated."""
    with pytest.raises(ValueError, match=field):
        AdaptiveTauLeapingConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reactants", "message"),
    [
        (np.ones((2, 1)), "shape"),
        (np.asarray([[-1]]), "non-negative"),
        (np.asarray([[0.5]]), "integers"),
        (np.asarray([[4]]), "orders through three"),
    ],
)
def test_adaptive_tau_validates_reactant_stoichiometry(
    reactants: NDArray[np.number],
    message: str,
) -> None:
    """Reactant structure is explicit, integral, compatible, and elementary."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match=message):
        AdaptiveTauLeapingSolver(core, np.asarray([[1]]), reactants)


def test_adaptive_tau_rejects_impossible_product_stoichiometry() -> None:
    """Reactants plus net changes must imply non-negative products."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="incompatible"):
        AdaptiveTauLeapingSolver(
            core,
            np.asarray([[-2]]),
            np.asarray([[1]]),
        )


def test_cgp_selector_matches_analytic_first_order_tau() -> None:
    """A first-order death channel uses the published mean/variance bound."""
    core = _make_core(1, 1, np.asarray([0.0, 0.25]))
    core.set_initial_state(np.asarray([[100.0]]))
    poisson = _RecordingPoissonSampler()

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", state)

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[-1]]),
        np.asarray([[1]]),
    ).run(
        propensity,
        poisson,
        NumpySSASampler(1),
        config=AdaptiveTauLeapingConfig(
            leap_tolerance=0.1,
            critical_threshold=0,
            exact_fallback_multiplier=0.0,
        ),
    )

    np.testing.assert_allclose(
        [float(mean[0, 0]) for mean in poisson.means],
        [10.0, 10.0, 5.0],
    )
    assert poisson.draw_indices == [0, 1, 2]


def test_cgp_selector_applies_repeated_reactant_scaling() -> None:
    """A dimer reaction uses g=2+1/(x-1) for its reactant species."""
    core = _make_core(1, 1, np.asarray([0.0, 3.0]))
    core.set_initial_state(np.asarray([[100.0]]))
    poisson = _RecordingPoissonSampler()

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.ones((1, 1)))

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[-2]]),
        np.asarray([[2]]),
    ).run(
        propensity,
        poisson,
        NumpySSASampler(2),
        config=AdaptiveTauLeapingConfig(
            leap_tolerance=0.1,
            critical_threshold=0,
            exact_fallback_multiplier=0.0,
        ),
    )

    scaling = 2.0 + 1.0 / 99.0
    expected_tau = (10.0 / scaling) / 2.0
    assert float(poisson.means[0][0, 0]) == pytest.approx(expected_tau)


def test_critical_reaction_uses_exact_event_across_output_boundary() -> None:
    """A near-exhaustion reaction is retained and fired exactly once."""
    core = _make_core(1, 1, np.asarray([0.0, 0.2, 0.5]))
    core.set_initial_state(np.asarray([[3.0]]))
    poisson = _RecordingPoissonSampler()
    exact = _RecordingSSASampler(_sample(0.25, 0), _sample(1.0, 0))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", state)

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[-1]]),
        np.asarray([[1]]),
    ).run(propensity, poisson, exact)

    assert core.state_array is not None
    np.testing.assert_array_equal(core.state_array[:, 0, 0], [3.0, 3.0, 2.0])
    assert poisson.draw_indices == []
    assert exact.draw_indices == [0, 1]


def test_tight_tolerance_crosses_to_exact_ssa_fallback() -> None:
    """Tightening epsilon below the cost boundary selects an exact event."""
    core = _make_core(1, 1, np.asarray([0.0, 0.1]))
    core.set_initial_state(np.asarray([[100.0]]))
    poisson = _RecordingPoissonSampler()
    exact = _RecordingSSASampler(_sample(1.0, 0))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", 0.7 * state)

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[-1]]),
        np.asarray([[1]]),
    ).run(
        propensity,
        poisson,
        exact,
        config=AdaptiveTauLeapingConfig(
            leap_tolerance=0.001,
            critical_threshold=0,
        ),
    )

    assert poisson.draw_indices == []
    assert exact.draw_indices == [0]
    np.testing.assert_array_equal(core.get_current_state(), [[100.0]])


def test_adaptive_tau_supports_nonleading_reaction_axis() -> None:
    """Critical event flattening respects a nonleading species axis."""
    core = _make_core(2, 3, np.asarray([0.0, 0.5]))
    core.set_initial_state(np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    exact = _RecordingSSASampler(_sample(0.1, 1), _sample(1.0, 0))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", state[:, 0:1])

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[-1], [1], [0]]),
        np.asarray([[1], [0], [0]]),
        reaction_axis="subgroup",
    ).run(propensity, _RecordingPoissonSampler(), exact)

    np.testing.assert_array_equal(
        core.get_current_state(),
        np.asarray([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
    )


def test_mixed_leap_allows_at_most_one_critical_event() -> None:
    """Noncritical Poisson firings combine with one selected critical event."""
    core = _make_core(1, 1, np.asarray([0.0, 0.5]))
    core.set_initial_state(np.asarray([[1.0]]))
    poisson = _RecordingPoissonSampler(np.asarray([[2], [0]]))
    exact = _RecordingSSASampler(_sample(0.2, 1), _sample(1.0, 1))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", np.asarray([[4.0], [float(state[0, 0])]]))

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[1, -1]]),
        np.asarray([[0, 1]]),
    ).run(propensity, poisson, exact)

    np.testing.assert_array_equal(core.get_current_state(), [[2.0]])
    np.testing.assert_allclose(poisson.means[0], [[0.8], [0.0]])
    np.testing.assert_allclose(exact.probabilities[0], [[0.0], [1.0]])


def test_negative_post_leap_retries_without_clipping() -> None:
    """An overshoot is rejected, halved, and redrawn with stable indices."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))
    poisson = _RecordingPoissonSampler(
        np.asarray([[2]]),
        np.asarray([[0]]),
        np.asarray([[0]]),
    )

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", state)

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[-1]]),
        np.asarray([[1]]),
    ).run(
        propensity,
        poisson,
        NumpySSASampler(3),
        config=AdaptiveTauLeapingConfig(
            leap_tolerance=0.9,
            critical_threshold=0,
            exact_fallback_multiplier=0.0,
        ),
    )

    np.testing.assert_array_equal(core.get_current_state(), [[1.0]])
    np.testing.assert_allclose(
        [float(mean[0, 0]) for mean in poisson.means],
        [1.0, 0.5, 0.5],
    )
    assert poisson.draw_indices == [0, 1, 2]


def test_negative_post_leap_enforces_retry_limit() -> None:
    """Repeated overshoots fail rather than clipping populations or counts."""
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", state)

    with pytest.raises(RuntimeError, match="max_retries"):
        AdaptiveTauLeapingSolver(
            core,
            np.asarray([[-1]]),
            np.asarray([[1]]),
        ).run(
            propensity,
            _AlwaysOvershootSampler(),
            NumpySSASampler(4),
            config=AdaptiveTauLeapingConfig(
                leap_tolerance=0.9,
                critical_threshold=0,
                exact_fallback_multiplier=0.0,
                max_retries=1,
            ),
        )


def test_zero_total_propensity_is_absorbing() -> None:
    """No random sampler is called once the network is absorbing."""
    core = _make_core(1, 1, np.asarray([0.0, 0.5, 1.0]))
    core.set_initial_state(np.asarray([[7.0]]))
    propensity_calls = 0

    def propensity(_time: float, _state: Array) -> Array:
        nonlocal propensity_calls
        propensity_calls += 1
        return cast("Array", np.zeros((1, 1)))

    def unexpected_poisson(_mean: Array, _draw_index: int) -> Array:
        pytest.fail("absorbing state must not sample Poisson firings")

    def unexpected_exact(
        _total_rate: Array,
        _probabilities: Array,
        _draw_index: int,
    ) -> SSASample:
        pytest.fail("absorbing state must not sample exact events")

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[1]]),
        np.asarray([[0]]),
    ).run(propensity, unexpected_poisson, unexpected_exact)

    assert propensity_calls == 1
    assert core.state_array is not None
    np.testing.assert_array_equal(core.state_array[:, 0, 0], 7.0)


def test_adaptive_leap_uses_fewer_propensity_calls_than_fixed_tau() -> None:
    """A constant high-count birth process leaps over conservative fixed taus."""
    times = np.asarray([0.0, 1.0])
    adaptive_core = _make_core(1, 1, times)
    adaptive_core.set_initial_state(np.asarray([[1_000.0]]))
    fixed_core = _make_core(1, 1, times)
    fixed_core.set_initial_state(np.asarray([[1_000.0]]))
    adaptive_calls = 0
    fixed_calls = 0

    def adaptive_propensity(_time: float, _state: Array) -> Array:
        nonlocal adaptive_calls
        adaptive_calls += 1
        return cast("Array", np.asarray([[100.0]]))

    def fixed_propensity(_time: float, _state: Array) -> Array:
        nonlocal fixed_calls
        fixed_calls += 1
        return cast("Array", np.asarray([[100.0]]))

    AdaptiveTauLeapingSolver(
        adaptive_core,
        np.asarray([[1]]),
        np.asarray([[0]]),
    ).run(
        adaptive_propensity,
        _RecordingPoissonSampler(),
        NumpySSASampler(5),
    )
    TauLeapingSolver(fixed_core, np.asarray([[1]])).run(
        fixed_propensity,
        _RecordingPoissonSampler(),
        config=TauLeapingConfig(max_step=0.01),
    )

    assert adaptive_calls == 1
    assert fixed_calls == 100


def test_adaptive_birth_process_matches_analytic_poisson_distribution() -> None:
    """A state-independent birth process is one exact Poisson leap."""
    n_replicates = 50_000
    rate = 4.0
    core = _make_core(1, n_replicates, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.zeros((1, n_replicates)))

    def propensity(_time: float, state: Array) -> Array:
        xp = state.__array_namespace__()
        return cast("Array", xp.full(state.shape, rate, dtype=state.dtype))

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[1]]),
        np.asarray([[0]]),
    ).run(
        propensity,
        NumpyPoissonSampler(90210),
        NumpySSASampler(6),
    )
    samples = np.asarray(core.get_current_state())[0]

    assert float(np.mean(samples)) == pytest.approx(rate, rel=0.02)
    assert float(np.var(samples)) == pytest.approx(rate, rel=0.04)


def _run_low_count_death(
    seed: int,
    *,
    adaptive: bool,
) -> float:
    """Run one low-count death trajectory.

    Returns:
        Final population.
    """
    core = _make_core(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[4.0]]))

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", 0.7 * state)

    if adaptive:
        AdaptiveTauLeapingSolver(
            core,
            np.asarray([[-1]]),
            np.asarray([[1]]),
        ).run(
            propensity,
            NumpyPoissonSampler(seed + 10_000),
            NumpySSASampler(seed),
            config=AdaptiveTauLeapingConfig(leap_tolerance=0.001),
        )
    else:
        DirectSSASolver(core, np.asarray([[-1]])).run(
            propensity,
            NumpySSASampler(seed),
        )
    return float(np.asarray(core.get_current_state())[0, 0])


def test_tight_adaptive_low_count_paths_recover_direct_ssa() -> None:
    """Tight low-count runs cross the documented exact-SSA boundary."""
    direct = np.asarray([
        _run_low_count_death(seed, adaptive=False) for seed in range(100)
    ])
    adaptive = np.asarray([
        _run_low_count_death(seed, adaptive=True) for seed in range(100)
    ])

    np.testing.assert_array_equal(adaptive, direct)


def test_adaptive_tau_preserves_jax_namespace_with_explicit_keys() -> None:
    """JAX supplies eager Poisson draws without a backend dependency."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(19)
    core = _make_core(
        1,
        2,
        np.asarray([0.0, 0.5, 1.0]),
        dtype=np.float32,
    )
    core.set_initial_state(jnp.zeros((1, 2), dtype=jnp.float32))
    draw_indices: list[int] = []

    def propensity(_time: float, state: Array) -> Array:
        return cast("Array", jnp.full(state.shape, 2.0, dtype=state.dtype))

    def poisson(mean: Array, draw_index: int) -> Array:
        draw_indices.append(draw_index)
        draw_key = jax.random.fold_in(key, draw_index)
        return cast("Array", jax.random.poisson(draw_key, mean))

    def unexpected_exact(
        _total_rate: Array,
        _probabilities: Array,
        _draw_index: int,
    ) -> SSASample:
        pytest.fail("state-independent births must stay on the leap path")

    AdaptiveTauLeapingSolver(
        core,
        np.asarray([[1]]),
        np.asarray([[0]]),
    ).run(propensity, poisson, unexpected_exact)

    state = core.get_current_state()
    assert state.__array_namespace__() is jnp
    assert core.state_array is not None
    assert core.state_array.__array_namespace__() is jnp
    assert bool(jnp.all(state >= 0.0))
    assert draw_indices == [0, 1]
