"""Conformance and distribution checks for bounded thinning SSA."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    DirectSSASolver,
    ModelCore,
    NumpySSASampler,
    NumpyThinningSampler,
    ThinningSample,
    ThinningSSAConfig,
    ThinningSSASolver,
    TotalRateBound,
    array_namespace,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _core(
    times: Sequence[float] = (0.0, 1.0),
    initial: NDArray[np.floating] | None = None,
) -> ModelCore:
    """Return a history-storing core with floating populations."""
    initial = np.zeros((1, 1)) if initial is None else initial
    core = ModelCore(
        *initial.shape,
        np.asarray(times),
        options=ModelCoreOptions(dtype=initial.dtype),
    )
    core.set_initial_state(initial)
    return core


class _Samples:
    """Record inputs and return prescribed namespace-native candidates."""

    def __init__(self, *draws: tuple[float, float]) -> None:
        self.draws = iter(draws)
        self.indices: list[int] = []
        self.rates: list[float] = []

    def __call__(self, rate: Array, index: int, /) -> ThinningSample:
        """Return the next prescribed exponential/uniform pair."""
        self.indices.append(index)
        self.rates.append(float(rate.item()))
        wait, uniform = next(self.draws)
        xp = array_namespace(rate)
        return ThinningSample(
            cast("Array", xp.asarray(wait, dtype=rate.dtype)),
            cast("Array", xp.asarray(uniform, dtype=rate.dtype)),
        )


def _smooth_birth(t: float, state: Array) -> Array:
    """Return independent birth rates growing smoothly from zero."""
    return cast(
        "Array", array_namespace(state).full(state.shape, 2 * t, dtype=state.dtype)
    )


def test_candidate_time_channels_rejections_and_pending_observations() -> None:
    """Channel probabilities use candidate-time rates across reaction/batch axes."""
    core = _core((0, 0.1, 0.25, 0.6, 0.75, 1), np.zeros((2, 2)))
    sampler = _Samples((0.25, 0.8), (0.25, 0.1), (0.25, 0.45), (2, 0))
    bounds: list[tuple[float, float, float]] = []

    def rates(t: float, _state: Array) -> Array:
        return cast("Array", np.asarray([[0, 0], [t, 1 - t]]))

    def bound(t: float, state: Array, limit: float) -> TotalRateBound:
        bounds.append((t, float(np.sum(state)), limit))
        return TotalRateBound(2, limit)

    solver = ThinningSSASolver(core, np.eye(2))
    assert solver.n_reactions == 2
    solver.run(rates, sampler, rate_bound=bound)
    np.testing.assert_array_equal(
        core.state_array,
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ],
    )
    assert sampler.indices == [0, 1, 2, 3]
    assert bounds == [(0, 0, 1), (0.5, 1, 1), (0.75, 2, 1)]


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_candidate_limits(value: object) -> None:
    """The per-run candidate limit must be a positive integer."""
    with pytest.raises(ValueError, match="max_candidates"):
        ThinningSSAConfig(max_candidates=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("points", [(0.5, 0.5), (1, 0), (np.inf,), (True,)])
def test_invalid_forcing_schedules(points: tuple[float, ...]) -> None:
    """Thinning uses the same strict forcing schedule as direct SSA."""
    with pytest.raises(ValueError, match="forcing_breakpoints"):
        ThinningSSAConfig(forcing_breakpoints=points)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "1", None, [1]])
def test_invalid_constant_and_callback_bound_rates(value: object) -> None:
    """Neither bound interface accepts malformed or non-finite rates."""
    with pytest.raises(ValueError, match="rate_bound"):
        ThinningSSASolver(_core(), np.asarray([[1]])).run(
            _smooth_birth,
            _Samples(),
            rate_bound=value,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="bound rate"):
        TotalRateBound(value, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("endpoint", [-1, 0, 2, np.nan, np.inf, True])
def test_invalid_bound_endpoints(endpoint: float) -> None:
    """Callback endpoints must be finite and strictly advance within the limit."""

    def bound(_t: float, _state: Array, _limit: float) -> TotalRateBound:
        return TotalRateBound(2, endpoint)

    with pytest.raises(ValueError, match="valid_until"):
        ThinningSSASolver(_core(), np.asarray([[1]])).run(
            _smooth_birth, _Samples(), rate_bound=bound
        )


@pytest.mark.parametrize(
    ("wait", "uniform"),
    [
        (0, 0.5),
        (-1, 0.5),
        (np.nan, 0.5),
        (np.inf, 0.5),
        (0.25, -0.1),
        (0.25, 1),
        (0.25, np.nan),
        (0.25, np.inf),
    ],
)
def test_invalid_candidate_randomness(wait: float, uniform: float) -> None:
    """Invalid exponential or uniform results fail before applying an event."""
    with pytest.raises(ValueError, match=r"waiting_time|uniform"):
        ThinningSSASolver(_core(), np.asarray([[1]])).run(
            _smooth_birth,
            _Samples((wait, uniform)),
            rate_bound=2,
        )


@pytest.mark.parametrize("initial_violation", [False, True])
def test_violated_bounds_fail_at_interval_start_and_candidate_time(
    *, initial_violation: bool
) -> None:
    """Rates exceeding a bound fail visibly even when the candidate would reject."""

    def rates(t: float, _state: Array) -> Array:
        return cast("Array", np.asarray([[2 if initial_violation or t > 0 else 0.0]]))

    sampler = _Samples((0.25, 0.99))
    with pytest.raises(
        ValueError, match=r"Total propensity 2\.0 exceeds bound rate 1\.0"
    ):
        ThinningSSASolver(_core(), np.asarray([[1]])).run(rates, sampler, rate_bound=1)
    assert sampler.indices == ([] if initial_violation else [0])


def test_impossible_reaction_and_nonadvancing_clock_fail() -> None:
    """Impossible consumption and waits below clock precision cannot be hidden."""
    rates = lambda _t, _y: cast("Array", np.ones((1, 1)))  # noqa: E731
    with pytest.raises(RuntimeError, match="invalid populations"):
        ThinningSSASolver(_core(), np.asarray([[-1]])).run(
            rates, _Samples((0.25, 0)), rate_bound=1
        )
    with pytest.raises(RuntimeError, match="failed to advance"):
        ThinningSSASolver(_core((1e16, 1e16 + 4)), np.asarray([[1]])).run(
            rates,
            _Samples((0.1, 0)),
            rate_bound=1,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_sampling_laws_and_dtype(dtype: type[np.floating]) -> None:
    """The sampler supplies independent exponential and uniform scalar laws."""
    sampler = NumpyThinningSampler(1730)
    rate = cast("Array", np.asarray(3, dtype=dtype))
    draws = [sampler(rate, index) for index in range(12_000)]
    values = np.asarray(draws)
    assert values.dtype == dtype
    assert np.all(values[:, 0] > 0)
    assert np.all((values[:, 1] >= 0) & (values[:, 1] < 1))
    assert np.mean(values[:, 0]) == pytest.approx(1 / 3, rel=0.03)
    assert np.var(values[:, 0]) == pytest.approx(1 / 9, rel=0.08)
    assert np.mean(values[:, 1]) == pytest.approx(0.5, abs=0.01)
    assert np.var(values[:, 1]) == pytest.approx(1 / 12, rel=0.04)
    assert abs(np.corrcoef(values.T)[0, 1]) < 0.03


@pytest.mark.parametrize(
    "value",
    [np.asarray([0.5]), np.asarray(1, dtype=bool), np.asarray(1), np.asarray(0.5j)],
)
@pytest.mark.parametrize("field", ["waiting_time", "uniform"])
def test_sampler_requires_real_floating_scalar_arrays(value: Array, field: str) -> None:
    """Non-scalar and non-floating samples cannot be interpreted as random draws."""

    def sampler(_rate: Array, _index: int, /) -> ThinningSample:
        valid = cast("Array", np.asarray(0.25))
        return (
            ThinningSample(value, valid)
            if field == "waiting_time"
            else ThinningSample(valid, value)
        )

    with pytest.raises(ValueError, match="real floating scalar"):
        ThinningSSASolver(_core(), np.asarray([[1]])).run(
            _smooth_birth, sampler, rate_bound=2
        )


def test_callback_and_sampler_return_types_are_checked() -> None:
    """Malformed records fail with actionable contract errors."""

    def bad_bound(_t: float, _state: Array, _limit: float) -> TotalRateBound:
        return cast("TotalRateBound", (2, 1))

    def bad_sample(_rate: Array, _index: int, /) -> ThinningSample:
        return cast("ThinningSample", (np.asarray(0.25), np.asarray(0.5)))

    solver = ThinningSSASolver(_core(), np.asarray([[1]]))
    with pytest.raises(TypeError, match="return a TotalRateBound"):
        solver.run(_smooth_birth, _Samples(), rate_bound=bad_bound)
    with pytest.raises(TypeError, match="return a ThinningSample"):
        solver.run(_smooth_birth, bad_sample, rate_bound=2)


def test_jax_sampler_cannot_return_numpy_randomness() -> None:
    """Array namespaces are enforced at the injected sampler boundary."""
    jnp = pytest.importorskip("jax.numpy")
    core = _core(initial=np.zeros((1, 1), dtype=np.float32))
    core.set_initial_state(jnp.zeros((1, 1), dtype=jnp.float32))

    def sampler(_rate: Array, _index: int, /) -> ThinningSample:
        return ThinningSample(
            cast("Array", np.asarray(0.25)), cast("Array", jnp.asarray(0.5))
        )

    with pytest.raises(TypeError, match="preserve the state array namespace"):
        ThinningSSASolver(core, np.asarray([[1]])).run(
            _smooth_birth, sampler, rate_bound=2
        )


@pytest.mark.parametrize("smooth", [False, True])
def test_birth_distribution_matches_integrated_intensity(*, smooth: bool) -> None:
    """Smooth births match analytic Poisson moments; constant births agree with SSA."""
    core = _core(initial=np.zeros((1, 1200)))
    rates = (
        _smooth_birth if smooth else lambda _t, y: cast("Array", np.full(y.shape, 2.0))
    )
    ThinningSSASolver(core, np.asarray([[1]])).run(
        rates,
        NumpyThinningSampler(1731),
        rate_bound=2400,
    )
    sample = np.asarray(core.get_current_state()).ravel()
    intensity = 1 if smooth else 2
    assert np.mean(sample) == pytest.approx(intensity, rel=0.08)
    assert np.var(sample) == pytest.approx(intensity, rel=0.15)
    if not smooth:
        reference = _core(initial=np.zeros((1, 1200)))
        DirectSSASolver(reference, np.asarray([[1]])).run(rates, NumpySSASampler(1732))
        direct = np.asarray(reference.get_current_state()).ravel()
        assert abs(np.mean(sample) - np.mean(direct)) < 0.15
        assert abs(np.var(sample) - np.var(direct)) < 0.3


@pytest.mark.parametrize("rate", [1e100, 1e-100])
def test_bound_must_be_representable_in_state_dtype(rate: float) -> None:
    """Casting a finite bound cannot overflow or silently become zero."""
    core = _core(initial=np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="representable"):
        ThinningSSASolver(core, np.asarray([[1]])).run(
            _smooth_birth, _Samples(), rate_bound=rate
        )


@pytest.mark.parametrize("times", [(0,), (0, 1)])
def test_final_boundary_and_singleton_grid_do_not_fire(
    times: tuple[float, ...],
) -> None:
    """The final exclusive endpoint discards a tied candidate without resampling."""
    core = _core(times)
    sampler = _Samples((1, 0))
    ThinningSSASolver(core, np.asarray([[1]])).run(_smooth_birth, sampler, rate_bound=2)
    np.testing.assert_array_equal(core.get_current_state(), [[0]])
    assert sampler.indices == ([] if len(times) == 1 else [0])


def test_nondefault_reaction_axis() -> None:
    """A reaction axis other than the first state dimension is respected."""
    core = _core(initial=np.zeros((1, 2)))
    ThinningSSASolver(core, np.eye(2), reaction_axis=1).run(
        lambda _t, _y: cast("Array", np.asarray([[0, 1.0]])),
        _Samples((0.25, 0), (1, 0)),
        rate_bound=1,
    )
    np.testing.assert_array_equal(core.get_current_state(), [[0, 1]])


def test_large_population_transfer_mean_and_drift_match_deterministic_solution() -> (
    None
):
    """A generic A-to-B network matches its smoothly forced mean and interval drift."""
    times = np.asarray([0, 0.5, 1])
    core = _core(times, np.asarray([np.full(64, 2000.0), np.zeros(64)]))

    def rates(t: float, state: Array) -> Array:
        return cast("Array", (0.02 + 0.03 * t) * np.asarray(state)[0:1])

    def bound(_t: float, state: Array, limit: float) -> TotalRateBound:
        return TotalRateBound(
            (0.02 + 0.03 * limit) * float(np.sum(np.asarray(state)[0])), limit
        )

    ThinningSSASolver(core, np.asarray([[-1], [1]])).run(
        rates, NumpyThinningSampler(1733), rate_bound=bound
    )
    history = np.asarray(core.state_array)
    np.testing.assert_array_equal(history.sum(axis=1), np.full((3, 64), 2000))
    mean = history.mean(axis=2)
    expected_a = 2000 * np.exp(-0.02 * times - 0.015 * times**2)
    expected = np.stack((expected_a, 2000 - expected_a), axis=1)
    np.testing.assert_allclose(mean, expected, atol=3, rtol=0)
    np.testing.assert_allclose(
        np.diff(mean, axis=0), np.diff(expected, axis=0), atol=3, rtol=0
    )


@pytest.mark.parametrize("forcing", [False, True])
@pytest.mark.parametrize("wait", [0.5, 0.75])
def test_expiry_and_forcing_discard_stale_candidates(
    *, forcing: bool, wait: float
) -> None:
    """Tied and later candidates never fire using an expired bound or channel."""
    core = _core((0, 0.5, 0.625, 1), np.zeros((2, 1)))
    sampler = _Samples((wait, 0), (0.125, 0), (1, 0))
    limits: list[float] = []

    def rates(t: float, _state: Array) -> Array:
        return cast(
            "Array", np.asarray([[1], [0]]) if t < 0.5 else np.asarray([[0], [1]])
        )

    def bound(t: float, _state: Array, limit: float) -> TotalRateBound:
        limits.append(limit)
        return TotalRateBound(1, min(0.5, limit) if t < 0.5 else limit)

    ThinningSSASolver(core, np.eye(2)).run(
        rates,
        sampler,
        rate_bound=bound,
        config=ThinningSSAConfig(forcing_breakpoints=(0.5,) if forcing else ()),
    )
    np.testing.assert_array_equal(
        core.state_array, [[[0], [0]], [[0], [0]], [[0], [1]], [[0], [1]]]
    )
    assert sampler.indices == [0, 1, 2]
    assert limits == ([0.5, 1, 1] if forcing else [1, 1, 1])


@pytest.mark.parametrize("zero_bound", [False, True])
def test_inactive_intervals_resume_without_observation_refresh(
    *,
    zero_bound: bool,
) -> None:
    """Zero actual rates can resume with either zero or positive initial bounds."""
    core = _core((0, 0.1, 0.5, 0.625, 1))
    draws = ((0.125, 0), (2, 0))
    sampler = _Samples(*draws if zero_bound else ((0.25, 0), (0.25, 0), *draws))

    def rates(t: float, state: Array) -> Array:
        return cast("Array", np.full(state.shape, 0 if t < 0.5 else 1.0))

    def bound(t: float, _state: Array, limit: float) -> TotalRateBound:
        return TotalRateBound(0 if zero_bound and t < 0.5 else 1, limit)

    ThinningSSASolver(core, np.asarray([[1]])).run(
        rates,
        sampler,
        rate_bound=bound,
        config=ThinningSSAConfig(forcing_breakpoints=(0.5,)),
    )
    np.testing.assert_array_equal(np.asarray(core.state_array).ravel(), [0, 0, 0, 1, 1])
    assert sampler.indices == list(range(2 if zero_bound else 4))


def test_accepted_events_refresh_state_dependent_bounds() -> None:
    """An accepted event invalidates the old bound; an absorbing bound draws nothing."""
    core = _core((0, 0.25, 0.5, 1), np.asarray([[2.0]]))
    sampler = _Samples((0.25, 0.1), (0.25, 0.1))

    def rates(t: float, state: Array) -> Array:
        return cast("Array", t * np.asarray(state))

    def bound(_t: float, state: Array, limit: float) -> TotalRateBound:
        return TotalRateBound(float(np.sum(state)), limit)

    ThinningSSASolver(core, np.asarray([[-1]])).run(rates, sampler, rate_bound=bound)
    np.testing.assert_array_equal(np.asarray(core.state_array).ravel(), [2, 1, 0, 0])
    assert sampler.rates == [2, 1]
    assert sampler.indices == [0, 1]


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_seeded_path_and_callback_history_ignore_added_observations(
    backend: str,
) -> None:
    """Injected NumPy/JAX draws preserve native arrays and global draw indices."""

    def run_grid(
        times: tuple[float, ...],
    ) -> tuple[NDArray[np.floating], list[int], list[tuple[float, tuple[float, ...]]]]:
        """Return the shared path, draw indices, and propensity evaluations."""
        core = _core(times, np.zeros((1, 3), dtype=np.float32))
        indices: list[int] = []
        evaluations: list[tuple[float, tuple[float, ...]]] = []
        if backend == "jax":
            jax = pytest.importorskip("jax")
            jnp = pytest.importorskip("jax.numpy")
            core.set_initial_state(jnp.zeros((1, 3), dtype=jnp.float32))
            key = jax.random.key(173)
        numpy_sampler = NumpyThinningSampler(173)

        def sampler(rate: Array, index: int, /) -> ThinningSample:
            indices.append(index)
            if backend == "numpy":
                return numpy_sampler(rate, index)
            wait_key, uniform_key = jax.random.split(jax.random.fold_in(key, index))
            return ThinningSample(
                cast(
                    "Array", jax.random.exponential(wait_key, dtype=rate.dtype) / rate
                ),
                cast("Array", jax.random.uniform(uniform_key, dtype=rate.dtype)),
            )

        def rates(t: float, state: Array) -> Array:
            evaluations.append((t, tuple(np.asarray(state).ravel())))
            return _smooth_birth(t, state)

        ThinningSSASolver(core, np.asarray([[1]])).run(
            rates,
            sampler,
            rate_bound=6,
            config=ThinningSSAConfig(forcing_breakpoints=(0.5,)),
        )
        assert indices == list(range(len(indices)))
        assert array_namespace(core.get_current_state()) is array_namespace(
            core.get_state_at(0)
        )
        if backend == "jax":
            assert core.get_current_state().__array_namespace__() is jnp
        shared = np.asarray(core.state_array)[[times.index(t) for t in (0, 0.5, 1)]]
        assert shared[-1].sum() > 0
        return shared, indices, evaluations

    histories = [
        run_grid(times) for times in ((0, 0.5, 1), (0, 0.125, 0.25, 0.5, 0.75, 1))
    ]
    np.testing.assert_array_equal(histories[0][0], histories[1][0])
    assert histories[0][1:] == histories[1][1:]


@pytest.mark.parametrize("expired", [False, True])
def test_candidate_guard_counts_rejections_and_expiry_across_observations(
    *,
    expired: bool,
) -> None:
    """Neither output times nor changing bounds reset the per-run draw guard."""
    core = _core((0, 0.1, 0.2, 0.3, 1))
    sampler = _Samples(*[(0.2 if expired else 0.125, 0.9)] * 4)

    def bound(t: float, _state: Array, limit: float) -> TotalRateBound:
        return TotalRateBound(1, min(t + 0.1, limit) if expired else limit)

    with pytest.raises(RuntimeError, match="max_candidates"):
        ThinningSSASolver(core, np.asarray([[1]])).run(
            lambda _t, y: cast("Array", np.zeros(y.shape)),
            sampler,
            rate_bound=bound,
            config=ThinningSSAConfig(max_candidates=4),
        )
    assert sampler.indices == [0, 1, 2, 3]
