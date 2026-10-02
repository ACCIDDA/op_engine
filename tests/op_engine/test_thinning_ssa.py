"""Conformance and distribution checks for bounded thinning SSA."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    ModelCore,
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
