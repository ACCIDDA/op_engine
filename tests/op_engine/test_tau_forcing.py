"""Forcing-boundary contracts for fixed and adaptive tau-leaping."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import (
    AdaptiveTauLeapingConfig,
    AdaptiveTauLeapingSolver,
    Array,
    ModelCore,
    NumpyPoissonSampler,
    NumpySSASampler,
    TauLeapingConfig,
    TauLeapingSolver,
)
from op_engine.model_core import ModelCoreOptions


class _RecordingPoisson:
    """Record means and global indices while returning zero counts."""

    def __init__(self) -> None:
        self.means: list[float] = []
        self.indices: list[int] = []

    def __call__(self, mean: Array, index: int, /) -> Array:
        """Record a draw.

        Returns:
            Zero counts with the requested shape.
        """
        self.means.append(float(np.asarray(mean).item()))
        self.indices.append(index)
        return cast("Array", np.zeros(mean.shape, dtype=np.int64))


def _core(times: list[float], *, initial: float = 0.0) -> ModelCore:
    """Construct a scalar population history.

    Returns:
        NumPy core with double-precision state.
    """
    core = ModelCore(
        1, 1, np.asarray(times), options=ModelCoreOptions(dtype=np.float64)
    )
    core.set_initial_state(np.asarray([[initial]]))
    return core


@pytest.mark.parametrize(
    "points",
    [
        None,
        0.5,
        (True,),
        ("0.5",),
        (np.nan,),
        (np.inf,),
        ((0.5,),),
        (0.5, 0.5),
        (1.0, 0.5),
    ],
)
@pytest.mark.parametrize("config_type", [TauLeapingConfig, AdaptiveTauLeapingConfig])
def test_tau_config_rejects_invalid_schedule(
    points: object, config_type: type[TauLeapingConfig | AdaptiveTauLeapingConfig]
) -> None:
    """Tau schedules use the same strict validation as exact SSA."""
    with pytest.raises(ValueError, match="forcing_breakpoints"):
        config_type(forcing_breakpoints=points)  # type: ignore[arg-type]


@pytest.mark.parametrize("config_type", [TauLeapingConfig, AdaptiveTauLeapingConfig])
def test_tau_config_snapshots_mutable_schedule(
    config_type: type[TauLeapingConfig | AdaptiveTauLeapingConfig],
) -> None:
    """Changing the caller's list cannot move a solver boundary."""
    points = [-1.0, 0.5, 1.0]
    config = config_type(forcing_breakpoints=points)  # type: ignore[arg-type]
    points.append(2.0)
    assert config.forcing_breakpoints == (-1.0, 0.5, 1.0)
    assert isinstance(config.forcing_breakpoints, tuple)


@pytest.mark.parametrize(
    ("max_step", "expected_times", "expected_means"),
    [
        (None, [0.0, 0.3, 0.5, 0.7], [0.3, 0.4, 0.4, 1.2]),
        (0.2, [0.0, 0.2, 0.3, 0.5, 0.7, 0.9], [0.2, 0.1, 0.4, 0.4, 0.8, 0.4]),
    ],
)
def test_fixed_leaps_end_at_forcing_and_observation_times(
    max_step: float | None, expected_times: list[float], expected_means: list[float]
) -> None:
    """Step caps integrate each rate segment and preserve global draw indices."""
    core = _core([0.0, 0.5, 1.0])
    sampler = _RecordingPoisson()
    times: list[float] = []

    def propensity(time: float, _state: Array) -> Array:
        times.append(time)
        rate = 1.0 if time < 0.3 else (2.0 if time < 0.7 else 4.0)
        return cast("Array", np.asarray([[rate]]))

    TauLeapingSolver(core, np.asarray([[1]])).run(
        propensity,
        sampler,
        config=TauLeapingConfig(max_step=max_step, forcing_breakpoints=(0.3, 0.7)),
    )
    np.testing.assert_allclose(times, expected_times)
    np.testing.assert_allclose(sampler.means, expected_means)
    assert sampler.indices == list(range(len(expected_times)))


def test_fixed_zero_rate_resumes_and_global_schedule_respects_endpoints() -> None:
    """Dormant intervals advance without applying rates from the next segment."""
    core = _core([1.0, 1.25, 2.0])
    sampler = _RecordingPoisson()
    times: list[float] = []

    def propensity(time: float, _state: Array) -> Array:
        times.append(time)
        return cast("Array", np.asarray([[0.0 if time < 1.5 else 2.0]]))

    TauLeapingSolver(core, np.asarray([[1]])).run(
        propensity,
        sampler,
        config=TauLeapingConfig(forcing_breakpoints=(-1.0, 1.0, 1.5, 2.0, 3.0)),
    )
    assert times == [1.0, 1.25, 1.5]
    assert sampler.means == [0.0, 0.0, 1.0]
    assert sampler.indices == [0, 1, 2]


def test_fixed_forcing_does_not_reset_interval_step_limit() -> None:
    """Each boundary-capped leap counts toward the whole output interval."""
    core = _core([0.0, 1.0])
    sampler = _RecordingPoisson()
    with pytest.raises(RuntimeError, match="max_steps"):
        TauLeapingSolver(core, np.asarray([[1]])).run(
            lambda _t, _y: cast("Array", np.ones((1, 1))),
            sampler,
            config=TauLeapingConfig(max_steps=2, forcing_breakpoints=(0.25, 0.5)),
        )
    assert sampler.indices == [0, 1]


@pytest.mark.parametrize(
    ("adaptive", "seed", "expected"),
    [
        (False, 1234, [0, 2, 2]),
        (False, 2026, [0, 1, 3]),
        (True, 1234, [0, 3, 4]),
        (True, 2026, [0, 0, 2]),
    ],
)
@pytest.mark.parametrize("explicit_empty", [False, True])
def test_empty_schedule_preserves_pre_forcing_seeded_history(
    seed: int,
    expected: list[int],
    *,
    adaptive: bool,
    explicit_empty: bool,
) -> None:
    """Golden histories from the merged baseline preserve default randomness."""
    core = _core([0.0, 0.5, 1.0])

    def propensity(_time: float, _state: Array) -> Array:
        return cast("Array", np.full((1, 1), 3.0))

    if adaptive:
        config = (
            AdaptiveTauLeapingConfig(forcing_breakpoints=()) if explicit_empty else None
        )
        AdaptiveTauLeapingSolver(core, np.asarray([[1]]), np.asarray([[0]])).run(
            propensity,
            NumpyPoissonSampler(seed),
            NumpySSASampler(seed + 1),
            config=config,
        )
    else:
        fixed_config = (
            TauLeapingConfig(max_step=0.2, forcing_breakpoints=())
            if explicit_empty
            else TauLeapingConfig(max_step=0.2)
        )
        TauLeapingSolver(core, np.asarray([[1]])).run(
            propensity,
            NumpyPoissonSampler(seed),
            config=fixed_config,
        )
    np.testing.assert_array_equal(np.asarray(core.state_array).ravel(), expected)


@pytest.mark.parametrize("adaptive", [False, True])
def test_piecewise_birth_counts_match_integrated_poisson_law(*, adaptive: bool) -> None:
    """Both methods reproduce the analytic mean and variance of forced births."""
    n_paths = 16_000
    core = ModelCore(
        1,
        n_paths,
        np.asarray([0.0, 0.5, 1.0]),
        options=ModelCoreOptions(dtype=np.float64),
    )
    core.set_initial_state(np.zeros((1, n_paths)))

    def propensity(time: float, state: Array) -> Array:
        rate = 0.0 if time < 0.25 else (4.0 if time < 0.75 else 2.0)
        return cast("Array", np.full(state.shape, rate))

    sampler = NumpyPoissonSampler(104729)
    if adaptive:
        AdaptiveTauLeapingSolver(core, np.asarray([[1]]), np.asarray([[0]])).run(
            propensity,
            sampler,
            NumpySSASampler(104730),
            config=AdaptiveTauLeapingConfig(forcing_breakpoints=(0.25, 0.75)),
        )
    else:
        TauLeapingSolver(core, np.asarray([[1]])).run(
            propensity,
            sampler,
            config=TauLeapingConfig(max_step=0.2, forcing_breakpoints=(0.25, 0.75)),
        )
    counts = np.asarray(core.get_current_state()).ravel()
    expected = 2.5
    assert abs(float(counts.mean()) - expected) < 5 * np.sqrt(expected / n_paths)
    assert abs(float(counts.var()) - expected) < 5 * np.sqrt(
        (2 * expected**2 + expected) / n_paths
    )


def test_fixed_tiny_step_fails_before_drawing() -> None:
    """An unrepresentable time advance fails before mutating the state."""
    core = _core([1.0, 2.0])
    sampler = _RecordingPoisson()
    with pytest.raises(RuntimeError, match="underflowed"):
        TauLeapingSolver(core, np.asarray([[1]])).run(
            lambda _t, _y: cast("Array", np.ones((1, 1))),
            sampler,
            config=TauLeapingConfig(max_step=np.finfo(float).tiny),
        )
    assert sampler.indices == []
