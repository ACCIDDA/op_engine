"""Forcing-boundary contracts for fixed and adaptive tau-leaping."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import (
    Array,
    ModelCore,
    NumpyPoissonSampler,
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
def test_fixed_config_rejects_invalid_schedule(points: object) -> None:
    """Tau schedules use the same strict validation as exact SSA."""
    with pytest.raises(ValueError, match="forcing_breakpoints"):
        TauLeapingConfig(forcing_breakpoints=points)  # type: ignore[arg-type]


def test_fixed_config_snapshots_mutable_schedule() -> None:
    """Changing the caller's list cannot move a solver boundary."""
    points = [-1.0, 0.5, 1.0]
    config = TauLeapingConfig(forcing_breakpoints=points)  # type: ignore[arg-type]
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


@pytest.mark.parametrize("seed", [1234, 2026])
def test_fixed_empty_schedule_preserves_seeded_history(seed: int) -> None:
    """Explicit empty schedules preserve the default random draw sequence."""
    histories = []
    for config in [
        TauLeapingConfig(max_step=0.2),
        TauLeapingConfig(max_step=0.2, forcing_breakpoints=()),
    ]:
        core = _core([0.0, 0.5, 1.0])
        TauLeapingSolver(core, np.asarray([[1]])).run(
            lambda _t, _y: cast("Array", np.full((1, 1), 3.0)),
            NumpyPoissonSampler(seed),
            config=config,
        )
        histories.append(np.asarray(core.state_array).copy())
    np.testing.assert_array_equal(*histories)


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
