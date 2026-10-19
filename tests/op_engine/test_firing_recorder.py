"""Accepted counts agree with passive counters and preserve seeded paths."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import (
    Array,
    DirectSSAConfig,
    DirectSSASolver,
    FiringRecorder,
    ModelCore,
    SSASample,
)


def test_counts_match_passive_rows_with_replenishment_and_forcing() -> None:
    """State differences miss incidence; selected counts match passive rows."""
    matrix = np.array([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    initial = np.array([[90.0, 70.0], [10.0, 20.0], [0.0, 10.0]])
    times = np.array([0.0, 0.75, 2.0, 2.5, 7.0])

    def propensity(time: float, state: Array) -> Array:
        s, i, r = np.asarray(state)[:3]
        beta = 0.5 if time < 2.0 else 0.2
        return cast("Array", np.stack((beta * s * i / (s + i + r), 0.2 * i, 0.3 * r)))

    histories = []
    recorder = FiringRecorder((0, 2))
    for passive, record in [(False, None), (False, recorder), (True, None)]:
        stoich = (
            np.vstack((matrix, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
            if passive
            else matrix
        )
        start = np.vstack((initial, np.zeros((2, 2)))) if passive else initial
        core = ModelCore(stoich.shape[0], 2, times)
        core.set_initial_state(cast("Array", start))
        result = DirectSSASolver(core, cast("Array", stoich)).run(
            propensity,
            rng=np.random.default_rng(3),
            config=DirectSSAConfig(forcing_breakpoints=(2.0,)),
            recorder=record,
        )
        assert result is None
        histories.append(np.asarray(core.state_array).copy())
    np.testing.assert_array_equal(histories[0], histories[1])
    np.testing.assert_array_equal(histories[0], histories[2][:, :3])
    np.testing.assert_array_equal(recorder.counts, np.diff(histories[2][:, 3:], axis=0))
    np.testing.assert_array_equal(recorder.cumulative, histories[2][:, 3:])
    assert recorder.channel_indices == (0, 2)
    assert not recorder.counts.flags.writeable


def test_output_ties_count_and_forcing_ties_discard() -> None:
    """An accepted output-time event is counted; a forcing-time candidate is not."""
    samples = iter((0.5, 0.5, 0.25, 10.0))

    def sampler(total: Array, probabilities: Array, draw: int) -> SSASample:
        del total, probabilities, draw
        return SSASample(
            cast("Array", np.asarray(next(samples))), cast("Array", np.asarray(0))
        )

    core = ModelCore(1, 1, np.array([0.0, 0.5, 1.0, 2.0]))
    core.set_initial_state(cast("Array", np.zeros((1, 1))))
    recorder = FiringRecorder()
    DirectSSASolver(core, cast("Array", np.ones((1, 1)))).run(
        lambda _t, _y: cast("Array", np.ones((1, 1))),
        sampler,
        config=DirectSSAConfig(forcing_breakpoints=(1.0,)),
        recorder=recorder,
    )
    np.testing.assert_array_equal(recorder.counts[:, 0, 0], [1, 0, 1])
    np.testing.assert_array_equal(np.asarray(core.state_array)[:, 0, 0], [0, 1, 1, 2])


def test_absorbing_run_resets_recorder() -> None:
    """Reusing a recorder replaces its history, including absorbing intervals."""
    recorder = FiringRecorder((0,))
    recorder.start(1, (1, 1), 0)
    recorder.record(0, 0)
    core = ModelCore(1, 1, np.array([0.0, 1.0, 2.0]))
    core.set_initial_state(cast("Array", np.zeros((1, 1))))
    DirectSSASolver(core, cast("Array", np.ones((1, 1)))).run(
        lambda _t, _y: cast("Array", np.zeros((1, 1))),
        rng=np.random.default_rng(1),
        recorder=recorder,
    )
    np.testing.assert_array_equal(recorder.counts, np.zeros((2, 1, 1)))


@pytest.mark.parametrize("channels", [(True,), (-1,), (0, 0), (1.5,)])
def test_invalid_channel_selections(channels: tuple[int, ...]) -> None:
    """Invalid selections fail before recording can silently target a wrong row."""
    with pytest.raises(ValueError, match="channels"):
        FiringRecorder(channels)


def test_out_of_range_selection_and_unstarted_access() -> None:
    """A layout validates the channel range, and absent records are explicit."""
    recorder = FiringRecorder((2,))
    with pytest.raises(RuntimeError, match="not started"):
        _ = recorder.counts
    with pytest.raises(ValueError, match="outside"):
        recorder.start(1, (2, 1), 0)
