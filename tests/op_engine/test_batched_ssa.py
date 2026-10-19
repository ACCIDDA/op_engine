"""Independent-clock banks: analytic laws, replay and boundary semantics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import (
    Array,
    BatchedDirectSSASolver,
    BatchedSSAConfig,
    DirectSSASolver,
    FiringRecorder,
    ModelCore,
)


def _birth_bank(
    ids: list[int],
    times: list[float],
    *,
    record: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    rates = np.asarray(ids) * 0.03 + 0.2
    knots = 1.0 + np.asarray(ids) % 3
    core = ModelCore(1, len(ids), np.asarray(times))
    core.set_initial_state(cast("Array", np.zeros((1, len(ids)))))
    recorder = FiringRecorder() if record else None

    def propensity(clock: np.ndarray, _state: np.ndarray) -> np.ndarray:
        return np.where(clock < knots, 0.0, rates)[None, :]

    BatchedDirectSSASolver(core, np.ones((1, 1))).run(
        propensity,
        rng=np.random.SeedSequence(83),
        replicate_ids=ids,
        config=BatchedSSAConfig(
            replicate_breakpoints=tuple((float(k),) for k in knots)
        ),
        recorder=recorder,
    )
    return np.asarray(
        core.state_array
    ).copy(), None if recorder is None else recorder.counts.copy()


def test_replay_is_invariant_to_order_shards_and_output_grid() -> None:
    """Stable IDs keep a path fixed across heterogeneous clocks and bank layouts."""
    ids = [8, 0, 17, 3]
    full, counts = _birth_bank(ids, [0.0, 1.0, 2.0, 3.0, 7.0])
    reordered, reordered_counts = _birth_bank(ids[::-1], [0.0, 1.0, 2.0, 3.0, 7.0])
    np.testing.assert_array_equal(full, reordered[:, :, ::-1])
    np.testing.assert_array_equal(counts, reordered_counts[:, :, ::-1])
    coarse, _ = _birth_bank(ids, [0.0, 7.0], record=False)
    np.testing.assert_array_equal(full[-1], coarse[-1])
    for column, identity in enumerate(ids):
        shard, _ = _birth_bank([identity], [0.0, 1.0, 2.0, 3.0, 7.0])
        np.testing.assert_array_equal(full[:, :, column], shard[:, :, 0])
    assert counts is not None
    np.testing.assert_array_equal(np.diff(full, axis=0), counts)
    without, _ = _birth_bank(ids, [0.0, 1.0, 2.0, 3.0, 7.0], record=False)
    np.testing.assert_array_equal(full, without)


def test_death_distribution_matches_analytic_and_scalar_reference() -> None:
    """Independent pure deaths follow Binomial(n0, exp(-rate*t))."""
    size = 1200
    initial, rate, duration = 12.0, 0.4, 3.0
    core = ModelCore(1, size, np.array([0.0, duration]))
    core.set_initial_state(cast("Array", np.full((1, size), initial)))
    solver = BatchedDirectSSASolver(core, -np.ones((1, 1)))
    solver.run(lambda _t, y: rate * y, rng=np.random.SeedSequence(19))
    values = np.asarray(core.state_array)[-1, 0]
    survival = np.exp(-rate * duration)
    expected_mean = initial * survival
    expected_variance = initial * survival * (1.0 - survival)
    assert abs(values.mean() - expected_mean) < 5.0 * np.sqrt(expected_variance / size)
    assert abs(values.var() - expected_variance) < 0.4
    scalar_values = []
    for identity in range(300):
        scalar = ModelCore(1, 1, np.array([0.0, duration]))
        scalar.set_initial_state(cast("Array", np.array([[initial]])))
        DirectSSASolver(scalar, cast("Array", -np.ones((1, 1)))).run(
            lambda _t, y: cast("Array", rate * np.asarray(y)),
            rng=np.random.default_rng([37, identity]),
        )
        scalar_values.append(np.asarray(scalar.state_array)[-1, 0, 0])
    se = np.sqrt(expected_variance * (1.0 / size + 1.0 / len(scalar_values)))
    assert abs(values.mean() - np.mean(scalar_values)) < 5.0 * se
    assert solver.events == int(size * initial - values.sum())


def test_native_streams_replay_and_reused_generators_are_rejected() -> None:
    """Caller-owned streams advance, while SeedSequences replay fresh runs."""

    def run(source: object) -> np.ndarray:
        core = ModelCore(1, 2, np.array([0.0, 4.0]))
        core.set_initial_state(cast("Array", np.zeros((1, 2))))
        BatchedDirectSSASolver(core, np.ones((1, 1))).run(
            lambda _t, y: np.ones_like(y),
            rng=source,
        )
        return np.asarray(core.state_array)

    sources = [np.random.SeedSequence(4), np.random.SeedSequence(5)]
    np.testing.assert_array_equal(run(sources), run(sources))
    generator = np.random.default_rng(6)
    with pytest.raises(ValueError, match="distinct"):
        run([generator, generator])
    with pytest.raises(ValueError, match="distinct"):
        run([np.random.SeedSequence(4), np.random.SeedSequence(4)])
    with pytest.raises(TypeError, match="SeedSequence"):
        run(generator)


@pytest.mark.parametrize("initial", [-1.0, 0.5, np.inf, np.nan])
def test_invalid_initial_counts_fail_before_callback(initial: float) -> None:
    """Fractional and non-finite counts cannot enter integer-jump execution."""
    core = ModelCore(1, 1, np.array([0.0, 1.0]))
    core.set_initial_state(cast("Array", np.array([[initial]])))

    def never(_t: np.ndarray, _y: np.ndarray) -> np.ndarray:
        pytest.fail("callback must not run")

    with pytest.raises(ValueError, match="Initial counts"):
        BatchedDirectSSASolver(core, np.ones((1, 1))).run(
            never, rng=np.random.SeedSequence(1)
        )


@pytest.mark.parametrize("bad", [-1.0, np.inf, np.nan])
def test_invalid_hazards_fail(bad: float) -> None:
    """Every callback result must be finite and non-negative."""
    core = ModelCore(1, 1, np.array([0.0, 1.0]))
    core.set_initial_state(cast("Array", np.zeros((1, 1))))
    with pytest.raises(ValueError, match="propensities"):
        BatchedDirectSSASolver(core, np.ones((1, 1))).run(
            lambda _t, y: np.full_like(y, bad),
            rng=np.random.SeedSequence(1),
        )


@pytest.mark.parametrize("forcing", [False, True])
def test_exact_boundary_precedence(*, forcing: bool) -> None:
    """Forcing discards a tied candidate; a pure output observes its acceptance."""
    seed = 5
    wait = float(np.random.default_rng(seed).exponential())
    core = ModelCore(1, 1, np.array([0.0, wait, 2 * wait]))
    core.set_initial_state(cast("Array", np.zeros((1, 1))))
    recorder = FiringRecorder()
    BatchedDirectSSASolver(core, np.ones((1, 1))).run(
        lambda _t, y: np.ones_like(y),
        rng=[np.random.default_rng(seed)],
        config=BatchedSSAConfig(forcing_breakpoints=(wait,) if forcing else ()),
        recorder=recorder,
    )
    assert recorder.counts[0, 0, 0] == (0 if forcing else 1)


def test_event_guards_and_impossible_consumption() -> None:
    """A pathological replicate fails visibly rather than silently clipping."""
    core = ModelCore(1, 1, np.array([0.0, 1.0]))
    core.set_initial_state(cast("Array", np.zeros((1, 1))))
    with pytest.raises(RuntimeError, match="max_events"):
        BatchedDirectSSASolver(core, np.ones((1, 1))).run(
            lambda _t, y: np.full_like(y, 1e6),
            rng=np.random.SeedSequence(1),
            config=BatchedSSAConfig(max_events=1),
        )
    with pytest.raises(RuntimeError, match="negative"):
        BatchedDirectSSASolver(core, -np.ones((1, 1))).run(
            lambda _t, y: np.full_like(y, 1e6),
            rng=np.random.SeedSequence(1),
        )
