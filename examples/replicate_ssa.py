"""Run independent exact SSA clocks with a forcing jump and firing records.

Run with ``python examples/replicate_ssa.py``. This example needs only the
NumPy core; compiler-certified specification wiring is in the replicate guide.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from op_engine import (
    BatchedDirectSSASolver,
    BatchedSSAConfig,
    FiringRecorder,
    ModelCore,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def simulate(
    replicate_ids: tuple[int, ...] = (10, 20),
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Simulate two-state transfers, preserving paths when the bank is reordered.

    Args:
        replicate_ids: Stable identities from the example's two parameter sets.

    Returns:
        State history (time, state, replicate) and interval firing counts.
    """
    parameters = {10: (12.0, 0.4), 20: (17.0, 0.7)}
    initial = np.array([
        [parameters[i][0] for i in replicate_ids],
        [0.0] * len(replicate_ids),
    ])
    rates = np.array([parameters[i][1] for i in replicate_ids])
    core = ModelCore(2, len(replicate_ids), np.linspace(0.0, 2.0, 5))
    core.set_initial_state(initial)
    recorder = FiringRecorder((0,))

    def hazards(
        times: NDArray[np.float64], state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        multiplier = np.where(times < 1.0, 1.0, 2.0)
        return rates[None, :] * multiplier[None, :] * state[:1]

    solver = BatchedDirectSSASolver(core, np.array([[-1.0], [1.0]]))
    solver.run(
        hazards,
        rng=np.random.SeedSequence(2026),
        replicate_ids=replicate_ids,
        config=BatchedSSAConfig(forcing_breakpoints=(1.0,)),
        recorder=recorder,
    )
    history = np.asarray(core.state_array)
    np.testing.assert_array_equal(
        history.sum(axis=1),
        np.broadcast_to(initial.sum(axis=0), (5, len(replicate_ids))),
    )
    np.testing.assert_array_equal(recorder.cumulative[:, 0], history[:, 1])
    return history, recorder.counts


def main() -> None:
    """Check bank-order and shard replay, then print the final transfer counts."""
    history, counts = simulate()
    reordered, reordered_counts = simulate((20, 10))
    np.testing.assert_array_equal(history, reordered[:, :, ::-1])
    np.testing.assert_array_equal(counts, reordered_counts[:, :, ::-1])
    alone, _ = simulate((10,))
    np.testing.assert_array_equal(history[:, :, :1], alone)
    print(  # noqa: T201 - executable example output
        json.dumps(
            {"final_state": history[-1].tolist(), "interval_counts": counts.tolist()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
