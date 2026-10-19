"""Retain per-member output-state flags without changing numerical paths."""

from __future__ import annotations

import json

import numpy as np

from op_engine import (
    CoreSolver,
    ModelCore,
    OutputStateCheckConfig,
    diagnose_trajectory,
)
from op_engine.core_solver import RunConfig
from op_engine.model_core import ModelCoreOptions


def main() -> None:
    """Check a coarse Euler bank and a decreasing positive cumulative counter.

    Raises:
        RuntimeError: If the requested diagnostics are missing.
    """
    core = ModelCore(
        1,
        2,
        np.asarray([0.0, 1.0, 2.0]),
        options=ModelCoreOptions(store_history=False),
    )
    core.set_initial_state(np.ones((1, 2)))
    solver = CoreSolver(core)
    solver.run(
        lambda _time, state: np.multiply(state, [[-0.1, -2.0]]),
        config=RunConfig(method="euler", state_checks=OutputStateCheckConfig()),
    )
    result = solver.last_state_diagnostics
    if result is None:
        msg = "Output-state checks did not produce a record"
        raise RuntimeError(msg)
    np.testing.assert_allclose(result.min_state, [0.81, -1.0])
    np.testing.assert_array_equal(result.n_negative_outputs, [0, 1])
    np.testing.assert_array_equal(result.valid, [True, False])

    cumulative = diagnose_trajectory(
        np.asarray([[[5.0]], [[4.0]], [[3.0]]]),
        OutputStateCheckConfig(monotone_indices=(0,)),
    )
    np.testing.assert_array_equal(cumulative.n_negative_outputs, [0])
    np.testing.assert_array_equal(cumulative.n_decreasing_intervals, [2])
    print(  # noqa: T201 - command-line example output
        json.dumps(
            {
                "bank_min_state": np.asarray(result.min_state).tolist(),
                "bank_negative_outputs": np.asarray(result.n_negative_outputs).tolist(),
                "bank_output_flags": np.asarray(result.valid).tolist(),
                "cumulative_decreasing_intervals": np.asarray(
                    cumulative.n_decreasing_intervals
                ).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
