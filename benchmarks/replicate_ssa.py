"""Correctness-gated independent-clock SIRS scaling measurements.

Run from the checkout with op_system 0.8.0+ installed. Timings are evidence,
never CI thresholds; compilation, network setup and execution are separated.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess  # noqa: S404
import time
from pathlib import Path

import numpy as np

from op_engine import (
    BatchedDirectSSASolver,
    DirectSSASolver,
    ModelCore,
    from_compiled_replicates,
    from_compiled_rhs,
)


def measure(size: int, days: int, *, flat: bool) -> dict[str, object]:  # noqa: PLR0914
    """Measure one bank after independently checking its mean drift.

    Returns:
        Dimensions, event counts and phase timings, with optional flat baseline.
    """
    from op_system import compile_spec  # noqa: PLC0415 - optional compiler benchmark

    spec = {
        "kind": "transitions",
        "factorize_axes": ["rep"],
        "axes": [{"name": "rep", "coords": [str(i) for i in range(size)]}],
        "state": ["S[rep]", "I[rep]", "R[rep]"],
        "aliases": {"N[rep]": "S[rep]+I[rep]+R[rep]"},
        "transitions": [
            {
                "name": "infect",
                "from": "S[rep]",
                "to": "I[rep]",
                "rate": "beta[rep]*I[rep]/N[rep]",
            },
            {"name": "recover", "from": "I[rep]", "to": "R[rep]", "rate": "gamma"},
            {"name": "wane", "from": "R[rep]", "to": "S[rep]", "rate": "omega"},
        ],
    }
    parameters = {
        "beta": np.random.default_rng(1).uniform(0.2, 0.4, size),
        "gamma": 0.1,
        "omega": 0.05,
    }
    start = time.perf_counter()
    compiled = compile_spec(spec)
    compiled_seconds = time.perf_counter() - start
    start = time.perf_counter()
    network = from_compiled_replicates(compiled, parameters, batch_axis="rep")
    setup_seconds = time.perf_counter() - start
    initial = np.tile([[90.0], [10.0], [0.0]], (1, size))
    hazards = network.propensity(np.zeros(size), initial)
    infection = parameters["beta"] * initial[0] * initial[1] / 100.0
    recovery, waning = 0.1 * initial[1], 0.05 * initial[2]
    expected = np.stack((-infection + waning, infection - recovery, recovery - waning))
    np.testing.assert_allclose(network.stoichiometry @ hazards, expected, rtol=1e-12)
    core = ModelCore(3, size, np.arange(days + 1, dtype=float))
    core.set_initial_state(initial)
    solver = BatchedDirectSSASolver(core, network.stoichiometry)
    start = time.perf_counter()
    solver.run(network.propensity, rng=np.random.SeedSequence(2))
    seconds = time.perf_counter() - start
    state = np.asarray(core.state_array)
    np.testing.assert_array_equal(state.sum(axis=1), 100.0)
    record = {
        "replicates": size,
        "days": days,
        "compile_seconds": compiled_seconds,
        "network_setup_seconds": setup_seconds,
        "batch_seconds": seconds,
        "seconds_per_replicate": seconds / size,
        "events": solver.events,
        "sweeps": solver.sweeps,
        "local_stoichiometry_bytes": network.stoichiometry.nbytes,
        "final_infected_mean": float(state[-1, 1].mean()),
    }
    if flat:
        start = time.perf_counter()
        expanded = from_compiled_rhs(compiled, parameters)
        record["flat_network_setup_seconds"] = time.perf_counter() - start
        flat_core = ModelCore(expanded.n_state, 1, np.arange(days + 1, dtype=float))
        flat_core.set_initial_state(initial.reshape((-1, 1)))
        start = time.perf_counter()
        DirectSSASolver(flat_core, expanded.stoichiometry).run(
            expanded.propensity, rng=np.random.default_rng(3)
        )
        record["flat_seconds"] = time.perf_counter() - start
        record["flat_stoichiometry_bytes"] = expanded.stoichiometry.nbytes
    return record


def main() -> None:
    """Write standalone measurement JSON with environment and source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 64, 256, 1024])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--flat-max", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        measure(size, args.days, flat=size <= args.flat_max) for size in args.sizes
    ]
    executable = shutil.which("git")
    if executable is None:
        parser.error("git is required to record source provenance")
    sha = subprocess.run(  # noqa: S603 - resolved git executable, fixed read-only arguments
        [executable, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # noqa: S603 - resolved git executable, fixed read-only arguments
            [executable, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "commit": sha,
                "dirty": dirty,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
