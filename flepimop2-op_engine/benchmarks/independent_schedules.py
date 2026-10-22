# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026 Joshua Macdonald, Carl Pearson, Timothy Willard
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure CPU block discovery, compilation, replay and compiler memory estimates.

Every policy runs in an isolated worker and must pass the same analytic gate.
Timing is opt-in and never a test threshold. No application defaults are tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


POLICIES = ("shared", "independent-length", "independent-serial")


def _worker(options: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0914, PLR0915
    """Qualify one isolated policy and retain its full measurement context.

    Returns:
        One JSON-serializable receipt, including both setup and steady costs.

    Raises:
        RuntimeError: The CPU or common analytic accuracy contract fails.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from flepimop2.axis import ResolvedShape  # noqa: PLC0415
    from flepimop2.parameter.abc import ParameterValue  # noqa: PLC0415
    from flepimop2.typing import StateChangeEnum  # noqa: PLC0415

    from benchmarks.structured_blocks import _system  # noqa: PLC0415
    from flepimop2.engine.op_engine import (  # noqa: PLC0415
        IndependentBlockSchedule,
        OpEngineEngineConfig,
        OpEngineFlepimop2Engine,
        SolverMethod,
        StateLayout,
    )

    if jax.default_backend() != "cpu":
        msg = "This qualification script requires the CPU backend."
        raise RuntimeError(msg)
    count = options.blocks[0]
    policy = options.policies[0]
    setup_start = time.perf_counter()
    system, axes = _system(count)
    config = OpEngineEngineConfig(
        method=SolverMethod.DOPRI5,
        adaptive=True,
        state_layout=StateLayout.BLOCK,
        block_axis="loc",
        block_schedule="shared" if policy == "shared" else "independent",
        block_batching="serial" if policy.endswith("serial") else "length",
        block_batch_size=options.batch_size if policy != "shared" else None,
        rtol=1e-6,
        atol=1e-8,
        max_steps=1024,
    )
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=config)
    times = np.asarray([0.0, 0.2, 0.5, 1.0])
    rates = jnp.full((count,), -0.2, dtype=jnp.float32).at[-1].set(-50.0)
    initial = jnp.ones((2, count), dtype=jnp.float32)
    raw = {"rate": rates, "decay": jnp.asarray(0.15), "x0": initial, "y0": 2 * initial}
    params = {
        name: ParameterValue(
            value,
            ResolvedShape()
            if name == "decay"
            else axes.resolve_shape(("loc",) if name == "rate" else ("age", "loc")),
        )
        for name, value in raw.items()
    }
    model_state = system.model_state(axes)
    model_setup_seconds = time.perf_counter() - setup_start
    start = time.perf_counter()
    discovered = engine.run_adaptive(system, times, {}, params, model_state=model_state)
    cast("Any", discovered.trajectory).block_until_ready()
    discovery_seconds = time.perf_counter() - start
    schedule = discovered.schedule
    start = time.perf_counter()
    prepared = engine.prepare(
        system, times, {}, params, model_state=model_state, adaptive_schedule=schedule
    )
    prepare_seconds = time.perf_counter() - start
    jitted = jax.jit(lambda values: prepared({}, values))
    start = time.perf_counter()
    compiled = jitted.lower(raw).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = compiled(raw).block_until_ready()
    first_call_seconds = time.perf_counter() - start
    expected_x = np.asarray(initial)[None] * np.exp(
        times[:, None, None] * np.asarray(rates)[None, None]
    )
    expected_y = 2 * np.asarray(initial)[None] * np.exp(-0.15 * times[:, None, None])
    expected = np.concatenate(
        (
            times[:, None],
            expected_x.reshape(len(times), -1),
            expected_y.reshape(len(times), -1),
        ),
        axis=1,
    )
    errors = {}
    for name, trajectory in (("discovery", discovered.trajectory), ("replay", result)):
        error = float(
            np.max(
                np.abs(np.asarray(trajectory) - expected)
                / (2e-7 + 5e-5 * np.abs(expected))
            )
        )
        errors[name] = error
        if not np.isfinite(error) or error > 1:
            msg = f"{name} failed the common pointwise accuracy gate: {error}"
            raise RuntimeError(msg)
    validated = engine.run_adaptive(
        system, times, {}, params, model_state=model_state, schedule=schedule
    )
    validated.require_accurate()
    durations = []
    for _ in range(options.repeats):
        start = time.perf_counter()
        compiled(raw).block_until_ready()
        durations.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    memory_fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    if isinstance(schedule, IndependentBlockSchedule):
        step_counts = schedule.step_counts
        batches = schedule.batches(config)
        schedule_bytes = sum(
            len(json.dumps(block.step_schedule.step_sizes).encode())
            for block in schedule.blocks
        )
    else:
        step_counts = (sum(map(len, schedule.step_schedule.step_sizes)),) * count
        batches = (tuple(range(count)),)
        schedule_bytes = len(json.dumps(schedule.step_schedule.step_sizes).encode())
    sources = (
        Path(__file__),
        Path(__file__).with_name("structured_blocks.py"),
        Path(__file__).parents[1] / "src/flepimop2/engine/op_engine/__init__.py",
        Path(__file__).parents[1] / "src/flepimop2/engine/op_engine/config.py",
        Path(__file__).parents[2] / "src/op_engine/core_solver.py",
        Path(__file__).parents[2] / "src/op_engine/model_core.py",
        Path(__file__).parents[2] / "src/op_engine/_runge_kutta.py",
        Path(__file__).parents[2] / "src/op_engine/_forcing.py",
    )
    return {
        "status": "passed",
        "policy": policy,
        "blocks": count,
        "worker": options.worker,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "dtype": str(initial.dtype),
        "config": config.model_dump(mode="json"),
        "output_times": times.tolist(),
        "rates": np.asarray(rates).tolist(),
        "model_setup_seconds": model_setup_seconds,
        "discovery_seconds": discovery_seconds,
        "prepare_seconds": prepare_seconds,
        "compile_seconds": compile_seconds,
        "first_call_seconds": first_call_seconds,
        "replay_seconds": durations,
        "median_replay_seconds": statistics.median(durations),
        "pointwise_scaled_errors": errors,
        "step_counts": step_counts,
        "total_block_steps": sum(step_counts),
        "batches": batches,
        "padding_steps": 0,
        "mesh_json_bytes": schedule_bytes,
        "compiler_memory_estimates": {
            field: getattr(memory, field, None) for field in memory_fields
        },
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "source_hashes": {
            str(path.relative_to(Path(__file__).parents[2])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
    }


def main(args: Sequence[str] | None = None) -> None:
    """Launch isolated CPU workers and write one independently gated JSON row.

    Raises:
        ValueError: A size/count/policy combination is invalid.
        RuntimeError: At least one worker fails its common qualification gates.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", type=int, default=[4, 16])
    parser.add_argument(
        "--policies", nargs="+", choices=POLICIES, default=list(POLICIES)
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--workers-per-case", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(args)
    if min(*options.blocks, options.repeats, options.workers_per_case) < 1 or (
        options.batch_size is not None and options.batch_size < 1
    ):
        msg = "Sizes, counts and optional batch limits must be positive."
        raise ValueError(msg)
    if options.worker is not None:
        options.output.write_text(json.dumps(_worker(options), sort_keys=True) + "\n")
        return
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    failed = False
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w") as stream:
        for count in options.blocks:
            for policy in options.policies:
                for worker in range(options.workers_per_case):
                    worker_path = options.output.with_suffix(
                        f".{count}.{policy}.{worker}.json"
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.independent_schedules",
                        "--blocks",
                        str(count),
                        "--policies",
                        policy,
                        "--repeats",
                        str(options.repeats),
                        "--worker",
                        str(worker),
                        "--output",
                        str(worker_path),
                    ]
                    if options.batch_size is not None:
                        command.extend(("--batch-size", str(options.batch_size)))
                    if hasattr(os, "sched_getaffinity"):
                        command = [
                            "taskset",
                            "-c",
                            str(min(os.sched_getaffinity(0))),
                            *command,
                        ]
                    result = subprocess.run(  # noqa: S603
                        command,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        check=False,
                    )
                    if result.returncode:
                        failed = True
                        row = {
                            "status": "failed",
                            "blocks": count,
                            "policy": policy,
                            "worker": worker,
                            "stdout": result.stdout[-4000:],
                            "stderr": result.stderr[-8000:],
                        }
                    else:
                        row = json.loads(worker_path.read_text())
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
    if failed:
        msg = f"Worker qualification failed; all receipts retained in {options.output}."
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
