"""Measure frozen replay compilation and memory in isolated CPU processes.

Run with ``uv run python scripts/benchmark_replay.py``. Each JSON line records
one method, step count, iteration driver, and derivative. Add ``--loops scan
unroll`` for a comparison, and use ``--timeout`` to bound expensive unrolled
compilations. RSS includes backend startup; XLA buffer sizes describe the
compiled computation separately.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess  # noqa: S404
import sys
import time
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from op_engine import Array, CoreSolver, ModelCore
from op_engine.core_solver import AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable


def _parser() -> argparse.ArgumentParser:
    """Construct the reproducible benchmark's CLI.

    Returns:
        Parser for step counts, methods, drivers, and differentiation modes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=[8, 64, 160])
    parser.add_argument(
        "--methods", nargs="+", choices=["dopri5", "ros2"], default=["dopri5", "ros2"]
    )
    parser.add_argument(
        "--loops", nargs="+", choices=["scan", "unroll"], default=["scan"]
    )
    parser.add_argument(
        "--derivatives",
        nargs="+",
        choices=["grad", "hessian", "reverse-over-reverse"],
        default=["grad", "hessian", "reverse-over-reverse"],
    )
    parser.add_argument(
        "--checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _run_case(args: argparse.Namespace) -> dict[str, object]:  # noqa: PLR0914
    """Trace, compile, and execute one derivative without previous-case caches.

    Returns:
        Timings, process peak RSS, and XLA's compiled buffer requirements.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    count, method, loop, derivative = (
        args.steps[0],
        args.methods[0],
        args.loops[0],
        args.derivatives[0],
    )
    schedule = AdaptiveStepSchedule((0.0, 1.0), ((1.0 / count,) * count,))
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float64)

    def loss(values: Array) -> Array:
        rate, forcing = values
        core = ModelCore(
            3,
            1,
            np.asarray(schedule.output_times),
            options=ModelCoreOptions(store_history=False),
        )
        core.set_initial_state(jnp.asarray([[1.0], [0.75], [0.5]], dtype=values.dtype))
        CoreSolver(core).replay_adaptive_schedule(
            lambda t, y: rate * (1.0 + t) * y + forcing * t,
            schedule,
            config=RunConfig(
                method=method,
                adaptive=True,
                replay_loop=loop,
                replay_checkpoint=args.checkpoint,
                jacobian=lambda t, _y: rate * (1.0 + t) * jnp.eye(3),
            ),
        )
        final = core.get_current_state()
        return cast("Array", 0.5 * jnp.sum(final**2) + 2.0 * jnp.sum(values**2))

    def squared_gradient(values: Array) -> Array:
        return cast("Array", jnp.sum(jax.grad(loss)(values) ** 2))

    transformations: dict[str, Callable[[Array], Array]] = {
        "grad": jax.grad(loss),
        "hessian": jax.hessian(loss),
        "reverse-over-reverse": jax.grad(squared_gradient),
    }
    start = time.perf_counter()
    lowered = jax.jit(transformations[derivative]).lower(parameters)
    lowered_at = time.perf_counter()
    compiled = lowered.compile()
    compiled_at = time.perf_counter()
    result = jax.block_until_ready(compiled(parameters))
    finished_at = time.perf_counter()
    analysis = compiled.memory_analysis()
    rss_divisor = 1024**2 if sys.platform == "darwin" else 1024
    return {
        "method": method,
        "steps": count,
        "loop": loop,
        "derivative": derivative,
        "checkpoint": args.checkpoint and loop == "scan",
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "trace_lower_seconds": lowered_at - start,
        "compile_seconds": compiled_at - lowered_at,
        "run_seconds": finished_at - compiled_at,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / rss_divisor,
        "finite": bool(np.all(np.isfinite(np.asarray(result)))),
        "xla_buffer_bytes": {
            name: getattr(analysis, name, None)
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
        },
    }


def main() -> int:
    """Run isolated cases so peak memory and compilation costs remain comparable.

    Returns:
        Zero on success, or one if any worker fails or exceeds its timeout.
    """
    parser = _parser()
    args = parser.parse_args()
    if any(count < 1 for count in args.steps) or args.timeout <= 0:
        parser.error("Step counts and timeout must be positive")
    if args.worker:
        if any(
            len(values) != 1
            for values in (args.steps, args.methods, args.loops, args.derivatives)
        ):
            parser.error("A worker must receive exactly one case")
        print(json.dumps(_run_case(args)))
        return 0

    failed = False
    for count, method, loop, derivative in product(
        args.steps, args.methods, args.loops, args.derivatives
    ):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--steps",
            str(count),
            "--methods",
            method,
            "--loops",
            loop,
            "--derivatives",
            derivative,
            "--checkpoint" if args.checkpoint else "--no-checkpoint",
        ]
        try:
            worker = subprocess.run(  # noqa: S603
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.timeout,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
            )
            if worker.returncode == 0:
                result = json.loads(worker.stdout)
                failed = failed or not result["finite"]
                print(json.dumps(result), flush=True)
                continue
            error = worker.stderr[-2000:]
        except subprocess.TimeoutExpired:
            error = f"Worker exceeded {args.timeout:g} seconds"
        failed = True
        print(
            json.dumps({
                "method": method,
                "steps": count,
                "loop": loop,
                "derivative": derivative,
                "error": error,
            }),
            flush=True,
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
