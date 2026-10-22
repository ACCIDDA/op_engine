"""Measure nonlinear solves in isolated CPU workers under common accuracy gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess  # noqa: S404
import sys
import tracemalloc
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy

from op_engine import (
    ActionOperator,
    Array,
    DenseNewtonSolver,
    LinearSolveOptions,
    LinearSolveRequest,
    NewtonConfig,
    NewtonKrylovConfig,
    NewtonKrylovSolver,
    NewtonLinearization,
    NonlinearProblem,
    NonlinearSolver,
    array_namespace,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]

SOLVERS = ("dense", "gmres", "gmres-jacobi")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        msg = "sizes, repeats and worker counts must be positive"
        raise argparse.ArgumentTypeError(msg)
    return number


def _worker(  # noqa: PLR0914, PLR0915 - bounded numerical worker and receipt
    n: int, dtype: str, solver: str, repeats: int
) -> dict[str, object]:
    """Compare manufactured coupled roots with fixed spectral scale across sizes.

    Returns:
        Failure-preserving timing/memory, accuracy and source provenance record.
    """
    setup_start = perf_counter()
    rates = np.geomspace(2, 200, n).astype(dtype)
    beta = 2.0
    truth = (0.5 + 0.1 * np.sin(np.arange(n) * 0.3)).astype(dtype)

    def physical(value: FloatArray) -> FloatArray:
        result = rates * value + 0.1 * value**3
        result += 2 * beta * value
        result[1:] -= beta * value[:-1]
        result[:-1] -= beta * value[1:]
        return result

    forcing = physical(truth)

    def residual(state: Array) -> Array:
        return physical(cast("FloatArray", state)) - forcing

    def jvp(state: Array, vector: Array) -> Array:
        value, direction = cast("FloatArray", state), cast("FloatArray", vector)
        result = (rates + 2 * beta + 0.3 * value**2) * direction
        result[1:] -= beta * direction[:-1]
        result[:-1] -= beta * direction[1:]
        return result

    def jacobian(state: Array) -> Array:
        diagonal = rates + 2 * beta + 0.3 * cast("FloatArray", state) ** 2
        result = np.diag(diagonal)
        index = np.arange(n - 1)
        result[index, index + 1] = result[index + 1, index] = -beta
        return result

    def inverse(request: LinearSolveRequest) -> ActionOperator:
        context = cast("NewtonLinearization", request.context)
        diagonal = rates + 2 * beta + 0.3 * context.iterate**2
        return ActionOperator(
            (n, n), rates.dtype, array_namespace(rates), lambda v: v / diagonal
        )

    rtol, atol = (2e-6, 1e-7) if dtype == "float32" else (1e-10, 1e-12)
    pointwise_limit = 1e-5 if dtype == "float32" else 1e-8
    configured: NonlinearSolver = (
        DenseNewtonSolver(NewtonConfig(max_iterations=8, rtol=rtol, atol=atol))
        if solver == "dense"
        else NewtonKrylovSolver(
            NewtonKrylovConfig(
                max_iterations=8,
                rtol=rtol,
                atol=atol,
                linear_options=LinearSolveOptions(
                    rtol=1e-4 if dtype == "float32" else 1e-9,
                    atol=0.0,
                    max_iterations=500,
                ),
            ),
            preconditioner_factory=inverse if solver == "gmres-jacobi" else None,
        )
    )
    problem = NonlinearProblem(
        residual,
        jacobian=jacobian if solver == "dense" else None,
        jvp=jvp if solver != "dense" else None,
    )
    guess = truth + np.asarray(0.2, dtype=dtype)
    initial_rms = float(
        np.sqrt(np.mean(np.square(np.asarray(residual(guess), dtype=np.float64))))
    )
    residual_limit = atol + rtol * initial_rms
    setup_seconds = perf_counter() - setup_start
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elapsed, residuals, pointwise, iterations, actions = [], [], [], [], []
    success = True
    for repeat in range(repeats + 1):
        tick = perf_counter()
        result = configured.solve(problem, guess)
        duration = perf_counter() - tick
        # Gate every run, including warmup; independently recompute physical R.
        rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        physical(np.asarray(result.value)).astype(np.float64)
                        - forcing.astype(np.float64)
                    )
                )
            )
        )
        error = float(
            np.max(
                np.abs(
                    np.asarray(result.value).astype(np.float64)
                    - truth.astype(np.float64)
                )
            )
        )
        success = bool(
            success
            and result.diagnostics.converged
            and np.isfinite(rms)
            and rms <= residual_limit
            and error <= pointwise_limit
        )
        residuals.append(rms)
        pointwise.append(error)
        iterations.append(result.diagnostics.iterations)
        actions.append(result.diagnostics.jvp_evaluations)
        if repeat:
            elapsed.append(duration)
    # Trace a separate solve so allocation instrumentation never affects timing.
    tracemalloc.start()
    memory_result = configured.solve(problem, guess)
    _, allocation_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_value = np.asarray(memory_result.value)
    memory_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    physical(memory_value).astype(np.float64)
                    - forcing.astype(np.float64)
                )
            )
        )
    )
    memory_error = float(
        np.max(np.abs(memory_value.astype(np.float64) - truth.astype(np.float64)))
    )
    success = bool(
        success
        and memory_result.diagnostics.converged
        and np.isfinite(memory_rms)
        and memory_rms <= residual_limit
        and memory_error <= pointwise_limit
    )
    root = Path(__file__).resolve().parents[1]
    numerical_files = [
        root / "src/op_engine" / name
        for name in (
            "newton_krylov.py",
            "nonlinear_solver.py",
            "linear_operator.py",
            "scipy_operator.py",
        )
    ]
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = (
        next(
            (
                line.partition(":")[2].strip()
                for line in cpu_info.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if cpu_info.exists()
        else platform.processor()
    )
    return {
        "status": "success" if success else "accuracy-failure",
        "solver": solver,
        "n": n,
        "dtype": dtype,
        "repeats": repeats,
        "setup_seconds": setup_seconds,
        "solve_seconds": elapsed,
        "median_solve_seconds": float(np.median(elapsed)),
        "residual_rms": residuals,
        "residual_limit": residual_limit,
        "max_pointwise_error": pointwise,
        "pointwise_limit": pointwise_limit,
        "memory_run_residual_rms": memory_rms,
        "memory_run_pointwise_error": memory_error,
        "newton_updates": iterations,
        "jvp_evaluations": actions,
        "peak_tracemalloc_bytes": allocation_peak,
        "baseline_peak_rss_mib": baseline_rss
        / (1024 if sys.platform != "darwin" else 1024**2),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 if sys.platform != "darwin" else 1024**2),
        "config": repr(configured.config),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*numerical_files, Path(__file__).resolve()]
        },
    }


def main() -> int:
    """Retain all isolated workers, including failures and timeouts.

    Returns:
        Nonzero when require-success was requested and any worker failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_positive, nargs="+", default=[32, 128, 512])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--workers-per-case", type=_positive, default=2)
    parser.add_argument("--timeout", type=_positive, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(
            json.dumps(
                _worker(args.sizes[0], args.dtypes[0], args.solvers[0], args.repeats)
            )
        )
        return 0
    rows = []
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for n, dtype, solver, trial in product(
        args.sizes, args.dtypes, args.solvers, range(args.workers_per_case)
    ):
        metadata = {"n": n, "dtype": dtype, "solver": solver, "trial": trial}
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_newton_krylov",
            "--worker",
            "--sizes",
            str(n),
            "--dtypes",
            dtype,
            "--solvers",
            solver,
            "--repeats",
            str(args.repeats),
        ]
        tick = perf_counter()
        try:
            process = subprocess.run(  # noqa: S603 - validated CLI, no shell
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
                env=environment,
            )
            row = (
                {**metadata, **json.loads(process.stdout)}
                if process.returncode == 0
                else {
                    **metadata,
                    "status": "error",
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                }
            )
        except subprocess.TimeoutExpired as error:
            row = {
                **metadata,
                "status": "timeout",
                "stdout": str(error.stdout),
                "stderr": str(error.stderr),
            }
        row["worker_wall_seconds"] = perf_counter() - tick
        rows.append(row)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(row), flush=True)
    return int(args.require_success and any(r["status"] != "success" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
