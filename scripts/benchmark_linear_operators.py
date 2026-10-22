"""Measure original-residual-gated eager solves in isolated serial CPU workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess  # noqa: S404
import sys
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import diags, identity

from op_engine import (
    ActionOperator,
    BandedOperator,
    LinearSolveOptions,
    array_namespace,
    implicit_solve,
    register_sparse_adapter,
    scipy_banded_adapter,
    scipy_gmres_adapter,
    unregister_sparse_adapter,
)

SOLVERS = ("dense", "csr", "banded", "gmres", "gmres-jacobi")


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        msg = "sizes, RHS counts, workers and repeats must be positive"
        raise argparse.ArgumentTypeError(msg)
    return result


def _tolerance(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        msg = "tolerances must be finite and non-negative"
        raise argparse.ArgumentTypeError(msg)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=_positive, default=[64])
    parser.add_argument("--rhs", nargs="+", type=_positive, default=[1])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float64"], default=["float64"]
    )
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--repeats", type=_positive, default=5)
    parser.add_argument("--workers-per-case", type=_positive, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--rtol-float32", type=_tolerance, default=2e-5)
    parser.add_argument("--rtol-float64", type=_tolerance, default=1e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _worker(  # noqa: PLR0913, PLR0914, PLR0915 - bounded worker configuration
    n: int, columns: int, dtype: str, solver: str, repeats: int, *, rtol: float
) -> dict[str, object]:
    start = perf_counter()
    # Fixed spectral scale across sizes; this is a storage/solve experiment,
    # not a domain-refinement or application-calibration study.
    widths = np.geomspace(0.15, 1.5, n).astype(dtype)
    conductance = np.linspace(0.2, 1.2, max(n - 1, 0)).astype(dtype)
    reaction = np.geomspace(1.0, 8.0, n).astype(dtype)
    h = 0.2
    main = reaction.copy()
    main[:-1] += h * conductance / widths[:-1]
    main[1:] += h * conductance / widths[1:]
    lower = -h * conductance / widths[1:]
    upper = -h * conductance / widths[:-1]
    xp = array_namespace(main)

    def apply(v: np.ndarray) -> np.ndarray:
        flux = conductance * (v[1:] - v[:-1])
        net = np.zeros_like(v)
        net[:-1] += flux
        net[1:] -= flux
        return reaction * v - h * net / widths

    truth = np.asarray(
        2 + np.sin(np.arange(n)[:, None] * 0.3 + np.arange(columns)[None, :]),
        dtype=dtype,
    )
    # Reference RHS uses the physical cell-flux expression rather than packed
    # bands or a solver's materialized matrix.
    rhs = np.column_stack([apply(truth[:, j]) for j in range(columns)])
    options = LinearSolveOptions(
        rtol=rtol,
        atol=1e-7 if dtype == "float32" else 1e-12,
        max_iterations=100,
    )
    adapter_name = None
    if solver == "dense":
        left = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
        right = np.eye(n, dtype=dtype)
        storage = left.nbytes + right.nbytes
    elif solver == "csr":
        left = diags((lower, main, upper), (-1, 0, 1), format="csr")
        right = identity(n, dtype=dtype, format="csr")
        storage = sum(
            a.nbytes for op in (left, right) for a in (op.data, op.indices, op.indptr)
        )
    else:
        right = ActionOperator((n, n), main.dtype, xp, lambda v: v)
        if solver == "banded":
            bands = np.zeros((3, n), dtype=dtype)
            bands[0, 1:], bands[1], bands[2, :-1] = upper, main, lower
            # Use a diagonal singleton representation when n=1.
            left = BandedOperator(
                bands if n > 1 else bands[1:2], lower=min(1, n - 1), upper=min(1, n - 1)
            )
            adapter = scipy_banded_adapter(options=options)
            storage = left.bands.nbytes
        else:
            left = ActionOperator((n, n), main.dtype, xp, apply)
            inverse = (
                ActionOperator((n, n), main.dtype, xp, lambda v: v / main)
                if solver == "gmres-jacobi"
                else None
            )
            adapter = scipy_gmres_adapter(
                options=options,
                preconditioner=inverse,
                context=("synthetic-mesh-v1", n, h),
            )
            storage = (
                widths.nbytes
                + conductance.nbytes
                + reaction.nbytes
                + (main.nbytes if inverse is not None else 0)
            )
        register_sparse_adapter(adapter)
        adapter_name = adapter.ecosystem_id
    prepared = perf_counter() - start
    try:
        first_start = perf_counter()
        value = implicit_solve(left, right, rhs)
        first = perf_counter() - first_start
        samples = []
        for _ in range(repeats):
            tick = perf_counter()
            value = implicit_solve(left, right, rhs)
            samples.append(perf_counter() - tick)
        result = np.asarray(value)
        accuracy_passed = bool(
            np.allclose(
                result,
                truth,
                rtol=1e-4 if dtype == "float32" else 2e-9,
                atol=1e-5 if dtype == "float32" else 2e-10,
            )
        )
        residual = np.column_stack([apply(result[:, j]) for j in range(columns)]) - rhs
        norms = np.hypot.reduce(np.abs(residual.astype(np.float64)), axis=0)
        limits = np.maximum(
            options.atol,
            options.rtol * np.hypot.reduce(np.abs(rhs.astype(np.float64)), axis=0),
        )
        residual_passed = bool(np.all(np.isfinite(result)) and np.all(norms <= limits))
    finally:
        if adapter_name is not None:
            unregister_sparse_adapter(adapter_name)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "src/op_engine/__init__.py",
        *[
            root / f"src/op_engine/{name}.py"
            for name in (
                "scipy_operator",
                "linear_operator",
                "matrix_ops",
                "core_solver",
            )
        ],
    ]
    cpu_info = Path("/proc/cpuinfo")
    cpu = platform.processor()
    if cpu_info.is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_info.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "status": "success"
        if accuracy_passed and residual_passed
        else "validation_failed",
        "accuracy_passed": accuracy_passed,
        "residual_passed": residual_passed,
        "solve_rtol": options.rtol,
        "solve_atol": options.atol,
        "pointwise_rtol": 1e-4 if dtype == "float32" else 2e-9,
        "pointwise_atol": 1e-5 if dtype == "float32" else 2e-10,
        "solver": solver,
        "n": n,
        "rhs": columns,
        "dtype": dtype,
        "preparation_seconds": prepared,
        "first_solve_seconds": first,
        "warm_seconds": samples,
        "warm_median_seconds": float(np.median(samples)),
        "operator_storage_bytes": storage,
        "max_error": float(np.max(np.abs(result - truth))),
        "residual_norms": norms.tolist(),
        "residual_limits": limits.tolist(),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "notes": (
            "CSR warm solves reuse the historical LU cache; new adapters are "
            "uncached and include original-residual checks. Storage excludes "
            "cached LU/Krylov workspace, working arrays, RHS, truth and allocator "
            "overhead; peak RSS "
            "includes the whole fresh worker."
        ),
    }


def main() -> int:
    """Run serial fresh workers and retain failures/timeouts as records.

    Returns:
        Nonzero when --require-success is selected and any worker fails.
    """
    args = _parser().parse_args()
    if args.worker:
        row = _worker(
            args.sizes[0],
            args.rhs[0],
            args.dtypes[0],
            args.solvers[0],
            args.repeats,
            rtol=args.rtol_float32
            if args.dtypes[0] == "float32"
            else args.rtol_float64,
        )
        print(json.dumps(row))
        return 0
    rows: list[dict[str, object]] = []
    for n, columns, dtype, solver, trial in product(
        args.sizes, args.rhs, args.dtypes, args.solvers, range(args.workers_per_case)
    ):
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_linear_operators",
            "--worker",
            "--sizes",
            str(n),
            "--rhs",
            str(columns),
            "--dtypes",
            dtype,
            "--solvers",
            solver,
            "--repeats",
            str(args.repeats),
            "--rtol-float32",
            str(args.rtol_float32),
            "--rtol-float64",
            str(args.rtol_float64),
        ]
        metadata = {
            "solver": solver,
            "n": n,
            "rhs": columns,
            "dtype": dtype,
            "trial": trial,
        }
        try:
            tick = perf_counter()
            process = subprocess.run(  # noqa: S603 - validated CLI choices; no shell
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            if process.returncode:
                row = {
                    **metadata,
                    "status": "error",
                    "stderr": process.stderr,
                    "worker_wall_seconds": perf_counter() - tick,
                }
            else:
                row = {
                    **json.loads(process.stdout),
                    "trial": trial,
                    "worker_wall_seconds": perf_counter() - tick,
                }
        except subprocess.TimeoutExpired:
            row = {**metadata, "status": "timeout", "timeout_seconds": args.timeout}
        rows.append(row)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(row), flush=True)
    return int(args.require_success and any(r["status"] != "success" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
