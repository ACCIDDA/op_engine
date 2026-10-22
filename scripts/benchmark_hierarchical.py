"""Run correctness-gated hierarchical measurements in isolated serial CPU workers."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess  # noqa: S404
import sys
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


SOLVERS = (
    "core",
    "external",
    "diffrax-fixed",
    "diffrax-adaptive",
    "compiled-diffrax-fixed",
    "compiled-diffrax-adaptive",
    "numpy",
    "scipy",
)
DERIVATIVES = ("trajectory", "grad-fwd", "grad-rev", "hessian", "logdet-grad")


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        msg = "dimensions and minimum steps must be positive integers"
        raise argparse.ArgumentTypeError(msg)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=_positive, default=[2])
    parser.add_argument("--local-dimensions", nargs="+", type=_positive, default=[1])
    parser.add_argument("--steps", nargs="+", type=_positive, default=[16])
    parser.add_argument(
        "--regimes", nargs="+", choices=["nonstiff", "stiff"], default=["nonstiff"]
    )
    parser.add_argument(
        "--coupling",
        nargs="+",
        choices=["independent", "coupled"],
        default=["independent"],
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=SOLVERS,
        default=["core", "external", "diffrax-fixed"],
    )
    parser.add_argument(
        "--derivatives", nargs="+", choices=DERIVATIVES, default=list(DERIVATIVES)
    )
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _cases(args: argparse.Namespace) -> Iterator[dict[str, object]]:
    fields = (
        "groups",
        "local_dimensions",
        "steps",
        "regimes",
        "coupling",
        "solvers",
        "derivatives",
    )
    for values in product(*(getattr(args, field) for field in fields)):
        case = dict(zip(fields, values, strict=True))
        # Concrete reference solvers measure trajectories; they do not expose AD.
        if (
            case["solvers"] in {"numpy", "scipy"}
            and case["derivatives"] != "trajectory"
        ):
            continue
        yield case


def _isolated(case: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    command = [sys.executable, "-m", "scripts.benchmark_hierarchical", "--worker"]
    for field, value in case.items():
        command.extend(("--" + field.replace("_", "-"), str(value)))
    command.extend(("--repeats", str(args.repeats), "--timeout", str(args.timeout)))
    environment = os.environ.copy()
    environment.update({
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    try:
        process = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "case": case,
            "timeout_seconds": args.timeout,
            "censored": True,
            "timing_result": None,
            "interpretation": "No completed measurement; not a measured speedup",
        }
    if process.stdout.strip():
        try:
            row = json.loads(process.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            row = {"status": "error", "error": "Worker did not return valid JSON"}
    else:
        row = {"status": "error", "error": "Worker exited without a result"}
    if process.returncode:
        row.update({
            "status": "error",
            "returncode": process.returncode,
            "case": case,
            "stderr": process.stderr[-2000:],
        })
    return row


def main(argv: list[str] | None = None) -> int:
    """Emit one JSON record per independent measurement, including censored failures.

    Returns:
        Nonzero for failed required measurements or invalid timeout configuration.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.worker:
        try:
            from benchmarks.hierarchical_measurement import run_case  # noqa: PLC0415

            result = run_case(args)
        except (
            ImportError,
            RuntimeError,
            ValueError,
            TypeError,
            OverflowError,
        ) as error:
            print(
                json.dumps({
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
            )
            return 1
        print(json.dumps(result, allow_nan=False))
        return 0
    rows = [_isolated(case, args) for case in _cases(args)]
    text = "\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(rows)} cases to {args.output}")
    return int(args.require_success and any(row["status"] != "success" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
