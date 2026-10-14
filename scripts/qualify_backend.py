"""Run required CPU solver qualification and reject absent or skipped cases."""

from __future__ import annotations

import argparse
import importlib
import subprocess  # noqa: S404
import sys
import tempfile
import xml.etree.ElementTree as ET  # noqa: S405
from pathlib import Path


def main() -> int:
    """Execute the selected backend's mandatory numerical contracts.

    Returns:
        Zero only after selected tests execute successfully without skips.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["numpy", "jax", "torch"])
    parser.add_argument(
        "--compiler-contract",
        action="store_true",
        help="Qualify the separately installed pinned compiler observation contract",
    )
    args = parser.parse_args()
    try:
        module = importlib.import_module(args.backend)
    except ImportError:
        print(
            f"Required backend {args.backend!r} is missing; "
            f"run just qualify-{args.backend}",
            file=sys.stderr,
        )
        return 1
    print(
        f"Qualifying {args.backend} {module.__version__} "
        f"on Python {sys.version.split()[0]}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="op-engine-qualification-") as directory:
        report = Path(directory) / "results.xml"
        files = (
            ["tests/op_engine/test_compiled_observations.py"]
            if args.compiler_contract
            else [
                "tests/op_engine/test_backend_qualification.py",
                "tests/op_engine/test_operator_precision.py",
                "tests/op_engine/test_hierarchical_ode.py",
                "tests/op_engine/test_observations.py",
                *(
                    ["tests/op_engine/test_array_namespace.py"]
                    if args.backend == "torch"
                    else []
                ),
            ]
        )
        process = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pytest",
                *files,
                "-k",
                args.backend,
                "-q",
                "--tb=short",
                f"--junitxml={report}",
            ],
            check=False,
        )
        if process.returncode:
            return process.returncode
        cases = ET.parse(report).findall(".//testcase")  # noqa: S314
        if not cases or any(case.find("skipped") is not None for case in cases):
            print(
                "Backend qualification must execute selected tests with zero skips",
                file=sys.stderr,
            )
            return 1
        print(f"Qualified {args.backend}: {len(cases)} tests, zero skips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
