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
    contracts = parser.add_mutually_exclusive_group()
    contracts.add_argument(
        "--compiler-contract",
        action="store_true",
        help="Qualify the separately installed pinned compiler observation contract",
    )
    contracts.add_argument(
        "--diffrax",
        action="store_true",
        help="Require the optional current-compiler Diffrax CPU adapter contract",
    )
    contracts.add_argument(
        "--torch-compile",
        action="store_true",
        help="Require the separately qualified CPU fullgraph Inductor adapter",
    )
    args = parser.parse_args()
    if args.torch_compile and args.backend != "torch":
        parser.error("--torch-compile requires the torch backend")
    if args.diffrax and args.backend != "jax":
        parser.error("--diffrax requires the jax backend")
    try:
        module = importlib.import_module(args.backend)
    except ImportError:
        print(
            f"Required backend {args.backend!r} is missing; "
            f"run just qualify-{args.backend}",
            file=sys.stderr,
        )
        return 1
    if args.diffrax:
        for name in ["diffrax", "op_system"]:
            importlib.import_module(name)
    print(
        f"Qualifying {args.backend} {module.__version__} "
        f"on Python {sys.version.split()[0]}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="op-engine-qualification-") as directory:
        report = Path(directory) / "results.xml"
        files = (
            ["tests/op_engine/test_jax_diffrax.py"]
            if args.diffrax
            else ["tests/op_engine/test_torch_loop.py"]
            if args.torch_compile
            else [
                "tests/op_engine/test_compiled_observations.py",
                "tests/op_engine/test_compiled_forcing.py",
                "tests/op_engine/test_compiled_sensitivities.py",
            ]
            if args.compiler_contract
            else [
                "tests/op_engine/test_backend_qualification.py",
                "tests/op_engine/test_sensitivities.py",
                "tests/op_engine/test_explicit_forcing.py",
                "tests/op_engine/test_operator_precision.py",
                "tests/op_engine/test_hierarchical_ode.py",
                "tests/op_engine/test_observations.py",
                "tests/op_engine/test_observation_reductions.py",
                "tests/op_engine/test_random_streams.py",
                "tests/op_engine/test_state_diagnostics.py",
                *(
                    ["tests/op_engine/test_numpy_event_update.py"]
                    if args.backend == "numpy"
                    else []
                ),
                *(
                    [
                        "tests/op_engine/test_array_namespace.py",
                        "tests/op_engine/test_model_core_autograd.py",
                    ]
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
