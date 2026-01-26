"""Integration test for the flepimop2 simulate pipeline using the op_engine provider.

This test invokes the public `flepimop2` CLI via subprocess to avoid coupling to
internal pipeline implementation details.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: TC002
import yaml


def _write_sir_stepper(script_path: Path) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        (
            '"""SIR model plugin for flepimop2 demo."""\n\n'
            "import numpy as np\n"
            "from numpy.typing import NDArray\n\n\n"
            "def stepper(\n"
            "    t: float,  # noqa: ARG001\n"
            "    y: NDArray[np.float64],\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            ") -> NDArray[np.float64]:\n"
            '    """dY/dt for the SIR model."""\n'
            "    y_s, y_i, _ = np.asarray(y, dtype=float)\n"
            "    infection = (beta * y_s * y_i) / np.sum(y)\n"
            "    recovery = gamma * y_i\n"
            "    dydt = [-infection, infection - recovery, recovery]\n"
            "    return np.array(dydt, dtype=float)\n"
        ),
        encoding="utf-8",
    )


def _write_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": "SIR_op_engine",
        "system": [{"module": "wrapper", "script": "model_input/plugins/SIR.py"}],
        "engine": [
            {
                "module": "op_engine",
                "config": {
                    "method": "heun",
                    "adaptive": False,
                    "strict": True,
                    "rtol": 1.0e-6,
                    "atol": 1.0e-9,
                },
            }
        ],
        "simulate": {"demo": {"times": [0, 10, 20]}},
        "backend": [{"module": "csv"}],
        "parameter": {
            "beta": {"module": "fixed", "value": 0.3},
            "gamma": {"module": "fixed", "value": 0.1},
            "s0": {"module": "fixed", "value": 999},
            "i0": {"module": "fixed", "value": 1},
            "r0": {"module": "fixed", "value": 0},
        },
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _run_flepimop2_simulate(config_path: Path) -> subprocess.CompletedProcess[str]:
    """
    Run `flepimop2 simulate` in a subprocess using the current environment.

    flepimop2 exposes a console script entrypoint (`flepimop2`) rather than
    supporting `python -m flepimop2` (no flepimop2.__main__).

    Args:
        config_path: Path to the flepimop2 YAML configuration file.

    Returns:
        CompletedProcess instance with execution results.
    """
    exe = Path(sys.executable)

    # Resolve the venv scripts directory in a cross-platform way.
    if exe.parent.name == "bin":
        cli = exe.parent / "flepimop2"
    else:
        # Windows layout: .../Scripts/python.exe
        cli = exe.parent / "flepimop2.exe"

    cmd = [str(cli), "simulate", str(config_path)]

    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")

    return subprocess.run(  # noqa:S603
        cmd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_simulate_pipeline_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end integration test invoking the public `flepimop2` CLI.

    Validates:
      - YAML config parsing
      - wrapper system loading from a script path
      - provider engine resolution: `engine: module: op_engine`
      - op_engine-backed execution
      - csv backend writes output
    """
    monkeypatch.chdir(tmp_path)

    (tmp_path / "model_output").mkdir(parents=True, exist_ok=True)
    _write_sir_stepper(tmp_path / "model_input" / "plugins" / "SIR.py")
    _write_config(tmp_path / "configs" / "SIR_op_engine.yml")

    proc = _run_flepimop2_simulate(Path("configs/SIR_op_engine.yml"))

    assert proc.returncode == 0, (
        "flepimop2 simulate failed\n\n"
        f"STDOUT:\n{proc.stdout}\n\n"
        f"STDERR:\n{proc.stderr}\n"
    )

    csv_files = sorted((tmp_path / "model_output").glob("*.csv"))
    assert csv_files, "expected csv backend to write at least one output file"

    arr = np.loadtxt(csv_files[-1], delimiter=",")
    assert arr.ndim == 2
    assert arr.shape[1] == 1 + 3  # time + 3-state SIR
    assert arr.shape[0] == 3  # times: [0, 10, 20]
