"""Integration test for external provider functionality."""

import re
from pathlib import Path

import numpy as np
import pytest
from flepimop2.testing import external_provider_package, flepimop2_run


def test_external_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run a simulation with an installed external op_engine provider package."""
    cwd = Path(__file__).parent.resolve()
    external_provider_package(
        tmp_path,
        copy_files={
            cwd / "data" / "config.yaml": Path("config.yaml"),
            cwd / "data" / "sir.py": Path("external_provider")
            / "src"
            / "flepimop2"
            / "system"
            / "sir.py",
        },
        dependencies=["op-engine"],
    )
    monkeypatch.chdir(tmp_path)

    assert len(list((tmp_path / "model_output").iterdir())) == 0
    result = flepimop2_run("simulate", args=["config.yaml"], cwd=tmp_path)
    assert result.returncode == 0

    model_output = list((tmp_path / "model_output").iterdir())
    assert len(model_output) == 1
    csv = model_output[0]
    assert re.match(r"^simulate_\d{8}_\d{6}\.csv$", csv.name)
    assert csv.stat().st_size > 0

    arr = np.loadtxt(csv, delimiter=",")
    assert arr.ndim == 2
    assert arr.shape[1] == 4
