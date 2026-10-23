# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Integration test for external provider functionality."""

import os
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from flepimop2.testing import external_provider_package, flepimop2_run


def _provider_core_pin(pyproject: Path) -> tuple[str, str]:
    """Read the provider version and its exact core pin.

    Returns:
        The provider version and required core version.
    """
    metadata = tomllib.loads(pyproject.read_text("utf-8"))
    provider_version = str(metadata["project"]["version"])
    core_dependencies = [
        dependency
        for dependency in metadata["project"]["dependencies"]
        if dependency.startswith("op-engine")
    ]
    assert len(core_dependencies) == 1
    dependency = core_dependencies[0]
    prefix = "op-engine=="
    assert dependency.startswith(prefix)
    required_core_version = dependency.removeprefix(prefix)
    assert required_core_version
    assert dependency == f"{prefix}{required_core_version}"
    return provider_version, required_core_version


def _candidate_core(cwd: Path, dist_dir: str | None) -> tuple[str, str]:
    """Resolve the candidate core requirement and version under test.

    Returns:
        The install requirement URI and candidate core version.
    """
    if dist_dir is None:
        core_root = cwd.parents[1]
        metadata = tomllib.loads((core_root / "pyproject.toml").read_text("utf-8"))
        return core_root.as_uri(), str(metadata["project"]["version"])
    wheels = sorted(Path(dist_dir).glob("op_engine-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    wheel_components = wheel.name.removesuffix(".whl").split("-")
    assert len(wheel_components) >= 5
    assert wheel_components[0] == "op_engine"
    return wheel.as_uri(), wheel_components[1]


def _incompatible_core_reason(
    *, provider_version: str, required_core_version: str, candidate_core_version: str
) -> str | None:
    """Explain a provider/core metadata mismatch, if one exists.

    Returns:
        ``None`` for an exact pair, otherwise a version-specific reason.
    """
    if candidate_core_version == required_core_version:
        return None
    return (
        "external-provider install requires a metadata-compatible pair: "
        f"provider {provider_version} pins op-engine=={required_core_version}, "
        f"but candidate core is {candidate_core_version}"
    )


@pytest.mark.parametrize(
    "versions_match",
    [True, False],
)
def test_external_provider_pair_gate(*, versions_match: bool) -> None:
    """Run the external install only for an exactly pinned core pair."""
    provider_version, required_core_version = _provider_core_pin(
        Path(__file__).parent.parent / "pyproject.toml"
    )
    candidate_core_version = required_core_version if versions_match else "999.0.0"
    reason = _incompatible_core_reason(
        provider_version=provider_version,
        required_core_version=required_core_version,
        candidate_core_version=candidate_core_version,
    )
    assert (reason is None) is versions_match
    if reason is not None:
        assert (
            f"provider {provider_version} pins op-engine=={required_core_version}"
            in reason
        )
        assert f"candidate core is {candidate_core_version}" in reason


def test_candidate_core_uses_wheel_version(tmp_path: Path) -> None:
    """A release clean room evaluates the constrained candidate wheel."""
    wheel = tmp_path / "op_engine-0.9.1-py3-none-any.whl"
    wheel.touch()
    requirement, version = _candidate_core(tmp_path, str(tmp_path))
    assert requirement == wheel.as_uri()
    assert version == "0.9.1"


def test_external_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run a simulation with an installed external op_engine provider package."""
    cwd = Path(__file__).parent.resolve()
    core_requirement, candidate_core_version = _candidate_core(
        cwd, os.environ.get("OP_ENGINE_TEST_DIST")
    )
    provider_version, required_core_version = _provider_core_pin(
        cwd.parent / "pyproject.toml"
    )
    reason = _incompatible_core_reason(
        provider_version=provider_version,
        required_core_version=required_core_version,
        candidate_core_version=candidate_core_version,
    )
    if reason is not None:
        pytest.skip(reason)

    constraints = tmp_path / "constraints.txt"
    constraints.write_text(f"op-engine @ {core_requirement}\n")
    monkeypatch.setenv("PIP_CONSTRAINT", str(constraints))
    monkeypatch.setenv("PIP_NO_CACHE_DIR", "1")
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
