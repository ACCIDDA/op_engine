"""Tests for core release artifact and documentation deployment boundaries."""

from __future__ import annotations

import io
import tarfile
import zipfile
from typing import TYPE_CHECKING

import pytest

from scripts.docs_deploy import deploy_docs, mike_deploy_command
from scripts.validate_core_artifacts import validate_core_artifacts

if TYPE_CHECKING:
    import pathlib


def test_development_docs_execute_without_duplicate_alias() -> None:
    """Mutable development docs pass only one `dev` argument to Mike."""
    calls: list[tuple[list[str], bool, bool]] = []

    def runner(command: list[str], *, check: bool, text: bool) -> object:
        calls.append((command, check, text))
        return object()

    deploy_docs("push", "0.9.0", "latest", runner=runner)
    assert calls == [(["mike", "deploy", "--push", "dev"], True, True)]


def test_release_docs_execute_with_distinct_version_and_alias() -> None:
    """An immutable release version can update its moving alias."""
    assert mike_deploy_command("release", "0.9.0", "latest") == [
        "mike",
        "deploy",
        "--push",
        "--update-aliases",
        "0.9.0",
        "latest",
    ]
    with pytest.raises(ValueError, match="distinct"):
        mike_deploy_command("release", "dev", "dev")


def test_core_wheel_and_sdist_exclude_provider(tmp_path: pathlib.Path) -> None:
    """Both core artifact formats can contain core files without provider files."""
    wheel = tmp_path / "op_engine-0.9.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr("op_engine/__init__.py", "")

    sdist = tmp_path / "op_engine-0.9.0.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        payload = b""
        info = tarfile.TarInfo("op_engine-0.9.0/src/op_engine/__init__.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    validate_core_artifacts([wheel, sdist])


@pytest.mark.parametrize("artifact_kind", ["wheel", "sdist"])
def test_core_artifact_rejects_provider_tree(
    tmp_path: pathlib.Path, artifact_kind: str
) -> None:
    """A held provider subtree fails either core artifact format."""
    if artifact_kind == "wheel":
        artifact = tmp_path / "op_engine-0.9.0-py3-none-any.whl"
        with zipfile.ZipFile(artifact, mode="w") as archive:
            archive.writestr("flepimop2-op_engine/src/provider.py", "")
    else:
        artifact = tmp_path / "op_engine-0.9.0.tar.gz"
        with tarfile.open(artifact, mode="w:gz") as archive:
            payload = b""
            info = tarfile.TarInfo(
                "op_engine-0.9.0/flepimop2-op_engine/src/provider.py"
            )
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(SystemExit, match="contains provider source"):
        validate_core_artifacts([artifact])
