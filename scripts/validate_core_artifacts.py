"""Reject core distribution artifacts that contain the held provider tree."""

from __future__ import annotations

import argparse
import pathlib
import tarfile
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PROVIDER_DIRECTORY = "flepimop2-op_engine"


def artifact_members(path: pathlib.Path) -> list[str]:
    """List normalized member names from a wheel or gzipped source archive.

    Returns:
        Archive member names using POSIX separators.

    Raises:
        ValueError: If the artifact format is unsupported.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            return archive.getnames()
    msg = f"Unsupported core artifact format: {path.name}."
    raise ValueError(msg)


def validate_core_artifacts(paths: Iterable[pathlib.Path]) -> None:
    """Require every artifact to exclude the FlepiMoP2 provider subtree.

    Raises:
        SystemExit: If no artifacts are supplied or a provider member is found.
    """
    artifacts = list(paths)
    if not artifacts:
        msg = "No core artifacts were supplied for inspection."
        raise SystemExit(msg)
    for artifact in artifacts:
        provider_members = [
            member
            for member in artifact_members(artifact)
            if PROVIDER_DIRECTORY in pathlib.PurePosixPath(member).parts
        ]
        if provider_members:
            preview = ", ".join(provider_members[:3])
            msg = f"Core artifact {artifact.name} contains provider source: {preview}"
            raise SystemExit(msg)
        print(f"Validated core artifact contents: {artifact.name}")


def main(argv: Sequence[str] | None = None) -> None:
    """Inspect core wheel and source-distribution paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=pathlib.Path)
    args = parser.parse_args(argv)
    validate_core_artifacts(args.artifacts)


if __name__ == "__main__":
    main()
