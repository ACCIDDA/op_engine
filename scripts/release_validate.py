"""Validate release version consistency for both distributable packages."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import tomllib
from typing import Final

REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]
SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+$")


def get_declared_versions() -> dict[str, str]:
    """Read each release version declaration used by this repository.

    Returns:
        A mapping from file path to declared release version.
    """
    core_version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))[
        "project"
    ]["version"]
    provider_version = tomllib.loads(
        (REPO_ROOT / "flepimop2-op_engine" / "pyproject.toml").read_text("utf-8")
    )["project"]["version"]

    return {
        "pyproject.toml": str(core_version),
        "flepimop2-op_engine/pyproject.toml": str(provider_version),
    }


def validate_release_version() -> str:
    """Validate that all release version declarations agree.

    Returns:
        The shared release version.

    Raises:
        SystemExit: If any declared versions differ or do not follow `X.Y.Z`.
    """
    versions = get_declared_versions()
    distinct_versions = sorted(set(versions.values()))
    if len(distinct_versions) != 1:
        rendered = ", ".join(f"{path}={version}" for path, version in versions.items())
        msg = f"Release versions do not match across packages: {rendered}"
        raise SystemExit(msg)

    version = distinct_versions[0]
    if SEMVER_PATTERN.fullmatch(version) is None:
        msg = f"Release version must be semantic X.Y.Z, got {version!r}."
        raise SystemExit(msg)

    return version


def write_github_outputs(version: str, output_path: pathlib.Path) -> None:
    """Write workflow outputs for downstream GitHub Actions jobs."""
    docs_version = ".".join(version.split(".")[:2])
    prerelease = str(version.startswith("0.")).lower()
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"version={version}\n")
        fh.write(f"docs-version={docs_version}\n")
        fh.write(f"prerelease={prerelease}\n")


def main() -> None:
    """Run release validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-output",
        type=pathlib.Path,
        default=None,
        help="Optional path to a GitHub Actions output file.",
    )
    args = parser.parse_args()

    version = validate_release_version()
    print(f"Validated release version: {version}")

    output_path = args.github_output
    if output_path is None:
        raw_output = os.environ.get("GITHUB_OUTPUT")
        if raw_output:
            output_path = pathlib.Path(raw_output)

    if output_path is not None:
        write_github_outputs(version, output_path)


if __name__ == "__main__":
    main()
