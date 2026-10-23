"""Validate core release metadata and optional provider-release contracts."""

from __future__ import annotations

import argparse
import os
import pathlib
import tomllib
from typing import Final, Literal

from packaging.version import InvalidVersion, Version

REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[1]
PROVIDER_PYPROJECT: Final[pathlib.Path] = (
    REPO_ROOT / "flepimop2-op_engine" / "pyproject.toml"
)
DEVELOPMENT_STATUS_PREFIX: Final[str] = "Development Status :: "
ReleaseScope = Literal["core", "provider", "all"]


def get_declared_versions() -> dict[str, str]:
    """Read each release version declaration used by this repository.

    Returns:
        A mapping from file path to declared release version.
    """
    core_version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))[
        "project"
    ]["version"]
    provider_version = tomllib.loads(PROVIDER_PYPROJECT.read_text("utf-8"))["project"][
        "version"
    ]

    return {
        "pyproject.toml": str(core_version),
        "flepimop2-op_engine/pyproject.toml": str(provider_version),
    }


def parse_release_version(version: str) -> Version:
    """Parse one canonical final, alpha, beta, or release-candidate version.

    Returns:
        The parsed PEP 440 version.

    Raises:
        SystemExit: If the version is invalid, noncanonical, or unsupported.
    """
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        msg = f"Release version is not valid PEP 440: {version!r}."
        raise SystemExit(msg) from exc

    unsupported = (
        parsed.epoch != 0
        or len(parsed.release) != 3
        or parsed.dev is not None
        or parsed.post is not None
        or parsed.local is not None
        or str(parsed) != version
    )
    if unsupported:
        msg = (
            "Release version must be canonical X.Y.Z or X.Y.ZaN/bN/rcN; "
            f"got {version!r}."
        )
        raise SystemExit(msg)

    return parsed


def validate_project_maturity(scope: ReleaseScope = "core") -> str:
    """Validate one matching maturity classifier for the release unit.

    Returns:
        The normalized maturity label for the selected release scope.

    Raises:
        SystemExit: If either project lacks one classifier or they differ.
    """
    projects = {
        "pyproject.toml": tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text("utf-8")
        )["project"],
    }
    if scope != "core":
        projects["flepimop2-op_engine/pyproject.toml"] = tomllib.loads(
            PROVIDER_PYPROJECT.read_text("utf-8")
        )["project"]
    maturities: dict[str, str] = {}
    for path, project in projects.items():
        matches = [
            str(value)
            for value in project.get("classifiers", [])
            if str(value).startswith(DEVELOPMENT_STATUS_PREFIX)
        ]
        if len(matches) != 1:
            msg = f"{path} must declare exactly one Development Status classifier."
            raise SystemExit(msg)
        maturities[path] = matches[0]

    if len(set(maturities.values())) != 1:
        rendered = ", ".join(f"{path}={value}" for path, value in maturities.items())
        msg = f"Project maturity classifiers do not match: {rendered}"
        raise SystemExit(msg)
    return next(iter(maturities.values())).split(" - ", maxsplit=1)[-1].lower()


def get_release_tag(version: str, scope: ReleaseScope) -> str:
    """Return the GitHub tag for one release scope.

    Returns:
        The exact tag used for the selected distribution scope.
    """
    if scope == "provider":
        return f"flepimop2-op_engine-v{version}"
    return f"v{version}"


def validate_release_tag(version: str, scope: ReleaseScope = "core") -> None:
    """Require an exact version tag when validation runs in a tag context.

    Raises:
        SystemExit: If the current tag does not match the package version.
    """
    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return
    expected = get_release_tag(version, scope)
    actual = os.environ.get("GITHUB_REF_NAME", "")
    if actual != expected:
        msg = f"Release tag must be {expected!r}, got {actual!r}."
        raise SystemExit(msg)


def validate_release_version(scope: ReleaseScope = "core") -> str:
    """Validate version declarations for the selected release scope.

    Returns:
        The selected release version.

    Raises:
        SystemExit: If required declarations differ or are not supported PEP
            440 release versions.
    """
    versions = get_declared_versions()
    core_version = versions["pyproject.toml"]
    provider_version = versions["flepimop2-op_engine/pyproject.toml"]
    if scope != "core" and core_version != provider_version:
        rendered = ", ".join(f"{path}={version}" for path, version in versions.items())
        msg = f"Provider release versions do not match across packages: {rendered}"
        raise SystemExit(msg)

    version = core_version if scope != "provider" else provider_version
    parse_release_version(version)
    validate_project_maturity(scope)
    validate_release_tag(version, scope)

    return version


def validate_provider_dependencies(version: str) -> None:
    """
    Validate provider dependencies that must match release metadata.

    Raises:
        SystemExit: If any provider dependencies are direct references.
        SystemExit: If the provider does not depend on its exact matching
            `op-engine` release version.
    """
    provider_project = tomllib.loads(PROVIDER_PYPROJECT.read_text("utf-8"))["project"]
    dependencies = [str(dependency) for dependency in provider_project["dependencies"]]

    direct_references = [
        dependency
        for dependency in dependencies
        if " @ " in dependency or "://" in dependency or "git+" in dependency
    ]
    if direct_references:
        rendered = ", ".join(direct_references)
        msg = f"Provider package dependencies must be publishable to PyPI: {rendered}"
        raise SystemExit(msg)

    expected_op_engine = f"op-engine=={version}"
    if expected_op_engine not in dependencies:
        rendered = ", ".join(dependencies)
        msg = (
            "Provider package must depend on the exact matching op-engine release "
            f"version {expected_op_engine!r}; got: {rendered}"
        )
        raise SystemExit(msg)


def write_github_outputs(
    version: str,
    output_path: pathlib.Path,
    scope: ReleaseScope = "core",
) -> None:
    """Write workflow outputs for downstream GitHub Actions jobs."""
    parsed = parse_release_version(version)
    prerelease = str(parsed.is_prerelease).lower()
    docs_alias = "pre" if parsed.is_prerelease else "latest"
    release_phase = "prerelease" if parsed.is_prerelease else "final"
    maturity = validate_project_maturity(scope)
    release_tag = get_release_tag(version, scope)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"version={version}\n")
        fh.write(f"release-scope={scope}\n")
        fh.write(f"release-tag={release_tag}\n")
        fh.write(f"docs-version={version}\n")
        fh.write(f"docs-alias={docs_alias}\n")
        fh.write(f"prerelease={prerelease}\n")
        fh.write(f"release-phase={release_phase}\n")
        fh.write(f"maturity={maturity}\n")


def main() -> None:
    """Run release validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-output",
        type=pathlib.Path,
        default=None,
        help="Optional path to a GitHub Actions output file.",
    )
    parser.add_argument(
        "--scope",
        choices=("core", "provider", "all"),
        default="core",
        help="Distribution scope to validate and release.",
    )
    args = parser.parse_args()
    scope: ReleaseScope = args.scope

    version = validate_release_version(scope)
    if scope != "core":
        validate_provider_dependencies(version)
    print(f"Validated {scope} release version: {version}")

    output_path = args.github_output
    if output_path is None:
        raw_output = os.environ.get("GITHUB_OUTPUT")
        if raw_output:
            output_path = pathlib.Path(raw_output)

    if output_path is not None:
        write_github_outputs(version, output_path, scope)


if __name__ == "__main__":
    main()
