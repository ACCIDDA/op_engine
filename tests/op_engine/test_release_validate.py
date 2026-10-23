"""Tests for release-version classification and workflow outputs."""

from __future__ import annotations

import pathlib

import pytest

from scripts import release_validate
from scripts.release_validate import (
    get_release_tag,
    parse_release_version,
    validate_project_maturity,
    validate_release_tag,
    validate_release_version,
    write_github_outputs,
)


@pytest.mark.parametrize(
    "version", ["0.8.0", "1.0.0", "0.9.0a1", "0.9.0b2", "0.9.0rc3"]
)
def test_parse_release_version_accepts_supported_versions(version: str) -> None:
    """Final and explicit pre-release versions are valid release inputs."""
    assert str(parse_release_version(version)) == version


@pytest.mark.parametrize(
    "version",
    ["0.8", "0.8.0.dev1", "0.8.0.post1", "0.8.0+local", "1!0.8.0", "0.8.0-alpha.1"],
)
def test_parse_release_version_rejects_unsupported_versions(version: str) -> None:
    """Development, post, local, epoch, and noncanonical versions are rejected."""
    with pytest.raises(SystemExit, match="Release version"):
        parse_release_version(version)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (
            "0.8.0",
            {
                "docs-alias": "latest",
                "docs-version": "0.8.0",
                "prerelease": "false",
                "release-phase": "final",
            },
        ),
        (
            "0.9.0rc1",
            {
                "docs-alias": "pre",
                "docs-version": "0.9.0rc1",
                "prerelease": "true",
                "release-phase": "prerelease",
            },
        ),
    ],
)
def test_write_github_outputs_separates_release_phase_from_maturity(
    tmp_path: pathlib.Path,
    version: str,
    expected: dict[str, str],
) -> None:
    """A zero major version is not itself evidence of a pre-release."""
    output_path = tmp_path / "github-output.txt"
    write_github_outputs(version, output_path)
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["version"] == version
    assert outputs["release-scope"] == "core"
    assert outputs["release-tag"] == f"v{version}"
    for key, value in expected.items():
        assert outputs[key] == value
    assert outputs["maturity"] == "alpha"


def test_release_tag_must_match_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tag-context validation cannot publish a differently named version."""
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.8.1")
    with pytest.raises(SystemExit, match="Release tag"):
        validate_release_tag("0.8.0")


def test_provider_release_uses_namespaced_tag() -> None:
    """A delayed provider release cannot collide with the core version tag."""
    assert get_release_tag("0.9.0", "provider") == ("flepimop2-op_engine-v0.9.0")


def test_core_release_allows_provider_to_remain_on_prior_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core release validation does not require a provider release."""
    monkeypatch.setattr(
        release_validate,
        "get_declared_versions",
        lambda: {
            "pyproject.toml": "0.9.0",
            "flepimop2-op_engine/pyproject.toml": "0.8.0",
        },
    )
    assert validate_release_version("core") == "0.9.0"
    with pytest.raises(SystemExit, match="do not match"):
        validate_release_version("provider")


def test_public_docs_distinguish_release_lifecycle_from_compatible_pair() -> None:
    """Docs must not equate an exact provider pin with simultaneous releases."""
    required = {
        pathlib.Path("docs/guides/capabilities.md"): (
            "independent release lifecycle",
            "pins the exactly matching `op-engine` version",
        ),
        pathlib.Path("README.md"): (
            "independent release lifecycle",
            "published 0.8.0 package requires `op-engine==0.8.0`",
        ),
        pathlib.Path("flepimop2-op_engine/README.md"): (
            "independent release lifecycles",
            "depends on the exact matching `op-engine` version",
        ),
    }
    optional_changelogs = (
        pathlib.Path("CHANGELOG.md"),
        pathlib.Path("flepimop2-op_engine/CHANGELOG.md"),
    )
    public_docs = [
        *required,
        *(path for path in optional_changelogs if path.is_file()),
        *pathlib.Path("docs").rglob("*.md"),
    ]
    sources = {path: path.read_text("utf-8") for path in public_docs}
    normalized = {path: " ".join(source.split()) for path, source in sources.items()}

    forbidden = (
        "shares its version",
        "share one release version",
        "shared package version",
        "shared release version",
    )
    for path, source in normalized.items():
        for phrase in forbidden:
            assert phrase not in source.casefold(), (path, phrase)

    for path, phrases in required.items():
        for phrase in phrases:
            assert phrase in normalized[path], (path, phrase)


def test_release_unit_has_one_matching_maturity() -> None:
    """Core and provider explicitly share a separate Alpha maturity."""
    assert validate_project_maturity() == "alpha"


def test_pages_writers_are_serialized_without_cancellation() -> None:
    """Development and release deployments cannot race or cancel each other."""
    workflow = pathlib.Path(".github/workflows/gh-pages.yaml").read_text("utf-8")
    assert "group: 'op-engine-pages'" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "Require authoritative main for development docs" in workflow


def test_reusable_release_docs_use_the_unified_inputs_context() -> None:
    """A called workflow must not branch on the caller's event name."""
    workflow = pathlib.Path(".github/workflows/gh-pages.yaml").read_text("utf-8")
    assert 'MODE="${INPUT_MODE:-push}"' in workflow
    assert "INPUT_MODE: '${{ inputs.deploy-mode }}'" in workflow
    assert "INPUT_RELEASE_REF: '${{ inputs.release-ref }}'" in workflow
    assert "github.event.inputs.deploy-mode" not in workflow
    assert "github.event.inputs.release-ref" not in workflow
    resolver = workflow.split("      - name: 'Resolve deployment inputs'", maxsplit=1)[
        1
    ].split("      - name: 'Install just'", maxsplit=1)[0]
    assert "'workflow_call'" not in resolver


def test_release_docs_require_exact_ref_before_checkout() -> None:
    """Release deployments reject an empty or mismatched tag before writing."""
    workflow = pathlib.Path(".github/workflows/gh-pages.yaml").read_text("utf-8")
    resolve_index = workflow.index("Resolve deployment inputs")
    checkout_index = workflow.index("Checkout repository")
    assert resolve_index < checkout_index
    assert "Release mode requires an exact v-prefixed release-ref." in workflow
    assert 'CHECKOUT_REF}" != "${EXPECTED_TAG}' in workflow
    assert "git tag --points-at HEAD" in workflow
    assert "already exists; exact versions are immutable" in workflow
    assert "mike set-default" not in workflow
    assert "scripts/docs_deploy.py" in workflow


def test_release_workflow_publishes_only_core_distribution() -> None:
    """The production workflow cannot accidentally publish the provider."""
    workflow = pathlib.Path(".github/workflows/release.yaml").read_text("utf-8")
    assert "Build core release artifacts" in workflow
    assert "packages-dir: 'dist'" in workflow
    assert "release-scope:" not in workflow
    assert "Publish flepimop2-op_engine" not in workflow
    assert "flepimop2-op-engine-dist" not in workflow
    assert "just build-all-core" in workflow
    assert "--scope core" in workflow
    assert 'release create "${RELEASE_TAG}"' in workflow
    assert "scripts/validate_core_artifacts.py" in workflow
