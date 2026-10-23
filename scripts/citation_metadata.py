# /// script
# requires-python = ">=3.11"
# dependencies = ["ruamel.yaml>=0.18"]
# ///
"""Check released citation attribution and identifiers against package records."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from ruamel.yaml import YAML  # type: ignore[import-not-found, unused-ignore]


def main() -> None:
    """Fail when core citation fields drift from released metadata.

    Raises:
        SystemExit: If attribution, version, date, or release URL disagree.
    """
    root = Path(__file__).resolve().parent.parent
    citation = YAML(typ="safe").load((root / "CITATION.cff").read_text("utf-8"))
    project = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"]
    authors = [
        {
            "name": f"{person['given-names']} {person['family-names']}",
            "email": person["email"],
        }
        for person in citation["authors"]
    ]
    if citation["license"] != "MIT":
        msg = "Citation license must match the core MIT grant"
        raise SystemExit(msg)
    version = str(citation["version"])
    if project["version"] != version or project["authors"] != authors:
        msg = "Citation attribution/version differs from op_engine package metadata"
        raise SystemExit(msg)
    changelog = (root / "CHANGELOG.md").read_text("utf-8")
    released = re.search(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if released is None or str(citation["date-released"]) != released.group(1):
        msg = (
            "Citation date-released must match the core version's dated changelog entry"
        )
        raise SystemExit(msg)
    repository = project["urls"]["Repository"]
    expected = f"{repository}/releases/tag/v{version}"
    if (
        citation["repository-code"] != repository
        or citation["repository-artifact"] != expected
    ):
        msg = "Citation repository and release URL must identify the core release"
        raise SystemExit(msg)
    print(f"Validated core citation attribution and release metadata: {version}")


if __name__ == "__main__":
    main()
