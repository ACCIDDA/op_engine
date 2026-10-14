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
    """Fail when shared software citation fields drift from released metadata.

    Raises:
        SystemExit: If attribution, version, date, or release URL disagree.
    """
    root = Path(__file__).resolve().parent.parent
    citation = YAML(typ="safe").load((root / "CITATION.cff").read_text("utf-8"))
    projects = [
        tomllib.loads((root / directory / "pyproject.toml").read_text("utf-8"))[
            "project"
        ]
        for directory in [".", "flepimop2-op_engine"]
    ]
    authors = [
        {
            "name": f"{person['given-names']} {person['family-names']}",
            "email": person["email"],
        }
        for person in citation["authors"]
    ]
    if citation["license"] != ["MIT", "GPL-3.0-or-later"]:
        msg = (
            "Citation licenses must match the core MIT and "
            "provider GPLv3-or-later grants"
        )
        raise SystemExit(msg)
    shared = str(citation["version"])
    for project in projects:
        if project["version"] != shared or project["authors"] != authors:
            msg = (
                "Citation attribution/version differs from "
                f"{project['name']} package metadata"
            )
            raise SystemExit(msg)
    changelog = (root / "CHANGELOG.md").read_text("utf-8")
    released = re.search(
        rf"^## \[{re.escape(shared)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.MULTILINE,
    )
    if released is None or str(citation["date-released"]) != released.group(1):
        msg = (
            "Citation date-released must match the shared version's "
            "dated changelog entry"
        )
        raise SystemExit(msg)
    repository = projects[0]["urls"]["Repository"]
    expected = f"{repository}/releases/tag/v{shared}"
    if (
        citation["repository-code"] != repository
        or citation["repository-artifact"] != expected
    ):
        msg = (
            "Citation repository and release URL must identify "
            "the shared software release"
        )
        raise SystemExit(msg)
    print(f"Validated citation attribution and release metadata: {shared}")


if __name__ == "__main__":
    main()
