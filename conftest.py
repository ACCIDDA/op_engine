"""Pytest test configuration."""

from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser


def documentation_setup(documentation_setup: dict[str, Any]) -> None:  # noqa: ARG001
    """Create and chnange to a temporary directory for documentation tests."""
    directory = Path(TemporaryDirectory().name)
    directory.mkdir(parents=True, exist_ok=True)
    chdir(directory)


pytest_collect_file = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    path=str(Path(__file__).parent / "docs"),
    pattern="**/*.md",
    setup=documentation_setup,
).pytest()
