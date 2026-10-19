"""Keep generated vignette tables and vector figures tied to executable sources.

This documentation helper is shared by the illustrated compiler, solver,
inference and trade-study examples; it adds no plotting dependency to a core.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class IllustrationData(NamedTuple):
    """Generated Markdown and the complete numerical inputs to its figure.

    Attributes:
        summary: Markdown table computed by the executable example.
        values: JSON-compatible plotted values.
    """

    summary: str
    values: Mapping[str, object]


def build_artifacts(
    output: Path,
    content: IllustrationData,
    sources: Sequence[Path],
    render: Callable[[Path], None],
    *,
    check: bool,
) -> None:
    """Generate a vignette or verify its source, table and exported-file identity.

    Args:
        output: Directory containing the committed figure and table.
        content: Generated table and complete plotted data.
        sources: Generator and fixture source files.
        render: Optional-dependency renderer, called only during generation.
        check: Verify existing files without importing a plotting library.

    Raises:
        ValueError: A committed table, source identity or export has changed.
    """
    digest = hashlib.sha256()
    for path in [Path(__file__), *sources]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    identity = digest.hexdigest()
    serialized = (
        json.dumps(content.values, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    data_identity = hashlib.sha256(serialized.encode()).hexdigest()
    table = output / "summary.txt"
    manifest = output / "manifest.json"
    names = ("figure.svg", "figure.pdf", "data.json")
    if check:
        recorded = json.loads(manifest.read_text())
        if (
            recorded["source_sha256"] != identity
            or recorded["data_sha256"] != data_identity
            or table.read_text() != content.summary
        ):
            msg = f"Regenerate the illustrated example in {output}"
            raise ValueError(msg)
        for name in names:
            actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
            if actual != recorded["exports"][name]:
                msg = f"Export changed without regeneration: {output / name}"
                raise ValueError(msg)
        return
    output.mkdir(parents=True, exist_ok=True)
    render(output)
    table.write_text(content.summary)
    (output / "data.json").write_text(serialized)
    manifest.write_text(
        json.dumps(
            {
                "source_sha256": identity,
                "data_sha256": data_identity,
                "sources": [path.name for path in sources],
                "rendered_with": {
                    name: version(name) for name in ("numpy", "matplotlib")
                },
                "exports": {
                    name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in names
                },
            },
            indent=2,
        )
        + "\n"
    )
