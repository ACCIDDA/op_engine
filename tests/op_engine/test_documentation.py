"""The real README, getting-started snippets and bank example stay executable."""

from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("index", [0, 1])
def test_readme_numpy_examples(index: int) -> None:
    """Both complete core examples preserve finite trajectories and output shapes."""
    blocks = re.findall(
        r"```python\n(.*?)\n```", (ROOT / "README.md").read_text(), re.DOTALL
    )
    namespace: dict[str, Any] = {}
    exec(blocks[index], namespace)  # noqa: S102 - trusted repository documentation
    history = np.asarray(namespace["core"].state_array)
    assert np.isfinite(history).all()
    if index == 0:
        np.testing.assert_allclose(history.sum(axis=1), 1.0, atol=1e-12)


def test_getting_started_has_expected_decay() -> None:
    """The actual first guide block agrees with the analytic decay solution."""
    blocks = re.findall(
        r"```python\n(.*?)\n```",
        (ROOT / "docs/guides/getting-started.md").read_text(),
        re.DOTALL,
    )
    namespace: dict[str, Any] = {}
    exec(blocks[0], namespace)  # noqa: S102 - trusted repository documentation
    history = namespace["solution"]
    np.testing.assert_allclose(history[-1, 0, 0], np.exp(-0.5), rtol=1e-4)


def test_replicate_example_preserves_counts_and_bank_addresses() -> None:
    """The standalone example runs with different bank orders and singleton shards."""
    namespace = runpy.run_path(str(ROOT / "examples/replicate_ssa.py"))
    namespace["main"]()


def test_observation_guide_has_expected_measured_total() -> None:
    """The actual guide callback preserves its stated physical quantity."""
    blocks = re.findall(
        r"```python\n(.*?)\n```",
        (ROOT / "docs/guides/observations.md").read_text(),
        re.DOTALL,
    )
    namespace: dict[str, Any] = {}
    exec(blocks[0], namespace)  # noqa: S102 -- trusted repository documentation
    np.testing.assert_allclose(
        namespace["observed"].series["total"].values, [2.0, 10.2, 18.9]
    )
