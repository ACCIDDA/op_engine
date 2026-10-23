"""Deploy development or immutable release documentation with Mike."""

from __future__ import annotations

import argparse
import subprocess  # noqa: S404
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def mike_deploy_command(mode: str, version: str, alias: str) -> list[str]:
    """Build the Mike command for one documented deployment mode.

    Returns:
        A complete command with no duplicate version/alias argument.

    Raises:
        ValueError: If the mode or release metadata is invalid.
    """
    if mode == "push":
        return ["mike", "deploy", "--push", "dev"]
    if mode != "release":
        msg = f"Unsupported documentation deployment mode: {mode!r}."
        raise ValueError(msg)
    if not version or not alias or version == alias:
        msg = "Release documentation requires distinct version and alias values."
        raise ValueError(msg)
    return ["mike", "deploy", "--push", "--update-aliases", version, alias]


def deploy_docs(
    mode: str,
    version: str,
    alias: str,
    *,
    runner: Callable[..., object] = subprocess.run,
) -> None:
    """Execute one checked Mike deployment command."""
    runner(mike_deploy_command(mode, version, alias), check=True, text=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse workflow inputs and deploy documentation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("push", "release"), required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--alias", default="")
    args = parser.parse_args(argv)
    deploy_docs(args.mode, args.version, args.alias)


if __name__ == "__main__":
    main()
