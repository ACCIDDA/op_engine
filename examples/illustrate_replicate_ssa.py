"""Plot output-time SSA observations, analytic means and accepted firing counts.

Regenerate with Python 3.12 and Matplotlib: uv run --with matplotlib==3.11.2
python -m examples.illustrate_replicate_ssa. Use --check without plotting dependencies.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from examples.illustration_artifacts import IllustrationData, build_artifacts
from examples.replicate_ssa import simulate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _render(
    output: Path, history: NDArray[np.float64], counts: NDArray[np.int64]
) -> None:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update({"svg.hashsalt": "op-engine-replicates", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    times = np.linspace(0.0, 2.0, history.shape[0])
    reference_times = np.linspace(0.0, 2.0, 201)
    integrated_multiplier = np.minimum(reference_times, 1.0) + 2 * np.maximum(
        reference_times - 1.0, 0.0
    )
    for i, (identity, initial, rate, color) in enumerate([
        (10, 12, 0.4, "#24678e"),
        (20, 17, 0.7, "#0d6b6e"),
    ]):
        axes[0].plot(
            times,
            history[:, 1, i],
            "o",
            color=color,
            label=f"ID {identity}: observations",
        )
        axes[0].plot(
            reference_times,
            initial * (1 - np.exp(-rate * integrated_multiplier)),
            color=color,
            alpha=0.7,
            label=f"ID {identity}: analytic mean",
        )
        axes[1].bar(
            (times[:-1] + times[1:]) / 2 + (i - 0.5) * 0.15,
            counts[:, 0, i],
            width=0.15,
            color=color,
            label=f"ID {identity}",
        )
    for axis in axes:
        axis.axvline(1.0, color="#536774", linestyle=":", label="Forcing change")
        axis.set_xlabel("Time")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set(
        ylabel="Transferred population B", title="Output samples and mean references"
    )
    axes[1].set(ylabel="Accepted firings per output interval", title="Recorder ledger")
    figure.savefig(output / "figure.svg", metadata={"Date": None})
    figure.savefig(
        output / "figure.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(figure)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/replicate-ssa",
    )
    args = parser.parse_args(argv)
    history, counts = simulate()
    lines = [
        (
            "| Replicate ID | Initial A | Base rate | Final A | Final B | "
            "Accepted firings |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (identity, initial, rate) in enumerate([(10, 12, 0.4), (20, 17, 0.7)]):
        lines.append(
            f"| {identity} | {initial} | {rate:g} | {history[-1, 0, i]:g} | "
            f"{history[-1, 1, i]:g} | {counts[:, 0, i].sum()} |"
        )
    build_artifacts(
        args.output,
        IllustrationData(
            "\n".join(lines) + "\n",
            {"history": history.tolist(), "counts": counts.tolist()},
        ),
        [Path(__file__), Path(__file__).with_name("replicate_ssa.py")],
        lambda output: _render(output, history, counts),
        check=args.check,
    )


if __name__ == "__main__":
    _main()
