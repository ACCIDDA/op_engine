# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Benchmark flat, PyTree, and block-PyTree provider scaling.

Run from the provider directory after installing the ``op-system`` and ``jax``
extras::

    uv run python benchmarks/structured_blocks.py --blocks 8 32 128

The CSV output distinguishes Python host allocation peaks from device-reported
peak bytes. Some JAX devices do not expose memory statistics; their device
column is left empty rather than substituting host memory.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from flepimop2.axis import Axis, AxisCollection, ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import (
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    SolverMethod,
    StateLayout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flepimop2.typing import Array


@dataclass(frozen=True, slots=True)
class Measurement:
    """One layout/block-count benchmark result."""

    blocks: int
    state_cells: int
    layout: StateLayout
    compile_seconds: float
    runtime_seconds: float
    compile_host_peak_bytes: int
    runtime_host_peak_bytes: int
    device_peak_bytes: int | None


class _Blockable(Protocol):
    """Synchronous completion surface exposed by JAX arrays."""

    def block_until_ready(self) -> object:
        """Wait for pending device work and return the value."""
        ...


def _system(blocks: int) -> tuple[OpSystemSystem, AxisCollection]:
    """Build a two-template model with independent location blocks.

    Returns:
        Compiled op_system provider and its resolved axes.
    """
    locations = tuple(f"loc_{index}" for index in range(blocks))
    axes = AxisCollection({
        "age": Axis(
            name="age",
            kind="categorical",
            size=2,
            labels=("young", "old"),
        ),
        "loc": Axis(
            name="loc",
            kind="categorical",
            size=blocks,
            labels=locations,
        ),
    })
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [
                {"name": "age", "coords": ["young", "old"]},
                {"name": "loc", "coords": list(locations)},
            ],
            "state": ["X[age, loc]", "Y[age, loc]"],
            "equations": {
                "X[age, loc]": "rate[loc] * X[age, loc]",
                "Y[age, loc]": "-decay * Y[age, loc]",
            },
            "initial_state": {
                "X[age, loc]": {"shaped": "x0", "axes": ["age", "loc"]},
                "Y[age, loc]": {"shaped": "y0", "axes": ["age", "loc"]},
            },
            "factorize_axes": ["loc"],
        }
    )
    return system, axes


def _device_peak_bytes() -> int | None:
    """Return a device-reported peak when the active backend supplies one."""
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use")
    return int(peak) if peak is not None else None


def _host_peak(action: Callable[[], Any]) -> tuple[Any, int]:
    """Execute one action and report peak Python-tracked host allocations.

    Returns:
        Action result and peak traced host bytes.
    """
    tracemalloc.start()
    try:
        result = action()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _measure(  # noqa: PLR0914
    blocks: int,
    layout: StateLayout,
    *,
    repeats: int,
    internal_step: float,
) -> Measurement:
    """Compile and time one public provider execution closure.

    Returns:
        Timing and peak-memory measurements for the selected layout.
    """
    system, axes = _system(blocks)
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=SolverMethod.RK4,
            fixed_max_step=internal_step,
            state_layout=layout,
            block_axis="loc" if layout is StateLayout.BLOCK else None,
        ),
    )
    times = np.linspace(0.0, 10.0, 41, dtype=np.float64)
    state_shape = axes.resolve_shape(("age", "loc"))
    rate_shape = axes.resolve_shape(("loc",))
    model_state = system.model_state(axes)
    decay = ParameterValue(jnp.asarray(0.15), ResolvedShape())

    def simulate(rate: Array, x0: Array, y0: Array) -> Array:
        return engine.run(
            system,
            times,
            {},
            {
                "rate": ParameterValue(rate, rate_shape),
                "x0": ParameterValue(x0, state_shape),
                "y0": ParameterValue(y0, state_shape),
                "decay": decay,
            },
            model_state=model_state,
        )

    rate = jnp.linspace(-0.05, -0.25, blocks)
    x0 = jnp.ones((2, blocks))
    y0 = 2.0 * jnp.ones((2, blocks))
    jitted = jax.jit(simulate)

    compile_start = time.perf_counter()
    compiled_obj, compile_host_peak = _host_peak(
        lambda: jitted.lower(rate, x0, y0).compile()
    )
    compile_seconds = time.perf_counter() - compile_start
    compiled = compiled_obj

    runtimes: list[float] = []
    runtime_host_peaks: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result, host_peak = _host_peak(lambda: compiled(rate, x0, y0))
        cast("_Blockable", result).block_until_ready()
        runtimes.append(time.perf_counter() - start)
        runtime_host_peaks.append(host_peak)

    return Measurement(
        blocks=blocks,
        state_cells=4 * blocks,
        layout=layout,
        compile_seconds=compile_seconds,
        runtime_seconds=statistics.median(runtimes),
        compile_host_peak_bytes=compile_host_peak,
        runtime_host_peak_bytes=max(runtime_host_peaks),
        device_peak_bytes=_device_peak_bytes(),
    )


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--internal-step", type=float, default=0.05)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Run the benchmark and write machine-readable CSV to stdout.

    Raises:
        ValueError: If a block count, repeat count, or step size is invalid.
    """
    options = _parse_args(args)
    if options.repeats < 1 or options.internal_step <= 0.0:
        msg = "--repeats and --internal-step must be positive."
        raise ValueError(msg)
    if any(blocks < 1 for blocks in options.blocks):
        msg = "Every --blocks value must be positive."
        raise ValueError(msg)

    writer = csv.DictWriter(sys.stdout, fieldnames=tuple(Measurement.__annotations__))
    writer.writeheader()
    for blocks in options.blocks:
        for layout in StateLayout:
            result = _measure(
                blocks,
                layout,
                repeats=options.repeats,
                internal_step=options.internal_step,
            )
            row = {
                **{name: getattr(result, name) for name in Measurement.__annotations__},
                "layout": result.layout.value,
                "device_peak_bytes": result.device_peak_bytes or "",
            }
            writer.writerow(row)


if __name__ == "__main__":
    main()
