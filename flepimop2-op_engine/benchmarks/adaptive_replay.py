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

"""Benchmark adaptive discovery and differentiated replay strategies.

Run from the provider directory with the JAX development dependency installed::

    uv run python benchmarks/adaptive_replay.py --output-counts 17 65 129

The benchmark emits CSV. ``reverse_temp_bytes`` comes from the compiled JAX
value-and-gradient executable's memory analysis, so it measures backend
temporary storage rather than Python allocations. Host peaks are reported
separately for compilation and execution.
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
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
from flepimop2.system.abc import SystemABC
from flepimop2.typing import StateChangeEnum
from typing_extensions import override

from flepimop2.engine.op_engine import (
    AdaptiveReplayMode,
    AdaptiveSchedule,
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    ReplayCheckpoint,
    SolverMethod,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flepimop2.typing import Array, IdentifierString, SystemProtocol


class _Blockable(Protocol):
    """Synchronous completion surface exposed by JAX arrays."""

    def block_until_ready(self) -> object:
        """Wait for pending device work and return the value."""
        ...


class _MemoryAnalysis(Protocol):
    """Compiled-memory fields used by this benchmark."""

    temp_size_in_bytes: int


class _Compiled(Protocol):
    """Small compiled-executable surface used by the timing loop."""

    def __call__(self, *args: object) -> object:
        """Execute with array arguments."""
        ...

    def memory_analysis(self) -> _MemoryAnalysis:
        """Return backend memory estimates."""
        ...


class _RateSystem(SystemABC, module="adaptive_replay_benchmark"):
    """Scalar exponential-growth system with one dynamic parameter."""

    state_change: StateChangeEnum = StateChangeEnum.FLOW

    @override
    def _bind_impl(
        self,
        params: dict[IdentifierString, Any] | None = None,
    ) -> SystemProtocol:
        bound = params or {}
        rate = bound["rate"]

        def step(_time: object, state: Array) -> Array:
            xp = cast("Any", state.__array_namespace__())
            return cast("Array", xp.multiply(state, rate))

        return cast("SystemProtocol", step)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One accepted-mesh/replay-policy benchmark row."""

    output_count: int
    accepted_steps: int
    mode: str
    discovery_seconds: float
    trace_equations: int
    trace_characters: int
    compile_seconds: float
    runtime_seconds: float
    compile_host_peak_bytes: int
    runtime_host_peak_bytes: int
    reverse_temp_bytes: int


def _host_peak(action: Callable[[], Any]) -> tuple[Any, int]:
    """Execute one action and measure peak Python-tracked allocations.

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


def _block_until_ready(value: object) -> None:
    """Synchronize every array leaf in a transformed result."""
    if isinstance(value, tuple | list):
        for item in value:
            _block_until_ready(item)
        return
    cast("_Blockable", value).block_until_ready()


def _config(
    mode: AdaptiveReplayMode,
    checkpoint: ReplayCheckpoint = ReplayCheckpoint.NONE,
) -> OpEngineEngineConfig:
    """Build the common numerical configuration for one replay policy.

    Returns:
        Provider configuration with identical discovery controls.
    """
    return OpEngineEngineConfig(
        method=SolverMethod.HEUN,
        adaptive=True,
        adaptive_replay=mode,
        replay_checkpoint=checkpoint,
        schedule_tag="adaptive-replay-benchmark-v1",
        rtol=1e-3,
        atol=1e-6,
    )


def _discover(
    system: _RateSystem,
    times: np.ndarray,
    model_state: ModelStateSpecification,
) -> tuple[AdaptiveSchedule, float]:
    """Discover one accepted mesh eagerly with NumPy arrays.

    Returns:
        Provider schedule and discovery wall time.
    """
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=_config(AdaptiveReplayMode.AUTO),
    )
    start = time.perf_counter()
    schedule = engine.run_adaptive(
        system,
        times,
        {"x": ParameterValue(np.asarray(1.0), ResolvedShape())},
        {"rate": ParameterValue(np.asarray(-0.2), ResolvedShape())},
        model_state=model_state,
    ).schedule
    return schedule, time.perf_counter() - start


def _measure(  # noqa: PLR0913, PLR0914
    *,
    system: _RateSystem,
    times: np.ndarray,
    model_state: ModelStateSpecification,
    schedule: AdaptiveSchedule,
    discovery_seconds: float,
    mode: AdaptiveReplayMode,
    checkpoint: ReplayCheckpoint,
    repeats: int,
) -> Measurement:
    """Compile and time one differentiated replay policy.

    Returns:
        Trace, compile, runtime, and memory measurements.
    """
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=_config(mode, checkpoint),
    )

    def final_state(rate: Array, initial: Array) -> Array:
        trajectory = engine.run(
            system,
            times,
            {"x": ParameterValue(initial, ResolvedShape())},
            {"rate": ParameterValue(rate, ResolvedShape())},
            model_state=model_state,
            adaptive_schedule=schedule,
        )
        return cast("Array", trajectory[-1, 1])

    transformed = jax.value_and_grad(final_state, argnums=(0, 1))
    rate = jnp.asarray(-0.2, dtype=jnp.float32)
    initial = jnp.asarray(1.0, dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(transformed)(rate, initial)
    jitted = jax.jit(transformed)

    compile_start = time.perf_counter()
    compiled_obj, compile_host_peak = _host_peak(
        lambda: jitted.lower(rate, initial).compile()
    )
    compile_seconds = time.perf_counter() - compile_start
    compiled = cast("_Compiled", compiled_obj)

    runtimes: list[float] = []
    runtime_host_peaks: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result, host_peak = _host_peak(lambda: compiled(rate, initial))
        _block_until_ready(result)
        runtimes.append(time.perf_counter() - start)
        runtime_host_peaks.append(host_peak)

    memory = compiled.memory_analysis()
    return Measurement(
        output_count=len(times),
        accepted_steps=sum(map(len, schedule.step_schedule.step_sizes)),
        mode=("checkpointed" if checkpoint is ReplayCheckpoint.STEP else mode.value),
        discovery_seconds=discovery_seconds,
        trace_equations=len(jaxpr.jaxpr.eqns),
        trace_characters=len(str(jaxpr)),
        compile_seconds=compile_seconds,
        runtime_seconds=statistics.median(runtimes),
        compile_host_peak_bytes=compile_host_peak,
        runtime_host_peak_bytes=max(runtime_host_peaks),
        reverse_temp_bytes=int(memory.temp_size_in_bytes),
    )


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-counts",
        nargs="+",
        type=int,
        default=[17, 65, 129],
    )
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Run the replay benchmark and write machine-readable CSV.

    Raises:
        ValueError: If an output or repeat count is invalid.
    """
    options = _parse_args(args)
    if options.repeats < 1 or any(count < 2 for count in options.output_counts):
        msg = "--repeats must be positive and every output count at least two."
        raise ValueError(msg)

    writer = csv.DictWriter(sys.stdout, fieldnames=tuple(Measurement.__annotations__))
    writer.writeheader()
    policies = (
        (AdaptiveReplayMode.UNROLLED, ReplayCheckpoint.NONE),
        (AdaptiveReplayMode.COMPACT, ReplayCheckpoint.NONE),
        (AdaptiveReplayMode.COMPACT, ReplayCheckpoint.STEP),
    )
    system = _RateSystem()
    model_state = ModelStateSpecification(parameter_names=("x",))
    for output_count in options.output_counts:
        times = np.linspace(0.0, 1.0, output_count, dtype=np.float64)
        schedule, discovery_seconds = _discover(system, times, model_state)
        for mode, checkpoint in policies:
            result = _measure(
                system=system,
                times=times,
                model_state=model_state,
                schedule=schedule,
                discovery_seconds=discovery_seconds,
                mode=mode,
                checkpoint=checkpoint,
                repeats=options.repeats,
            )
            writer.writerow({
                name: getattr(result, name) for name in Measurement.__annotations__
            })
            sys.stdout.flush()


if __name__ == "__main__":
    main()
