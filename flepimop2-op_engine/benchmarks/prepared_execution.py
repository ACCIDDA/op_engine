# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Compare ordinary and prepared public-provider staging boundaries.

The benchmark uses the same analytic exponential system for both paths. It
reports preparation/cache-hit cost, system binding count, JAX trace/lower/
compile phases, synchronized execution, and final-state accuracy separately.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
from flepimop2.system.abc import SystemABC
from flepimop2.typing import StateChangeEnum
from op_engine import array_namespace
from pydantic import PrivateAttr
from typing_extensions import override

try:
    from .solver_matrix import _environment
except ImportError:  # pragma: no cover - direct script execution
    from solver_matrix import _environment  # noqa: PLC2701

from flepimop2.engine.op_engine import (
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    SolverMethod,
    StateLayout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flepimop2.typing import Array, IdentifierString, SystemProtocol


@dataclass(frozen=True, slots=True)
class _BlockAxis:
    """Synthetic factorization metadata consumed by the provider."""

    name: str
    size: int
    state_axis_pos: dict[str, int]
    param_axis_pos: dict[str, int | None]


class _Compiled(Protocol):
    """Compiled JAX callable used by synchronized timing."""

    def __call__(self, *args: object) -> object:
        """Execute the compiled program."""


class _RateSystem(SystemABC, module="prepared_execution_benchmark"):
    """Dynamic exponential system instrumenting parameter binding."""

    state_change: StateChangeEnum = StateChangeEnum.FLOW
    _bind_calls: int = PrivateAttr(default=0)

    @property
    def bind_calls(self) -> int:
        """Return the number of system bindings since the last reset."""
        return self._bind_calls

    def reset_bind_calls(self) -> None:
        """Reset system-binding instrumentation."""
        self._bind_calls = 0

    @override
    def _bind_impl(
        self,
        params: dict[IdentifierString, Any] | None = None,
    ) -> SystemProtocol:
        self._bind_calls += 1
        bound = params or {}

        def step(_time: object, state: Array, **dynamic: object) -> Array:
            rate = dynamic.get("rate", bound.get("rate"))
            if rate is None:
                msg = "rate is required"
                raise ValueError(msg)
            xp = array_namespace(state)
            return cast("Array", xp.multiply(state, rate))

        return cast("SystemProtocol", step)


def _configure_structured_options(system: _RateSystem, batch_size: int) -> None:
    """Publish one analytic PyTree and block-PyTree execution contract."""

    def step(
        _time: object,
        state: dict[str, Array],
        **params: object,
    ) -> dict[str, Array]:
        value = state["x"]
        xp = array_namespace(value)
        return {"x": cast("Array", xp.multiply(value, params["rate"]))}

    system.options = {
        "template_shapes": {"x": (batch_size,)},
        "pytree_stepper_fn": step,
        "block_template_shapes": {"x": ()},
        "block_pytree_stepper_fn": step,
        "block_axes": (
            _BlockAxis(
                name="batch",
                size=batch_size,
                state_axis_pos={"x": 0},
                param_axis_pos={"rate": 0},
            ),
        ),
    }


@dataclass(frozen=True, slots=True)
class Measurement:
    """One provider lifecycle measurement."""

    execution_path: str
    construction_seconds: float
    cache_hit_seconds: float | None
    trace_seconds: float
    lower_seconds: float
    compile_seconds: float
    first_execution_seconds: float
    warm_execution_seconds: float
    trace_equations: int
    stablehlo_characters: int
    system_bind_calls: int
    final_state_abs_error: float


def _elapsed(action: Callable[[], object]) -> tuple[object, float]:
    """Return an action result and elapsed wall time."""
    start = time.perf_counter()
    result = action()
    return result, time.perf_counter() - start


def _execute(action: Callable[[], object]) -> tuple[object, float]:
    """Return an action result and synchronized execution time."""
    start = time.perf_counter()
    result = action()
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def _measure(
    *,
    path: str,
    solve: Callable[[Array, Array], Array],
    rate: Array,
    initial: Array,
    construction_seconds: float,
    cache_hit_seconds: float | None,
    system: _RateSystem,
    repeats: int,
    exact: np.ndarray,
) -> Measurement:
    """Measure a stable callable from trace through warm execution.

    Returns:
        Phase-separated lifecycle result for one path.
    """
    system.reset_bind_calls()
    trace_samples: list[float] = []
    jaxpr: Any | None = None
    for _ in range(repeats):
        # A fresh wrapper deliberately exercises repeated caller staging.
        def staged(active_rate: Array, active_initial: Array) -> Array:
            return solve(active_rate, active_initial)

        trace_start = time.perf_counter()
        jaxpr = jax.make_jaxpr(staged)(rate, initial)
        trace_samples.append(time.perf_counter() - trace_start)
    if jaxpr is None:  # pragma: no cover - repeats is validated before entry
        msg = "At least one trace sample is required."
        raise RuntimeError(msg)
    lowered, lower_seconds = _elapsed(lambda: jax.jit(solve).lower(rate, initial))
    stablehlo_characters = len(str(lowered.compiler_ir(dialect="stablehlo")))
    compiled_value, compile_seconds = _elapsed(lowered.compile)
    compiled = cast("_Compiled", compiled_value)
    first, first_seconds = _execute(lambda: compiled(rate, initial))
    warm_samples = [
        _execute(lambda: compiled(rate, initial))[1] for _ in range(repeats)
    ]
    return Measurement(
        execution_path=path,
        construction_seconds=construction_seconds,
        cache_hit_seconds=cache_hit_seconds,
        trace_seconds=statistics.median(trace_samples),
        lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,
        first_execution_seconds=first_seconds,
        warm_execution_seconds=statistics.median(warm_samples),
        trace_equations=len(jaxpr.jaxpr.eqns),
        stablehlo_characters=stablehlo_characters,
        system_bind_calls=system.bind_calls,
        final_state_abs_error=float(np.max(np.abs(np.asarray(first) - exact))),
    )


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=30.0)
    parser.add_argument("--output-count", type=int, default=121)
    parser.add_argument("--fixed-max-step", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--layout",
        choices=tuple(layout.value for layout in StateLayout),
        default=StateLayout.FLAT.value,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Run matched ordinary/prepared measurements and emit JSON."""
    options = _parse_args(args)
    if (
        options.horizon <= 0.0
        or options.output_count < 2
        or options.fixed_max_step <= 0.0
        or options.repeats < 1
        or options.batch_size < 1
    ):
        msg = (
            "horizon, step, repeats, and batch-size must be positive; "
            "output-count must exceed one"
        )
        raise ValueError(msg)

    layout = StateLayout(options.layout)
    config = OpEngineEngineConfig(
        method=SolverMethod.DOPRI5,
        fixed_max_step=options.fixed_max_step,
        state_layout=layout,
        block_axis="batch" if layout is StateLayout.BLOCK else None,
    )
    engine_value, engine_seconds = _elapsed(
        lambda: OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=config,
        )
    )
    engine = cast("OpEngineFlepimop2Engine", engine_value)
    system = _RateSystem()
    _configure_structured_options(system, options.batch_size)
    times = np.linspace(0.0, options.horizon, options.output_count)
    model_state = ModelStateSpecification(parameter_names=("x",))
    batch_shape = ResolvedShape(
        axis_names=("batch",),
        sizes=(options.batch_size,),
    )
    rate = jnp.linspace(-0.08, -0.02, options.batch_size)
    initial = jnp.linspace(1.0, 1.5, options.batch_size)

    def ordinary(active_rate: Array, active_initial: Array) -> Array:
        trajectory = engine.run(
            system,
            times,
            {"x": ParameterValue(active_initial, batch_shape)},
            {"rate": ParameterValue(active_rate, batch_shape)},
            model_state=model_state,
        )
        return cast("Array", trajectory[-1, 1:])

    prepared_value, prepare_seconds = _elapsed(
        lambda: engine.prepare(
            system,
            times,
            {"x": ParameterValue(initial, batch_shape)},
            {"rate": ParameterValue(rate, batch_shape)},
            model_state=model_state,
        )
    )
    prepared = prepared_value
    alternate_initial = initial + 1.0
    alternate_rate = rate - 0.01
    cache_samples = [
        _elapsed(
            lambda: engine.prepare(
                system,
                times,
                {"x": ParameterValue(alternate_initial, batch_shape)},
                {"rate": ParameterValue(alternate_rate, batch_shape)},
                model_state=model_state,
            )
        )[1]
        for _ in range(options.repeats)
    ]

    def prepared_solve(active_rate: Array, active_initial: Array) -> Array:
        trajectory = prepared(
            {"x": active_initial},
            {"rate": active_rate},
        )
        return cast("Array", trajectory[-1, 1:])

    exact = np.asarray(initial) * np.exp(np.asarray(rate) * options.horizon)
    results = [
        _measure(
            path="ordinary",
            solve=ordinary,
            rate=rate,
            initial=initial,
            construction_seconds=engine_seconds,
            cache_hit_seconds=None,
            system=system,
            repeats=options.repeats,
            exact=exact,
        ),
        _measure(
            path="prepared",
            solve=prepared_solve,
            rate=rate,
            initial=initial,
            construction_seconds=prepare_seconds,
            cache_hit_seconds=statistics.median(cache_samples),
            system=system,
            repeats=options.repeats,
            exact=exact,
        ),
    ]
    root = Path(__file__).resolve().parents[2]
    document = {
        "schema_version": 1,
        "settings": {
            "method": config.method.value,
            "horizon": options.horizon,
            "output_count": options.output_count,
            "fixed_max_step": options.fixed_max_step,
            "repeats": options.repeats,
            "layout": layout.value,
            "batch_size": options.batch_size,
        },
        "environment": _environment(root),
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if options.output is None:
        sys.stdout.write(rendered)
    else:
        options.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
