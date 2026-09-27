# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Benchmark solver cost, accuracy, and differentiation independently.

The benchmark uses the analytic vector problem x' = rate * x. It emits one
JSON document so every timing remains attached to its numerical settings,
accuracy, revision, and execution environment. JAX trace, lower, compile,
first-execution, and warm-execution times are deliberately separate.

Run this file from the provider directory. Diffrax is an optional reference;
when it is requested but unavailable the JSON document records that fact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
from flepimop2.system.abc import SystemABC
from flepimop2.typing import StateChangeEnum
from op_engine import array_namespace
from pydantic import PrivateAttr
from scipy.integrate import solve_ivp
from typing_extensions import override

from flepimop2.engine.op_engine import (
    AdaptiveSchedule,
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    SolverMethod,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flepimop2.typing import Array, IdentifierString, SystemProtocol


SCHEMA_VERSION = 1
_MODEL_STATE = ModelStateSpecification(parameter_names=("x",))


def _parameter(value: Array) -> ParameterValue:
    """Attach a synthetic batch axis to one benchmark array."""
    size = int(value.shape[0])
    return ParameterValue(
        value,
        ResolvedShape(axis_names=("batch",), sizes=(size,)),
    )


class _Compiled(Protocol):
    """JAX compiled-executable surface used by the benchmark."""

    def __call__(self, *args: object) -> object:
        """Execute with array arguments."""

    def memory_analysis(self) -> object:
        """Return backend memory estimates."""


class _CountingRateSystem(SystemABC, module="solver_matrix_benchmark"):
    """Elementwise exponential system with eager RHS-call instrumentation."""

    state_change: StateChangeEnum = StateChangeEnum.FLOW
    _rhs_calls: int = PrivateAttr(default=0)

    @property
    def rhs_calls(self) -> int:
        """Return calls made through bound steppers since the last reset."""
        return self._rhs_calls

    def reset_rhs_calls(self) -> None:
        """Reset eager RHS-call instrumentation."""
        self._rhs_calls = 0

    @override
    def _bind_impl(
        self,
        params: dict[IdentifierString, Any] | None = None,
    ) -> SystemProtocol:
        bound = params or {}
        rate = bound["rate"]

        def step(_time: object, state: Array) -> Array:
            self._rhs_calls += 1
            xp = array_namespace(state)
            return cast("Array", xp.multiply(state, rate))

        return cast("SystemProtocol", step)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Numerical problem and execution-policy settings."""

    horizon: float
    batch_size: int
    output_count: int
    policy: str
    method: str
    fixed_max_step: float
    rtol: float
    atol: float


@dataclass(frozen=True, slots=True)
class Measurement:
    """One benchmark row with phase-separated timing and quality fields."""

    implementation: str
    backend: str
    policy: str
    method: str
    horizon: float
    batch_size: int
    output_count: int
    fixed_max_step: float | None
    rtol: float | None
    atol: float | None
    controller: str | None
    adjoint: str | None
    provider_construction_seconds: float | None
    discovery_seconds: float | None
    trace_seconds: float | None
    lower_seconds: float | None
    compile_seconds: float | None
    first_execution_seconds: float
    warm_execution_seconds: float
    grad_trace_seconds: float | None
    grad_lower_seconds: float | None
    grad_compile_seconds: float | None
    grad_first_execution_seconds: float | None
    grad_warm_execution_seconds: float | None
    trace_equations: int | None
    grad_trace_equations: int | None
    stablehlo_characters: int | None
    grad_stablehlo_characters: int | None
    forward_temp_bytes: int | None
    grad_temp_bytes: int | None
    accepted_steps: int | None
    rejected_steps: int | None
    discovery_rhs_evaluations: int | None
    execution_rhs_evaluations: int | None
    max_abs_error: float
    relative_l2_error: float
    objective_abs_error: float
    gradient_max_abs_error: float | None
    notes: str | None = None


def _elapsed(action: Callable[[], Any]) -> tuple[Any, float]:
    """Run an action and return its result and elapsed wall time."""
    start = time.perf_counter()
    result = action()
    return result, time.perf_counter() - start


def _timed_execution(action: Callable[[], object]) -> tuple[object, float]:
    """Run and synchronize an execution inside one wall-clock interval."""
    start = time.perf_counter()
    result = action()
    _block_until_ready(result)
    return result, time.perf_counter() - start


def _block_until_ready(value: object) -> None:
    """Synchronize arrays in a transformed result without importing JAX."""
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _block_until_ready(item)
        return
    blocker = getattr(value, "block_until_ready", None)
    if blocker is not None:
        blocker()


def _median_execution(
    action: Callable[[], object],
    *,
    repeats: int,
) -> float:
    """Return median synchronized execution time."""
    samples: list[float] = []
    for _ in range(repeats):
        _result, elapsed = _timed_execution(action)
        samples.append(elapsed)
    return statistics.median(samples)


def _inputs(case: BenchmarkCase, xp: Any) -> tuple[object, object, np.ndarray]:  # noqa: ANN401
    """Create rates, initial values, and output times for one backend."""
    rates = np.linspace(-0.35, 0.15, case.batch_size, dtype=np.float64)
    initial = np.linspace(0.75, 1.25, case.batch_size, dtype=np.float64)
    times = np.linspace(0.0, case.horizon, case.output_count, dtype=np.float64)
    return xp.asarray(rates), xp.asarray(initial), times


def _exact_trajectory(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact trajectory and final-objective gradients."""
    rates = np.linspace(-0.35, 0.15, case.batch_size, dtype=np.float64)
    initial = np.linspace(0.75, 1.25, case.batch_size, dtype=np.float64)
    times = np.linspace(0.0, case.horizon, case.output_count, dtype=np.float64)
    exact = initial[None, :] * np.exp(times[:, None] * rates[None, :])
    grad_rate = case.horizon * exact[-1]
    grad_initial = np.exp(case.horizon * rates)
    return exact, grad_rate, grad_initial


def _quality(
    case: BenchmarkCase,
    trajectory: object,
) -> tuple[float, float, float]:
    """Return max error, relative L2 error, and objective error."""
    observed = np.asarray(trajectory)[:, 1:]
    exact, _grad_rate, _grad_initial = _exact_trajectory(case)
    difference = observed - exact
    max_abs = float(np.max(np.abs(difference)))
    relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(exact))
    objective_error = abs(float(np.sum(observed[-1]) - np.sum(exact[-1])))
    return max_abs, relative_l2, objective_error


def _engine(
    case: BenchmarkCase,
) -> OpEngineFlepimop2Engine:
    """Construct the public provider for one benchmark case."""
    return OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=SolverMethod(case.method),
            adaptive=case.policy == "adaptive",
            fixed_max_step=(case.fixed_max_step if case.policy == "fixed" else None),
            rtol=case.rtol,
            atol=case.atol,
            schedule_tag="solver-matrix-v1",
        ),
    )


def _run_provider(  # noqa: PLR0913, PLR0917
    engine: OpEngineFlepimop2Engine,
    system: _CountingRateSystem,
    times: np.ndarray,
    rate: object,
    initial: object,
    schedule: AdaptiveSchedule | None = None,
) -> object:
    """Execute the provider using its public run boundary."""
    return engine.run(
        system,
        times,
        {"x": _parameter(initial)},
        {"rate": _parameter(rate)},
        model_state=_MODEL_STATE,
        adaptive_schedule=schedule,
    )


def _discover_schedule(
    case: BenchmarkCase,
    xp: object,
) -> tuple[AdaptiveSchedule, float, int, int | None, int | None]:
    """Discover an adaptive mesh in ``xp`` and return available diagnostics."""
    system = _CountingRateSystem()
    engine = _engine(case)
    rate, initial, times = _inputs(case, xp)
    system.reset_rhs_calls()
    result, elapsed = _elapsed(
        lambda: engine.run_adaptive(
            system,
            times,
            {"x": _parameter(initial)},
            {"rate": _parameter(rate)},
            model_state=_MODEL_STATE,
        )
    )
    schedule = result.schedule
    accepted = sum(map(len, schedule.step_schedule.step_sizes))
    if xp is np:
        calls: int | None = system.rhs_calls
        rejected: int | None = _adaptive_rejections(case.method, accepted, calls)
    else:
        calls = None
        rejected = None
    return schedule, elapsed, accepted, rejected, calls


def _adaptive_rejections(method: str, accepted: int, rhs_calls: int) -> int:
    """Recover rejected attempts from eager explicit-kernel call counts."""
    if method == SolverMethod.EULER.value:
        rejected = rhs_calls - 2 * accepted
    elif method == SolverMethod.HEUN.value:
        rejected = rhs_calls // 2 - accepted
    elif method == SolverMethod.RK4.value:
        rejected = (rhs_calls - 11 * accepted) // 10
    elif method == SolverMethod.DOPRI5.value:
        rejected = (rhs_calls - 1) // 6 - accepted
    else:
        msg = f"Unsupported adaptive benchmark method: {method}"
        raise ValueError(msg)
    if rejected < 0:
        msg = (
            "RHS-call instrumentation was inconsistent with the explicit "
            f"{method} kernel: accepted={accepted}, calls={rhs_calls}."
        )
        raise RuntimeError(msg)
    return rejected


def _execution_rhs(method: str, steps: int) -> int:
    """Return fixed or frozen-replay RHS calls for accepted steps."""
    if method == SolverMethod.EULER.value:
        return steps
    if method == SolverMethod.HEUN.value:
        return 2 * steps
    if method == SolverMethod.RK4.value:
        return 4 * steps
    if method == SolverMethod.DOPRI5.value:
        return 0 if steps == 0 else 1 + 6 * steps
    msg = f"Unsupported benchmark method: {method}"
    raise ValueError(msg)


def _fixed_step_count(case: BenchmarkCase) -> int:
    """Return internal steps in the fixed provider plan."""
    interval = case.horizon / (case.output_count - 1)
    return (case.output_count - 1) * math.ceil(interval / case.fixed_max_step)


def _measure_numpy(case: BenchmarkCase, *, repeats: int) -> Measurement:
    """Measure eager NumPy provider execution."""
    system = _CountingRateSystem()
    engine_value, construction = _elapsed(lambda: _engine(case))
    engine = cast("OpEngineFlepimop2Engine", engine_value)
    rate, initial, times = _inputs(case, np)
    schedule: AdaptiveSchedule | None = None
    discovery_seconds: float | None = None
    accepted: int
    rejected: int | None
    discovery_calls: int | None
    if case.policy == "adaptive":
        schedule, discovery_seconds, accepted, rejected, discovery_calls = (
            _discover_schedule(case, np)
        )
    else:
        accepted = _fixed_step_count(case)
        rejected = None
        discovery_calls = None

    action = lambda: _run_provider(  # noqa: E731
        engine, system, times, rate, initial, schedule
    )
    first, first_seconds = _elapsed(action)
    warm_seconds = _median_execution(action, repeats=repeats)
    max_abs, relative_l2, objective_error = _quality(case, first)
    return Measurement(
        implementation="op_engine",
        backend="numpy",
        policy=case.policy,
        method=case.method,
        horizon=case.horizon,
        batch_size=case.batch_size,
        output_count=case.output_count,
        fixed_max_step=(case.fixed_max_step if case.policy == "fixed" else None),
        rtol=(case.rtol if case.policy == "adaptive" else None),
        atol=(case.atol if case.policy == "adaptive" else None),
        controller=("op_engine PI controller" if case.policy == "adaptive" else None),
        adjoint=None,
        provider_construction_seconds=construction,
        discovery_seconds=discovery_seconds,
        trace_seconds=None,
        lower_seconds=None,
        compile_seconds=None,
        first_execution_seconds=first_seconds,
        warm_execution_seconds=warm_seconds,
        grad_trace_seconds=None,
        grad_lower_seconds=None,
        grad_compile_seconds=None,
        grad_first_execution_seconds=None,
        grad_warm_execution_seconds=None,
        trace_equations=None,
        grad_trace_equations=None,
        stablehlo_characters=None,
        grad_stablehlo_characters=None,
        forward_temp_bytes=None,
        grad_temp_bytes=None,
        accepted_steps=accepted,
        rejected_steps=rejected,
        discovery_rhs_evaluations=discovery_calls,
        execution_rhs_evaluations=_execution_rhs(case.method, accepted),
        max_abs_error=max_abs,
        relative_l2_error=relative_l2,
        objective_abs_error=objective_error,
        gradient_max_abs_error=None,
    )


def _temp_bytes(compiled: _Compiled) -> int | None:
    """Read temporary-byte estimates when the JAX backend exposes them."""
    analysis = compiled.memory_analysis()
    value = getattr(analysis, "temp_size_in_bytes", None)
    return int(value) if value is not None else None


def _measure_jax(case: BenchmarkCase, *, repeats: int) -> Measurement:
    """Measure phase-separated JAX forward and value-and-gradient calls."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    system = _CountingRateSystem()
    engine_value, construction = _elapsed(lambda: _engine(case))
    engine = cast("OpEngineFlepimop2Engine", engine_value)
    rate, initial, times = _inputs(case, jnp)
    schedule: AdaptiveSchedule | None = None
    discovery_seconds: float | None = None
    accepted: int
    rejected: int | None
    discovery_calls: int | None
    if case.policy == "adaptive":
        schedule, discovery_seconds, accepted, rejected, discovery_calls = (
            _discover_schedule(case, jnp)
        )
    else:
        accepted = _fixed_step_count(case)
        rejected = None
        discovery_calls = None

    def simulate(active_rate: Array, active_initial: Array) -> Array:
        return cast(
            "Array",
            _run_provider(
                engine,
                system,
                times,
                active_rate,
                active_initial,
                schedule,
            ),
        )

    jaxpr, trace_seconds = _elapsed(lambda: jax.make_jaxpr(simulate)(rate, initial))
    jitted = jax.jit(simulate)
    lowered, lower_seconds = _elapsed(lambda: jitted.lower(rate, initial))
    stablehlo_characters = len(str(lowered.compiler_ir(dialect="stablehlo")))
    compiled_value, compile_seconds = _elapsed(lowered.compile)
    compiled = cast("_Compiled", compiled_value)
    first, first_seconds = _timed_execution(lambda: compiled(rate, initial))
    warm_seconds = _median_execution(
        lambda: compiled(rate, initial),
        repeats=repeats,
    )
    max_abs, relative_l2, _objective_error = _quality(case, first)

    def objective(active_rate: Array, active_initial: Array) -> Array:
        trajectory = simulate(active_rate, active_initial)
        return cast("Array", jnp.sum(trajectory[-1, 1:]))

    transformed = jax.value_and_grad(objective, argnums=(0, 1))
    grad_jaxpr, grad_trace_seconds = _elapsed(
        lambda: jax.make_jaxpr(transformed)(rate, initial)
    )
    grad_lowered, grad_lower_seconds = _elapsed(
        lambda: jax.jit(transformed).lower(rate, initial)
    )
    grad_stablehlo_characters = len(str(grad_lowered.compiler_ir(dialect="stablehlo")))
    grad_compiled_value, grad_compile_seconds = _elapsed(grad_lowered.compile)
    grad_compiled = cast("_Compiled", grad_compiled_value)
    grad_first, grad_first_seconds = _timed_execution(
        lambda: grad_compiled(rate, initial)
    )
    grad_warm_seconds = _median_execution(
        lambda: grad_compiled(rate, initial),
        repeats=repeats,
    )
    _value, gradients = grad_first
    exact, exact_rate, exact_initial = _exact_trajectory(case)
    gradient_error = max(
        float(np.max(np.abs(np.asarray(gradients[0]) - exact_rate))),
        float(np.max(np.abs(np.asarray(gradients[1]) - exact_initial))),
    )

    return Measurement(
        implementation="op_engine",
        backend=f"jax-{jax.default_backend()}",
        policy=case.policy,
        method=case.method,
        horizon=case.horizon,
        batch_size=case.batch_size,
        output_count=case.output_count,
        fixed_max_step=(case.fixed_max_step if case.policy == "fixed" else None),
        rtol=(case.rtol if case.policy == "adaptive" else None),
        atol=(case.atol if case.policy == "adaptive" else None),
        controller=("op_engine PI controller" if case.policy == "adaptive" else None),
        adjoint="jax reverse-mode through fixed plan or frozen adaptive replay",
        provider_construction_seconds=construction,
        discovery_seconds=discovery_seconds,
        trace_seconds=trace_seconds,
        lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,
        first_execution_seconds=first_seconds,
        warm_execution_seconds=warm_seconds,
        grad_trace_seconds=grad_trace_seconds,
        grad_lower_seconds=grad_lower_seconds,
        grad_compile_seconds=grad_compile_seconds,
        grad_first_execution_seconds=grad_first_seconds,
        grad_warm_execution_seconds=grad_warm_seconds,
        trace_equations=len(jaxpr.jaxpr.eqns),
        grad_trace_equations=len(grad_jaxpr.jaxpr.eqns),
        stablehlo_characters=stablehlo_characters,
        grad_stablehlo_characters=grad_stablehlo_characters,
        forward_temp_bytes=_temp_bytes(compiled),
        grad_temp_bytes=_temp_bytes(grad_compiled),
        accepted_steps=accepted,
        rejected_steps=rejected,
        discovery_rhs_evaluations=discovery_calls,
        execution_rhs_evaluations=_execution_rhs(case.method, accepted),
        max_abs_error=max_abs,
        relative_l2_error=relative_l2,
        objective_abs_error=abs(
            float(np.asarray(grad_first[0])) - float(np.sum(exact[-1]))
        ),
        gradient_max_abs_error=gradient_error,
    )


def _measure_scipy(case: BenchmarkCase, *, repeats: int) -> Measurement:
    """Measure SciPy RK45 against the same analytic problem."""
    rate, initial, times = _inputs(case, np)

    def solve() -> object:
        return solve_ivp(
            lambda _time, state: np.asarray(rate) * state,
            (0.0, case.horizon),
            np.asarray(initial),
            method="RK45",
            t_eval=times,
            rtol=case.rtol,
            atol=case.atol,
        )

    result, first_seconds = _elapsed(solve)
    if not result.success:
        raise RuntimeError(result.message)
    warm_seconds = _median_execution(solve, repeats=repeats)
    trajectory = np.concatenate((times[:, None], result.y.T), axis=1)
    max_abs, relative_l2, objective_error = _quality(case, trajectory)
    return Measurement(
        implementation="scipy",
        backend="numpy",
        policy="adaptive",
        method="RK45 (Dormand-Prince 5(4))",
        horizon=case.horizon,
        batch_size=case.batch_size,
        output_count=case.output_count,
        fixed_max_step=None,
        rtol=case.rtol,
        atol=case.atol,
        controller="SciPy RK45 controller",
        adjoint=None,
        provider_construction_seconds=None,
        discovery_seconds=None,
        trace_seconds=None,
        lower_seconds=None,
        compile_seconds=None,
        first_execution_seconds=first_seconds,
        warm_execution_seconds=warm_seconds,
        grad_trace_seconds=None,
        grad_lower_seconds=None,
        grad_compile_seconds=None,
        grad_first_execution_seconds=None,
        grad_warm_execution_seconds=None,
        trace_equations=None,
        grad_trace_equations=None,
        stablehlo_characters=None,
        grad_stablehlo_characters=None,
        forward_temp_bytes=None,
        grad_temp_bytes=None,
        accepted_steps=None,
        rejected_steps=None,
        discovery_rhs_evaluations=None,
        execution_rhs_evaluations=int(result.nfev),
        max_abs_error=max_abs,
        relative_l2_error=relative_l2,
        objective_abs_error=objective_error,
        gradient_max_abs_error=None,
        notes=(
            "SciPy does not expose accepted/rejected step counts in solve_ivp results."
        ),
    )


def _measure_diffrax(case: BenchmarkCase, *, repeats: int) -> Measurement:
    """Measure Diffrax Tsit5 with an explicit controller and adjoint policy."""
    import diffrax  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    rate, initial, times = _inputs(case, jnp)
    dt0 = min(case.fixed_max_step, case.horizon / (case.output_count - 1))

    def construct() -> tuple[object, object, object, object]:
        term = diffrax.ODETerm(lambda _time, state, args: args * state)
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=case.rtol, atol=case.atol)
        adjoint = diffrax.RecursiveCheckpointAdjoint()
        return term, solver, controller, adjoint

    components, construction = _elapsed(construct)
    term, solver, controller, adjoint = components
    saveat = diffrax.SaveAt(ts=jnp.asarray(times))

    def solve_payload(
        active_rate: Array,
        active_initial: Array,
    ) -> tuple[Array, Array, Array]:
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=case.horizon,
            dt0=dt0,
            y0=active_initial,
            args=active_rate,
            saveat=saveat,
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=1_000_000,
        )
        return (
            cast("Array", solution.ys),
            cast("Array", solution.stats["num_accepted_steps"]),
            cast("Array", solution.stats["num_rejected_steps"]),
        )

    def solve_array(active_rate: Array, active_initial: Array) -> Array:
        return solve_payload(active_rate, active_initial)[0]

    jaxpr, trace_seconds = _elapsed(
        lambda: jax.make_jaxpr(solve_payload)(rate, initial)
    )
    lowered, lower_seconds = _elapsed(
        lambda: jax.jit(solve_payload).lower(rate, initial)
    )
    stablehlo_characters = len(str(lowered.compiler_ir(dialect="stablehlo")))
    compiled_value, compile_seconds = _elapsed(lowered.compile)
    compiled = cast("_Compiled", compiled_value)
    first, first_seconds = _timed_execution(lambda: compiled(rate, initial))
    warm_seconds = _median_execution(
        lambda: compiled(rate, initial),
        repeats=repeats,
    )
    trajectory = np.concatenate((times[:, None], np.asarray(first[0])), axis=1)
    max_abs, relative_l2, _objective_error = _quality(case, trajectory)

    def objective(active_rate: Array, active_initial: Array) -> Array:
        return cast("Array", jnp.sum(solve_array(active_rate, active_initial)[-1]))

    transformed = jax.value_and_grad(objective, argnums=(0, 1))
    grad_jaxpr, grad_trace_seconds = _elapsed(
        lambda: jax.make_jaxpr(transformed)(rate, initial)
    )
    grad_lowered, grad_lower_seconds = _elapsed(
        lambda: jax.jit(transformed).lower(rate, initial)
    )
    grad_stablehlo_characters = len(str(grad_lowered.compiler_ir(dialect="stablehlo")))
    grad_compiled_value, grad_compile_seconds = _elapsed(grad_lowered.compile)
    grad_compiled = cast("_Compiled", grad_compiled_value)
    grad_first, grad_first_seconds = _timed_execution(
        lambda: grad_compiled(rate, initial)
    )
    grad_warm_seconds = _median_execution(
        lambda: grad_compiled(rate, initial),
        repeats=repeats,
    )
    _value, gradients = grad_first
    exact, exact_rate, exact_initial = _exact_trajectory(case)
    gradient_error = max(
        float(np.max(np.abs(np.asarray(gradients[0]) - exact_rate))),
        float(np.max(np.abs(np.asarray(gradients[1]) - exact_initial))),
    )

    return Measurement(
        implementation="diffrax",
        backend=f"jax-{jax.default_backend()}",
        policy="adaptive",
        method="Tsit5",
        horizon=case.horizon,
        batch_size=case.batch_size,
        output_count=case.output_count,
        fixed_max_step=None,
        rtol=case.rtol,
        atol=case.atol,
        controller="diffrax.PIDController",
        adjoint="diffrax.RecursiveCheckpointAdjoint",
        provider_construction_seconds=construction,
        discovery_seconds=None,
        trace_seconds=trace_seconds,
        lower_seconds=lower_seconds,
        compile_seconds=compile_seconds,
        first_execution_seconds=first_seconds,
        warm_execution_seconds=warm_seconds,
        grad_trace_seconds=grad_trace_seconds,
        grad_lower_seconds=grad_lower_seconds,
        grad_compile_seconds=grad_compile_seconds,
        grad_first_execution_seconds=grad_first_seconds,
        grad_warm_execution_seconds=grad_warm_seconds,
        trace_equations=len(jaxpr.jaxpr.eqns),
        grad_trace_equations=len(grad_jaxpr.jaxpr.eqns),
        stablehlo_characters=stablehlo_characters,
        grad_stablehlo_characters=grad_stablehlo_characters,
        forward_temp_bytes=_temp_bytes(compiled),
        grad_temp_bytes=_temp_bytes(grad_compiled),
        accepted_steps=int(np.asarray(first[1])),
        rejected_steps=int(np.asarray(first[2])),
        discovery_rhs_evaluations=None,
        execution_rhs_evaluations=None,
        max_abs_error=max_abs,
        relative_l2_error=relative_l2,
        objective_abs_error=abs(
            float(np.asarray(grad_first[0])) - float(np.sum(exact[-1]))
        ),
        gradient_max_abs_error=gradient_error,
        notes=f"Initial step dt0={dt0}; SaveAt uses the common output grid.",
    )


def _version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git_metadata(root: Path) -> dict[str, object]:
    """Return revision and dirty-state metadata without failing exports."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() or None,
        "dirty": bool(status.stdout.strip()),
    }


def _environment(root: Path) -> dict[str, object]:
    """Build reproducibility metadata for one benchmark document."""
    packages = {
        name: _version(name)
        for name in (
            "array-api-compat",
            "diffrax",
            "flepimop2",
            "flepimop2-op_engine",
            "jax",
            "jaxlib",
            "numpy",
            "op_engine",
            "scipy",
        )
    }
    devices: list[str] = []
    jax_enable_x64: bool | None = None
    try:
        import jax  # noqa: PLC0415

        devices = [str(device) for device in jax.devices()]
        jax_enable_x64 = bool(jax.config.x64_enabled)
    except ImportError:
        pass
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "jax_devices": devices,
        "jax_enable_x64": jax_enable_x64,
        "git": _git_metadata(root),
    }


def _parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=("numpy", "jax"), default=["numpy", "jax"]
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("fixed", "adaptive"),
        default=["fixed", "adaptive"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(method.value for method in SolverMethod if method.is_explicit),
        default=["rk4", "dopri5"],
    )
    parser.add_argument("--horizons", nargs="+", type=float, default=[1.0, 30.0])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--output-count", type=int, default=33)
    parser.add_argument("--fixed-max-step", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--references", nargs="*", choices=("scipy", "diffrax"), default=["scipy"]
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(args)


def _validate_options(options: argparse.Namespace) -> None:
    """Validate CLI ranges before any expensive compilation."""
    if options.repeats < 1:
        msg = "--repeats must be positive."
        raise ValueError(msg)
    if options.output_count < 2:
        msg = "--output-count must be at least two."
        raise ValueError(msg)
    if options.fixed_max_step <= 0.0:
        msg = "--fixed-max-step must be positive."
        raise ValueError(msg)
    if options.rtol < 0.0 or options.atol < 0.0:
        msg = "--rtol and --atol must be non-negative."
        raise ValueError(msg)
    if any(value <= 0.0 for value in options.horizons):
        msg = "Every --horizons value must be positive."
        raise ValueError(msg)
    if any(value < 1 for value in options.batch_sizes):
        msg = "Every --batch-sizes value must be positive."
        raise ValueError(msg)


def _cases(options: argparse.Namespace) -> list[BenchmarkCase]:
    """Expand CLI settings into deterministic benchmark cases."""
    return [
        BenchmarkCase(
            horizon=horizon,
            batch_size=batch_size,
            output_count=options.output_count,
            policy=policy,
            method=method,
            fixed_max_step=options.fixed_max_step,
            rtol=options.rtol,
            atol=options.atol,
        )
        for horizon in options.horizons
        for batch_size in options.batch_sizes
        for policy in options.policies
        for method in options.methods
    ]


def main(args: Sequence[str] | None = None) -> None:
    """Run the requested matrix and write one machine-readable JSON document."""
    options = _parse_args(args)
    _validate_options(options)
    root = Path(__file__).resolve().parents[2]
    measurements: list[Measurement] = []
    skipped: list[dict[str, str]] = []
    cases = _cases(options)
    for case in cases:
        for backend in options.backends:
            measure = _measure_numpy if backend == "numpy" else _measure_jax
            measurements.append(measure(case, repeats=options.repeats))

    reference_cases = {
        (case.horizon, case.batch_size): case
        for case in cases
        if case.policy == "adaptive" and case.method == "dopri5"
    }
    if "scipy" in options.references:
        measurements.extend(
            _measure_scipy(case, repeats=options.repeats)
            for case in reference_cases.values()
        )
    if "diffrax" in options.references:
        try:
            measurements.extend(
                _measure_diffrax(case, repeats=options.repeats)
                for case in reference_cases.values()
            )
        except ImportError:
            skipped.append({
                "implementation": "diffrax",
                "reason": "Diffrax is not installed in this benchmark environment.",
            })

    document = {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment(root),
        "results": [asdict(result) for result in measurements],
        "skipped": skipped,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if options.output is None:
        sys.stdout.write(rendered)
    else:
        options.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
