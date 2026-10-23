"""Independent piecewise solutions and transform contracts for known jumps."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from op_engine import CoreSolver, ModelCore, array_namespace
from op_engine._forcing import _ForcingSchedule  # noqa: PLC2701
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    RunConfig,
    fixed_step_sizes,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from op_engine import Array, Scalar


def _module(backend: str) -> Any:  # noqa: ANN401
    return (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )


def _precision(backend: str) -> Any:  # noqa: ANN401
    if backend == "jax":
        jax = pytest.importorskip("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        return enable()
    return nullcontext()


def _host(value: Any) -> np.ndarray:  # noqa: ANN401
    return np.asarray(value.detach().numpy() if hasattr(value, "detach") else value)


def _solver(initial: Array, times: tuple[float, ...]) -> CoreSolver:
    core = ModelCore(
        1,
        1,
        np.asarray(times),
        options=ModelCoreOptions(dtype=np.dtype(str(initial.dtype).split(".")[-1])),
    )
    core.set_initial_state(initial)
    return CoreSolver(core)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_known_jumps_preserve_native_piecewise_solution(
    backend: str, dtype: str, method: str, *, adaptive: bool
) -> None:
    """Internal off-grid jumps, outgoing stages and native precision agree."""
    with _precision(backend):
        module = _module(backend)
        initial = module.asarray([[1.0]], dtype=getattr(module, dtype))
        xp = array_namespace(initial)

        def rhs(time: Scalar, state: Array) -> Array:
            t = xp.asarray(time, dtype=state.dtype)
            rate = xp.where(t < 0.3, 1.0, xp.where(t < 0.8, 0.0, 2.0))
            return cast("Array", xp.ones_like(state) * rate)

        config = RunConfig(
            method=method,
            adaptive=adaptive,
            forcing_breakpoints=(-1.0, 0.0, 0.3, 0.8, 1.5, 2.0),
            adaptive_cfg=AdaptiveConfig(rtol=1e-3, atol=1e-5, max_steps=100),
        )
        solver = _solver(initial, (0.0, 1.5))
        solver.run(rhs, config=config)
        result = solver.core.get_current_state()
        assert result.dtype == initial.dtype
        assert array_namespace(result) is xp
        np.testing.assert_allclose(
            _host(result), [[2.7]], rtol=2e-6 if dtype == "float32" else 1e-13
        )
        assert solver.core.state_array is not None
        assert solver.core.state_array.shape == (2, 1, 1)
        if adaptive:
            schedule = solver.last_adaptive_schedule
            assert schedule is not None
            assert schedule.provenance is not None
            assert schedule.provenance.forcing_breakpoints == config.forcing_breakpoints
            replay = _solver(initial, (0.0, 1.5))
            replay.replay_adaptive_schedule(rhs, schedule, config=config)
            np.testing.assert_allclose(
                _host(replay.core.get_current_state()),
                [[2.7]],
                rtol=2e-6 if dtype == "float32" else 1e-13,
            )


@pytest.mark.parametrize(("method", "minimum_ratio"), [("rk4", 15.0), ("dopri5", 30.0)])
@pytest.mark.parametrize("outputs", [(0.0, 2.0), (0.0, 1.0, 2.0)])
def test_decay_recovers_high_order_at_hold_boundary(
    method: str, minimum_ratio: float, outputs: tuple[float, ...]
) -> None:
    """A known exponential exposes the original first-order endpoint error."""
    errors = []
    for step in (0.25, 0.125, 0.0625):
        solver = _solver(np.ones((1, 1)), outputs)
        solver.run(
            lambda t, y: np.where(t < 1.0, -y, 0.0 * y),
            config=RunConfig(
                method=method, fixed_max_step=step, forcing_breakpoints=(1.0,)
            ),
        )
        errors.append(
            abs(float(solver.core.get_current_state().item()) - math.exp(-1.0))
        )
    assert errors[-2] / errors[-1] > minimum_ratio
    assert errors[-1] < 5e-8


@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_adaptive_decay_and_replay_match_piecewise_exponential(method: str) -> None:
    """Both regimes contribute to the integral, including after an output jump."""
    times = (0.0, 0.6, 1.0, 1.4, 2.0)
    config = RunConfig(
        method=method,
        adaptive=True,
        forcing_breakpoints=(1.0,),
        adaptive_cfg=AdaptiveConfig(rtol=1e-8, atol=1e-10),
    )

    def rhs(t: Scalar, y: Array) -> Array:
        return np.where(t < 1.0, -y, -2.0 * y)

    solver = _solver(np.ones((1, 1)), times)
    solver.run(rhs, config=config)
    expected = np.exp(
        -np.minimum(times, 1.0) - 2.0 * np.maximum(np.asarray(times) - 1.0, 0.0)
    )
    np.testing.assert_allclose(
        np.asarray(solver.core.state_array)[:, 0, 0], expected, rtol=2e-7
    )
    schedule = solver.last_adaptive_schedule
    assert schedule is not None
    replay = _solver(np.ones((1, 1)), times)
    replay.replay_adaptive_schedule(rhs, schedule, config=config)
    np.testing.assert_allclose(
        replay.core.state_array, solver.core.state_array, rtol=1e-13
    )
    with pytest.raises(ValueError, match="forcing_breakpoints mismatch"):
        _solver(np.ones((1, 1)), times).replay_adaptive_schedule(
            rhs, schedule, config=replace(config, forcing_breakpoints=(0.8,))
        )


def test_crossing_replay_fails_before_mutation() -> None:
    """A legacy mesh must land at jumps, rather than silently changing its map."""
    solver = _solver(np.ones((1, 1)), (0.0, 2.0))
    schedule = AdaptiveStepSchedule((0.0, 2.0), ((0.5, 1.5),))
    with pytest.raises(ValueError, match="crosses a forcing breakpoint"):
        solver.replay_adaptive_schedule(
            lambda _t, y: y,
            schedule,
            config=RunConfig(method="rk4", adaptive=True, forcing_breakpoints=(1.0,)),
        )
    assert solver.core.current_time == 0.0
    np.testing.assert_array_equal(solver.core.get_current_state(), [[1.0]])


@pytest.mark.parametrize(
    "points", [(1.0, 1.0), (2.0, 1.0), (math.nan,), (True,), ("1",)]
)
def test_invalid_forcing_declarations_fail(points: tuple[Any, ...]) -> None:
    """ODE declarations use the same strict schedule contract as stochastic ones."""
    with pytest.raises(ValueError, match="forcing_breakpoints"):
        RunConfig(forcing_breakpoints=points)


@pytest.mark.parametrize(
    "method", ["imex-euler", "imex-ark3", "bdf2", "ros2", "sdirk2"]
)
def test_other_methods_reject_declared_jumps(method: str) -> None:
    """Unsupported stage/history policies cannot silently discard metadata."""
    with pytest.raises(ValueError, match="require an explicit method"):
        RunConfig(method=method, forcing_breakpoints=(1.0,))


def test_fixed_partition_includes_jumps_without_changing_outputs() -> None:
    """Output and forcing coordinates independently constrain internal steps."""
    assert fixed_step_sizes(0.0, 2.0, 0.75, forcing_breakpoints=(0.5, 1.0)) == (
        0.5,
        0.5,
        0.75,
        0.25,
    )


def test_jax_forcing_lookup_remains_loop_free_inside_scan_gradient() -> None:
    """A forcing lookup does not nest a search loop inside a replay scan."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    schedule = _ForcingSchedule((0.7, 1.0))
    reference = jnp.asarray(0.0)

    times = jnp.asarray([0.2, 0.8, 1.2])

    def objective(scale: Any) -> Any:  # noqa: ANN401
        def body(total: Any, time: Any) -> tuple[Any, None]:  # noqa: ANN401
            incoming = schedule.rhs_for_step(
                lambda stage_time, _state: scale * stage_time,
                time,
                reference,
            )(time, reference)
            return total + incoming, None

        return jax.lax.scan(body, reference, times)[0]

    scale = jnp.asarray(1.0)
    gradient = jax.jit(jax.grad(objective))(scale)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(times).sum())

    jaxpr = str(jax.make_jaxpr(jax.grad(objective))(scale))
    assert "scan[" in jaxpr
    assert "searchsorted" not in jaxpr.lower()
    assert "while[" not in jaxpr


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_forced_replay_native_parameter_gradient(backend: str, method: str) -> None:
    """Frozen loops differentiate parameters across two independently known regimes."""
    with _precision(backend):
        module = _module(backend)
        step = 0.0625
        schedule = AdaptiveStepSchedule((0.0, 2.0), ((step,) * 32,))
        config = RunConfig(
            method=method,
            adaptive=True,
            forcing_breakpoints=(1.0,),
            replay_loop="scan" if backend == "jax" else "unroll",
            replay_checkpoint=backend == "jax",
        )

        def solve(rate: Array) -> Array:
            solver = _solver(module.ones((1, 1), dtype=module.float64), (0.0, 2.0))
            xp = array_namespace(rate)
            solver.replay_adaptive_schedule(
                lambda t, y: cast(
                    "Array", -rate * xp.where(xp.asarray(t) < 1.0, 1.0, 2.0) * y
                ),
                schedule,
                config=config,
            )
            return cast("Array", xp.sum(solver.core.get_current_state()))

        if backend == "jax":
            jax = pytest.importorskip("jax")
            value, gradient = jax.jit(jax.value_and_grad(solve))(
                module.asarray(1.0, dtype=module.float64)
            )
            batched = jax.jit(jax.vmap(solve))(
                module.asarray([0.5, 1.0], dtype=module.float64)
            )
            np.testing.assert_allclose(
                batched, np.exp(-3 * np.asarray([0.5, 1.0])), rtol=2e-6
            )
        else:
            rate = module.tensor(1.0, dtype=module.float64, requires_grad=True)
            value = solve(rate)
            value.backward()
            gradient = rate.grad
        np.testing.assert_allclose(_host(value), math.exp(-3.0), rtol=2e-6)
        np.testing.assert_allclose(_host(gradient), -3 * math.exp(-3.0), rtol=3e-6)


def test_forcing_coordinates_must_remain_distinct_in_native_precision() -> None:
    """Collapsing two real jumps onto one float32 time cannot silently lose one."""
    solver = _solver(np.ones((1, 1), dtype=np.float32), (0.0, 2.0))
    with pytest.raises(ValueError, match="finite and distinct"):
        solver.run(
            lambda _t, y: y, config=RunConfig(forcing_breakpoints=(1.0, 1.0 + 1e-10))
        )


def test_long_replay_mesh_lands_on_jump_at_large_time_offset() -> None:
    """Thousands of exact scheduled durations do not create a false crossing."""
    solver = _solver(np.ones((1, 1)), (364.0, 366.0))
    schedule = AdaptiveStepSchedule((364.0, 366.0), ((0.001,) * 2000,))
    solver.replay_adaptive_schedule(
        lambda t, y: np.full_like(y, 1.0 if t < 365.0 else 2.0),
        schedule,
        config=RunConfig(method="euler", adaptive=True, forcing_breakpoints=(365.0,)),
    )
    # One unit from the incoming day, two from the outgoing day.
    np.testing.assert_allclose(solver.core.get_current_state(), [[4.0]], rtol=1e-12)
