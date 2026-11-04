"""Independent known-time state-map solutions and conditional derivatives."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from op_engine import CoreSolver, ModelCore, StateImpulse, array_namespace
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions
from op_engine.sensitivities import SensitivityOptions

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
def test_impulses_native_endpoints_and_once_only(
    backend: str, dtype: str, method: str, *, adaptive: bool
) -> None:
    """Maps act once on (start,end], including off-grid and output endpoints."""
    with _precision(backend):
        module = _module(backend)
        initial = module.asarray([[1.0]], dtype=getattr(module, dtype))
        xp = array_namespace(initial)
        calls: list[float] = []

        def add(time: Scalar, state: Array) -> Array:
            calls.append(float(time))
            return state + 4.0

        def multiply(time: Scalar, state: Array) -> Array:
            calls.append(float(time))
            return state * (3.0 if time == 0.3 else 2.0)

        config = RunConfig(
            method=method,
            adaptive=adaptive,
            replay_model_id="constant-flow-add-multiply-v1",
            adaptive_cfg=AdaptiveConfig(rtol=1e-3, atol=1e-5, max_steps=100),
            impulses=(
                StateImpulse(-1.0, add),
                StateImpulse(0.0, add),
                StateImpulse(0.3, multiply),
                StateImpulse(0.8, add),
                StateImpulse(1.5, multiply),
                StateImpulse(2.0, add),
            ),
        )
        solver = _solver(initial, (0.0, 0.8, 1.5))

        def rhs(_time: Scalar, state: Array) -> Array:
            return cast("Array", xp.ones_like(state) * 2.0)

        solver.run(rhs, config=config)
        assert calls == [0.3, 0.8, 1.5]
        assert solver.core.state_array is not None
        values = solver.core.state_array
        np.testing.assert_allclose(
            _host(values).reshape(-1), [1.0, 9.8, 22.4], rtol=2e-6
        )
        assert values.dtype == initial.dtype
        assert array_namespace(values) is xp
        if adaptive:
            schedule = solver.last_adaptive_schedule
            assert schedule is not None
            calls.clear()
            replay = _solver(initial, (0.0, 0.8, 1.5))
            replay.replay_adaptive_schedule(rhs, schedule, config=config)
            assert calls == [0.3, 0.8, 1.5]
            np.testing.assert_allclose(
                _host(replay.core.state_array), _host(values), rtol=2e-6
            )


@pytest.mark.parametrize(("method", "minimum_ratio"), [("rk4", 12), ("dopri5", 24)])
def test_numpy_impulse_refinement_and_fsal(method: str, minimum_ratio: int) -> None:
    """Decay after an additive map retains each method's smooth order."""
    errors = []
    exact = math.exp(-1.4) + 1.3 * math.exp(-0.7 * 1.3)
    for step in (0.2, 0.1):
        solver = _solver(np.ones((1, 1)), (0.0, 0.7, 2.0))
        solver.run(
            lambda _time, state: state * -0.7,
            config=RunConfig(
                method=method,
                fixed_max_step=step,
                impulses=(StateImpulse(0.7, lambda _time, state: state + 1.3),),
            ),
        )
        errors.append(abs(float(solver.core.get_current_state()[0, 0]) - exact))
    assert errors[0] / errors[1] > minimum_ratio


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_impulses_with_coincident_forcing(backend: str, *, adaptive: bool) -> None:
    """A forcing jump and state replacement share one boundary and one update."""
    with _precision(backend):
        module = _module(backend)
        initial = module.asarray([[1.0]], dtype=module.float64)
        xp = array_namespace(initial)

        def rhs(time: Scalar, state: Array) -> Array:
            return state * xp.where(xp.asarray(time) < 0.7, -0.3, -0.8)

        solver = _solver(initial, (0.0, 2.0))
        solver.run(
            rhs,
            config=RunConfig(
                method="dopri5",
                adaptive=adaptive,
                fixed_max_step=None if adaptive else 0.05,
                forcing_breakpoints=(0.7,),
                impulses=(StateImpulse(0.7, lambda _time, state: state + 1.2),),
                replay_model_id="piecewise-decay-add-v1",
                adaptive_cfg=AdaptiveConfig(rtol=1e-9, atol=1e-11),
            ),
        )
        exact = (math.exp(-0.21) + 1.2) * math.exp(-1.04)
        np.testing.assert_allclose(
            _host(solver.core.get_current_state()), exact, rtol=2e-8
        )


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("replay", [False, True])
def test_impulse_native_parameter_gradients(backend: str, *, replay: bool) -> None:
    """Static unrolled maps retain rate, update and initial-state derivatives."""
    with _precision(backend):
        module = _module(backend)
        base = RunConfig(
            method="dopri5",
            adaptive=replay,
            fixed_max_step=None if replay else 0.08,
            replay_model_id="decay-add-v1",
            adaptive_cfg=AdaptiveConfig(rtol=1e-9, atol=1e-11),
            impulses=(StateImpulse(0.7, lambda _time, state: state + 1.2),),
        )
        discovery = _solver(np.ones((1, 1)), (0.0, 0.7, 2.0))
        discovery.run(lambda _time, state: state * -0.4, config=base)
        schedule = discovery.last_adaptive_schedule

        def solve(parameters: Any) -> Any:  # noqa: ANN401
            rate, jump, initial_value = parameters
            initial = initial_value * module.ones((1, 1), dtype=module.float64)
            solver = _solver(initial, (0.0, 0.7, 2.0))
            config = replace(
                base,
                impulses=(StateImpulse(0.7, lambda _time, state: state + jump),),
            )
            if replay:
                assert schedule is not None
                solver.replay_adaptive_schedule(
                    lambda _time, state: state * -rate, schedule, config=config
                )
            else:
                solver.run(lambda _time, state: state * -rate, config=config)
            return solver.core.get_current_state().sum()

        values = [0.4, 1.2, 1.0]
        parameters = module.asarray(values, dtype=module.float64)
        if backend == "jax":
            jax = pytest.importorskip("jax")
            value, gradient = jax.jit(jax.value_and_grad(solve))(parameters)
        else:
            parameters.requires_grad_()
            value = solve(parameters)
            (gradient,) = module.autograd.grad(value, parameters)
        decay, tail = math.exp(-0.8), math.exp(-0.52)
        np.testing.assert_allclose(_host(value), decay + 1.2 * tail, rtol=2e-8)
        np.testing.assert_allclose(
            _host(gradient), [-2.0 * decay - 1.56 * tail, tail, decay], rtol=2e-7
        )


def test_numpy_impulse_replay_identity_and_mesh_guards() -> None:
    """Same names cannot replace required semantic identity and mesh boundaries."""
    event = StateImpulse(0.5, lambda _time, state: state + 1.0, "jump")
    with pytest.raises(ValueError, match="replay_model_id"):
        RunConfig(adaptive=True, impulses=(event,))
    config = RunConfig(adaptive=True, impulses=(event,), replay_model_id="add-v1")
    solver = _solver(np.ones((1, 1)), (0.0, 1.0))
    solver.run(lambda _time, state: state * 0.0, config=config)
    schedule = solver.last_adaptive_schedule
    assert schedule is not None
    replay = _solver(np.ones((1, 1)), (0.0, 1.0))
    with pytest.raises(ValueError, match=r"config\.adaptive=True"):
        replay.replay_adaptive_schedule(
            lambda _time, state: state,
            schedule,
            config=replace(config, adaptive=False, replay_model_id=None),
        )
    with pytest.raises(ValueError, match="model_id mismatch"):
        replay.replay_adaptive_schedule(
            lambda _time, state: state * 0.0,
            schedule,
            config=replace(
                config,
                impulses=(StateImpulse(0.5, lambda _time, state: state * 2.0, "jump"),),
                replay_model_id="multiply-v1",
            ),
        )
    with pytest.raises(ValueError, match="impulse_signature mismatch"):
        replay.replay_adaptive_schedule(
            lambda _time, state: state * 0.0,
            schedule,
            config=replace(config, impulses=(replace(event, time=0.4),)),
        )
    with pytest.raises(ValueError, match="no provenance"):
        replay.replay_adaptive_schedule(
            lambda _time, state: state,
            replace(schedule, provenance=None),
            config=config,
        )
    with pytest.raises(ValueError, match="crosses a forcing breakpoint"):
        replay.replay_adaptive_schedule(
            lambda _time, state: state,
            AdaptiveStepSchedule((0.0, 1.0), ((1.0,),), schedule.provenance),
            config=config,
        )


@pytest.mark.parametrize("time", [True, math.nan, math.inf, [1.0], 10**400])
def test_numpy_impulse_invalid_time(time: object) -> None:
    """Dynamic/invalid coordinates cannot be interpreted as a static schedule."""
    with pytest.raises(ValueError, match="finite real"):
        StateImpulse(time, lambda _time, state: state)  # type: ignore[arg-type]


def test_numpy_impulse_configuration_guards() -> None:
    """Reject unsupported policies instead of partially executing the maps."""
    event = StateImpulse(0.5, lambda _time, state: state)
    for method in ("bdf2", "imex-euler", "ros2", "sdirk2"):
        with pytest.raises(ValueError, match="explicit method"):
            RunConfig(method=method, impulses=(event,))
    for options in ({"fixed_loop": "scan"}, {"replay_loop": "scan"}):
        with pytest.raises(ValueError, match="eager unrolled"):
            RunConfig(impulses=(event,), **options)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly increasing"):
        RunConfig(impulses=(event, event))
    with pytest.raises(TypeError, match="StateImpulse"):
        RunConfig(impulses=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="smooth fixed explicit"):
        SensitivityOptions(solver=RunConfig(impulses=(event,)))


def test_numpy_impulse_normalized_times_remain_distinct() -> None:
    """Distinct integers that coalesce in solver time cannot silently merge."""
    with pytest.raises(ValueError, match="strictly increasing"):
        RunConfig(
            impulses=(
                StateImpulse(2**53, lambda _time, state: state),
                StateImpulse(2**53 + 1, lambda _time, state: state),
            )
        )


@pytest.mark.parametrize("bad_result", [np.ones(2), np.ones((1, 1), dtype=np.float32)])
def test_numpy_impulse_result_contract(bad_result: np.ndarray) -> None:
    """Maps cannot silently change the numerical state contract."""
    solver = _solver(np.ones((1, 1)), (0.0, 1.0))
    with pytest.raises((ValueError, TypeError), match="preserve state"):
        solver.run(
            lambda _time, state: state * 0.0,
            config=RunConfig(impulses=(StateImpulse(0.5, lambda _t, _y: bad_result),)),
        )
