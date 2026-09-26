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

"""Unit tests for flepimop2.engine.op_engine."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
from flepimop2.system.abc import SystemABC
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import (
    AdaptiveReplayMode,
    AdaptiveSchedule,
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    ReplayCheckpoint,
    SolverMethod,
)

if TYPE_CHECKING:
    from flepimop2.typing import IdentifierString, SystemProtocol
    from op_engine import Array

# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


class _GoodStepper:
    """Simple stepper returning dy/dt = y."""

    def __call__(
        self,
        time: np.float64,
        state: np.ndarray,
        **params: object,
    ) -> np.ndarray:
        _ = time
        _ = params
        return state


class _GoodSystem(SystemABC, module="test_good"):
    """SystemABC implementation exposing a valid stepper via bind()."""

    state_change: StateChangeEnum = StateChangeEnum.FLOW

    def __init__(self) -> None:
        super().__init__()
        self._stepper: SystemProtocol = _GoodStepper()
        self.options = {
            "operators": {
                "default": (np.eye(1, dtype=np.float64), np.eye(1, dtype=np.float64))
            }
        }

    def _bind_impl(
        self, params: dict[IdentifierString, Any] | None = None
    ) -> SystemProtocol:
        return functools.partial(self._stepper, **(params or {}))


class _DeltaSystem(_GoodSystem, module="test_delta"):
    """SystemABC implementation with incompatible state_change."""

    state_change: StateChangeEnum = StateChangeEnum.DELTA


def _initial_state(
    *values: float,
) -> tuple[dict[str, ParameterValue], ModelStateSpecification]:
    """Build scalar state entries and their declared ordering.

    Returns:
        Parameter values and their model-state specification.
    """
    names = tuple(f"x{idx}" for idx in range(len(values)))
    entries = {
        name: ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in zip(names, values, strict=True)
    }
    return entries, ModelStateSpecification(parameter_names=names)


def _identity_rhs_jacobian(_time: float, state: Array) -> Array:
    """Return a full flattened identity Jacobian in the state namespace."""
    xp = cast("Any", state.__array_namespace__())
    return cast("Array", xp.eye(math.prod(state.shape), dtype=state.dtype))


# -----------------------------------------------------------------------------
# Engine construction
# -----------------------------------------------------------------------------


def test_public_engine_wrapper_defines_module() -> None:
    """Public engine wrapper satisfies flepimop2's concrete module contract."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)

    assert isinstance(engine, OpEngineFlepimop2Engine)
    assert engine.module == "flepimop2.engine.op_engine"


# -----------------------------------------------------------------------------
# Engine run behavior
# -----------------------------------------------------------------------------


def test_engine_run_basic_shape_and_dtype() -> None:
    """Engine returns correctly shaped float64 output array."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()

    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0, 2.0)

    out = engine.run(system, times, initial_state, {}, model_state=model_state)

    assert out.shape == (3, 3)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out[:, 0], times)


def test_engine_run_identity_rhs_behavior() -> None:
    """
    With dy/dt = y, state values should grow monotonically.

    This test validates wiring correctness, not numerical accuracy.
    """
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()

    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)

    out = engine.run(system, times, initial_state, {}, model_state=model_state)

    state_values = out[:, 1]
    assert state_values[1] >= state_values[0]
    assert state_values[2] >= state_values[1]


@pytest.mark.parametrize("method", [SolverMethod.RK4, SolverMethod.DOPRI5])
def test_jax_fixed_explicit_trajectory_is_one_differentiable_scan(
    method: SolverMethod,
) -> None:
    """Long JAX trajectories stay compact and differentiable."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=method, fixed_max_step=0.00075),
    )
    system = _GoodSystem()
    times = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    _initial, model_state = _initial_state(1.0)

    def solve(initial: object) -> object:
        result = engine.run(
            system,
            times,
            {"x0": ParameterValue(initial, ResolvedShape())},
            {},
            model_state=model_state,
        )
        return result[-1, 1]

    initial = jnp.asarray(1.0, dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(solve)(initial)
    scan_equations = [
        equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "scan"
    ]
    assert len(scan_equations) == 1

    value, derivative = jax.jit(jax.value_and_grad(solve))(initial)
    assert value == pytest.approx(np.e, rel=2e-5)
    assert derivative == pytest.approx(np.e, rel=2e-5)


def test_adaptive_schedule_discovery_and_replay_are_explicit() -> None:
    """Provider discovery returns a reusable artifact and matching replay."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=SolverMethod.HEUN,
            adaptive=True,
            rtol=1e-5,
            atol=1e-8,
        ),
    )
    system = _GoodSystem()
    times = np.asarray([0.0, 0.3, 1.0], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)

    discovered = engine.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
    )

    assert isinstance(discovered.schedule, AdaptiveSchedule)
    assert discovered.schedule.method is SolverMethod.HEUN
    assert discovered.schedule.step_schedule.output_times == tuple(times)
    assert discovered.diagnostics is None
    assert discovered.require_converged() is discovered
    assert engine.last_adaptive_schedule is discovered.schedule

    replayed = engine.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
        schedule=discovered.schedule,
    )
    ordinary_replay = engine.run(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
        adaptive_schedule=discovered.schedule,
    )

    assert replayed.schedule is discovered.schedule
    assert engine.last_adaptive_schedule is discovered.schedule
    np.testing.assert_allclose(replayed.trajectory, discovered.trajectory)
    np.testing.assert_allclose(ordinary_replay, discovered.trajectory)


def test_adaptive_schedule_replay_validates_method_controls_and_grid() -> None:
    """A frozen mesh cannot silently cross incompatible provider settings."""
    system = _GoodSystem()
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.HEUN, adaptive=True),
    )
    schedule = engine.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
    ).schedule

    wrong_method = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.RK4, adaptive=True),
    )
    with pytest.raises(ValueError, match="does not match configured method"):
        wrong_method.run(
            system,
            times,
            initial_state,
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )

    wrong_controls = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=SolverMethod.HEUN,
            adaptive=True,
            rtol=1e-4,
        ),
    )
    with pytest.raises(ValueError, match="controller settings"):
        wrong_controls.run(
            system,
            times,
            initial_state,
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )

    for changed_control in (
        {"dt_init": 0.1},
        {"max_reject": 7},
        {"max_steps": 20},
    ):
        changed_engine = OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=OpEngineEngineConfig(
                method=SolverMethod.HEUN,
                adaptive=True,
                **changed_control,  # type: ignore[arg-type]
            ),
        )
        with pytest.raises(ValueError, match="controller settings"):
            changed_engine.run(
                system,
                times,
                initial_state,
                {},
                model_state=model_state,
                adaptive_schedule=schedule,
            )

    with pytest.raises(ValueError, match="output_times"):
        engine.run(
            system,
            np.asarray([0.0, 1.0]),
            initial_state,
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )


def test_compact_replay_scan_and_gradient_parity() -> None:  # noqa: PLR0914
    """Long explicit meshes stay compact without changing values or gradients."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.linspace(0.0, 2.0, 33, dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)
    system = _GoodSystem()

    def make_engine(mode: AdaptiveReplayMode) -> OpEngineFlepimop2Engine:
        return OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=OpEngineEngineConfig(
                method=SolverMethod.HEUN,
                adaptive=True,
                rtol=1e-5,
                atol=1e-8,
                adaptive_replay=mode,
            ),
        )

    discovery_engine = make_engine(AdaptiveReplayMode.AUTO)
    schedule = discovery_engine.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
    ).schedule
    compact_engine = make_engine(AdaptiveReplayMode.COMPACT)
    unrolled_engine = make_engine(AdaptiveReplayMode.UNROLLED)

    def final_state(engine: OpEngineFlepimop2Engine, initial: Array) -> Array:
        trajectory = engine.run(
            system,
            times,
            {"x0": ParameterValue(initial, ResolvedShape())},
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )
        return cast("Array", trajectory[-1, 1])

    initial = jnp.asarray(1.0, dtype=jnp.float32)
    compact = functools.partial(final_state, compact_engine)
    unrolled = functools.partial(final_state, unrolled_engine)
    compact_jaxpr = jax.make_jaxpr(compact)(initial)
    scan_equations = [
        equation
        for equation in compact_jaxpr.jaxpr.eqns
        if equation.primitive.name == "scan"
    ]
    compact_value, compact_gradient = jax.jit(jax.value_and_grad(compact))(initial)
    unrolled_value, unrolled_gradient = jax.value_and_grad(unrolled)(initial)
    epsilon = 1e-3
    finite_gradient = (compact(initial + epsilon) - compact(initial - epsilon)) / (
        2.0 * epsilon
    )

    assert len(scan_equations) == 1
    assert compact_value == pytest.approx(unrolled_value, rel=1e-6)
    assert compact_gradient == pytest.approx(unrolled_gradient, rel=2e-6)
    assert compact_gradient == pytest.approx(finite_gradient, rel=2e-3)


def test_checkpointed_compact_replay_rematerializes_the_scan_step() -> None:
    """The step checkpoint policy emits rematerialization and keeps gradients."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)
    system = _GoodSystem()
    discovery = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(adaptive=True),
    )
    schedule = discovery.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
    ).schedule
    checkpointed = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            adaptive=True,
            adaptive_replay=AdaptiveReplayMode.COMPACT,
            replay_checkpoint=ReplayCheckpoint.STEP,
        ),
    )

    def solve(initial: Array) -> Array:
        trajectory = checkpointed.run(
            system,
            times,
            {"x0": ParameterValue(initial, ResolvedShape())},
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )
        return cast("Array", trajectory[-1, 1])

    initial = jnp.asarray(1.0, dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(solve)(initial)
    value, gradient = jax.jit(jax.value_and_grad(solve))(initial)

    assert "remat" in str(jaxpr)
    assert value == pytest.approx(np.e, rel=2e-5)
    assert gradient == pytest.approx(value, rel=2e-5)


def test_schedule_context_and_forced_compact_backend_are_validated() -> None:
    """Caller tags catch stale schedules and forced compact mode requires JAX."""
    system = _GoodSystem()
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)
    discovery = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(adaptive=True, schedule_tag="model-v1"),
    )
    schedule = discovery.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
    ).schedule

    stale = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(adaptive=True, schedule_tag="model-v2"),
    )
    with pytest.raises(ValueError, match="schedule_tag"):
        stale.run(
            system,
            times,
            initial_state,
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )

    forced_compact = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            adaptive=True,
            adaptive_replay=AdaptiveReplayMode.COMPACT,
            schedule_tag="model-v1",
        ),
    )
    with pytest.raises(TypeError, match="requires JAX"):
        forced_compact.run(
            system,
            times,
            initial_state,
            {},
            model_state=model_state,
            adaptive_schedule=schedule,
        )


def test_adaptive_entry_point_requires_adaptive_configuration() -> None:
    """Schedule discovery is unavailable for a fixed-step configuration."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(adaptive=False),
    )
    initial_state, model_state = _initial_state(1.0)

    with pytest.raises(ValueError, match="adaptive=True"):
        engine.run_adaptive(
            _GoodSystem(),
            np.asarray([0.0, 1.0]),
            initial_state,
            {},
            model_state=model_state,
        )


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_engine_rejects_non_increasing_times() -> None:
    """Engine rejects non-strictly-increasing time grids."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()

    times = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.run(system, times, initial_state, {}, model_state=model_state)


def test_engine_requires_model_state_ordering() -> None:
    """Engine requires semantic ordering for structured initial state."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()

    times = np.array([0.0, 1.0], dtype=np.float64)
    initial_state, _model_state = _initial_state(1.0, 2.0)

    with pytest.raises(ValueError, match="model_state must be provided"):
        engine.run(system, times, initial_state, {})


def test_validate_system_checks_state_change() -> None:
    """Engine validates state_change compatibility via validate_system."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    good = _GoodSystem()
    assert engine.validate_system(good) is None

    bad = _DeltaSystem()
    issues = engine.validate_system(bad)
    assert issues is not None
    assert issues[0].kind == "incompatible_system"


# -----------------------------------------------------------------------------
# validate_system: IMEX + missing operators
# -----------------------------------------------------------------------------


def test_validate_imex_missing_operators() -> None:
    """IMEX method without operators in config or system → missing_operators."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.IMEX_EULER),
    )
    system = _GoodSystem()
    system.options = {}

    issues = engine.validate_system(system)
    assert issues is not None
    kinds = [i.kind for i in issues]
    assert "missing_operators" in kinds


def test_validate_imex_system_provides_operators() -> None:
    """IMEX method + system.option('operators') provided → no operator warning."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.IMEX_EULER),
    )
    system = _GoodSystem()
    # _GoodSystem already has operators in options

    issues = engine.validate_system(system)
    assert issues is None


def test_validate_imex_config_provides_operators() -> None:
    """IMEX method + operators in engine config → no operator warning."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=SolverMethod.IMEX_EULER,
            operators={
                "default": [np.eye(1).tolist(), np.eye(1).tolist()],
            },
        ),
    )
    system = _GoodSystem()
    system.options = {}

    issues = engine.validate_system(system)
    assert issues is None


# -----------------------------------------------------------------------------
# validate_system: implicit/Rosenbrock + missing jacobian
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        SolverMethod.IMPLICIT_EULER,
        SolverMethod.TRAPEZOIDAL,
        SolverMethod.BDF2,
        SolverMethod.ROS2,
    ],
)
def test_validate_implicit_missing_jacobian(method: SolverMethod) -> None:
    """Implicit/Rosenbrock method without system jacobian → missing_jacobian."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=method),
    )
    system = _GoodSystem()
    # no "jacobian" in system.options

    issues = engine.validate_system(system)
    assert issues is not None
    kinds = [i.kind for i in issues]
    assert "missing_jacobian" in kinds


@pytest.mark.parametrize(
    "method",
    [
        SolverMethod.IMPLICIT_EULER,
        SolverMethod.TRAPEZOIDAL,
        SolverMethod.BDF2,
        SolverMethod.ROS2,
    ],
)
def test_validate_implicit_with_jacobian(method: SolverMethod) -> None:
    """Implicit/Rosenbrock method + system provides jacobian → no warning."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=method),
    )
    system = _GoodSystem()
    system.options = {
        **(system.options or {}),
        "jacobian": lambda _t, y: -np.eye(len(y)),
    }

    issues = engine.validate_system(system)
    assert issues is None


def test_validate_sdirk2_requires_full_rhs_jacobian() -> None:
    """SDIRK2 reports its distinct full-system Jacobian requirement."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.SDIRK2),
    )
    system = _GoodSystem()

    issues = engine.validate_system(system)
    assert issues is not None
    assert "missing_rhs_jacobian" in {issue.kind for issue in issues}

    system.options = {
        **(system.options or {}),
        "rhs_jacobian": _identity_rhs_jacobian,
    }
    assert engine.validate_system(system) is None


def test_validate_explicit_no_extra_issues() -> None:
    """Explicit methods do not trigger operator or jacobian warnings."""
    for method in (
        SolverMethod.EULER,
        SolverMethod.HEUN,
        SolverMethod.RK4,
        SolverMethod.DOPRI5,
    ):
        engine = OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=OpEngineEngineConfig(method=method),
        )
        system = _GoodSystem()
        system.options = {}
        assert engine.validate_system(system) is None


# -----------------------------------------------------------------------------
# Bind API integration
# -----------------------------------------------------------------------------


def test_engine_uses_bind_not_stepper() -> None:
    """Engine calls system.bind() rather than accessing system._stepper."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()
    bind_called = False
    original_bind = system.bind

    def tracking_bind(
        params: dict[IdentifierString, Any] | None = None, **kwargs: object
    ) -> SystemProtocol:
        nonlocal bind_called
        bind_called = True
        return original_bind(params, **kwargs)

    system.bind = tracking_bind  # type: ignore[method-assign]

    times = np.array([0.0, 0.1], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)
    engine.run(system, times, initial_state, {}, model_state=model_state)

    assert bind_called, "Engine should call system.bind()"


# -----------------------------------------------------------------------------
# Jacobian wiring for implicit methods
# -----------------------------------------------------------------------------


def test_run_implicit_method_uses_system_jacobian() -> None:
    """Implicit method retrieves jacobian from system.option and runs."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.IMPLICIT_EULER),
    )
    system = _GoodSystem()

    def neg_identity_jac(_t: float, y: np.ndarray) -> np.ndarray:
        return -np.eye(len(y), dtype=np.float64)

    system.options = {**(system.options or {}), "jacobian": neg_identity_jac}

    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)

    out = engine.run(system, times, initial_state, {}, model_state=model_state)

    assert out.shape == (3, 2)
    assert out.dtype == np.float64


def test_sdirk2_runs_and_replays_adaptively_with_jax_gradients() -> None:
    """The provider exposes nonlinear discovery and differentiable replay."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=SolverMethod.SDIRK2,
            adaptive=True,
            dt_init=0.05,
            max_reject=10,
            max_steps=100,
            rtol=1e-4,
            atol=1e-7,
        ),
    )
    system = _GoodSystem()
    system.options = {
        **(system.options or {}),
        "rhs_jacobian": _identity_rhs_jacobian,
    }
    times = np.asarray([0.0, 0.2, 0.4], dtype=np.float64)
    initial_state, model_state = _initial_state(1.0)

    discovered = engine.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
    )
    replayed = engine.run_adaptive(
        system,
        times,
        initial_state,
        {},
        model_state=model_state,
        schedule=discovered.schedule,
    )

    assert discovered.diagnostics is not None
    assert replayed.require_converged() is replayed
    np.testing.assert_allclose(replayed.trajectory, discovered.trajectory)

    def solve(initial: Array) -> Array:
        trajectory = engine.run(
            system,
            times,
            {"x0": ParameterValue(initial, ResolvedShape())},
            {},
            model_state=model_state,
            adaptive_schedule=discovered.schedule,
        )
        return cast("Array", trajectory[-1, 1])

    initial = jnp.asarray(1.0, dtype=jnp.float32)
    value, derivative = jax.jit(jax.value_and_grad(solve))(initial)

    assert np.isfinite(float(value))
    assert derivative == pytest.approx(value, rel=2e-5)
