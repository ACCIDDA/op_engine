"""Unit tests for flepimop2.engine.op_engine."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from flepimop2.system.abc import SystemABC
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import (
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    SolverMethod,
)

if TYPE_CHECKING:
    from flepimop2.typing import IdentifierString, SystemProtocol

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
        return np.asarray(state, dtype=np.float64)


class _GoodSystem(SystemABC):
    """SystemABC implementation exposing a valid stepper via bind()."""

    module = "flepimop2.system.test_good"
    state_change = StateChangeEnum.FLOW

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


class _DeltaSystem(_GoodSystem):
    """SystemABC implementation with incompatible state_change."""

    module = "flepimop2.system.test_delta"
    state_change = StateChangeEnum.DELTA


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
    y0 = np.array([1.0, 2.0], dtype=np.float64)

    params: dict[str, object] = {}

    out = engine.run(system, times, y0, params)

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
    y0 = np.array([1.0], dtype=np.float64)

    params: dict[str, object] = {}

    out = engine.run(system, times, y0, params)

    state_values = out[:, 1]
    assert state_values[1] >= state_values[0]
    assert state_values[2] >= state_values[1]


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_engine_rejects_non_increasing_times() -> None:
    """Engine rejects non-strictly-increasing time grids."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()

    times = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y0 = np.array([1.0], dtype=np.float64)

    params: dict[str, object] = {}

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.run(system, times, y0, params)


def test_engine_rejects_non_1d_initial_state() -> None:
    """Engine rejects non-1D initial state arrays."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    system = _GoodSystem()

    times = np.array([0.0, 1.0], dtype=np.float64)
    y0 = np.array([[1.0, 2.0]], dtype=np.float64)

    params: dict[str, object] = {}

    with pytest.raises(ValueError, match="1D"):
        engine.run(system, times, y0, params)


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


def test_validate_explicit_no_extra_issues() -> None:
    """Explicit methods do not trigger operator or jacobian warnings."""
    for method in (SolverMethod.EULER, SolverMethod.HEUN):
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
    y0 = np.array([1.0], dtype=np.float64)
    engine.run(system, times, y0, {})

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
    y0 = np.array([1.0], dtype=np.float64)

    out = engine.run(system, times, y0, {})

    assert out.shape == (3, 2)
    assert out.dtype == np.float64
