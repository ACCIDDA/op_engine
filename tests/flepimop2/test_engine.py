# tests/flepimop2/test_engine.py
"""Unit tests for op_engine.flepimop2.engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from flepimop2.system.abc import SystemABC, SystemProtocol

    from op_engine.flepimop2.types import IdentifierString

import numpy as np
import pytest

from op_engine.flepimop2.engine import OpEngineFlepimop2Engine, build

pytestmark = pytest.mark.flepimop2


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


class _GoodSystem:
    """System-like object exposing a valid stepper via _stepper."""

    def __init__(self) -> None:
        self._stepper: SystemProtocol = _GoodStepper()


class _BadSystem:
    """System-like object exposing an invalid _stepper."""

    def __init__(self) -> None:
        self._stepper: object = object()


# -----------------------------------------------------------------------------
# Engine construction
# -----------------------------------------------------------------------------


def test_build_from_dict() -> None:
    """Engine can be constructed from a plain config dict."""
    cfg: dict[str, object] = {"method": "heun"}

    engine = build(cfg)

    assert isinstance(engine, OpEngineFlepimop2Engine)
    assert engine.config.method == "heun"


# -----------------------------------------------------------------------------
# Engine run behavior
# -----------------------------------------------------------------------------


def test_engine_run_basic_shape_and_dtype() -> None:
    """Engine returns correctly shaped float64 output array."""
    engine = OpEngineFlepimop2Engine()
    system = cast("SystemABC", _GoodSystem())

    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    y0 = np.array([1.0, 2.0], dtype=np.float64)

    params: dict[IdentifierString, object] = {}

    out = engine.run(system, times, y0, params)

    assert out.shape == (3, 3)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out[:, 0], times)


def test_engine_run_identity_rhs_behavior() -> None:
    """
    With dy/dt = y, state values should grow monotonically.

    This test validates wiring correctness, not numerical accuracy.
    """
    engine = OpEngineFlepimop2Engine()
    system = cast("SystemABC", _GoodSystem())

    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    y0 = np.array([1.0], dtype=np.float64)

    params: dict[IdentifierString, object] = {}

    out = engine.run(system, times, y0, params)

    state_values = out[:, 1]
    assert state_values[1] >= state_values[0]
    assert state_values[2] >= state_values[1]


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_engine_rejects_non_increasing_times() -> None:
    """Engine rejects non-strictly-increasing time grids."""
    engine = OpEngineFlepimop2Engine()
    system = cast("SystemABC", _GoodSystem())

    times = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y0 = np.array([1.0], dtype=np.float64)

    params: dict[IdentifierString, object] = {}

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.run(system, times, y0, params)


def test_engine_rejects_non_1d_initial_state() -> None:
    """Engine rejects non-1D initial state arrays."""
    engine = OpEngineFlepimop2Engine()
    system = cast("SystemABC", _GoodSystem())

    times = np.array([0.0, 1.0], dtype=np.float64)
    y0 = np.array([[1.0, 2.0]], dtype=np.float64)

    params: dict[IdentifierString, object] = {}

    with pytest.raises(ValueError, match="1D"):
        engine.run(system, times, y0, params)


def test_engine_rejects_missing_stepper() -> None:
    """Engine raises TypeError if system does not expose a valid stepper."""
    engine = OpEngineFlepimop2Engine()
    system = cast("SystemABC", _BadSystem())

    times = np.array([0.0, 1.0], dtype=np.float64)
    y0 = np.array([1.0], dtype=np.float64)

    params: dict[IdentifierString, object] = {}

    with pytest.raises(TypeError, match="SystemProtocol"):
        engine.run(system, times, y0, params)
