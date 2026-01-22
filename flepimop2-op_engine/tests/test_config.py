# tests/flepimop2/test_config.py
"""Tests for op_engine.flepimop2.config."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")
from op_engine.core_solver import RunConfig  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from flepimop2.engine.op_engine.config import OpEngineEngineConfig  # noqa: E402


def test_engine_config_defaults_to_run_config() -> None:
    """Engine config defaults produce expected RunConfig."""
    cfg = OpEngineEngineConfig()
    run = cfg.to_run_config()

    assert isinstance(run, RunConfig)
    assert run.method == "heun"
    assert run.adaptive is False
    assert run.strict is True

    # Adaptive config defaults
    assert run.adaptive_cfg.rtol == pytest.approx(1e-6)
    assert float(run.adaptive_cfg.atol) == pytest.approx(1e-9)

    # dt controller defaults
    assert run.dt_controller.dt_min == pytest.approx(0.0)
    assert run.dt_controller.dt_max == pytest.approx(float("inf"))
    assert run.dt_controller.safety == pytest.approx(0.9)
    assert run.dt_controller.fac_min == pytest.approx(0.2)
    assert run.dt_controller.fac_max == pytest.approx(5.0)

    # Operators intentionally omitted for now
    assert run.operators.default is None
    assert run.operators.tr is None
    assert run.operators.bdf2 is None

    # Gamma defaults
    assert run.gamma is None


def test_engine_config_round_trips_selected_fields() -> None:
    """Engine config round-trips selected fields correctly."""
    cfg = OpEngineEngineConfig(
        method="euler",
        adaptive=True,
        strict=False,
        rtol=1e-4,
        atol=1e-7,
        dt_min=1e-6,
        dt_max=0.25,
        safety=0.95,
        fac_min=0.5,
        fac_max=2.0,
    )
    run = cfg.to_run_config()

    assert run.method == "euler"
    assert run.adaptive is True
    assert run.strict is False

    assert run.adaptive_cfg.rtol == pytest.approx(1e-4)
    assert float(run.adaptive_cfg.atol) == pytest.approx(1e-7)

    assert run.dt_controller.dt_min == pytest.approx(1e-6)
    assert run.dt_controller.dt_max == pytest.approx(0.25)
    assert run.dt_controller.safety == pytest.approx(0.95)
    assert run.dt_controller.fac_min == pytest.approx(0.5)
    assert run.dt_controller.fac_max == pytest.approx(2.0)


def test_engine_config_allows_unknown_fields() -> None:
    """Engine config should allow unknown fields without error."""
    cfg = OpEngineEngineConfig(  # type: ignore[call-arg]
        method="heun",
        adaptive=False,
        some_unknown_key=123,
        nested_unknown={"a": 1},
    )
    run = cfg.to_run_config()
    assert run.method == "heun"
    assert run.adaptive is False


def test_engine_config_rejects_unknown_method() -> None:
    """Engine config validates method name."""
    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method="rk4")  # type: ignore[arg-type]


def test_engine_config_gamma_bounds_validation() -> None:
    """Engine config validates gamma bounds for imex-trbdf2 method."""
    cfg = OpEngineEngineConfig(method="imex-trbdf2", gamma=0.6)
    run = cfg.to_run_config()
    assert run.method == "imex-trbdf2"
    assert run.gamma == pytest.approx(0.6)

    # invalid: gamma must be in (0, 1)
    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method="imex-trbdf2", gamma=0.0)

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method="imex-trbdf2", gamma=1.0)

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method="imex-trbdf2", gamma=-0.1)

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method="imex-trbdf2", gamma=1.1)
