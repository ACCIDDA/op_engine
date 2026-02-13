# tests/flepimop2/test_config.py
"""Tests for op_engine.flepimop2.config."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from op_engine.core_solver import OperatorSpecs, RunConfig  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from flepimop2.engine.op_engine.config import OpEngineEngineConfig  # noqa: E402


def _has_any_operator_specs(specs: OperatorSpecs) -> bool:
    return any(getattr(specs, name) is not None for name in ("default", "tr", "bdf2"))


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

    # Operators exist and are empty by default
    assert isinstance(run.operators, OperatorSpecs)
    assert not _has_any_operator_specs(run.operators)

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
    # IMEX requires operators at parse-time.
    cfg = OpEngineEngineConfig(
        method="imex-trbdf2",
        gamma=0.6,
        operators={"default": "sentinel"},
    )
    run = cfg.to_run_config()
    assert run.method == "imex-trbdf2"
    assert run.gamma == pytest.approx(0.6)

    # invalid: gamma must be in (0, 1)
    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method="imex-trbdf2",
            gamma=0.0,
            operators={"default": "sentinel"},
        )

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method="imex-trbdf2",
            gamma=1.0,
            operators={"default": "sentinel"},
        )

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method="imex-trbdf2",
            gamma=-0.1,
            operators={"default": "sentinel"},
        )

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method="imex-trbdf2",
            gamma=1.1,
            operators={"default": "sentinel"},
        )


def test_engine_config_imex_allows_deferred_operators() -> None:
    """IMEX configs may omit operators to defer to system-provided specs."""
    cfg = OpEngineEngineConfig(method="imex-euler")
    run = cfg.to_run_config()

    assert run.method == "imex-euler"
    assert isinstance(run.operators, OperatorSpecs)
    assert not _has_any_operator_specs(run.operators)


def test_engine_config_imex_rejects_explicitly_empty_operator_block() -> None:
    """Providing operators with no stages populated should raise."""
    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method="imex-heun-tr", operators={})

    # Providing operators should pass validation.
    cfg = OpEngineEngineConfig(
        method="imex-euler",
        operators={"default": "sentinel"},
    )
    run = cfg.to_run_config()
    assert run.method == "imex-euler"
    assert isinstance(run.operators, OperatorSpecs)
    assert _has_any_operator_specs(run.operators)
    assert run.operators.default == "sentinel"


def test_engine_config_operator_dict_coerces_to_specs() -> None:
    """Operator dict input is coerced into OperatorSpecs with stage keys."""
    cfg = OpEngineEngineConfig(
        method="imex-trbdf2",
        operators={"default": 1, "tr": 2, "bdf2": 3},
        gamma=0.6,
    )
    run = cfg.to_run_config()

    assert isinstance(run.operators, OperatorSpecs)
    assert run.operators.default == 1
    assert run.operators.tr == 2
    assert run.operators.bdf2 == 3
