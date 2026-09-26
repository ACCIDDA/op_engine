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

"""Tests for op_engine.flepimop2.config."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")
from op_engine.core_solver import OperatorSpecs, RunConfig  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from flepimop2.engine.op_engine import (  # noqa: E402
    ExecutionMode,
    OpEngineEngineConfig,
    SolverMethod,
    StochasticMethod,
)


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
        method=SolverMethod.EULER,
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
        method=SolverMethod.HEUN,
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
        method=SolverMethod.IMEX_TRBDF2,
        gamma=0.6,
        operators={"default": "sentinel"},
    )
    run = cfg.to_run_config()
    assert run.method == "imex-trbdf2"
    assert run.gamma == pytest.approx(0.6)

    # invalid: gamma must be in (0, 1)
    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method=SolverMethod.IMEX_TRBDF2,
            gamma=0.0,
            operators={"default": "sentinel"},
        )

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method=SolverMethod.IMEX_TRBDF2,
            gamma=1.0,
            operators={"default": "sentinel"},
        )

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method=SolverMethod.IMEX_TRBDF2,
            gamma=-0.1,
            operators={"default": "sentinel"},
        )

    with pytest.raises(ValidationError):
        OpEngineEngineConfig(
            method=SolverMethod.IMEX_TRBDF2,
            gamma=1.1,
            operators={"default": "sentinel"},
        )


def test_engine_config_imex_allows_deferred_operators() -> None:
    """IMEX methods may omit operators to defer to system options at runtime."""
    cfg = OpEngineEngineConfig(method=SolverMethod.IMEX_EULER)
    run = cfg.to_run_config()
    assert run.method == "imex-euler"
    assert isinstance(run.operators, OperatorSpecs)
    assert not _has_any_operator_specs(run.operators)


def test_engine_config_exposes_higher_order_imex_ark3() -> None:
    """The provider passes the paired ARK method through to core config."""
    run = OpEngineEngineConfig(method=SolverMethod.IMEX_ARK3).to_run_config()

    assert run.method == "imex-ark3"


def test_engine_config_imex_rejects_explicitly_empty_operator_block() -> None:
    """Providing an empty operator block should raise validation errors."""
    with pytest.raises(ValidationError):
        OpEngineEngineConfig(method=SolverMethod.IMEX_HEUN_TR, operators={})


def test_engine_config_imex_with_operators_still_valid() -> None:
    """Providing IMEX operators explicitly should still validate."""
    cfg = OpEngineEngineConfig(
        method=SolverMethod.IMEX_EULER,
        operators={"default": "sentinel"},
    )
    run = cfg.to_run_config()
    assert run.method == "imex-euler"

    assert isinstance(run.operators, OperatorSpecs)
    assert _has_any_operator_specs(run.operators)


def test_engine_config_defaults_to_deterministic_execution() -> None:
    """Existing configurations retain deterministic Heun behavior."""
    config = OpEngineEngineConfig()

    assert config.mode is ExecutionMode.DETERMINISTIC
    assert config.stochastic_method is StochasticMethod.TAU_LEAPING


def test_engine_config_accepts_pure_direct_ssa() -> None:
    """Pure stochastic execution can select exact direct SSA."""
    config = OpEngineEngineConfig(
        mode=ExecutionMode.STOCHASTIC,
        stochastic_method=StochasticMethod.DIRECT_SSA,
        random_seed=42,
    )

    assert config.ssa_max_events == 1_000_000


def test_hybrid_mode_requires_a_unique_jump_partition() -> None:
    """Hybrid execution cannot silently select no channels or duplicates."""
    with pytest.raises(ValidationError, match="requires at least one"):
        OpEngineEngineConfig(mode=ExecutionMode.HYBRID)
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        OpEngineEngineConfig(
            mode=ExecutionMode.HYBRID,
            stochastic_reactions=("infect", "infect"),
        )


def test_stochastic_reaction_selection_is_hybrid_only() -> None:
    """A pure stochastic run always consumes the complete typed network."""
    with pytest.raises(ValidationError, match="hybrid mode only"):
        OpEngineEngineConfig(
            mode=ExecutionMode.STOCHASTIC,
            stochastic_reactions=("infect",),
        )


@pytest.mark.parametrize("method", list(SolverMethod))
def test_hybrid_mode_rejects_methods_requiring_the_full_jacobian(
    method: SolverMethod,
) -> None:
    """The full deterministic Jacobian is invalid after jump-drift subtraction."""
    if not method.is_implicit:
        OpEngineEngineConfig(
            mode=ExecutionMode.HYBRID,
            method=method,
            stochastic_reactions=("infect",),
        )
        return
    with pytest.raises(ValidationError, match="cannot reuse the full-system Jacobian"):
        OpEngineEngineConfig(
            mode=ExecutionMode.HYBRID,
            method=method,
            stochastic_reactions=("infect",),
        )
