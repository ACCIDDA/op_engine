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

"""Pure stochastic runs must cover every transition and operator (#182)."""

from __future__ import annotations

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

AGE = {"name": "age", "coords": ["c", "a"]}
IMM = {"name": "imm", "type": "ordinal", "coords": ["x0", "x1", "x2"]}


def _sir(*, recovery_named: bool) -> OpSystemSystem:
    """Build a two-age S->I->R model whose recovery may lack a name.

    Returns:
        The provider system.
    """
    recovery: dict[str, object] = {"from": "I[age]", "to": "R[age]", "rate": "gamma"}
    if recovery_named:
        recovery["name"] = "recover"
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [AGE],
            "state": ["S[age]", "I[age]", "R[age]"],
            "transitions": [
                {
                    "name": "infect",
                    "from": "S[age]",
                    "to": "I[age]",
                    "rate": "beta * I[age]",
                },
                recovery,
            ],
        }
    )


def _run(
    system: OpSystemSystem,
    config: OpEngineEngineConfig,
    values: list[float],
    *,
    gamma: float = 5.0,
) -> np.ndarray:
    """Run ``system`` from ``values`` with scalar beta/gamma over [0, 1].

    Returns:
        The provider trajectory.
    """
    initial = {
        str(name): ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in zip(system.option("state_names"), values, strict=True)
    }
    params = {
        "beta": ParameterValue(np.asarray(0.01), ResolvedShape()),
        "gamma": ParameterValue(np.asarray(gamma), ResolvedShape()),
    }
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=config)
    return np.asarray(engine.run(system, np.asarray([0.0, 1.0]), initial, params))


_SSA = OpEngineEngineConfig(
    mode="stochastic", stochastic_method="direct-ssa", random_seed=182
)
_SIR_VALUES = [50.0, 50.0, 5.0, 5.0, 0.0, 0.0]


def test_unnamed_recovery_is_rejected_instead_of_never_firing() -> None:
    """The motivating model: recovery used to be dropped without a trace."""
    system = _sir(recovery_named=False)
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=_SSA)
    issues = engine.validate_system(system) or []
    assert [issue.kind for issue in issues] == ["uncovered_transitions"]
    assert "transitions[1] unnamed: I[age] -> R[age] (unnamed)" in issues[0].msg
    with pytest.raises(ValueError, match=r"1 transition\(s\) have none"):
        _run(system, _SSA, _SIR_VALUES)


def test_covered_model_runs_and_recovers() -> None:
    """Naming the transition gives it a reaction, so the run proceeds."""
    system = _sir(recovery_named=True)
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=_SSA)
    assert engine.validate_system(system) is None
    history = _run(system, _SSA, _SIR_VALUES)
    assert history[-1, 5:].sum() > 0


def test_hybrid_mode_still_integrates_uncovered_transitions() -> None:
    """Hybrid integrates everything outside the jump partition."""
    config = OpEngineEngineConfig(mode="hybrid", stochastic_reactions=("infect",))
    system = _sir(recovery_named=False)
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=config)
    assert engine.validate_system(system) is None
    # A gentle rate keeps the default deterministic step from overshooting.
    history = _run(system, config, _SIR_VALUES, gamma=0.5)
    assert history[-1, 5:].sum() > 0


def test_producers_without_coverage_records_are_not_checked() -> None:
    """Artifacts from op_system releases before #244 keep today's behavior."""
    system = _sir(recovery_named=False)
    del system.options["reaction_gaps"]
    history = _run(system, _SSA, _SIR_VALUES)
    np.testing.assert_array_equal(history[-1, 5:], 0)


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        pytest.param(
            {
                "kind": "transitions",
                "state": ["S", "R"],
                "chain": [
                    {
                        "name": "I",
                        "length": 2,
                        "entry": {"from": "S", "rate": "beta"},
                        "forward": ["gamma"],
                        "exit": {"to": "R", "rate": "gamma"},
                    }
                ],
                "transitions": [],
            },
            "chain[0].forward[0] unnamed: I1 -> I2 (unnamed)",
            id="chain",
        ),
        pytest.param(
            {
                "kind": "transitions",
                "axes": [IMM],
                "state": ["X[imm]"],
                "transitions": [
                    {
                        "name": "wane",
                        "from": "X[imm:i]",
                        "to": "X[imm:j]",
                        "rate": "gamma * K[imm:i, imm:j]",
                    }
                ],
            },
            "transitions[0] wane: X[imm:i] -> X[imm:j] (routing)",
            id="routing",
        ),
    ],
)
def test_other_uncovered_transitions_are_named(
    spec: dict[str, object], fragment: str
) -> None:
    """Each uncovered transition appears with its origin and reason."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=_SSA)
    issues = engine.validate_system(OpSystemSystem(spec=spec)) or []
    assert any(fragment in issue.msg for issue in issues)


def test_typed_operators_are_rejected_in_pure_stochastic_mode() -> None:
    """Operators have no discrete counterpart, so they cannot be dropped."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [IMM],
            "state": ["X[imm]"],
            "transitions": [
                {"name": "decay", "from": "X[imm]", "to": "X[imm]", "rate": "gamma"}
            ],
            "operators": [
                {
                    "kind": "axis_kernel",
                    "axis": "imm",
                    "velocity": 0.5,
                    "kernel": {
                        "form": "generator",
                        "params": {"matrix": "G"},
                        "param_axes": {"G": ["imm", "imm"]},
                    },
                }
            ],
        }
    )
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=_SSA)
    issues = engine.validate_system(system) or []
    assert "stochastic_operators" in [issue.kind for issue in issues]
