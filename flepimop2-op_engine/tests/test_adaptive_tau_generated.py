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

"""Adaptive tau-leaping on generated chain and coord_shift reactions (#188)."""

from __future__ import annotations

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

AGE = {"name": "age", "type": "ordinal", "coords": ["young", "old"]}
CONFIG = OpEngineEngineConfig(
    mode="stochastic", stochastic_method="adaptive-tau-leaping", random_seed=188
)


def _system(*, declared: bool) -> OpSystemSystem:
    """Build age-structured SIR with a two-stage infectious chain and aging.

    Returns:
        The provider system, with or without catalyst declarations.
    """
    stages = ["I1[age]", "I2[age]"]
    entry: dict[str, object] = {
        "from": "S[age]",
        "rate": "beta * (I1[age] + I2[age]) / 100",
    }
    chain: dict[str, object] = {
        "name": "I[age]",
        "length": 2,
        "entry": entry,
        "forward": ["gamma"],
        "exit": {"to": "R[age]", "rate": "gamma"},
    }
    aging: dict[str, object] = {
        "name": "aging",
        "coord_shift": {"axis": "age", "boundary": "absorb"},
        "rate": "mu",
        "apply_to": ["S", "I1", "I2", "R"],
    }
    if declared:
        entry["catalysts"] = [{"state": stage, "order": 1} for stage in stages]
        chain["catalysts"] = []
        aging["catalysts"] = []
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [AGE],
            "state": ["S[age]", "R[age]"],
            "chain": [chain],
            "transitions": [aging],
        }
    )


def _run(system: OpSystemSystem) -> np.ndarray:
    """Run ``system`` from 90 susceptible and 10 infectious young individuals.

    Returns:
        The trajectory without its time column.
    """
    values = dict.fromkeys(system.option("state_names"), 0.0)
    values |= {"S__age_young": 90.0, "I1__age_young": 10.0}
    initial = {
        name: ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in values.items()
    }
    params = {
        name: ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in {"beta": 1.2, "gamma": 0.5, "mu": 0.05}.items()
    }
    history = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW, config=CONFIG
    ).run(system, np.linspace(0.0, 10.0, 6), initial, params)
    return np.asarray(history)[:, 1:]


def test_declared_generated_reactions_run_under_adaptive_tau() -> None:
    """Catalysts complete the chain and aging reactions for adaptive leaping."""
    system = _system(declared=True)
    reactions = system.option("reactions")
    assert {r.name for r in reactions} == {
        "I_entry",
        "I_advance_1",
        "I_exit",
        "aging_S",
        "aging_I1",
        "aging_I2",
        "aging_R",
    }
    assert all(r.reactants_complete for r in reactions)
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=CONFIG)
    assert engine.validate_system(system) is None
    history = _run(system)
    np.testing.assert_array_equal(history, np.floor(history))
    assert np.all(history >= 0)
    totals = history.sum(axis=1)
    # Aging out of the oldest bin is absorbing, so the living total can only
    # fall, and every individual not yet aged out is accounted for.
    assert totals[0] == 100
    assert np.all(np.diff(totals) <= 0)
    names = list(system.option("state_names"))
    assert (
        history[-1, names.index("R__age_young")]
        + history[-1, names.index("R__age_old")]
        > 0
    )


def test_undeclared_generated_reactions_name_the_remedy() -> None:
    """Validation and run name each incomplete reaction and the remedies.

    Since op_system 0.7.0 a reaction whose rate reads no state is complete
    without declarations, so only the entry, whose rate reads both stages,
    is named.
    """
    system = _system(declared=False)
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=CONFIG)
    issues = engine.validate_system(system) or []
    (issue,) = [issue for issue in issues if issue.kind == "incomplete_reactants"]
    assert "lack it: I_entry." in issue.msg
    for name in ("I_advance_1", "I_exit", "aging_S", "aging_R"):
        assert name not in issue.msg
    assert "'catalysts'" in issue.msg
    assert "'reactants'" in issue.msg
    assert "'reactants: auto'" in issue.msg
    with pytest.raises(ValueError, match=r"lack it: I_entry\."):
        _run(system)


@pytest.mark.parametrize(
    ("selected", "incomplete"),
    [
        pytest.param(("aging_S", "I_exit"), None, id="complete-partition"),
        pytest.param(("aging_S", "I_entry"), "I_entry", id="includes-entry"),
    ],
)
def test_hybrid_validation_checks_only_selected_reactions(
    selected: tuple[str, ...], incomplete: str | None
) -> None:
    """A hybrid jump partition is checked for its own reactions only."""
    config = OpEngineEngineConfig(
        mode="hybrid",
        stochastic_method="adaptive-tau-leaping",
        stochastic_reactions=selected,
    )
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=config)
    issues = engine.validate_system(_system(declared=False)) or []
    messages = [issue.msg for issue in issues if issue.kind == "incomplete_reactants"]
    if incomplete is None:
        assert messages == []
    else:
        (message,) = messages
        assert f"lack it: {incomplete}." in message
