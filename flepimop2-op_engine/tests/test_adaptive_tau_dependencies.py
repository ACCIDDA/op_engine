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

"""Adaptive tau-leaping on frequency-dependent op_system reactions (#193)."""

from __future__ import annotations

import numpy as np
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine
from flepimop2.engine.op_engine.reactions import compile_reaction_network

AGE = {"name": "age", "coords": ["young", "old"]}
CONFIG = OpEngineEngineConfig(
    mode="stochastic", stochastic_method="adaptive-tau-leaping", random_seed=193
)
FORCE_OF_INFECTION = (
    "beta * sum_over(I[age:a], age=a) / sum_over(S[age:a] + I[age:a] + R[age:a], age=a)"
)
PARAMS = {"beta": 1.5, "gamma": 0.5}


def _sir(*, auto: bool) -> OpSystemSystem:
    """Build age-structured frequency-dependent SIR.

    Returns:
        The provider system, with or without ``reactants: auto`` on infection.
    """
    infection: dict[str, object] = {
        "name": "infect",
        "from": "S[age]",
        "to": "I[age]",
        "rate": FORCE_OF_INFECTION,
    }
    if auto:
        infection["reactants"] = "auto"
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [AGE],
            "state": ["S[age]", "I[age]", "R[age]"],
            "transitions": [
                infection,
                {"name": "recover", "from": "I[age]", "to": "R[age]", "rate": "gamma"},
            ],
        }
    )


def _run(system: OpSystemSystem, initial_values: dict[str, float]) -> np.ndarray:
    """Run ``system`` under adaptive tau-leaping.

    Returns:
        The trajectory without its time column.
    """
    values = dict.fromkeys(system.option("state_names"), 0.0) | initial_values
    initial = {
        name: ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in values.items()
    }
    params = {
        name: ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in PARAMS.items()
    }
    history = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW, config=CONFIG
    ).run(system, np.linspace(0.0, 10.0, 6), initial, params)
    return np.asarray(history)[:, 1:]


def test_frequency_dependent_infection_runs_under_adaptive_tau() -> None:
    """``reactants: auto`` publishes dependencies the leap selector uses."""
    system = _sir(auto=True)
    network = compile_reaction_network(
        system, {k: np.asarray(v) for k, v in PARAMS.items()}, n_state=6
    )
    assert network.reactants_complete is True
    np.testing.assert_array_equal(network.propensity_orders, [3, 3, 0, 0])

    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=CONFIG)
    assert engine.validate_system(system) is None
    history = _run(
        system, {"S__age_young": 480.0, "S__age_old": 490.0, "I__age_young": 30.0}
    )
    np.testing.assert_array_equal(history, np.floor(history))
    assert np.all(history >= 0)
    np.testing.assert_array_equal(history.sum(axis=1), 1000.0)
    names = list(system.option("state_names"))
    assert history[-1, names.index("R__age_old")] > 0


def test_frequency_dependent_infection_without_auto_is_incomplete() -> None:
    """Omitted reactants on a state-reading rate still block adaptive tau."""
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=CONFIG)
    issues = engine.validate_system(_sir(auto=False)) or []
    (issue,) = [issue for issue in issues if issue.kind == "incomplete_reactants"]
    assert "lack it: infect." in issue.msg


def test_chain_entry_with_catalysts_auto_runs_under_adaptive_tau() -> None:
    """A chain entry reading a sum of stages is described by dependencies."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [AGE],
            "state": ["S[age]", "R[age]"],
            "chain": [
                {
                    "name": "I[age]",
                    "length": 2,
                    "entry": {
                        "from": "S[age]",
                        "rate": "beta * (I1[age] + I2[age]) / 100",
                        "catalysts": "auto",
                    },
                    "forward": ["gamma"],
                    "exit": {"to": "R[age]", "rate": "gamma"},
                }
            ],
        }
    )
    entry = next(r for r in system.option("reactions") if r.name == "I_entry")
    assert entry.dependencies_complete is True
    assert entry.propensity_order == 2
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=CONFIG)
    assert engine.validate_system(system) is None
    history = _run(system, {"S__age_young": 90.0, "I1__age_young": 10.0})
    np.testing.assert_array_equal(history.sum(axis=1), 100.0)
