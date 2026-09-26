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

"""Tests for typed op_system reaction compilation."""

from __future__ import annotations

import numpy as np
import pytest
from flepimop2.system.op_system import OpSystemSystem

from flepimop2.engine.op_engine.reactions import (
    CompiledReactionNetwork,
    compile_reaction_network,
)


def _system() -> OpSystemSystem:
    """Build one system exercising every supported reaction layout.

    Returns:
        Compiled flepimop2 op_system provider.
    """
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [
                {"name": "age", "coords": ["a0", "a1"]},
                {"name": "vax", "coords": ["u", "v"]},
            ],
            "state": ["S[age,vax]", "I[age,vax]", "R[age,vax]"],
            "transitions": [
                {
                    "name": "infect",
                    "from": "S[age,vax]",
                    "to": "I[age,vax]",
                    "rate": "beta",
                },
                {
                    "name": "seed",
                    "to": "I[age,vax]",
                    "rate": "imp",
                },
                {
                    "name": "vaccinate",
                    "from": "S[age,vax=u]",
                    "to": "S[age,vax=v]",
                    "rate": "nu",
                },
                {
                    "name": "recover",
                    "from": "I[age,vax]",
                    "to": "R[age,vax=v]",
                    "rate": "gamma",
                },
            ],
        }
    )


@pytest.fixture
def network() -> CompiledReactionNetwork:
    """Compile the shared reaction fixture.

    Returns:
        Flat stochastic reaction network.
    """
    return compile_reaction_network(
        _system(),
        {
            "beta": np.asarray(0.1),
            "imp": np.asarray(0.2),
            "nu": np.asarray(0.3),
            "gamma": np.asarray(0.4),
        },
        n_state=12,
    )


def test_transition_channels_move_matching_cells(
    network: CompiledReactionNetwork,
) -> None:
    """An ordinary transition expands to one source/destination column per cell."""
    stoichiometry = network.stoichiometry
    expected = np.zeros((12, 4), dtype=np.int64)
    for cell in range(4):
        expected[cell, cell] = -1
        expected[4 + cell, cell] = 1
    np.testing.assert_array_equal(stoichiometry[:, :4], expected)


def test_source_only_channels_do_not_deplete_state(
    network: CompiledReactionNetwork,
) -> None:
    """Exogenous source channels contain only their destination increment."""
    stoichiometry = network.stoichiometry
    expected = np.zeros((12, 4), dtype=np.int64)
    for cell in range(4):
        expected[4 + cell, cell] = 1
    np.testing.assert_array_equal(stoichiometry[:, 4:8], expected)


def test_pinned_axis_channels_use_distinct_source_and_target_cells(
    network: CompiledReactionNetwork,
) -> None:
    """Point-to-point pinned reactions preserve their unpinned age coordinate."""
    stoichiometry = network.stoichiometry
    expected = np.zeros((12, 2), dtype=np.int64)
    expected[0, 0] = -1
    expected[1, 0] = 1
    expected[2, 1] = -1
    expected[3, 1] = 1
    np.testing.assert_array_equal(stoichiometry[:, 8:10], expected)


def test_summed_axis_channels_share_the_pinned_destination(
    network: CompiledReactionNetwork,
) -> None:
    """Collapsed source cells remain distinct channels with shared target rows."""
    stoichiometry = network.stoichiometry
    expected = np.zeros((12, 4), dtype=np.int64)
    expected[4, 0] = -1
    expected[9, 0] = 1
    expected[5, 1] = -1
    expected[9, 1] = 1
    expected[6, 2] = -1
    expected[11, 2] = 1
    expected[7, 3] = -1
    expected[11, 3] = 1
    np.testing.assert_array_equal(stoichiometry[:, 10:14], expected)


def test_legacy_reactions_keep_source_fallback_but_are_incomplete(
    network: CompiledReactionNetwork,
) -> None:
    """Omitted declarations remain usable without claiming catalyst safety."""
    expected = np.zeros((12, 4), dtype=np.int64)
    for cell in range(4):
        expected[cell, cell] = 1

    assert network.reactants_complete is False
    np.testing.assert_array_equal(network.reactant_stoichiometry[:, :4], expected)


def test_explicit_reactants_expand_multiplicity_and_grouped_catalysts() -> None:
    """Provider channels preserve molecular order beyond net stoichiometry."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [
                {"name": "age", "coords": ["a0", "a1"]},
                {"name": "vax", "coords": ["u", "v"]},
            ],
            "state": ["S[age,vax]", "E[age,vax]", "I[age]"],
            "transitions": [
                {
                    "name": "infect",
                    "from": "S[age,vax]",
                    "to": "E[age,vax]",
                    "rate": "beta * I[age]",
                    "reactants": [
                        {"state": "S[age,vax]", "order": 2},
                        {"state": "I[age]", "order": 1},
                    ],
                },
            ],
        }
    )
    compiled = compile_reaction_network(
        system,
        {"beta": np.asarray(0.1)},
        n_state=10,
    )
    expected = np.zeros((10, 4), dtype=np.int64)
    expected[[0, 1, 2, 3], [0, 1, 2, 3]] = 2
    expected[8, [0, 1]] = 1
    expected[9, [2, 3]] = 1

    assert compiled.reactants_complete is True
    np.testing.assert_array_equal(compiled.reactant_stoichiometry, expected)


def test_propensities_follow_flat_channel_order(
    network: CompiledReactionNetwork,
) -> None:
    """Typed callbacks are evaluated on reshaped state and concatenated in order."""
    state = np.arange(1.0, 13.0).reshape(12, 1)
    got = network.propensity(0.0, state)
    expected = np.concatenate((
        0.1 * state[:4, 0],
        np.full(4, 0.2),
        0.3 * state[[0, 2], 0],
        0.4 * state[4:8, 0],
    ))
    np.testing.assert_allclose(got[:, 0], expected, rtol=0.0, atol=0.0)


def test_hybrid_selection_keeps_all_cells_of_named_reactions() -> None:
    """Selecting parent names retains every expanded source-cell channel."""
    network = compile_reaction_network(
        _system(),
        {"nu": np.asarray(0.3), "gamma": np.asarray(0.4)},
        n_state=12,
        reaction_names=("vaccinate", "recover"),
    )

    assert network.channel_reactions == (
        "vaccinate",
        "vaccinate",
        "recover",
        "recover",
        "recover",
        "recover",
    )


def test_unknown_hybrid_reaction_fails_visibly() -> None:
    """A misspelled jump partition cannot silently become deterministic."""
    with pytest.raises(ValueError, match="Unknown stochastic reaction"):
        compile_reaction_network(
            _system(),
            {},
            n_state=12,
            reaction_names=("typo",),
        )
