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

"""Renewal births and pinned aging through real producer artifacts (#178)."""

from __future__ import annotations

import numpy as np
import pytest
from flepimop2.system.op_system import OpSystemSystem

from flepimop2.engine.op_engine.reactions import compile_reaction_network


def _renewal_system(
    n_groups: int | None = 2, *, age_first: bool = True, demography: bool = False
) -> OpSystemSystem:
    """Build a reduced birth flux, optionally with aging and departures.

    Returns:
        A system with living N, additional fertile M, and departure counter D.
    """
    selector = "age" if n_groups is None else "age,group" if age_first else "group,age"
    bound = selector.replace("age", "age:a")
    axes = [{"name": "age", "coords": ["a0", "a1", "a2"]}]
    if n_groups is not None:
        axes.append({"name": "group", "coords": [str(i) for i in range(n_groups)]})
    transitions: list[dict[str, object]] = [
        {
            "name": "renewal",
            "from": None,
            "to": f"N[{selector.replace('age', 'age=a0')}]",
            "rate": f"sum_over(B[age:a] * (N[{bound}] + M[{bound}]), age=a)",
            "reactants": [],
        },
    ]
    if demography:
        transitions.append({
            "name": "depart",
            "from": f"N[{selector}]",
            "to": f"D[{selector}]",
            "rate": "mu",
            "reactants": [{"state": f"N[{selector}]", "order": 1}],
        })
        for age in range(2):
            source = f"N[{selector.replace('age', f'age=a{age}')}]"
            transitions.append({
                "name": f"age_{age}",
                "from": source,
                "to": f"N[{selector.replace('age', f'age=a{age + 1}')}]",
                "rate": "aging",
                "reactants": [{"state": source, "order": 1}],
            })
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": axes,
            "state": [f"{base}[{selector}]" for base in ("N", "M", "D")],
            "transitions": transitions,
        }
    )


@pytest.mark.parametrize(
    ("n_groups", "age_first"), [(None, True), (2, True), (2, False)]
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_reduced_birth_channels_preserve_groups_and_state_layout(
    n_groups: int | None, *, age_first: bool, backend: str
) -> None:
    """Birth hazards reduce all fertile ages and ignore the departure counter."""
    system = _renewal_system(n_groups, age_first=age_first)
    groups = n_groups or 1
    shape = (3,) if n_groups is None else (3, groups) if age_first else (groups, 3)
    n_cells = 3 * groups
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    population = np.arange(n_cells, dtype=float).reshape(shape) + 1
    params = {"B": xp.asarray([0.1, 0.2, 0.3])}
    flat = xp.asarray(
        np.concatenate([
            population.ravel(),
            (population + 10).ravel(),
            np.full(n_cells, 1000.0),
        ])
    )
    network = compile_reaction_network(system, params, n_state=3 * n_cells)
    assert network.n_channels == groups
    assert network.channel_reactions == ("renewal",) * groups
    assert network.reactants_complete is True
    np.testing.assert_array_equal(network.reactant_stoichiometry, 0)

    target_rows = np.arange(groups) if age_first else 3 * np.arange(groups)
    expected_stoichiometry = np.zeros((3 * n_cells, groups), dtype=int)
    expected_stoichiometry[target_rows, np.arange(groups)] = 1
    np.testing.assert_array_equal(network.stoichiometry, expected_stoichiometry)
    weighted = (2 * population + 10) * np.array([0.1, 0.2, 0.3]).reshape(
        (3,) if n_groups is None else (3, 1) if age_first else (1, 3)
    )
    expected = weighted.sum(axis=0 if age_first else 1).reshape(groups, 1)
    propensity = network.propensity(0.0, flat.reshape(-1, 1))
    assert propensity.__array_namespace__() is xp
    np.testing.assert_allclose(np.asarray(propensity), expected)
    np.testing.assert_allclose(
        np.asarray(network.mean_drift(0.0, flat)),
        np.asarray(system.step(0.0, flat, **params)),
    )
    np.testing.assert_allclose(
        np.asarray(network.mean_drift(0.0, flat)),
        expected_stoichiometry @ expected[:, 0],
    )


@pytest.mark.parametrize(
    ("n_groups", "age_first"), [(None, True), (2, True), (2, False)]
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_pinned_aging_and_departures_reconstruct_producer_drift(
    n_groups: int | None, *, age_first: bool, backend: str
) -> None:
    """Each aging column moves the donor count into its successor coordinate."""
    system = _renewal_system(n_groups, age_first=age_first, demography=True)
    groups = n_groups or 1
    n_cells = 3 * groups
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    params = {
        "B": xp.asarray([0.1, 0.2, 0.3]),
        "mu": xp.asarray(0.4),
        "aging": xp.asarray(0.5),
    }
    network = compile_reaction_network(system, params, n_state=3 * n_cells)
    assert network.channel_reactions == (
        ("renewal",) * groups
        + ("depart",) * n_cells
        + ("age_0",) * groups
        + ("age_1",) * groups
    )
    for age in range(2):
        first = groups + n_cells + age * groups
        source = (
            age * groups + np.arange(groups)
            if age_first
            else 3 * np.arange(groups) + age
        )
        target = source + (groups if age_first else 1)
        expected = np.zeros((3 * n_cells, groups), dtype=int)
        expected[source, np.arange(groups)] = -1
        expected[target, np.arange(groups)] = 1
        np.testing.assert_array_equal(
            network.stoichiometry[:, first : first + groups], expected
        )
        np.testing.assert_array_equal(
            network.reactant_stoichiometry[source, first : first + groups],
            np.eye(groups, dtype=int),
        )

    rng = np.random.default_rng(178)
    for _ in range(3):
        state = xp.asarray(rng.integers(1, 100, size=3 * n_cells).astype(float))
        drift = network.mean_drift(0.0, state.reshape(-1, 1))
        assert drift.__array_namespace__() is xp
        assert drift.shape == (3 * n_cells, 1)
        np.testing.assert_allclose(
            np.asarray(drift)[:, 0],
            np.asarray(system.step(0.0, state, **params)),
            rtol=1e-6,
            atol=1e-5,
        )
