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

"""Axis-wide ``coord_shift`` offset reactions in stochastic execution (#180)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from flepimop2.system.op_system import OpSystemSystem

from flepimop2.engine.op_engine.reactions import compile_reaction_network

N_AGE = 4


def _aging_system(
    boundary: str,
    *,
    step: int = 1,
    n_groups: int | None = None,
    age_first: bool = True,
) -> OpSystemSystem:
    """Build an aging chain over ``N_AGE`` bins from one axis-wide entry.

    Returns:
        A system with one aging reaction per state template.
    """
    axes: list[dict[str, object]] = [
        {"name": "age", "type": "ordinal", "coords": [f"a{k}" for k in range(N_AGE)]}
    ]
    selector = "age"
    if n_groups is not None:
        axes.append({"name": "group", "coords": [str(i) for i in range(n_groups)]})
        selector = "age,group" if age_first else "group,age"
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": axes,
            "state": [f"S[{selector}]"],
            "transitions": [
                {
                    "name": "aging",
                    "coord_shift": {"axis": "age", "step": step, "boundary": boundary},
                    "rate": "aging_rate[age]",
                    "apply_to": ["S"],
                }
            ],
        }
    )


def _fires(boundary: str, step: int, age: int) -> bool:
    """Return whether a source bin has a nonzero hazard.

    Returns:
        ``False`` only for ``stay`` sources whose target leaves the axis.
    """
    return boundary == "absorb" or 0 <= age + step < N_AGE


@pytest.mark.parametrize("boundary", ["absorb", "stay"])
@pytest.mark.parametrize("step", [1, -2])
@pytest.mark.parametrize(
    ("n_groups", "age_first"), [(None, True), (2, True), (2, False)]
)
def test_offset_channels_shift_each_source_cell(
    boundary: str, step: int, n_groups: int | None, *, age_first: bool
) -> None:
    """Each source cell moves to ``age + step``; off-axis firings only deplete."""
    system = _aging_system(boundary, step=step, n_groups=n_groups, age_first=age_first)
    groups = n_groups or 1
    n_state = N_AGE * groups
    rates = np.linspace(0.2, 0.8, N_AGE)
    network = compile_reaction_network(system, {"aging_rate": rates}, n_state=n_state)
    assert network.n_channels == n_state
    assert network.channel_reactions == ("aging_S",) * n_state

    def cell(age: int, group: int) -> int:
        # Channels follow the template's axis order, as do state cells.
        return age * groups + group if age_first else group * N_AGE + age

    expected = np.zeros((n_state, n_state), dtype=np.int64)
    for age in range(N_AGE):
        for group in range(groups):
            expected[cell(age, group), cell(age, group)] = -1
            if 0 <= age + step < N_AGE:
                expected[cell(age + step, group), cell(age, group)] = 1
    np.testing.assert_array_equal(network.stoichiometry, expected)
    np.testing.assert_array_equal(
        network.reactant_stoichiometry, -np.minimum(expected, 0)
    )

    state = np.arange(1.0, n_state + 1.0)
    propensity = np.asarray(network.propensity(0.0, state))
    for age in range(N_AGE):
        for group in range(groups):
            np.testing.assert_allclose(
                propensity[cell(age, group)],
                rates[age] * state[cell(age, group)] * _fires(boundary, step, age),
            )


@pytest.mark.parametrize("boundary", ["absorb", "stay"])
@pytest.mark.parametrize("step", [1, 2, -1])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_offset_drift_reconstructs_producer_rhs(
    boundary: str, step: int, backend: str
) -> None:
    """Propensity-weighted offset columns equal the producer's deterministic RHS."""
    system = _aging_system(boundary, step=step, n_groups=3, age_first=False)
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    params = {"aging_rate": xp.asarray(np.linspace(0.1, 0.9, N_AGE))}
    network = compile_reaction_network(system, params, n_state=3 * N_AGE)
    rng = np.random.default_rng(180)
    for _ in range(3):
        state = xp.asarray(rng.integers(0, 50, size=3 * N_AGE).astype(float))
        drift = network.mean_drift(0.0, state.reshape(-1, 1))
        assert drift.__array_namespace__() is xp
        np.testing.assert_allclose(
            np.asarray(drift)[:, 0],
            np.asarray(system.step(0.0, state, **params)),
            rtol=1e-6,
            atol=1e-5,
        )


class _WithoutOffsets:
    """A reaction artifact from an op_system release that predates offsets."""

    def __init__(self, reaction: object) -> None:
        self._reaction = reaction

    def __getattr__(self, name: str) -> object:
        if name == "offsets":
            raise AttributeError(name)
        return getattr(self._reaction, name)


def test_artifacts_without_offsets_compile_unchanged() -> None:
    """Older producer artifacts keep their existing channel layout."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [{"name": "age", "coords": ["a0", "a1"]}],
            "state": ["S[age]", "I[age]"],
            "transitions": [
                {"name": "infect", "from": "S[age]", "to": "I[age]", "rate": "beta"}
            ],
        }
    )
    params = {"beta": np.asarray(0.5)}
    current = compile_reaction_network(system, params, n_state=4)
    system.options["reactions"] = tuple(
        _WithoutOffsets(r) for r in system.option("reactions")
    )
    legacy = compile_reaction_network(system, params, n_state=4)
    np.testing.assert_array_equal(legacy.stoichiometry, current.stoichiometry)
    assert legacy.channel_names == current.channel_names


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"offsets": (("age", 0),)}, "nonzero step"),
        ({"offsets": (("age", 1),), "to_axes": ("age",)}, "neither copied"),
        ({"offsets": (("age", 1), ("age", 2))}, "more than once"),
        ({"sum_axes": ("age",)}, "inconsistent sum_axes"),
        ({"offsets": ()}, "inconsistent sum_axes|incomplete destination"),
    ],
)
def test_malformed_offsets_fail_visibly(changes: dict[str, object], match: str) -> None:
    """Offsets must shift channel axes that no other destination field covers."""
    system = _aging_system("absorb")
    system.options["reactions"] = tuple(
        dataclasses.replace(r, **changes) for r in system.option("reactions")
    )
    with pytest.raises(ValueError, match=match):
        compile_reaction_network(system, {"aging_rate": np.ones(N_AGE)}, n_state=N_AGE)
