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

"""Routing and fan-out reactions in stochastic execution (#186)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine
from flepimop2.engine.op_engine.reactions import compile_reaction_network

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_IMM = 3
IMM = {"name": "imm", "type": "ordinal", "coords": [f"x{k}" for k in range(N_IMM)]}
VAX = {"name": "vax", "coords": ["u", "v"]}


def _generator(rate: float) -> NDArray[np.float64]:
    """Return a forward waning generator with an absorbing last bin.

    Returns:
        Row-source, column-target matrix whose rows sum to zero.
    """
    generator = rate * np.eye(N_IMM, k=1)
    return generator - np.diag(generator.sum(axis=1))


def _waning(*, groups: int | None = None) -> OpSystemSystem:
    """Build self-routing waning, optionally replicated over a group axis.

    Returns:
        The provider system.
    """
    axes: list[dict[str, object]] = [IMM]
    sel = "{imm}"
    if groups is not None:
        axes.append({"name": "group", "coords": [str(g) for g in range(groups)]})
        sel = "{imm},group"
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": axes,
            "state": [f"X[{sel.format(imm='imm')}]"],
            "transitions": [
                {
                    "name": "wane",
                    "from": f"X[{sel.format(imm='imm:i')}]",
                    "to": f"X[{sel.format(imm='imm:j')}]",
                    "rate": "G[imm:i, imm:j]",
                }
            ],
        }
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_self_routing_channels_skip_the_negative_diagonal(backend: str) -> None:
    """Each (source, target) pair is a channel; diagonal channels never fire."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = _waning()
    generator = _generator(0.7)
    params = {"G": xp.asarray(generator)}
    network = compile_reaction_network(system, params, n_state=N_IMM)
    assert network.n_channels == N_IMM * N_IMM
    assert network.channel_names[1] == "wane[imm=0,to:imm=1]"
    expected = np.zeros((N_IMM, N_IMM * N_IMM), dtype=np.int64)
    for source in range(N_IMM):
        for target in range(N_IMM):
            channel = source * N_IMM + target
            expected[source, channel] -= 1
            expected[target, channel] += 1
    np.testing.assert_array_equal(network.stoichiometry, expected)
    state = xp.asarray([10.0, 20.0, 30.0])
    propensity = np.asarray(network.propensity(0.0, state))
    assert np.all(propensity >= 0)
    np.testing.assert_array_equal(propensity[:: N_IMM + 1], 0)
    np.testing.assert_allclose(
        np.asarray(network.mean_drift(0.0, state)),
        np.asarray(system.step(0.0, state, **params)),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("spec", "params", "n_state"),
    [
        pytest.param(
            {
                "kind": "transitions",
                "axes": [VAX, IMM],
                "state": ["X[vax, imm]"],
                "transitions": [
                    {
                        "name": "vaccinate",
                        "from": "X[vax=u, imm:i]",
                        "to": "X[vax=v, imm:j]",
                        "rate": "nu * K[imm:i, imm:j]",
                    }
                ],
            },
            {"nu": 0.4, "K": np.arange(1.0, 10.0).reshape(3, 3) / 10},
            2 * N_IMM,
            id="routing-with-pinned-flip",
        ),
        pytest.param(
            {
                "kind": "transitions",
                "axes": [VAX, IMM],
                "state": ["I[vax]", "X[vax, imm]"],
                "transitions": [
                    {
                        "name": "reset",
                        "from": "I[vax]",
                        "to": "X[vax, imm:j]",
                        "rate": "r * w[imm:j]",
                    }
                ],
            },
            {"r": 0.6, "w": np.array([0.2, 0.3, 0.5])},
            2 + 2 * N_IMM,
            id="fan-out",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_routed_reactions_reconstruct_producer_drift(
    spec: dict[str, object],
    params: dict[str, object],
    n_state: int,
    backend: str,
) -> None:
    """Routed deposits land in each channel's target coordinate."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = OpSystemSystem(spec=spec)
    bound = {name: xp.asarray(value) for name, value in params.items()}
    network = compile_reaction_network(system, bound, n_state=n_state)
    np.testing.assert_array_equal(network.stoichiometry.sum(axis=0), 0)
    rng = np.random.default_rng(186)
    for _ in range(3):
        state = xp.asarray(rng.integers(1, 50, size=n_state).astype(float))
        np.testing.assert_allclose(
            np.asarray(network.mean_drift(0.0, state)),
            np.asarray(system.step(0.0, state, **bound)),
            rtol=1e-6,
        )


class _WithoutRoutedAxes:
    """A reaction artifact from an op_system release before routed_axes."""

    def __init__(self, reaction: object) -> None:
        self._reaction = reaction

    def __getattr__(self, name: str) -> object:
        if name == "routed_axes":
            raise AttributeError(name)
        return getattr(self._reaction, name)


def test_reactions_without_routed_axes_compile_unchanged() -> None:
    """Older artifacts have no routed dimensions."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [IMM],
            "state": ["S[imm]", "I[imm]"],
            "transitions": [
                {"name": "infect", "from": "S[imm]", "to": "I[imm]", "rate": "b"}
            ],
        }
    )
    params = {"b": np.asarray(0.5)}
    current = compile_reaction_network(system, params, n_state=6)
    system.options["reactions"] = tuple(
        _WithoutRoutedAxes(r) for r in system.option("reactions")
    )
    legacy = compile_reaction_network(system, params, n_state=6)
    np.testing.assert_array_equal(legacy.stoichiometry, current.stoichiometry)
    assert legacy.channel_names == current.channel_names


def _waning_ensemble(
    rate: float, per_group: int, times: NDArray[np.float64], groups: int = 64
) -> NDArray[np.float64]:
    """Run seeded direct-SSA waning paths over independent group channels.

    Returns:
        Occupancy shaped as (time, bin, replicate).
    """
    system = _waning(groups=groups)
    population = np.zeros((N_IMM, groups))
    population[0] = per_group
    initial = {
        str(name): ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in zip(
            system.option("state_names"), population.ravel(), strict=True
        )
    }
    params = {
        "G": ParameterValue(
            _generator(rate), ResolvedShape(("imm", "imm"), (N_IMM, N_IMM))
        )
    }
    histories = []
    for seed in range(186, 202):
        engine = OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=OpEngineEngineConfig(
                mode="stochastic", stochastic_method="direct-ssa", random_seed=seed
            ),
        )
        history = np.asarray(engine.run(system, times, initial, params))
        histories.append(history[:, 1:].reshape(times.size, N_IMM, groups))
    return np.concatenate(histories, axis=-1)


def test_routed_waning_ssa_matches_poisson_occupancy() -> None:
    """A routing generator wanes each unit through the bins at rate r.

    Before the absorbing last bin, a unit occupies bin ``j`` with Poisson
    probability ``exp(-r t) (r t)**j / j!``; the last bin holds the rest.
    """
    rate, per_group = 1.5, 5
    times = np.asarray([0.0, 0.5, 1.0, 2.0])
    occupancy = _waning_ensemble(rate, per_group, times)
    np.testing.assert_array_equal(occupancy.sum(axis=1), per_group)
    replicates = occupancy.shape[-1]
    elapsed = rate * times[1:, None]
    bins = np.arange(N_IMM)[None, :]
    p = np.exp(-elapsed) * elapsed**bins / np.vectorize(math.factorial)(bins)
    p[:, -1] = 1.0 - p[:, :-1].sum(axis=1)
    errors = np.sqrt(per_group * p * (1 - p) / replicates)
    assert np.all(np.abs(occupancy[1:].mean(axis=-1) - per_group * p) <= 5 * errors)
