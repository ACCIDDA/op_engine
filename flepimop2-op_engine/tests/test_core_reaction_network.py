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

"""op_engine's core reaction network from real op_system artifacts (#191).

Core op_engine has no op_system dependency, so these checks live in the
provider suite, which installs op_system.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from flepimop2.system.op_system import OpSystemSystem
from op_engine import DirectSSASolver, ModelCore, NumpySSASampler, from_compiled_rhs
from op_system import compile_spec

from flepimop2.engine.op_engine.reactions import compile_reaction_network

SPEC: dict[str, Any] = {
    "kind": "transitions",
    "axes": [{"name": "age", "coords": ["child", "adult"]}],
    "state": ["S[age]", "I[age]", "R[age]", "Z"],
    "aliases": {"lam": "beta * Z / 100"},
    "transitions": [
        {"name": "infect", "from": "S[age]", "to": "I[age]", "rate": "b * I[age]"},
        {"name": "spill", "from": "S[age]", "to": "I[age]", "rate": "lam"},
        {"name": "recover", "from": "I[age]", "to": "R[age]", "rate": "g"},
        {"name": "wane", "from": "R[age]", "to": "S[age]", "rate": "w"},
    ],
}
PARAMS = {"b": 0.002, "beta": 0.5, "g": 0.1, "w": 0.01}


def test_mean_drift_matches_the_deterministic_rhs() -> None:
    """``stoichiometry @ propensity`` is the compiled RHS for a covered model."""
    compiled = compile_spec(SPEC)
    network = from_compiled_rhs(compiled, PARAMS)
    rng = np.random.default_rng(3)
    for _ in range(5):
        state = rng.uniform(1.0, 100.0, network.n_state)
        np.testing.assert_allclose(
            network.mean_drift(0.0, state),
            np.asarray(compiled.eval_fn(0.0, state, **PARAMS)),
            rtol=1e-12,
        )


def test_provider_and_core_build_the_same_network() -> None:
    """The flepimop2 shim compiles exactly what the core function does."""
    compiled = compile_spec(SPEC)
    core = from_compiled_rhs(compiled, PARAMS)
    provider = compile_reaction_network(
        OpSystemSystem(spec=SPEC), PARAMS, n_state=core.n_state
    )
    np.testing.assert_array_equal(provider.stoichiometry, core.stoichiometry)
    np.testing.assert_array_equal(
        provider.reactant_stoichiometry, core.reactant_stoichiometry
    )
    assert provider.channel_names == core.channel_names
    assert provider.incomplete_reactions == core.incomplete_reactions


def test_direct_ssa_runs_from_a_spec_without_flepimop2() -> None:
    """A spec, op_system, and op_engine core are enough to simulate."""
    compiled = compile_spec(SPEC)
    network = from_compiled_rhs(compiled, PARAMS)
    times = np.linspace(0.0, 20.0, 5)
    core = ModelCore(network.n_state, 1, times)
    initial = np.asarray([90.0, 80.0, 5.0, 5.0, 0.0, 0.0, 10.0])
    core.set_initial_state(initial[:, None])
    DirectSSASolver(core, network.stoichiometry).run(
        network.propensity, NumpySSASampler(seed=7)
    )
    final = np.asarray(core.get_current_state())[:, 0]
    assert np.all(final >= 0.0)
    np.testing.assert_array_equal(final, np.round(final))
    # Every channel moves one unit between S, I, and R; Z only catalyzes.
    assert final[:6].sum() == initial[:6].sum()
    assert final[6] == initial[6]


def test_reaction_gaps_from_op_system_are_refused() -> None:
    """An unnamed transition has no reaction, so the network would drop it."""
    spec = {
        **SPEC,
        "transitions": [*SPEC["transitions"], {"from": "Z", "to": "Z", "rate": "0"}],
    }
    compiled = compile_spec(spec)
    assert compiled.reaction_gaps
    with pytest.raises(ValueError, match="without a reaction artifact"):
        from_compiled_rhs(compiled, PARAMS)
