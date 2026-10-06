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

"""Steady states of compiled op_system models (#192).

Core op_engine has no op_system dependency, so this end-to-end check lives in
the provider suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from op_engine import (
    conserved_quantities,
    from_compiled_rhs,
    linear_invariants,
    require_steady_state,
    sink_states,
    steady_state,
)
from op_system import compile_spec
from scipy.integrate import solve_ivp

POPULATION = "sum_over(S[age:a] + I[age:a] + R[age:a], age=a)"
SPEC: dict[str, Any] = {
    "kind": "transitions",
    "axes": [{"name": "age", "coords": ["young", "old"]}],
    "state": ["S[age]", "I[age]", "R[age]", "D"],
    "transitions": [
        {
            "name": "infect",
            "from": "S[age]",
            "to": "I[age]",
            "rate": f"beta * sum_over(I[age:a], age=a) / {POPULATION}",
        },
        {"name": "recover", "from": "I[age]", "to": "R[age]", "rate": "gamma"},
        {"name": "wane", "from": "R[age]", "to": "S[age]", "rate": "omega"},
        {"name": "birth", "to": "S[age]", "rate": f"mu * {POPULATION} / 2"},
        {"name": "die_S", "from": "S[age]", "to": "D", "rate": "mu"},
        {"name": "die_I", "from": "I[age]", "to": "D", "rate": "mu"},
        {"name": "die_R", "from": "R[age]", "to": "D", "rate": "mu"},
    ],
}
PARAMS = {"beta": 0.4, "gamma": 0.1, "omega": 1 / 365, "mu": 1 / (70 * 365)}


def test_endemic_steady_state_matches_a_long_integration() -> None:
    """Detect the sink and the rate-balanced total, then solve.

    Births break every structural conservation law, so only
    :func:`linear_invariants` finds the conserved living population. The
    steady state agrees with a 200-year integration.
    """
    compiled = compile_spec(SPEC)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(compiled.eval_fn(t, y, **PARAMS))

    network = from_compiled_rhs(compiled, PARAMS)
    assert conserved_quantities(network.stoichiometry).shape == (0, 7)

    rng = np.random.default_rng(1)
    samples = [rng.uniform(10.0, 3000.0, 7) for _ in range(3)]
    sinks = sink_states(rhs, samples)
    np.testing.assert_array_equal(sinks, [False] * 6 + [True])  # D
    (living,) = linear_invariants(rhs, samples, fixed=sinks)
    np.testing.assert_allclose(
        np.abs(living), [1 / np.sqrt(6.0)] * 6 + [0.0], atol=1e-8
    )

    y0 = np.array([2495.0, 2495.0, 5.0, 5.0, 0.0, 0.0, 0.0])
    result = require_steady_state(
        steady_state(rhs, y0, fixed=sinks, invariants=living[None, :])
    )
    assert float(result.invariant_drift) < 1e-14
    long_run = solve_ivp(
        rhs, (0.0, 200 * 365.0), y0, method="LSODA", rtol=1e-10, atol=1e-8
    ).y[:, -1]
    np.testing.assert_allclose(result.state[:6], long_run[:6], rtol=1e-8)
