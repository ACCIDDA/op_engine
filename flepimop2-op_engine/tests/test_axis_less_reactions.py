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

"""Reactions between templates with different axes (#184)."""

from __future__ import annotations

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine
from flepimop2.engine.op_engine.reactions import compile_reaction_network

IMM = {"name": "imm", "coords": ["x0", "x1", "x2"]}

SIR: dict[str, object] = {
    "kind": "transitions",
    "state": ["S", "I", "R"],
    "transitions": [
        {
            "name": "infect",
            "from": "S",
            "to": "I",
            "rate": "beta * I / (S + I + R)",
        },
        {"name": "recover", "from": "I", "to": "R", "rate": "gamma"},
    ],
}


@pytest.mark.parametrize(
    ("spec", "params", "stoichiometry"),
    [
        pytest.param(
            SIR,
            {"beta": 0.5, "gamma": 0.2},
            [[-1, 0], [1, -1], [0, 1]],
            id="axis-less-sir",
        ),
        pytest.param(
            {
                "kind": "transitions",
                "axes": [IMM],
                "state": ["I", "X[imm]"],
                "transitions": [
                    {"name": "seed", "from": "I", "to": "X[imm=x1]", "rate": "r"}
                ],
            },
            {"r": 0.3},
            [[-1], [0], [1], [0]],
            id="scalar-to-pinned-cell",
        ),
        pytest.param(
            {
                "kind": "transitions",
                "axes": [IMM],
                "state": ["X[imm]", "R"],
                "transitions": [
                    {"name": "collapse", "from": "X[imm]", "to": "R", "rate": "r"}
                ],
            },
            {"r": 0.3},
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1]],
            id="templated-to-scalar",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_mixed_template_reactions_reconstruct_producer_drift(
    spec: dict[str, object],
    params: dict[str, float],
    stoichiometry: list[list[int]],
    backend: str,
) -> None:
    """Each target is indexed with its own axes, so drift matches the RHS."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = OpSystemSystem(spec=spec)
    n_state = len(system.option("state_names"))
    bound = {name: xp.asarray(value) for name, value in params.items()}
    network = compile_reaction_network(system, bound, n_state=n_state)
    np.testing.assert_array_equal(network.stoichiometry, stoichiometry)
    rng = np.random.default_rng(184)
    for _ in range(3):
        state = xp.asarray(rng.integers(1, 50, size=n_state).astype(float))
        drift = network.mean_drift(0.0, state)
        assert drift.__array_namespace__() is xp
        np.testing.assert_allclose(
            np.asarray(drift),
            np.asarray(system.step(0.0, state, **bound)),
            rtol=1e-6,
        )


class _WithoutTargetAxes:
    """A reaction artifact from an op_system release before to_full_axes."""

    def __init__(self, reaction: object) -> None:
        self._reaction = reaction

    def __getattr__(self, name: str) -> object:
        if name == "to_full_axes":
            raise AttributeError(name)
        return getattr(self._reaction, name)


def test_reactions_without_target_axes_compile_unchanged() -> None:
    """Older artifacts share one axis order between source and target."""
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [IMM],
            "state": ["S[imm]", "I[imm]"],
            "transitions": [
                {"name": "infect", "from": "S[imm]", "to": "I[imm=x0]", "rate": "b"}
            ],
        }
    )
    params = {"b": np.asarray(0.5)}
    current = compile_reaction_network(system, params, n_state=6)
    system.options["reactions"] = tuple(
        _WithoutTargetAxes(r) for r in system.option("reactions")
    )
    legacy = compile_reaction_network(system, params, n_state=6)
    np.testing.assert_array_equal(legacy.stoichiometry, current.stoichiometry)


def _run(
    spec: dict[str, object],
    values: list[float],
    params: dict[str, float],
    times: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Run one direct-SSA path of a scalar model.

    Returns:
        The trajectory without its time column.
    """
    system = OpSystemSystem(spec=spec)
    initial = {
        str(name): ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in zip(system.option("state_names"), values, strict=True)
    }
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            mode="stochastic", stochastic_method="direct-ssa", random_seed=seed
        ),
    )
    bound = {
        k: ParameterValue(np.asarray(v), ResolvedShape()) for k, v in params.items()
    }
    history = np.asarray(engine.run(system, times, initial, bound))
    np.testing.assert_array_equal(history[:, 0], times)
    return history[:, 1:]


def test_axis_less_pure_death_matches_binomial_survival() -> None:
    """Survivors of an axis-less death process are Binomial(X0, exp(-mu t))."""
    spec: dict[str, object] = {
        "kind": "transitions",
        "state": ["X", "D"],
        "transitions": [{"name": "die", "from": "X", "to": "D", "rate": "mu"}],
    }
    survivors0, mu, runs = 20, 0.5, 400
    times = np.asarray([0.0, 0.5, 1.0, 2.0])
    paths = np.stack([
        _run(spec, [float(survivors0), 0.0], {"mu": mu}, times, seed)
        for seed in range(runs)
    ])
    np.testing.assert_array_equal(paths.sum(axis=-1), survivors0)
    survivors = paths[:, 1:, 0]
    p = np.exp(-mu * times[1:])
    mean = survivors0 * p
    variance = survivors0 * p * (1 - p)
    assert np.all(np.abs(survivors.mean(axis=0) - mean) <= 5 * np.sqrt(variance / runs))
    # Sampling error of the unbiased variance from the binomial fourth
    # central moment n p q (1 + 3 (n - 2) p q).
    fourth = survivors0 * p * (1 - p) * (1 + 3 * (survivors0 - 2) * p * (1 - p))
    variance_errors = np.sqrt((fourth - (runs - 3) / (runs - 1) * variance**2) / runs)
    assert np.all(
        np.abs(survivors.var(axis=0, ddof=1) - variance) <= 5 * variance_errors
    )


def test_axis_less_sir_runs_in_direct_ssa() -> None:
    """The textbook CTMC runs and conserves its integer population."""
    history = _run(
        SIR, [95.0, 5.0, 0.0], {"beta": 1.5, "gamma": 0.5}, np.linspace(0, 5, 6), 184
    )
    np.testing.assert_array_equal(history.sum(axis=1), 100)
    np.testing.assert_array_equal(history, np.floor(history))
    assert history[-1, 2] > 0
