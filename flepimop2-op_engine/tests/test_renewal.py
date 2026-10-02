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

from typing import TYPE_CHECKING

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum
from op_engine import SSASample

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine
from flepimop2.engine.op_engine.reactions import compile_reaction_network

if TYPE_CHECKING:
    from flepimop2.typing import Array
    from numpy.typing import NDArray


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


def _population_parameters(
    fertility: float, *, aging: float = 0.25
) -> dict[str, ParameterValue]:
    """Bind age-indexed fertility and scalar departure/aging coefficients.

    Returns:
        Provider parameters with their declared shapes.
    """
    return {
        "B": ParameterValue(np.full(3, fertility), ResolvedShape(("age",), (3,))),
        "mu": ParameterValue(np.asarray(0.25), ResolvedShape()),
        "aging": ParameterValue(np.asarray(aging), ResolvedShape()),
    }


def _ensemble_history(
    system: OpSystemSystem,
    population: NDArray[np.float64],
    params: dict[str, ParameterValue],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Run independently seeded direct-SSA paths across disjoint group channels.

    Returns:
        History shaped as (time, state template, age, independent replicate).
    """
    values = np.concatenate([population.ravel(), np.zeros(2 * population.size)])
    initial = {
        str(name): ParameterValue(np.asarray(value), ResolvedShape())
        for name, value in zip(system.option("state_names"), values, strict=True)
    }
    histories = []
    for seed in range(178, 194):
        engine = OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=OpEngineEngineConfig(
                mode="stochastic", stochastic_method="direct-ssa", random_seed=seed
            ),
        )
        history = np.asarray(engine.run(system, times, initial, params))
        np.testing.assert_array_equal(history[:, 0], times)
        histories.append(history[:, 1:].reshape(times.size, 3, 3, population.shape[1]))
    return np.concatenate(histories, axis=-1)


@pytest.mark.parametrize(
    ("aging", "profile"),
    [
        pytest.param(0.25, (10, 5, 5), id="exponential-fitted"),
        pytest.param(0.5, (3, 2, 4), id="upwind"),
    ],
)
def test_stationary_living_age_means_and_critical_total_variance(
    aging: float, profile: tuple[int, int, int]
) -> None:
    """Balanced renewal preserves the mean profile while total counts fluctuate."""
    system = _renewal_system(64, demography=True)
    initial = np.asarray(profile, dtype=float)
    q = aging / (aging + 0.25)
    np.testing.assert_allclose(initial / initial.sum(), [1 - q, q * (1 - q), q**2])
    times = np.asarray([0.0, 0.25, 0.5, 1.0])
    history = _ensemble_history(
        system,
        np.broadcast_to(initial[:, None], (3, 64)),
        _population_parameters(0.25, aging=aging),
        times,
    )
    assert np.all(history >= 0)
    np.testing.assert_array_equal(history, np.floor(history))
    np.testing.assert_array_equal(history[:, 1], 0)
    living = history[:, 0]
    np.testing.assert_array_equal(
        living[0], np.broadcast_to(initial[:, None], living[0].shape)
    )
    replicates = living.shape[-1]
    # A fixed integer initial profile is stationary in expectation, not a
    # stationary joint distribution. Each group evolves as its own process.
    standard_errors = np.sqrt(living[1:].var(axis=-1, ddof=1) / replicates)
    assert np.all(np.abs(living[1:].mean(axis=-1) - initial) <= 5 * standard_errors)

    total = living.sum(axis=1)[1:]
    elapsed = 0.25 * times[1:]
    variance = 2 * initial.sum() * elapsed
    assert np.all(
        np.abs(total.mean(axis=-1) - initial.sum())
        <= 5 * np.sqrt(variance / replicates)
    )
    # For the critical linear birth-death total, E[(X-E[X])**4] is
    # V + 3*V**2 + 24*X0*(mu*t)**3. This determines the sampling error
    # of the unbiased sample variance independently of the solver.
    fourth_moment = variance + 3 * variance**2 + 24 * initial.sum() * elapsed**3
    variance_errors = np.sqrt(
        (fourth_moment - (replicates - 3) / (replicates - 1) * variance**2) / replicates
    )
    assert np.all(np.abs(total.var(axis=-1, ddof=1) - variance) <= 5 * variance_errors)

    departed = history[:, 2]
    assert np.all(np.diff(departed, axis=0) >= 0)
    departure_errors = np.sqrt(departed[1:].var(axis=-1, ddof=1) / replicates)
    assert np.all(
        np.abs(departed[1:].mean(axis=-1) - elapsed[:, None] * initial)
        <= 5 * departure_errors
    )


def test_births_only_increase_total_with_yule_mean_and_variance() -> None:
    """A population-dependent birth flux grows only the youngest living bin."""
    system = _renewal_system(64)
    initial = np.asarray([2.0, 3.0, 5.0])
    times = np.asarray([0.0, 0.25, 0.5, 1.0])
    history = _ensemble_history(
        system,
        np.broadcast_to(initial[:, None], (3, 64)),
        _population_parameters(0.3),
        times,
    )
    np.testing.assert_array_equal(history[:, 1:], 0)
    np.testing.assert_array_equal(history, np.floor(history))
    living = history[:, 0]
    assert np.all(np.diff(living[:, 0], axis=0) >= 0)
    np.testing.assert_array_equal(
        living[:, 1:], np.broadcast_to(initial[None, 1:, None], living[:, 1:].shape)
    )
    total = living.sum(axis=1)[1:]
    replicates = total.shape[-1]
    growth = np.exp(0.3 * times[1:])
    mean = initial.sum() * growth
    variance = initial.sum() * growth * (growth - 1)
    assert np.all(
        np.abs(total.mean(axis=-1) - mean) <= 5 * np.sqrt(variance / replicates)
    )
    # A Yule total is the sum of X0 geometric counts. Its fourth central
    # moment yields a fixed analytic error bar for the variance estimate.
    fourth_moment = 3 * variance**2 + variance * (6 * growth**2 - 6 * growth + 1)
    variance_errors = np.sqrt(
        (fourth_moment - (replicates - 3) / (replicates - 1) * variance**2) / replicates
    )
    assert np.all(np.abs(total.var(axis=-1, ddof=1) - variance) <= 5 * variance_errors)
    assert total[-1].mean() > initial.sum()


@pytest.mark.parametrize(
    ("n_groups", "age_first"), [(None, True), (2, True), (2, False)]
)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_prescribed_birth_aging_and_departure_fire_into_the_selected_cells(
    n_groups: int | None, *, age_first: bool, backend: str
) -> None:
    """A birth can fill an empty youngest bin before aging and departure."""
    system = _renewal_system(n_groups, age_first=age_first, demography=True)
    groups = n_groups or 1
    group = groups - 1
    stride = groups if age_first else 1
    youngest = group if age_first else 3 * group
    values = np.zeros(9 * groups)
    values[youngest + stride] = 10
    values[3 * groups + youngest + 2 * stride] = 3
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    initial = {
        str(name): ParameterValue(xp.asarray(value), ResolvedShape())
        for name, value in zip(system.option("state_names"), values, strict=True)
    }
    params = _population_parameters(0.1)
    params["B"] = ParameterValue(
        xp.asarray([0.1, 0.2, 0.3]), ResolvedShape(("age",), (3,))
    )
    choices = (group, 4 * groups + group, groups + youngest + stride, group)
    draws: list[int] = []
    rates: list[float] = []

    def sampler(rate: Array, probabilities: Array, index: int, /) -> SSASample:
        """Record each global draw and select the prescribed live channel.

        Returns:
            Waiting time and channel index in the state's namespace.
        """
        draws.append(index)
        rates.append(float(rate.item()))
        assert rate.__array_namespace__() is xp
        assert np.asarray(probabilities).ravel()[choices[index]] > 0
        return SSASample(
            xp.asarray(0.2 if index < 3 else 2.0, dtype=rate.dtype),
            xp.asarray(choices[index], dtype=xp.int32),
        )

    times = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    history = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(mode="stochastic", stochastic_method="direct-ssa"),
    ).run(system, times, initial, params, ssa_sampler=sampler)
    assert history.__array_namespace__() is xp
    expected = np.broadcast_to(values, (5, values.size)).copy()
    expected[1, youngest] += 1
    expected[2, youngest + stride] += 1
    expected[3:, 6 * groups + youngest + stride] += 1
    np.testing.assert_array_equal(np.asarray(history)[:, 1:], expected)
    np.testing.assert_array_equal(np.asarray(history)[:, 0], times)
    assert draws == [0, 1, 2, 3]
    np.testing.assert_allclose(rates, [7.9, 8.5, 8.6, 7.9], rtol=1e-6)
