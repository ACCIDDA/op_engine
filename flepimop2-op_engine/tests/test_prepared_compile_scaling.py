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

"""The traced program of a prepared solve must not grow with the state size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import (
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    _gather_shaped_run,  # noqa: PLC2701
)

if TYPE_CHECKING:
    from collections.abc import Iterator

N_AGE = 3
N_TIME = 5


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    """Run in float64 so round-off comparisons are meaningful, then restore."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    yield
    jax.config.update("jax_enable_x64", previous)


def _system(n_loc: int) -> OpSystemSystem:
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "time_interpolation": "previous",
            "axes": [
                {"name": "loc", "coords": [f"l{i}" for i in range(n_loc)]},
                {"name": "age", "coords": [f"a{i}" for i in range(N_AGE)]},
                {
                    "name": "time",
                    "type": "continuous",
                    "domain": {"lb": 0.0, "ub": float(N_TIME - 1)},
                    "size": N_TIME,
                },
            ],
            "state": ["S[loc,age]", "I[loc,age]", "R[loc,age]", "Z"],
            "aliases": {
                "force[loc]": (
                    "apply_along(I[loc,age:a], age=a, kernel=sum) / population[loc]"
                ),
            },
            "transitions": [
                {
                    "name": "infection",
                    "from": "S[loc,age]",
                    "to": "I[loc,age]",
                    "rate": "beta[time,loc] * force[loc]",
                },
                {
                    "name": "recovery",
                    "from": "I[loc,age]",
                    "to": "R[loc,age]",
                    "rate": "gamma",
                },
            ],
            "initial_state": {
                "S[loc,age]": {"shaped": "s_init", "axes": ["loc", "age"]},
                "I[loc,age]": {"shaped": "i_init", "axes": ["loc", "age"]},
                "R[loc,age]": "r_start",
                "Z": "z_start",
            },
        }
    )


def _inputs(n_loc: int, backend: str = "jax") -> dict[str, ParameterValue]:
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(n_loc)
    shape = ResolvedShape(("loc", "age"), (n_loc, N_AGE))
    population = rng.uniform(1e3, 1e4, n_loc)

    def pv(value: object, axes: tuple[str, ...] = ()) -> ParameterValue:
        array = xp.asarray(value, dtype=xp.float64)
        return ParameterValue(array, ResolvedShape(axes, tuple(array.shape)))

    infected = rng.uniform(1.0, 5.0, (n_loc, N_AGE))
    return {
        "gamma": pv(0.2),
        "z_start": pv(0.0),
        "r_start": pv(0.0),
        "population": pv(population, ("loc",)),
        "beta": pv(rng.uniform(0.2, 0.6, (N_TIME, n_loc)), ("time", "loc")),
        "s_init": ParameterValue(
            xp.asarray(population[:, None] / N_AGE - infected), shape
        ),
        "i_init": ParameterValue(xp.asarray(infected), shape),
    }


def _engine() -> OpEngineFlepimop2Engine:
    return OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method="rk4", fixed_max_step=0.5),
    )


def _count_equations(jaxpr: Any) -> int:  # noqa: ANN401
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for param in equation.params.values():
            for item in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    total += _count_equations(inner)
    return total


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    "cells",
    [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 0)],
        [(1, 2), (0, 0), (1, 1)],
        [(1, 1), (1, 1), (0, 2)],
        [(0, 0)],
    ],
)
def test_gather_shaped_run_matches_cellwise_indexing(
    backend: str, cells: list[tuple[int, ...]]
) -> None:
    """One slice or gather returns the same cells as indexing each one."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    values = np.arange(6.0).reshape(2, 3) * 1.5 + 0.25
    gathered = _gather_shaped_run(xp.asarray(values), cells, xp=xp, dtype=xp.float64)
    assert gathered is not None
    expected = np.asarray([values[cell] for cell in cells])
    np.testing.assert_array_equal(np.asarray(gathered), expected)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    "cells", [[(0,)], [(0, 0, 0)], [(2, 0)], [(0, 3)], [(-1, 0)], []]
)
def test_gather_shaped_run_declines_malformed_cells(
    backend: str, cells: list[tuple[int, ...]]
) -> None:
    """Indices that are not scalar elements fall back to the per-cell errors."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    values = xp.zeros((2, 3))
    assert _gather_shaped_run(values, cells, xp=xp, dtype=xp.float64) is None


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_initial_state_assembly_matches_cellwise_reference(backend: str) -> None:
    """Shaped, literal and parameter-valued cells keep their declared order."""
    n_loc = 4
    params = _inputs(n_loc, backend)
    system = _system(n_loc)
    result = _engine().run(system, np.asarray([0.0, 0.5]), {}, params, model_state=None)
    expected = np.concatenate([
        np.asarray(params["s_init"].value).reshape(-1),
        np.asarray(params["i_init"].value).reshape(-1),
        np.zeros(n_loc * N_AGE),
        np.zeros(1),
    ])
    np.testing.assert_array_equal(np.asarray(result)[0, 1:], expected)


def test_prepared_traced_program_does_not_grow_with_locations() -> None:
    """The jaxpr of a prepared solve is the same size for every number of blocks."""
    jax = pytest.importorskip("jax")
    sizes = []
    for n_loc in (2, 9):
        params = _inputs(n_loc)
        prepared = _engine().prepare(
            _system(n_loc), np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), {}, params
        )
        raw = {name: value.value for name, value in params.items()}
        jaxpr = jax.make_jaxpr(prepared.__call__)({}, raw)
        sizes.append(_count_equations(jaxpr))
    assert sizes[0] == sizes[1]


def test_prepared_solve_matches_run_and_vmaps_over_initial_state() -> None:
    """Prepared and vmapped solves agree with the ordinary run to round-off."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    n_loc = 5
    times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    params = _inputs(n_loc)
    engine = _engine()
    system = _system(n_loc)
    reference = np.asarray(engine.run(system, times, {}, _inputs(n_loc, "numpy")))
    prepared = engine.prepare(system, times, {}, params)
    raw = {name: value.value for name, value in params.items()}
    solved = np.asarray(jax.jit(lambda p: prepared({}, p))(raw))
    np.testing.assert_allclose(solved, reference, rtol=1e-12, atol=1e-12)

    def shifted(shift: Any) -> Any:  # noqa: ANN401
        moved = raw["s_init"] * shift
        return prepared(
            {},
            {**raw, "s_init": raw["s_init"] - moved, "i_init": raw["i_init"] + moved},
        )

    shifts = jnp.asarray([0.0, 0.01, 0.02])
    batch = np.asarray(jax.jit(jax.vmap(shifted))(shifts))
    for index, shift in enumerate(np.asarray(shifts)):
        np.testing.assert_allclose(
            batch[index], np.asarray(jax.jit(shifted)(shift)), rtol=1e-12, atol=1e-12
        )
    np.testing.assert_allclose(batch[0], reference, rtol=1e-12, atol=1e-12)
