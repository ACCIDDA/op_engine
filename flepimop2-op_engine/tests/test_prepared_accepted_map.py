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

"""Independent accepted-step maps for flat prepared adaptive replay."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum
from op_engine.core_solver import AdaptiveStepSchedule

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

if TYPE_CHECKING:
    from op_engine import Array


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_prepared_replay_preserves_accepted_map(backend: str, method: str) -> None:  # noqa: PLR0914
    """The accepted duration is two quarter steps, with its matching derivative."""
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["x"],
            "equations": {"x": "-rate * x"},
            "initial_state": {"x": "start"},
        }
    )
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=method, adaptive=True, rtol=1e-2, atol=1e-4, max_steps=100
        ),
    )
    params = {
        name: ParameterValue(xp.asarray(value), ResolvedShape())
        for name, value in {"start": 1.0, "rate": 1.0}.items()
    }
    times = np.asarray([0.0, 1.0])
    discovery = engine.run_adaptive(system, times, {}, params)
    mesh = AdaptiveStepSchedule((0.0, 1.0), ((0.5, 0.5),))
    schedule = replace(discovery.schedule, step_schedule=mesh)
    ordinary = engine.run_adaptive(system, times, {}, params, schedule=schedule)
    prepared = engine.prepare(system, times, {}, params, adaptive_schedule=schedule)

    def final(rate: Array) -> Array:
        return prepared({}, {"start": xp.asarray(1.0), "rate": rate})[-1, 1]

    quarter = 0.25
    factor = 1.0 - quarter
    factor_derivative = -quarter
    if method == "rk4":
        factor += quarter**2 / 2.0 - quarter**3 / 6.0 + quarter**4 / 24.0
        factor_derivative *= 1.0 - quarter + quarter**2 / 2.0 - quarter**3 / 6.0
    expected = factor**4
    expected_derivative = 4.0 * factor**3 * factor_derivative
    if backend == "jax":
        jax = pytest.importorskip("jax")
        value, derivative = jax.jit(jax.value_and_grad(final))(xp.asarray(1.0))
        np.testing.assert_allclose(derivative, expected_derivative, rtol=3e-6)
    else:
        value = final(xp.asarray(1.0))
    np.testing.assert_allclose(value, expected, rtol=2e-6)
    np.testing.assert_allclose(value, np.asarray(ordinary.trajectory)[-1, 1], rtol=2e-6)
