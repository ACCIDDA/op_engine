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


"""Real compiler/provider/solver contract for known-time state maps."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine


def _system() -> OpSystemSystem:
    """Return a real two-coordinate compiler fixture with one selected map."""
    return OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["x[unit]"],
            "axes": [{"name": "unit", "coords": ["a", "b"]}],
            "equations": {"x[unit]": "-k*x[unit]"},
            "impulses": [
                {
                    "time": 1.0,
                    "updates": [
                        {
                            "target": "x[unit=a]",
                            "operation": "add",
                            "expr": "boost[unit]",
                        }
                    ],
                }
            ],
        }
    )


def _run(
    system: OpSystemSystem,
    engine: OpEngineFlepimop2Engine,
    values: Any,  # noqa: ANN401
    *,
    replay: object = None,
) -> Any:  # noqa: ANN401
    """Return a native provider trajectory with map-only shaped parameters."""
    initial = {
        name: ParameterValue(value, ResolvedShape())
        for name, value in zip(system.option("state_names"), values[3:], strict=True)
    }
    params = {
        "k": ParameterValue(values[0], ResolvedShape()),
        "boost": ParameterValue(values[1:3], ResolvedShape(("unit",), (2,))),
    }
    return engine.run(
        system, np.asarray([0.0, 1.0, 2.0]), initial, params, adaptive_schedule=replay
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_compiled_impulses_use_declared_selector_and_post_event_state(
    backend: str, *, adaptive: bool
) -> None:
    """Native maps preserve selector semantics, endpoint outputs and derivatives."""
    if backend == "jax":
        xp = jnp
        enable = getattr(jax, "enable_x64", None)
        context = (enable if enable is not None else jax.experimental.enable_x64)()
    else:
        xp, context = np, nullcontext()
    with context:
        system = _system()
        engine = OpEngineFlepimop2Engine(
            state_change=StateChangeEnum.FLOW,
            config=OpEngineEngineConfig(
                method="dopri5",
                adaptive=adaptive,
                fixed_max_step=None if adaptive else 0.05,
                rtol=1e-9,
                atol=1e-11,
                schedule_tag="decay-selected-add-v1",
            ),
        )
        values = xp.asarray([0.5, 5.0, 8.0, 2.0, 3.0], dtype=xp.float64)
        result = _run(system, engine, values)
        expected = np.array([
            [0, 2, 3],
            [1, 2 * np.exp(-0.5) + 5, 3 * np.exp(-0.5)],
            [2, 2 * np.exp(-1) + 5 * np.exp(-0.5), 3 * np.exp(-1)],
        ])
        np.testing.assert_allclose(result, expected, rtol=1e-8)
        if adaptive:
            schedule = engine.last_adaptive_schedule
            assert schedule is not None
            np.testing.assert_allclose(
                _run(system, engine, values, replay=schedule), expected, rtol=1e-8
            )
        elif backend == "jax":
            value, grad = jax.jit(
                jax.value_and_grad(lambda p: _run(system, engine, p)[-1, 1:].sum())
            )(values)
            np.testing.assert_allclose(value, expected[-1, 1:].sum(), rtol=1e-8)
            np.testing.assert_allclose(
                grad,
                [
                    -10 * np.exp(-1) - 5 * np.exp(-0.5),
                    np.exp(-0.5),
                    0.0,
                    np.exp(-1),
                    np.exp(-1),
                ],
                rtol=1e-8,
            )
