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

"""Actual compiler tables through ordinary, prepared and adaptive execution."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

if TYPE_CHECKING:
    from op_engine import Array


def _system(policy: str = "previous") -> OpSystemSystem:
    return OpSystemSystem(
        spec={
            "kind": "transitions",
            "time_interpolation": policy,
            "axes": [{"name": "time", "type": "continuous", "coords": [0.0, 0.7, 1.0]}],
            "state": ["x", "z"],
            "initial_state": {"x": "start", "z": "zero"},
            "transitions": [
                {"name": "transfer", "from": "x", "to": "z", "rate": "rate[time]"}
            ],
        }
    )


def _inputs(
    backend: str, dtype: str = "float32"
) -> tuple[dict[str, ParameterValue], dict[str, ParameterValue]]:
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    return {}, {
        "zero": ParameterValue(
            xp.asarray(0.0, dtype=getattr(xp, dtype)), ResolvedShape()
        ),
        "start": ParameterValue(
            xp.asarray(1.0, dtype=getattr(xp, dtype)), ResolvedShape()
        ),
        "rate": ParameterValue(
            xp.asarray([1.0, 0.0, 2.0], dtype=getattr(xp, dtype)),
            ResolvedShape(("time",), (3,)),
        ),
    }


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("method", ["rk4", "dopri5"])
@pytest.mark.parametrize("layout", ["flat", "pytree"])
@pytest.mark.parametrize("prepared", [False, True])
def test_real_previous_table_fixed_and_prepared(
    backend: str, method: str, layout: str, *, prepared: bool
) -> None:
    """A float32 outgoing coordinate below float64 0.7 still changes regimes."""
    system = _system()
    initial, params = _inputs(backend)
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=method,
            state_layout=layout,
            fixed_max_step=0.05,
            forcing_breakpoints=(0.5,),
        ),
    )
    times = np.asarray([0.0, 0.9, 1.5])
    if prepared:
        executor = engine.prepare(system, times, initial, params)
        run = pytest.importorskip("jax").jit(executor) if backend == "jax" else executor
        result = run({}, {key: value.value for key, value in params.items()})
    else:
        result = engine.run(system, times, initial, params)
    expected_x = np.exp(-np.asarray([0.0, 0.7, 1.7]))
    np.testing.assert_allclose(np.asarray(result)[:, 1], expected_x, rtol=3e-6)
    np.testing.assert_allclose(np.asarray(result)[:, 1:].sum(axis=1), 1.0, rtol=1e-6)
    assert np.asarray(result).shape == (3, 3)
    assert engine.config.forcing_breakpoints == (0.5,)


@pytest.mark.parametrize(
    ("backend", "layout"), [("numpy", "flat"), ("jax", "flat"), ("jax", "pytree")]
)
@pytest.mark.parametrize("method", ["rk4", "dopri5"])
def test_real_previous_table_adaptive_and_prepared_replay(
    backend: str, method: str, layout: str
) -> None:
    """The producer/config union survives discovery, replay and preparation."""
    initial, params = _inputs(backend)
    system = _system()
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method=method,
            state_layout=layout,
            adaptive=True,
            rtol=2e-5,
            atol=1e-7,
            max_steps=200,
            forcing_breakpoints=(0.5,),
        ),
    )
    times = np.asarray([0.0, 1.5])
    discovered = engine.run_adaptive(system, times, initial, params)
    assert discovered.schedule.forcing_breakpoints == (0.5, 0.7, 1.0)
    replay = engine.run_adaptive(
        system, times, initial, params, schedule=discovered.schedule
    )
    executor = engine.prepare(
        system, times, initial, params, adaptive_schedule=discovered.schedule
    )
    run = pytest.importorskip("jax").jit(executor) if backend == "jax" else executor
    prepared = run({}, {key: value.value for key, value in params.items()})
    for trajectory in (discovered.trajectory, replay.trajectory, prepared):
        np.testing.assert_allclose(
            np.asarray(trajectory)[-1, 1], math.exp(-1.7), rtol=1e-4
        )
    changed = replace(discovered.schedule, forcing_breakpoints=(0.7, 1.0))
    with pytest.raises(ValueError, match="forcing_breakpoints changed"):
        engine.run_adaptive(system, times, initial, params, schedule=changed)


@pytest.mark.parametrize("method", ["imex-euler", "bdf2", "sdirk2"])
def test_producer_jumps_reject_unsupported_deterministic_methods(method: str) -> None:
    """Validation and execution consistently reject unsupported history policies."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW, config=OpEngineEngineConfig(method=method)
    )
    system = _system()
    issues = engine.validate_system(system)
    assert issues is not None
    assert "invalid_forcing" in [issue.kind for issue in issues]
    initial, params = _inputs("numpy")
    with pytest.raises(ValueError, match="forcing_breakpoints require"):
        engine.run(system, np.asarray([0.0, 1.5]), initial, params)


def test_deterministic_producer_metadata_is_validated() -> None:
    """A deterministic path can no longer silently ignore malformed declarations."""
    system = _system()
    system.options["forcing_breakpoints"] = (1.0, 0.7)
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW)
    initial, params = _inputs("numpy")
    with pytest.raises(ValueError, match=r"Invalid system.option"):
        engine.run(system, np.asarray([0.0, 1.5]), initial, params)


def test_prepared_previous_table_parameter_gradient() -> None:
    """JIT/grad retains table-value sensitivity across both declared jumps."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method="dopri5", fixed_max_step=0.05),
    )
    initial, params = _inputs("jax")
    prepared = engine.prepare(_system(), np.asarray([0.0, 1.5]), initial, params)

    def final(rates: Array) -> Array:
        return prepared(
            {}, {"start": jnp.asarray(1.0), "zero": jnp.asarray(0.0), "rate": rates}
        )[-1, 1]

    value, gradient = jax.jit(jax.value_and_grad(final))(jnp.asarray([1.0, 0.0, 2.0]))
    expected = math.exp(-1.7)
    np.testing.assert_allclose(value, expected, rtol=3e-6)
    np.testing.assert_allclose(
        gradient, -expected * np.asarray([0.7, 0.3, 0.5]), rtol=4e-6
    )


def test_numpy_prepared_executor_explains_eager_jax_arrays() -> None:
    """Eager JAX arrays on a NumPy-prepared plan get an actionable error."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method="rk4", fixed_max_step=0.05),
    )
    initial, params = _inputs("numpy")
    prepared = engine.prepare(_system(), np.asarray([0.0, 1.5]), initial, params)

    def final(rates: Array) -> Array:
        return prepared(
            {}, {"start": jnp.asarray(1.0), "zero": jnp.asarray(0.0), "rate": rates}
        )[-1, 1]

    rates = jnp.asarray([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match=r"prepared with non-JAX") as excinfo:
        final(rates)
    message = str(excinfo.value)
    for expected in ("NumPy", "inside a JAX", "jax.jit", "prepare it with JAX"):
        assert expected in message
    assert "'rate'" in message

    expected_final = math.exp(-1.7)
    np.testing.assert_allclose(jax.jit(final)(rates), expected_final, rtol=1e-4)
    numpy_rates = {k: v.value for k, v in params.items()}
    eager = prepared({}, numpy_rates)[-1, 1]
    np.testing.assert_allclose(eager, expected_final, rtol=1e-4)

    jax_initial, jax_params = _inputs("jax")
    jax_prepared = engine.prepare(
        _system(), np.asarray([0.0, 1.5]), jax_initial, jax_params
    )
    with pytest.raises(ValueError, match=r"Mismatched params entries"):
        jax_prepared({}, numpy_rates)


@pytest.mark.parametrize("adaptive", [False, True])
def test_block_forcing_and_prepared_execution(*, adaptive: bool) -> None:
    """A held table with distinct block rates retains each independent regime."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "time_interpolation": "previous",
            "axes": [
                {"name": "group", "coords": ["a", "b"]},
                {"name": "time", "type": "continuous", "coords": [0.0, 0.7, 1.0]},
            ],
            "state": ["x[group]"],
            "equations": {"x[group]": "-rate[time, group] * x[group]"},
            "initial_state": {"x[group]": {"shaped": "start", "axes": ["group"]}},
            "factorize_axes": ["group"],
        }
    )
    params = {
        "start": ParameterValue(
            jnp.asarray([1.0, 2.0]), ResolvedShape(("group",), (2,))
        ),
        "rate": ParameterValue(
            jnp.asarray([[1.0, 2.0], [0.0, 0.0], [2.0, 4.0]]),
            ResolvedShape(("time", "group"), (3, 2)),
        ),
    }
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method="dopri5",
            state_layout="block",
            block_axis="group",
            adaptive=adaptive,
            fixed_max_step=None if adaptive else 0.05,
            rtol=1e-5,
            atol=1e-7,
            max_steps=200,
        ),
    )
    times = np.asarray([0.0, 1.5])
    if adaptive:
        discovered = engine.run_adaptive(system, times, {}, params)
        schedule = discovered.schedule
        result = engine.run_adaptive(
            system, times, {}, params, schedule=schedule
        ).trajectory
    else:
        schedule = None
        result = engine.run(system, times, {}, params)
    prepared = engine.prepare(system, times, {}, params, adaptive_schedule=schedule)
    output = jax.jit(prepared)({}, {key: value.value for key, value in params.items()})
    expected = np.asarray([1.0, 2.0]) * np.exp(-np.asarray([1.7, 3.4]))
    np.testing.assert_allclose(np.asarray(result)[-1, 1:], expected, rtol=8e-5)
    np.testing.assert_allclose(np.asarray(output)[-1, 1:], expected, rtol=8e-5)


def test_run_local_producer_union_is_not_retained_by_reused_engine() -> None:
    """A subsequent linear producer cannot inherit an earlier held-table mesh."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method="dopri5", adaptive=True, rtol=1e-7, atol=1e-9
        ),
    )
    initial, params = _inputs("numpy")
    times = np.asarray([0.0, 0.7, 1.0, 1.5])
    previous = engine.run_adaptive(_system(), times, initial, params)
    linear = engine.run_adaptive(_system("linear"), times, initial, params)
    assert previous.schedule.forcing_breakpoints == (0.7, 1.0)
    assert linear.schedule.forcing_breakpoints == ()
    assert engine.config.forcing_breakpoints == ()
    np.testing.assert_allclose(
        np.asarray(linear.trajectory)[-1, 1], math.exp(-1.65), rtol=2e-6
    )


def test_long_fixed_prepared_plan_preserves_declared_boundary() -> None:
    """Preparing fine fixed steps cannot spuriously reject their forcing grid."""
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(
            method="rk4", fixed_max_step=0.001, forcing_breakpoints=(365.0,)
        ),
    )
    initial, params = _inputs("numpy")
    prepared = engine.prepare(_system(), np.asarray([364.0, 366.0]), initial, params)
    assert prepared.internal_step_count == 2000
