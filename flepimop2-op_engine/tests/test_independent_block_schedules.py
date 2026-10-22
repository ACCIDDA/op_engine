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

"""Independent compiler-proven block discovery, replay and differentiation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flepimop2.axis import Axis, AxisCollection, ResolvedShape
from flepimop2.parameter.abc import ParameterValue
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum
from op_engine.core_solver import AdaptiveStepSchedule

from flepimop2.engine.op_engine import (
    AdaptiveScheduleAccuracyError,
    IndependentBlockSchedule,
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    SolverMethod,
    StateLayout,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from flepimop2.typing import Array


def _case(count: int = 3) -> tuple[OpSystemSystem, AxisCollection]:
    """Resolve a non-leading block axis and differently shaped state leaves.

    Returns:
        Compiler-backed provider and its axis collection.
    """
    labels = tuple(str(index) for index in range(count))
    axes = AxisCollection({
        "age": Axis(name="age", kind="categorical", size=2, labels=("a", "b")),
        "loc": Axis(name="loc", kind="categorical", size=count, labels=labels),
    })
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [
                {"name": "age", "coords": ["a", "b"]},
                {"name": "loc", "coords": list(labels)},
            ],
            "state": ["X[age, loc]", "Y[loc]"],
            "equations": {
                "X[age, loc]": "-(shared + rate[loc]) * X[age, loc]",
                "Y[loc]": "-rate[loc] * Y[loc]",
            },
            "initial_state": {
                "X[age, loc]": {"shaped": "x0", "axes": ["age", "loc"]},
                "Y[loc]": {"shaped": "y0", "axes": ["loc"]},
            },
            "factorize_axes": ["loc"],
        }
    )
    return system, axes


def _engine(**options: object) -> OpEngineFlepimop2Engine:
    """Use a bounded CPU qualification configuration.

    Returns:
        An adaptive explicit engine with the independent policy.
    """
    return OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(**{
            "method": SolverMethod.DOPRI5,
            "adaptive": True,
            "state_layout": StateLayout.BLOCK,
            "block_axis": "loc",
            "block_schedule": "independent",
            "rtol": 2e-6,
            "atol": 1e-8,
            "max_steps": 512,
            **options,
        }),
    )


def _params(
    axes: AxisCollection, initial: Array, rates: Array, shared: Array
) -> dict[str, ParameterValue]:
    """Preserve global scalars and the declared runtime parameter axes.

    Returns:
        Ordinary provider inputs, including a shaped initial-condition parameter.
    """
    return {
        "x0": ParameterValue(initial, axes.resolve_shape(("age", "loc"))),
        "y0": ParameterValue(
            jnp.ones(rates.shape, dtype=rates.dtype), axes.resolve_shape(("loc",))
        ),
        "rate": ParameterValue(rates, axes.resolve_shape(("loc",))),
        "shared": ParameterValue(shared, ResolvedShape()),
    }


@pytest.fixture(params=("float32", "float64"))
def precision(
    request: pytest.FixtureRequest,
) -> Iterator[type[np.float32 | np.float64]]:
    """Qualify both declared real dtypes without leaking a JAX global setting.

    Yields:
        The requested dtype while its JAX precision mode is active.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    try:
        yield np.float64 if request.param == "float64" else np.float32
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    ("batching", "batch_size"), [("length", None), ("length", 1), ("serial", None)]
)
def test_heterogeneous_blocks_match_analytic_reference_and_gradients(  # noqa: PLR0914
    batching: str,
    batch_size: int | None,
    precision: type[np.float32 | np.float64],
) -> None:
    """Shared/local/initial-condition derivatives use each block's own mesh."""
    system, axes = _case()
    engine = _engine(block_batching=batching, block_batch_size=batch_size)
    times = np.asarray([0.0, 0.25, 0.7, 1.0])
    rates = jnp.asarray([0.2, 30.0, 0.2], dtype=precision)
    shared = jnp.asarray(0.15, dtype=precision)
    initial = jnp.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=precision)
    params = _params(axes, initial, rates, shared)
    discovered = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    )
    schedule = discovered.schedule
    assert discovered.trajectory.dtype == np.dtype(precision)
    assert isinstance(schedule, IndependentBlockSchedule)
    assert schedule.step_counts[0] == schedule.step_counts[2] < schedule.step_counts[1]
    replayed = engine.run_adaptive(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        schedule=schedule,
    )
    assert replayed.trajectory.dtype == np.dtype(precision)
    expected_x = np.asarray(initial)[None] * np.exp(
        -times[:, None, None] * np.asarray(rates + shared)[None, None]
    )
    expected_y = np.exp(-times[:, None] * np.asarray(rates)[None])
    expected = np.concatenate(
        (times[:, None], expected_x.reshape(len(times), -1), expected_y), axis=1
    )
    np.testing.assert_allclose(discovered.trajectory, expected, rtol=2e-5, atol=4e-7)
    np.testing.assert_allclose(replayed.trajectory, expected, rtol=2e-5, atol=4e-7)
    assert replayed.require_accurate() is replayed
    evidence = replayed.block_replay_diagnostics
    assert evidence is not None
    np.testing.assert_array_equal(evidence.step_count, schedule.step_counts)
    assert evidence.require_accurate() is evidence
    assert replayed.replay_diagnostics.step_count == sum(schedule.step_counts)
    prepared = engine.prepare(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        adaptive_schedule=schedule,
    )
    assert prepared.internal_step_count == sum(schedule.step_counts)

    def objective(global_value: Array, local: Array, x0: Array) -> Array:
        return prepared(
            {},
            {
                "shared": global_value,
                "rate": local,
                "x0": x0,
                "y0": jnp.ones_like(local),
            },
        )[-1, 1:].sum()

    value, (global_grad, local_grad, initial_grad) = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1, 2))
    )(shared, rates, initial)
    final_x = np.asarray(initial) * np.exp(-np.asarray(rates + shared))
    final_y = np.exp(-np.asarray(rates))
    assert value == pytest.approx(float(final_x.sum() + final_y.sum()), rel=2e-5)
    assert global_grad == pytest.approx(float(-final_x.sum()), rel=3e-5)
    np.testing.assert_allclose(
        local_grad, -final_x.sum(axis=0) - final_y, rtol=5e-4, atol=2e-7
    )
    np.testing.assert_allclose(
        initial_grad,
        np.broadcast_to(np.exp(-np.asarray(rates + shared)), initial.shape),
        rtol=5e-4,
        atol=2e-7,
    )


def test_unrelated_stiff_block_preserves_original_mesh_and_accuracy() -> None:
    """A second block cannot influence the first block's controller reduction."""
    results = []
    for count in (1, 2):
        system, axes = _case(count)
        initial = jnp.ones((2, count))
        rates = jnp.asarray([0.2, 80.0][:count])
        result = _engine().run_adaptive(
            system,
            np.asarray([0.0, 0.4, 1.0]),
            {},
            _params(axes, initial, rates, jnp.asarray(0.1)),
            model_state=system.model_state(axes),
        )
        results.append(result)
    assert (
        results[0].schedule.blocks[0].step_schedule
        == results[1].schedule.blocks[0].step_schedule
    )
    np.testing.assert_allclose(
        results[0].trajectory[:, 1], results[1].trajectory[:, 1], rtol=0, atol=0
    )
    np.testing.assert_allclose(results[1].trajectory[-1, 1], np.exp(-0.3), rtol=2e-5)


def test_equal_lengths_allow_distinct_meshes_and_output_slots() -> None:
    """A bucket member's plan, not its representative's mesh, selects outputs."""
    system, axes = _case(2)
    engine = _engine(rtol=1e-4)
    times = np.asarray([0.0, 0.4, 1.0])
    params = _params(axes, jnp.ones((2, 2)), jnp.asarray([0.2, 0.4]), jnp.asarray(0.1))
    result = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    )
    meshes = (
        AdaptiveStepSchedule(tuple(times), ((0.4,), (0.1, 0.2, 0.3))),
        AdaptiveStepSchedule(tuple(times), ((0.2, 0.2), (0.3, 0.3))),
    )
    blocks = tuple(
        replace(block, step_schedule=mesh)
        for block, mesh in zip(result.schedule.blocks, meshes, strict=True)
    )
    schedule = replace(result.schedule, blocks=blocks)
    prepared_original = engine.prepare(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        adaptive_schedule=result.schedule,
    )
    prepared_replaced = engine.prepare(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        adaptive_schedule=schedule,
    )
    assert prepared_replaced is not prepared_original
    assert prepared_replaced.signature != prepared_original.signature
    assert schedule.batches(engine.config) == ((0, 1),)
    replayed = engine.run_adaptive(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        schedule=schedule,
    )
    replayed.require_accurate()
    expected = np.exp(-times[:, None, None] * np.asarray([0.3, 0.5])[None, None])
    np.testing.assert_allclose(
        replayed.trajectory[:, 1:5].reshape(3, 2, 2),
        np.broadcast_to(expected, (3, 2, 2)),
        rtol=2e-5,
        atol=1e-7,
    )
    serial = _engine(rtol=1e-4, block_batching="serial").run_adaptive(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        schedule=schedule,
    )
    np.testing.assert_allclose(
        serial.trajectory, replayed.trajectory, rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("checkpoint", ["step", "chunk"])
def test_checkpointed_replay_and_outer_vmap_preserve_gradients(checkpoint: str) -> None:
    """Rematerialization and outer batches retain independent block plans."""
    system, axes = _case(2)
    engine = _engine(
        adaptive_replay="compact", replay_checkpoint=checkpoint, checkpoint_chunk_size=3
    )
    times = np.asarray([0.0, 0.4, 1.0])
    initial = jnp.ones((2, 2))
    rates = jnp.asarray([0.2, 12.0])
    shared = jnp.asarray(0.1)
    params = _params(axes, initial, rates, shared)
    schedule = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    ).schedule
    prepared = engine.prepare(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        adaptive_schedule=schedule,
    )

    def objective(value: Array) -> Array:
        return prepared(
            {},
            {
                "x0": initial,
                "y0": jnp.ones_like(value),
                "shared": shared,
                "rate": value,
            },
        )[-1, 1:].sum()

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(objective)))(
        jnp.stack((rates, rates + 0.01))
    )
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    expected = -(2 * np.exp(-np.asarray(rates + shared)) + np.exp(-np.asarray(rates)))
    np.testing.assert_allclose(gradients[0], expected, rtol=3e-4, atol=4e-7)


def test_changed_local_geometry_reports_each_blocks_accuracy() -> None:
    """A stale parameter mesh reports the affected member before host refusal."""
    system, axes = _case(2)
    engine = _engine()
    times = np.asarray([0.0, 0.4, 1.0])
    params = _params(axes, jnp.ones((2, 2)), jnp.asarray([0.2, 0.2]), jnp.asarray(0.1))
    schedule = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    ).schedule
    changed = _params(
        axes, jnp.ones((2, 2)), jnp.asarray([30.0, 0.2]), jnp.asarray(0.1)
    )
    result = engine.run_adaptive(
        system,
        times,
        {},
        changed,
        model_state=system.model_state(axes),
        schedule=schedule,
    )
    np.testing.assert_array_equal(
        result.block_replay_diagnostics.accurate, [False, True]
    )
    with pytest.raises(AdaptiveScheduleAccuracyError):
        result.require_accurate()
    with pytest.raises(AdaptiveScheduleAccuracyError):
        result.block_replay_diagnostics.require_accurate()


@pytest.mark.parametrize(
    "change",
    [
        {"rtol": 1e-4},
        {"schedule_tag": "different"},
        {"method": "heun"},
        {"block_schedule": "shared"},
    ],
)
def test_stale_policy_or_context_is_rejected(change: dict[str, object]) -> None:
    """Controller, method, context and mesh policy must agree before replay."""
    system, axes = _case(1)
    times = np.asarray([0.0, 1.0])
    params = _params(axes, jnp.ones((2, 1)), jnp.asarray([0.2]), jnp.asarray(0.1))
    schedule = (
        _engine()
        .run_adaptive(system, times, {}, params, model_state=system.model_state(axes))
        .schedule
    )
    with pytest.raises(ValueError, match=r"schedule|Schedule"):
        _engine(**change).run_adaptive(
            system,
            times,
            {},
            params,
            model_state=system.model_state(axes),
            schedule=schedule,
        )


def test_changed_grid_and_reordered_meshes_are_rejected() -> None:
    """Block contexts prevent accidental positional remapping of mesh banks."""
    system, axes = _case(2)
    engine = _engine()
    times = np.asarray([0.0, 0.4, 1.0])
    params = _params(axes, jnp.ones((2, 2)), jnp.asarray([0.2, 12.0]), jnp.asarray(0.1))
    schedule = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    ).schedule
    with pytest.raises(ValueError, match="context"):
        replace(schedule, blocks=tuple(reversed(schedule.blocks)))
    with pytest.raises(ValueError, match="output grid"):
        engine.run_adaptive(
            system,
            np.asarray([0.0, 0.5, 1.0]),
            {},
            params,
            model_state=system.model_state(axes),
            schedule=schedule,
        )
    with pytest.raises(ValueError, match="output grid"):
        engine.prepare(
            system,
            np.asarray([0.0, 0.5, 1.0]),
            {},
            params,
            model_state=system.model_state(axes),
            adaptive_schedule=schedule,
        )


def test_single_output_has_empty_block_plans() -> None:
    """Zero-step buckets preserve the initial trajectory and empty evidence."""
    system, axes = _case(2)
    engine = _engine()
    times = np.asarray([0.0])
    params = _params(axes, jnp.ones((2, 2)), jnp.asarray([0.2, 12.0]), jnp.asarray(0.1))
    result = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    )
    assert result.schedule.step_counts == (0, 0)
    replayed = engine.run_adaptive(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        schedule=result.schedule,
    )
    np.testing.assert_array_equal(replayed.trajectory, result.trajectory)
    assert replayed.replay_diagnostics.step_count == 0
    replayed.require_accurate()


@pytest.mark.parametrize(
    "options",
    [
        {"adaptive": False},
        {"state_layout": "flat"},
        {"adaptive_replay": "unrolled"},
        {"block_batch_size": 0},
        {"block_batch_size": True},
    ],
)
def test_invalid_independent_configuration_is_refused(
    options: dict[str, object],
) -> None:
    """Unsupported dispatches cannot silently select another numerical path."""
    with pytest.raises(ValueError, match="validation error"):
        _engine(**options)


def test_missing_compiler_proof_is_refused() -> None:
    """Declaring a block layout cannot manufacture a separability proof."""
    system, axes = _case(2)
    spec = dict(system.spec)
    spec.pop("factorize_axes")
    system = OpSystemSystem(spec=spec)
    params = _params(axes, jnp.ones((2, 2)), jnp.asarray([0.2, 12.0]), jnp.asarray(0.1))
    with pytest.raises((TypeError, ValueError), match=r"block|Block"):
        _engine().run_adaptive(
            system,
            np.asarray([0.0, 1.0]),
            {},
            params,
            model_state=system.model_state(axes),
        )


def test_global_reduction_cannot_supply_a_block_proof() -> None:
    """The actual compiler rejects a reduction across the proposed block axis."""
    from op_system._errors import UnsupportedFeatureError  # noqa: PLC0415, PLC2701

    system, _axes = _case(2)
    spec = dict(system.spec)
    spec["equations"] = {
        "X[age, loc]": "-X[age, loc] * total",
        "Y[loc]": "-rate[loc] * Y[loc]",
    }
    spec["aliases"] = {
        "total": (
            "apply_along(apply_along(X[age:a, loc:l], loc=l, kernel=sum), "
            "age=a, kernel=sum)"
        )
    }
    with pytest.raises(UnsupportedFeatureError, match="reduces over axis"):
        OpSystemSystem(spec=spec)


def test_non_jax_independent_discovery_is_rejected() -> None:
    """The public block feature cannot silently execute a NumPy fallback."""
    system, axes = _case(1)
    params = {
        "x0": ParameterValue(np.ones((2, 1)), axes.resolve_shape(("age", "loc"))),
        "y0": ParameterValue(np.ones(1), axes.resolve_shape(("loc",))),
        "rate": ParameterValue(np.asarray([0.2]), axes.resolve_shape(("loc",))),
        "shared": ParameterValue(np.asarray(0.1), ResolvedShape()),
    }
    with pytest.raises(TypeError, match="require JAX"):
        _engine().run_adaptive(
            system,
            np.asarray([0.0, 1.0]),
            {},
            params,
            model_state=system.model_state(axes),
        )


def test_discovery_does_not_hide_controller_exhaustion() -> None:
    """Failure in one block cannot yield a partial bank or a shared fallback."""
    system, axes = _case(1)
    params = _params(axes, jnp.ones((2, 1)), jnp.asarray([0.2]), jnp.asarray(0.1))
    engine = _engine(max_steps=1, dt_init=1e-3)
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        engine.run_adaptive(
            system,
            np.asarray([0.0, 1.0]),
            {},
            params,
            model_state=system.model_state(axes),
        )
    assert engine.last_adaptive_schedule is None


@pytest.mark.parametrize("method", ["euler", "heun", "rk4"])
def test_other_explicit_methods_preserve_their_accepted_candidates(method: str) -> None:
    """Step-doubling methods replay their accepted two-half-step candidates."""
    system, axes = _case(2)
    engine = _engine(method=method)
    times = np.asarray([0.0, 0.4, 1.0])
    params = _params(axes, jnp.ones((2, 2)), jnp.asarray([0.2, 0.5]), jnp.asarray(0.1))
    discovered = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    )
    replayed = engine.run_adaptive(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        schedule=discovered.schedule,
    )
    replayed.require_accurate()
    np.testing.assert_allclose(
        replayed.trajectory, discovered.trajectory, rtol=3e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        replayed.trajectory[-1, 1:3],
        np.exp(-np.asarray([0.3, 0.6])),
        rtol=5e-4,
        atol=2e-6,
    )


def test_complex_state_is_rejected_before_discovery_or_preparation() -> None:
    """Complex controller time arithmetic is outside this qualified contract."""
    system, axes = _case(1)
    engine = _engine()
    times = np.asarray([0.0, 1.0])
    good = _params(axes, jnp.ones((2, 1)), jnp.asarray([0.2]), jnp.asarray(0.1))
    schedule = engine.run_adaptive(
        system, times, {}, good, model_state=system.model_state(axes)
    ).schedule
    bad = _params(
        axes,
        jnp.ones((2, 1), dtype=jnp.complex64),
        jnp.asarray([1], dtype=jnp.complex64),
        jnp.asarray(1, dtype=jnp.complex64),
    )
    with pytest.raises(TypeError, match="real float32 or float64"):
        engine.run_adaptive(
            system, times, {}, bad, model_state=system.model_state(axes)
        )
    with pytest.raises(TypeError, match="real float32 or float64"):
        engine.prepare(
            system,
            times,
            {},
            bad,
            model_state=system.model_state(axes),
            adaptive_schedule=schedule,
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int32])
def test_provider_promotes_inputs_before_the_real_state_boundary(dtype: object) -> None:
    """Existing provider packing promotes low-precision/integer inputs safely."""
    system, axes = _case(1)
    engine = _engine()
    times = np.asarray([0.0, 1.0])
    params = _params(
        axes,
        jnp.ones((2, 1), dtype=dtype),
        jnp.asarray([1], dtype=dtype),
        jnp.asarray(1, dtype=dtype),
    )
    discovered = engine.run_adaptive(
        system, times, {}, params, model_state=system.model_state(axes)
    )
    prepared = engine.prepare(
        system,
        times,
        {},
        params,
        model_state=system.model_state(axes),
        adaptive_schedule=discovered.schedule,
    )
    result = prepared.run({}, params)
    assert str(result.dtype) in {"float32", "float64"}
    np.testing.assert_allclose(
        result[-1, 1:], np.exp(-np.asarray([2.0, 2.0, 1.0])), rtol=2e-5, atol=1e-7
    )


def test_held_table_runtime_axis_forcing_and_table_gradients() -> None:
    """Runtime time/group slicing preserves the compiler's declared held jumps."""
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
    rates = jnp.asarray([[1.0, 2.0], [0.0, 0.0], [2.0, 4.0]])
    initial = jnp.asarray([1.0, 2.0])
    params = {
        "start": ParameterValue(initial, ResolvedShape(("group",), (2,))),
        "rate": ParameterValue(rates, ResolvedShape(("time", "group"), (3, 2))),
    }
    engine = _engine(block_axis="group", rtol=1e-5, atol=1e-7)
    times = np.asarray([0.0, 0.4, 0.9, 1.5])
    discovered = engine.run_adaptive(system, times, {}, params)
    assert all(
        block.forcing_breakpoints == (0.7, 1.0) for block in discovered.schedule.blocks
    )
    replayed = engine.run_adaptive(
        system, times, {}, params, schedule=discovered.schedule
    )
    replayed.require_accurate()
    expected = np.asarray(initial) * np.exp(-np.asarray([1.7, 3.4]))
    np.testing.assert_allclose(replayed.trajectory[-1, 1:], expected, rtol=8e-5)
    prepared = engine.prepare(
        system, times, {}, params, adaptive_schedule=discovered.schedule
    )

    def objective(value: Array) -> Array:
        return prepared({}, {"start": initial, "rate": value})[-1, 1:].sum()

    gradient = jax.jit(jax.grad(objective))(rates)
    np.testing.assert_allclose(
        gradient,
        -np.asarray([0.7, 0.3, 0.5])[:, None] * expected[None],
        rtol=5e-4,
        atol=1e-6,
    )
