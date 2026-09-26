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

"""End-to-end coverage of the op_system to op_engine provider boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from flepimop2.axis import Axis, AxisCollection, ResolvedShape
from flepimop2.backend.abc import BackendABC
from flepimop2.configuration import SimulateSpecificationModel
from flepimop2.parameter.abc import ParameterValue
from flepimop2.parameter.sparse_table import SparseTableParameter
from flepimop2.simulator import Simulator
from flepimop2.system.op_system import OpSystemSystem
from flepimop2.typing import StateChangeEnum

from flepimop2.engine.op_engine import (
    OpEngineEngineConfig,
    OpEngineFlepimop2Engine,
    SolverMethod,
)

if TYPE_CHECKING:
    from flepimop2.meta import RunMeta
    from flepimop2.typing import Float64NDArray


class _NoopBackend(BackendABC, module="test_op_system_noop"):
    """Persistence sink for provider orchestration tests."""

    def _save(self, data: Float64NDArray, run_meta: RunMeta) -> None:
        """Accept the result without changing it."""

    def _read(self, run_meta: RunMeta) -> Float64NDArray:
        """Reading is outside these tests' scope."""
        msg = "The no-op test backend does not store results."
        raise NotImplementedError(msg)


def _scalar(value: float) -> ParameterValue:
    """Build a scalar parameter value.

    Returns:
        A scalar NumPy parameter value with no named axes.
    """
    return ParameterValue(np.asarray(value, dtype=np.float64), ResolvedShape())


def _engine() -> OpEngineFlepimop2Engine:
    """Build the fixed-step engine used by these integration tests.

    Returns:
        A Heun-configured op_engine provider.
    """
    return OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.HEUN),
    )


def test_simulator_runs_real_op_system_with_shared_seed_parameter() -> None:
    """Simulator resolves an empty model_state through op_system metadata."""
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["S", "I", "R"],
            "equations": {
                "S": "-rate * S",
                "I": "rate * S",
                "R": "0 * R",
            },
            "initial_state": {"S": "population", "I": "zero", "R": "zero"},
        }
    )
    simulator = Simulator(
        system,
        _engine(),
        _NoopBackend(),
        simulate_config=SimulateSpecificationModel(times=[0.0, 0.5]),
    )

    result = simulator.run(
        initial_state={},
        params={
            "population": _scalar(10.0),
            "zero": _scalar(0.0),
            "rate": _scalar(0.1),
        },
    )

    np.testing.assert_allclose(
        result,
        np.asarray([[0.0, 10.0, 0.0, 0.0], [0.5, 9.5125, 0.4875, 0.0]]),
        rtol=0.0,
        atol=1e-14,
    )


def test_routing_sparse_table_and_shaped_initial_state_run_end_to_end() -> None:
    """A routed op_system RHS consumes sparse_table and shaped state metadata."""
    axes = AxisCollection({
        "vax": Axis(name="vax", kind="categorical", size=2, labels=("u", "v")),
        "imm": Axis(name="imm", kind="categorical", size=2, labels=("x0", "x1")),
    })
    system = OpSystemSystem(
        spec={
            "kind": "transitions",
            "axes": [
                {"name": "vax", "coords": ["u", "v"]},
                {"name": "imm", "coords": ["x0", "x1"]},
            ],
            "state": ["X[vax, imm]"],
            "transitions": [
                {
                    "from": "X[vax=u, imm:i]",
                    "to": "X[vax=v, imm:j]",
                    "rate": "eta[imm:i, imm:j]",
                }
            ],
            "initial_state": {
                "X[vax, imm]": {
                    "shaped": "x_init",
                    "axes": ["vax", "imm"],
                }
            },
        }
    )
    requests = system.requested_parameters(axes)
    eta = SparseTableParameter(
        indices=("imm", "imm"),
        entries=[
            {"index": ["x0", "x1"], "value": 0.5},
            {"index": ["x1", "x0"], "value": 0.25},
        ],
    ).sample(axes=axes, request=requests["eta"])
    x_init = ParameterValue(
        np.asarray([[2.0, 4.0], [0.0, 0.0]], dtype=np.float64),
        axes.resolve_shape(("vax", "imm")),
    )
    params = {"eta": eta, "x_init": x_init}
    y0 = np.asarray([2.0, 4.0, 0.0, 0.0], dtype=np.float64)
    stepper = system.bind(params={name: value.value for name, value in params.items()})
    rhs_start = stepper(np.float64(0.0), y0)
    predictor = y0 + 0.5 * rhs_start
    rhs_end = stepper(np.float64(0.5), predictor)
    expected = y0 + 0.25 * (rhs_start + rhs_end)

    result = _engine().run(
        system,
        np.asarray([0.0, 0.5], dtype=np.float64),
        {},
        params,
        model_state=system.model_state(axes),
    )

    np.testing.assert_allclose(result[0, 1:], y0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result[1, 1:], expected, rtol=0.0, atol=1e-14)


def test_axis_kernel_generator_runs_as_flat_imex_operator() -> None:
    """A typed row-source generator drives a real op_system IMEX solve."""
    axes = AxisCollection({
        "imm": Axis(
            name="imm",
            kind="ordinal",
            size=3,
            labels=("x0", "x1", "x2"),
        ),
    })
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [
                {
                    "name": "imm",
                    "type": "ordinal",
                    "coords": ["x0", "x1", "x2"],
                }
            ],
            "state": ["X[imm]"],
            "equations": {"X[imm]": "0 * X[imm]"},
            "initial_state": {
                "X[imm]": {"shaped": "x_init", "axes": ["imm"]},
            },
            "operators": [
                {
                    "kind": "axis_kernel",
                    "axis": "imm",
                    "velocity": 0.5,
                    "kernel": {
                        "form": "generator",
                        "params": {"matrix": "G"},
                        "param_axes": {"G": ["imm", "imm"]},
                    },
                }
            ],
        }
    )
    generator = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    y0 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    params = {
        "G": ParameterValue(generator, axes.resolve_shape(("imm", "imm"))),
        "x_init": ParameterValue(y0, axes.resolve_shape(("imm",))),
    }
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.IMEX_EULER),
    )

    assert engine.validate_system(system) is None
    result = engine.run(
        system,
        np.asarray([0.0, 0.2], dtype=np.float64),
        {},
        params,
        model_state=system.model_state(axes),
    )

    base_operator = 0.5 * generator.T
    half_step_left = np.eye(3) - 0.1 * base_operator
    expected = np.linalg.solve(half_step_left, y0)
    expected = np.linalg.solve(half_step_left, expected)
    np.testing.assert_allclose(result[1, 1:], expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result[1, 1:].sum(), 1.0, rtol=0.0, atol=1e-14)


def test_axis_kernel_parameters_are_jittable_and_differentiable() -> None:  # noqa: PLR0914
    """Typed operator parameters remain dynamic through the JAX IMEX path."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    axes = AxisCollection({
        "imm": Axis(
            name="imm",
            kind="ordinal",
            size=2,
            labels=("x0", "x1"),
        ),
    })
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "axes": [
                {
                    "name": "imm",
                    "type": "ordinal",
                    "coords": ["x0", "x1"],
                }
            ],
            "state": ["X[imm]"],
            "equations": {"X[imm]": "0 * X[imm]"},
            "initial_state": {
                "X[imm]": {"shaped": "x_init", "axes": ["imm"]},
            },
            "operators": [
                {
                    "kind": "axis_kernel",
                    "axis": "imm",
                    "velocity": "speed",
                    "kernel": {
                        "form": "generator",
                        "params": {"matrix": "G"},
                        "param_axes": {"G": ["imm", "imm"]},
                    },
                }
            ],
        }
    )
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method=SolverMethod.IMEX_EULER),
    )
    times = np.asarray([0.0, 0.2], dtype=np.float64)
    scalar_shape = ResolvedShape()
    generator_shape = axes.resolve_shape(("imm", "imm"))
    state_shape = axes.resolve_shape(("imm",))
    generator_template = jnp.asarray([[-1.0, 1.0], [0.0, 0.0]], dtype=jnp.float32)
    initial = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    def final_target(speed: object, generator_rate: object) -> object:
        generator = generator_template * generator_rate
        result = engine.run(
            system,
            times,
            {},
            {
                "speed": ParameterValue(speed, scalar_shape),
                "G": ParameterValue(generator, generator_shape),
                "x_init": ParameterValue(initial, state_shape),
            },
            model_state=system.model_state(axes),
        )
        return result[-1, 2]

    speed = jnp.asarray(0.5, dtype=jnp.float32)
    generator_rate = jnp.asarray(1.2, dtype=jnp.float32)
    value, derivatives = jax.jit(jax.value_and_grad(final_target, argnums=(0, 1)))(
        speed, generator_rate
    )
    decay = 1.0 + 0.1 * float(speed) * float(generator_rate)
    expected_value = 1.0 - decay**-2
    expected_speed_derivative = 0.2 * float(generator_rate) * decay**-3
    expected_generator_derivative = 0.2 * float(speed) * decay**-3

    assert value == pytest.approx(expected_value, rel=2e-5)
    assert derivatives[0] == pytest.approx(expected_speed_derivative, rel=2e-5)
    assert derivatives[1] == pytest.approx(expected_generator_derivative, rel=2e-5)


def test_real_op_system_is_jittable_and_differentiable_through_provider() -> None:  # noqa: PLR0914
    """Portable Heun retains JAX state and parameter gradients end to end."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["X"],
            "equations": {"X": "rate * X"},
            "initial_state": {"X": 0.0},
        }
    )
    engine = _engine()
    times = np.asarray([0.0, 0.25, 1.0], dtype=np.float64)
    scalar_shape = ResolvedShape()

    def solve(rate: object, initial: object) -> object:
        result = engine.run(
            system,
            times,
            {"X": ParameterValue(initial, scalar_shape)},
            {"rate": ParameterValue(rate, scalar_shape)},
        )
        return result[-1, 1]

    rate = jnp.asarray(-0.3, dtype=jnp.float32)
    initial = jnp.asarray(1.2, dtype=jnp.float32)
    value, gradients = jax.jit(jax.value_and_grad(solve, argnums=(0, 1)))(
        rate,
        initial,
    )

    steps = (0.25, 0.75)
    factors = tuple(1.0 - 0.3 * dt + 0.5 * (-0.3 * dt) ** 2 for dt in steps)
    factor_derivatives = tuple(dt - 0.3 * dt**2 for dt in steps)
    expected_value = 1.2 * factors[0] * factors[1]
    expected_rate_grad = 1.2 * (
        factor_derivatives[0] * factors[1] + factors[0] * factor_derivatives[1]
    )
    expected_initial_grad = factors[0] * factors[1]

    assert value == pytest.approx(expected_value, rel=2e-6)
    assert gradients[0] == pytest.approx(expected_rate_grad, rel=2e-6)
    assert gradients[1] == pytest.approx(expected_initial_grad, rel=2e-6)

    result = engine.run(
        system,
        times,
        {"X": ParameterValue(initial, scalar_shape)},
        {"rate": ParameterValue(rate, scalar_shape)},
    )
    assert result.__array_namespace__() is jnp


def test_state_namespace_controls_mixed_parameter_inputs() -> None:
    """A declared JAX state seed controls mixed parameter evaluation."""
    jnp = pytest.importorskip("jax.numpy")
    system = OpSystemSystem(
        spec={
            "kind": "expr",
            "state": ["X"],
            "equations": {"X": "rate * X"},
            "initial_state": {"X": "x0"},
        }
    )
    scalar_shape = ResolvedShape()

    result = _engine().run(
        system,
        np.asarray([0.0, 1.0], dtype=np.float64),
        {},
        {
            "rate": ParameterValue(np.asarray(-0.3), scalar_shape),
            "x0": ParameterValue(jnp.asarray(1.0), scalar_shape),
        },
    )

    assert result.__array_namespace__() is jnp
    assert float(result[-1, 1]) == pytest.approx(0.745, rel=2e-6)
