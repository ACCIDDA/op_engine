"""Public CoreSolver contracts for paired additive Runge--Kutta integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    OperatorSpecs,
    RunConfig,
)
from op_engine.matrix_ops import (
    StageOperatorContext,
    build_implicit_euler_operators,
    make_constant_base_builder,
    make_stage_operator_factory,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable


def _numpy_factory(rate: float) -> Callable[..., Any]:
    """Build a constant implicit-Euler stage factory.

    Returns:
        Stage operator factory for the requested scalar rate.
    """
    return make_stage_operator_factory(
        make_constant_base_builder(np.asarray([[rate]], dtype=float))
    )


def _solve_split_problem(dt: float, *, nonlinear: bool) -> float:
    """Integrate a scalar linear/nonlinear split problem to time one.

    Returns:
        Final scalar state.
    """
    n_steps = round(1.0 / dt)
    core = ModelCore(1, 1, np.linspace(0.0, 1.0, n_steps + 1))
    core.set_initial_state(np.asarray([[1.0]]))
    rhs = (
        (lambda _time, state: cast("Array", np.negative(state * state)))
        if nonlinear
        else (lambda _time, state: cast("Array", np.multiply(state, -0.5)))
    )
    CoreSolver(core).run(
        rhs,
        config=RunConfig(
            method="imex-ark3",
            operators=OperatorSpecs(default=_numpy_factory(-2.0)),
        ),
    )
    return float(core.get_current_state()[0, 0])


@pytest.mark.parametrize("nonlinear", [False, True])
def test_imex_ark3_has_third_order_on_split_benchmarks(
    nonlinear: bool,  # noqa: FBT001
) -> None:
    """ARS443 attains third order for linear and nonlinear explicit splits."""
    exact = 2.0 * np.exp(-2.0) / (3.0 - np.exp(-2.0)) if nonlinear else np.exp(-2.5)
    errors = [
        abs(_solve_split_problem(dt, nonlinear=nonlinear) - exact)
        for dt in (0.2, 0.1, 0.05)
    ]
    orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(2)
    ]

    assert min(orders) > 2.7


def test_imex_ark3_damps_a_very_stiff_implicit_mode() -> None:
    """The L-stable implicit half damps a large negative real mode."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))

    CoreSolver(core).run(
        lambda _time, state: cast("Array", np.zeros_like(state)),
        config=RunConfig(
            method="imex-ark3",
            operators=OperatorSpecs(default=_numpy_factory(-1000.0)),
        ),
    )

    assert abs(float(core.get_current_state()[0, 0])) < 0.01


def test_imex_ark3_passes_each_actual_stage_context_to_factory() -> None:
    """DIRK factories receive full dt, diagonal scale, stage time, and base."""
    calls: list[tuple[float, float, StageOperatorContext]] = []

    def factory(
        dt: float,
        scale: float,
        context: StageOperatorContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((dt, scale, context))
        left, right = build_implicit_euler_operators(
            np.asarray([[-2.0]]),
            dt_scale=dt * scale,
        )
        return np.asarray(left), np.asarray(right)

    core = ModelCore(1, 1, np.asarray([0.0, 0.3]))
    core.set_initial_state(np.asarray([[1.0]]))
    CoreSolver(core).run(
        lambda _time, state: cast("Array", np.multiply(state, -0.5)),
        config=RunConfig(
            method="imex-ark3",
            operators=OperatorSpecs(default=factory),
        ),
    )

    assert len(calls) == 4
    assert np.allclose([item[0] for item in calls], 0.3)
    assert np.allclose([item[1] for item in calls], 0.5)
    assert np.allclose([item[2].t for item in calls], [0.15, 0.2, 0.15, 0.3])
    assert [item[2].stage for item in calls] == [
        "ark3-1",
        "ark3-2",
        "ark3-3",
        "ark3-4",
    ]
    assert all(item[2].y.shape == (1, 1) for item in calls)


def test_imex_ark3_requires_a_stage_operator_factory() -> None:
    """Paired DIRK stages cannot silently reuse one static propagator."""
    core = ModelCore(1, 1, np.asarray([0.0, 0.1]))
    core.set_initial_state(np.asarray([[1.0]]))

    with pytest.raises(ValueError, match="requires operators"):
        CoreSolver(core).run(
            lambda _time, state: state,
            config=RunConfig(method="imex-ark3"),
        )
    with pytest.raises(TypeError, match="StageOperatorFactory"):
        CoreSolver(core).run(
            lambda _time, state: state,
            config=RunConfig(
                method="imex-ark3",
                operators=OperatorSpecs(default=(np.eye(1), np.eye(1))),
            ),
        )


def test_adaptive_imex_ark3_rejects_and_records_accepted_mesh() -> None:
    """The shared-stage embedded estimate drives ordinary rejection/retry."""
    attempted_dt: list[float] = []

    def factory(
        dt: float,
        scale: float,
        _context: StageOperatorContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        attempted_dt.append(dt)
        left, right = build_implicit_euler_operators(
            np.asarray([[-2.0]]),
            dt_scale=dt * scale,
        )
        return np.asarray(left), np.asarray(right)

    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))
    solver = CoreSolver(core)
    solver.run(
        lambda _time, state: cast("Array", np.negative(state * state)),
        config=RunConfig(
            method="imex-ark3",
            adaptive=True,
            adaptive_cfg=AdaptiveConfig(
                rtol=1e-2,
                atol=1e-4,
                dt_init=1.0,
                max_reject=10,
            ),
            operators=OperatorSpecs(default=factory),
        ),
    )

    schedule = solver.last_adaptive_schedule
    assert schedule is not None
    accepted = schedule.step_sizes[0]
    assert accepted[0] < 1.0
    assert attempted_dt[:4] == [1.0] * 4
    assert len(attempted_dt) > 4 * len(accepted)
    assert np.isclose(sum(accepted), 1.0)


def test_numpy_and_jax_fixed_imex_ark3_agree() -> None:
    """The paired method preserves and agrees across array namespaces."""
    jnp = pytest.importorskip("jax.numpy")

    def solve(*, use_jax: bool) -> np.ndarray:
        xp = jnp if use_jax else np
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 0.1, 0.2]),
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(xp.asarray([[1.0]], dtype=np.float32))
        CoreSolver(core).run(
            lambda _time, state: cast("Array", xp.multiply(state, -0.5)),
            config=RunConfig(
                method="imex-ark3",
                operators=OperatorSpecs(default=_numpy_factory(-2.0)),
            ),
        )
        return np.asarray(core.get_current_state())

    assert np.allclose(solve(use_jax=True), solve(use_jax=False), rtol=2e-6)


def test_jax_compiled_imex_ark3_replay_is_differentiable() -> None:
    """A frozen ARK mesh supports JIT and gradients without a JAX config type."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    schedule = AdaptiveStepSchedule(
        output_times=(0.0, 0.5),
        step_sizes=((0.25, 0.25),),
    )

    def replay(rate: Array, initial: Array) -> Array:
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 0.5]),
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.reshape(initial, (1, 1)))

        def factory(
            dt: float,
            scale: float,
            context: StageOperatorContext,
        ) -> tuple[Array, Array]:
            identity = jnp.eye(1, dtype=context.y.dtype)
            operator = jnp.reshape(rate, (1, 1))
            left = jnp.subtract(identity, jnp.multiply(operator, dt * scale))
            return cast("Array", left), cast("Array", identity)

        CoreSolver(core).replay_adaptive_schedule(
            lambda _time, state: cast("Array", jnp.negative(state * state)),
            schedule,
            config=RunConfig(
                method="imex-ark3",
                adaptive=True,
                operators=OperatorSpecs(default=factory),
            ),
        )
        return cast("Array", core.get_current_state()[0, 0])

    rate = jnp.asarray(-2.0, dtype=jnp.float32)
    initial = jnp.asarray(1.0, dtype=jnp.float32)
    value, gradients = jax.jit(jax.value_and_grad(replay, argnums=(0, 1)))(
        rate, initial
    )
    epsilon = 1e-3
    finite_rate = (
        replay(rate + epsilon, initial) - replay(rate - epsilon, initial)
    ) / (2.0 * epsilon)
    finite_initial = (
        replay(rate, initial + epsilon) - replay(rate, initial - epsilon)
    ) / (2.0 * epsilon)

    assert np.isfinite(float(value))
    assert np.allclose(gradients[0], finite_rate, rtol=2e-3, atol=2e-4)
    assert np.allclose(gradients[1], finite_initial, rtol=2e-3, atol=2e-4)
