"""Qualification of the public explicit step in external JAX loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore, Scalar, array_namespace
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def jax_x64() -> Iterator[None]:
    """Scope double precision to finite-difference qualification tests."""
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        yield


def _rhs(parameters: Array) -> Callable[[Scalar, Array], Array]:
    """Bind a nonlinear, time-dependent two-parameter test problem.

    Returns:
        RHS preserving the state namespace.
    """
    xp = array_namespace(parameters)
    rate, forcing = parameters

    def rhs(time: Scalar, state: Array) -> Array:
        return cast(
            "Array",
            xp.add(
                xp.multiply(rate, state),
                xp.multiply(forcing * time, xp.square(state)),
            ),
        )

    return rhs


def _scan_trajectory(
    parameters: Array,
    times: Array,
    *,
    method: str,
    checkpoint: bool = False,
    reuse_fsal: bool = True,
) -> Array:
    """Drive only public core APIs in a rolled external loop.

    Returns:
        Full trajectory, including the initial state.
    """
    jax = pytest.importorskip("jax")
    xp = array_namespace(parameters)
    initial = xp.asarray([[1.2], [0.7]], dtype=parameters.dtype)
    solver = CoreSolver(ModelCore(2, 1, np.asarray([0.0, 1.0])))
    rhs = _rhs(parameters)
    sizes = xp.diff(times)
    first, stage = solver.fixed_explicit_step(
        rhs,
        method=method,
        t=times[0],
        dt=sizes[0],
        y=initial,
    )
    if not reuse_fsal:
        stage = None

    def advance(
        carry: tuple[Array, Array | None],
        step: tuple[Scalar, Scalar],
    ) -> tuple[tuple[Array, Array | None], Array]:
        state, cached = carry
        time, dt = step
        result, next_stage = solver.fixed_explicit_step(
            rhs,
            method=method,
            t=time,
            dt=dt,
            y=state,
            first_stage=cached,
        )
        return (result, next_stage if reuse_fsal else None), result

    body = jax.checkpoint(advance) if checkpoint else advance
    _final, tail = jax.lax.scan(body, (first, stage), (times[1:-1], sizes[1:]))
    return cast("Array", xp.concat((initial[None], first[None], tail), axis=0))


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
@pytest.mark.parametrize("reuse_fsal", [False, True])
def test_external_scan_matches_core_run(method: str, *, reuse_fsal: bool) -> None:
    """Traced times and optional FSAL carry reproduce nonuniform eager steps."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.asarray([0.0, 0.02, 0.11, 0.23, 0.4, 0.6], dtype=np.float32)
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float32)
    core = ModelCore(
        2,
        1,
        times,
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(jnp.asarray([[1.2], [0.7]], dtype=jnp.float32))
    CoreSolver(core).run(_rhs(parameters), config=RunConfig(method=method))

    trajectory = jax.jit(
        lambda values, grid: _scan_trajectory(
            values,
            grid,
            method=method,
            reuse_fsal=reuse_fsal,
        ),
    )(parameters, jnp.asarray(times))

    assert array_namespace(trajectory) is jnp
    assert trajectory.shape == (len(times), 2, 1)
    np.testing.assert_allclose(trajectory, core.state_array, rtol=2e-6, atol=2e-7)


def test_dopri5_fsal_reuses_derivative_without_changing_solution() -> None:
    """The public cache contract saves one RHS call on a subsequent step."""
    solver = CoreSolver(ModelCore(2, 1, np.asarray([0.0, 1.0])))
    calls: list[float] = []
    initial = np.asarray([[1.2], [0.7]])

    def rhs(time: float, state: Array) -> Array:
        calls.append(time)
        return cast("Array", -0.3 * state + 0.2 * time * np.square(state))

    first, stage = solver.fixed_explicit_step(
        rhs,
        method="dopri5",
        t=0.0,
        dt=0.2,
        y=initial,
    )
    assert stage is not None
    assert len(calls) == 7
    np.testing.assert_array_equal(stage, rhs(0.2, first))
    calls.clear()
    cached, _stage = solver.fixed_explicit_step(
        rhs,
        method="dopri5",
        t=0.2,
        dt=0.3,
        y=first,
        first_stage=stage,
    )
    assert len(calls) == 6
    calls.clear()
    uncached, _stage = solver.fixed_explicit_step(
        rhs,
        method="dopri5",
        t=0.2,
        dt=0.3,
        y=first,
    )
    assert len(calls) == 7
    np.testing.assert_array_equal(cached, uncached)


def _record_schedule(method: str) -> tuple[AdaptiveStepSchedule, RunConfig]:
    """Choose a nominal adaptive mesh outside all differentiation.

    Returns:
        Frozen mesh and its matching solver configuration.
    """
    core = ModelCore(2, 1, np.asarray([0.0, 0.2, 0.6]))
    core.set_initial_state(np.asarray([[1.2], [0.7]]))
    config = RunConfig(
        method=method,
        adaptive=True,
        adaptive_cfg=AdaptiveConfig(rtol=1e-3, atol=1e-6, dt_init=0.05),
    )
    solver = CoreSolver(core)
    solver.run(_rhs(np.asarray([-0.3, 0.2])), config=config)
    assert solver.last_adaptive_schedule is not None
    return solver.last_adaptive_schedule, config


def _replay_times(schedule: AdaptiveStepSchedule, method: str) -> Array:
    """Flatten the accepted mesh into equivalent fixed-step boundaries.

    Returns:
        Times including the half steps accepted by Euler and RK4.
    """
    times = [schedule.output_times[0]]
    for start, steps in zip(
        schedule.output_times[:-1], schedule.step_sizes, strict=True
    ):
        time = start
        for dt in steps:
            if method in {"euler", "rk4"}:
                times.append(time + 0.5 * dt)
            time += dt
            times.append(time)
    return np.asarray(times)


def _loss(parameters: Array, final: Array) -> Array:
    """Return a nonlinear scalar loss with a positive definite Hessian."""
    xp = array_namespace(parameters)
    return cast(
        "Array",
        0.5 * xp.sum(xp.square(final)) + 2.0 * xp.sum(xp.square(parameters)),
    )


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_scan_replay_hessian_matches_gradient_differences(
    method: str,
    *,
    checkpoint: bool,
) -> None:
    """Hessians of a frozen adaptive mesh agree with finite differences and core."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    schedule, config = _record_schedule(method)
    times = jnp.asarray(_replay_times(schedule, method))
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float64)

    def scan_loss(values: Array) -> Array:
        trajectory = _scan_trajectory(
            values, times, method=method, checkpoint=checkpoint
        )
        return _loss(values, trajectory[-1])

    def core_loss(values: Array) -> Array:
        core = ModelCore(2, 1, np.asarray(schedule.output_times))
        core.set_initial_state(jnp.asarray([[1.2], [0.7]], dtype=values.dtype))
        CoreSolver(core).replay_adaptive_schedule(_rhs(values), schedule, config=config)
        return _loss(values, core.get_current_state())

    hessian = np.asarray(jax.jit(jax.hessian(scan_loss))(parameters))
    gradient = jax.jit(jax.grad(scan_loss))
    perturbations = 1e-4 * np.eye(2)
    finite_difference = np.column_stack([
        (gradient(parameters + shift) - gradient(parameters - shift)) / 2e-4
        for shift in perturbations
    ])

    assert np.all(np.isfinite(hessian))
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hessian, finite_difference, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(scan_loss(parameters), core_loss(parameters), rtol=1e-12)


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
def test_checkpointed_scan_supports_reverse_over_reverse(method: str) -> None:
    """A 160-step solve retains higher-order derivatives under rematerialization."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = jnp.linspace(0.0, 0.6, 161, dtype=jnp.float64)
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float64)

    def loss(values: Array) -> Array:
        trajectory = _scan_trajectory(values, times, method=method, checkpoint=True)
        return _loss(values, trajectory[-1])

    def squared_gradient(values: Array) -> Array:
        return cast("Array", jnp.sum(jnp.square(jax.grad(loss)(values))))

    reverse = jax.jit(jax.grad(squared_gradient))(parameters)
    hessian = jax.jit(jax.hessian(loss))(parameters)
    expected = 2.0 * hessian.T @ jax.grad(loss)(parameters)
    assert np.all(np.isfinite(np.asarray(reverse)))
    np.testing.assert_allclose(reverse, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
def test_scan_hessian_includes_initial_state(method: str) -> None:
    """Mixed initial-state and RHS derivatives match an analytic solution."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    solver = CoreSolver(ModelCore(1, 1, np.asarray([0.0, 0.6])))
    times = jnp.asarray([0.0, 0.07, 0.2, 0.6], dtype=jnp.float64)

    def loss(values: Array) -> Array:
        rate, initial = values

        def advance(state: Array, step: tuple[Scalar, Scalar]) -> tuple[Array, None]:
            time, dt = step
            result, _stage = solver.fixed_explicit_step(
                lambda _time, y: jnp.full_like(y, rate),
                method=method,
                t=time,
                dt=dt,
                y=state,
            )
            return result, None

        final, _tail = jax.lax.scan(
            jax.checkpoint(advance),
            jnp.reshape(initial, (1, 1)),
            (times[:-1], jnp.diff(times)),
        )
        return cast("Array", 0.5 * jnp.sum(jnp.square(final)))

    parameters = jnp.asarray([-0.3, 1.2], dtype=jnp.float64)
    value, gradient = jax.jit(jax.value_and_grad(loss))(parameters)
    final = 1.2 - 0.3 * 0.6
    np.testing.assert_allclose(value, 0.5 * final**2, rtol=1e-12)
    np.testing.assert_allclose(gradient, final * np.asarray([0.6, 1.0]), rtol=1e-12)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(loss))(parameters),
        [[0.36, 0.6], [0.6, 1.0]],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.usefixtures("jax_x64")
def test_checkpointed_scan_supports_logdet_hessian_gradient() -> None:
    """The motivating Laplace derivative is finite and agrees with differences."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = jnp.linspace(0.0, 0.6, 101, dtype=jnp.float64)
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float64)

    def loss(values: Array) -> Array:
        trajectory = _scan_trajectory(values, times, method="dopri5", checkpoint=True)
        return _loss(values, trajectory[-1])

    def laplace(values: Array) -> Array:
        _sign, logdet = jnp.linalg.slogdet(jax.hessian(loss)(values))
        return cast("Array", 0.5 * logdet)

    eigenvalues = jnp.linalg.eigvalsh(jax.hessian(loss)(parameters))
    assert np.all(np.asarray(eigenvalues) > 0.0)
    gradient = jax.jit(jax.grad(laplace))(parameters)
    evaluate = jax.jit(laplace)
    expected = np.asarray([
        (evaluate(parameters + shift) - evaluate(parameters - shift)) / 2e-4
        for shift in 1e-4 * np.eye(2)
    ])
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(gradient, expected, rtol=2e-6, atol=2e-7)
