"""Qualification of the public explicit step in external JAX loops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore, Scalar, array_namespace
from op_engine.core_solver import RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable


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
