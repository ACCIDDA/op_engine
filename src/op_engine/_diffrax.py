"""Differentiable adaptive solve strategies backed by Diffrax."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import diffrax
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    from ._typing import Array


_JAX_STATE_ERROR = (
    "The diffrax-tsit5 method requires JAX state. Initialize ModelCore with "
    "a jax.Array instead of relying on an implicit host-to-device conversion."
)
_RHS_SHAPE_ERROR = "rhs shape {actual} does not match initial state shape {expected}"


def _is_jax_value(value: object) -> bool:
    """Return whether ``value`` is a concrete or traced JAX array."""
    return isinstance(value, jax.Array | jax.core.Tracer)


def solve_tsit5(  # noqa: C901, PLR0913
    rhs_func: Callable[..., Array],
    time_grid: NDArray[np.floating],
    initial_state: Array,
    *,
    rtol: float,
    atol: float | Array,
    dt_init: float | None,
    dt_min: float,
    dt_max: float,
    max_steps: int,
) -> Array:
    """Solve one JAX ODE trajectory with adaptive Diffrax Tsit5.

    This is an internal CoreSolver strategy. It intentionally accepts an
    already-assembled state and RHS: model/provider concerns such as parameter
    binding, PyTree layout, operators, batching, and history kernels remain at
    their own boundaries and can be added in later slices.

    Args:
        rhs_func: JAX-traceable vector field ``f(t, y)``.
        time_grid: Strictly increasing output times owned by ``ModelCore``.
        initial_state: JAX state at ``time_grid[0]``.
        rtol: Relative error tolerance for the PID controller.
        atol: Absolute error tolerance for the PID controller.
        dt_init: Optional initial internal step size.
        dt_min: Minimum internal step size; zero means no lower bound.
        dt_max: Maximum internal step size; infinity means no upper bound.
        max_steps: Maximum number of internal Diffrax steps.

    Returns:
        JAX trajectory with shape ``(len(time_grid), *initial_state.shape)``.

    Raises:
        TypeError: If ``initial_state`` is not a JAX array or tracer.
        ValueError: If controller limits or RHS shape are invalid.
    """
    if not _is_jax_value(initial_state):
        raise TypeError(_JAX_STATE_ERROR)
    if max_steps <= 0:
        msg = "Diffrax max_steps must be positive."
        raise ValueError(msg)
    if dt_min < 0.0:
        msg = "Diffrax dt_min must be nonnegative."
        raise ValueError(msg)
    if dt_max <= 0.0:
        msg = "Diffrax dt_max must be positive."
        raise ValueError(msg)
    if dt_min > 0.0 and dt_min > dt_max:
        msg = "Diffrax requires dt_min <= dt_max."
        raise ValueError(msg)

    times = jnp.asarray(time_grid, dtype=cast("Any", initial_state.dtype))
    if dt_init is None:
        dt0 = times[1] - times[0]
    else:
        if dt_init <= 0.0:
            msg = "Diffrax dt_init must be positive."
            raise ValueError(msg)
        dt0 = jnp.asarray(dt_init, dtype=times.dtype)

    controller_kwargs: dict[str, Any] = {"rtol": rtol, "atol": atol}
    if dt_min > 0.0:
        controller_kwargs["dtmin"] = dt_min
    if math.isfinite(dt_max):
        controller_kwargs["dtmax"] = dt_max

    expected_shape = initial_state.shape

    def vector_field(time: object, state: Array, _args: object) -> Array:
        result = rhs_func(time, state)
        if result.shape != expected_shape:
            raise ValueError(
                _RHS_SHAPE_ERROR.format(
                    actual=result.shape,
                    expected=expected_shape,
                )
            )
        if not _is_jax_value(result):
            msg = "diffrax-tsit5 rhs_func must return JAX state."
            raise TypeError(msg)
        return result

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=times[0],
        t1=times[-1],
        dt0=dt0,
        y0=initial_state,
        saveat=diffrax.SaveAt(ts=times),
        stepsize_controller=diffrax.PIDController(**controller_kwargs),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
        throw=True,
    )
    trajectory = cast("Array", solution.ys)
    expected_trajectory_shape = (time_grid.size, *expected_shape)
    if trajectory.shape != expected_trajectory_shape:
        msg = (
            "Diffrax returned trajectory shape "
            f"{trajectory.shape}; expected {expected_trajectory_shape}."
        )
        raise ValueError(msg)
    return trajectory


__all__: list[str] = []
