"""Tests for the optional differentiable Diffrax CoreSolver strategy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import AdaptiveConfig, DtControllerConfig, RunConfig
from op_engine.model_core import ModelCoreOptions


def _diffrax_config() -> RunConfig:
    """Build the tight adaptive configuration shared by Diffrax tests.

    Returns:
        A Tsit5 run configuration.
    """
    return RunConfig(
        method="diffrax-tsit5",
        adaptive=True,
        adaptive_cfg=AdaptiveConfig(
            rtol=1e-6,
            atol=1e-8,
            dt_init=0.05,
            max_steps=4096,
        ),
        dt_controller=DtControllerConfig(dt_min=1e-6, dt_max=0.5),
    )


def _jax_core() -> ModelCore:
    """Build a one-state JAX-backed core.

    Returns:
        Initialized model core with four requested output times.
    """
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 0.2, 0.7, 1.0], dtype=np.float32),
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(jnp.asarray([[2.0]], dtype=jnp.float32))
    return core


def test_diffrax_tsit5_preserves_jax_history_and_matches_decay() -> None:
    """Tsit5 saves every requested output and matches the analytic solution."""
    core = _jax_core()

    def rhs(_time: object, state: jax.Array) -> jax.Array:
        return jnp.multiply(state, -0.4)

    CoreSolver(core).run(rhs, config=_diffrax_config())

    assert core.state_array is not None
    assert core.state_array.__array_namespace__() is jnp
    expected = 2.0 * np.exp(-0.4 * np.asarray(core.time_grid))
    np.testing.assert_allclose(
        np.asarray(core.state_array[:, 0, 0]),
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


def test_diffrax_tsit5_requires_adaptive_configuration() -> None:
    """Selecting the adaptive method without opt-in fails before execution."""
    core = _jax_core()

    with pytest.raises(ValueError, match="requires adaptive=True"):
        CoreSolver(core).run(
            lambda _time, state: state,
            config=RunConfig(method="diffrax-tsit5"),
        )


def test_diffrax_tsit5_rejects_implicit_namespace_conversion() -> None:
    """A NumPy state is rejected instead of silently moving to a JAX device."""
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 1.0], dtype=np.float32),
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(np.asarray([[1.0]], dtype=np.float32))

    with pytest.raises(TypeError, match="requires JAX state"):
        CoreSolver(core).run(
            lambda _time, state: -state,
            config=_diffrax_config(),
        )


def test_diffrax_tsit5_is_jittable_and_differentiable() -> None:
    """Dynamic RHS parameters flow through JIT compilation and gradients."""
    times = np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float32)
    config = _diffrax_config()

    def final_state(rate: jax.Array) -> jax.Array:
        core = ModelCore(
            1,
            1,
            times,
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.asarray([[1.0]], dtype=jnp.float32))

        def rhs(_time: object, state: jax.Array) -> jax.Array:
            return jnp.multiply(state, rate)

        CoreSolver(core).run(rhs, config=config)
        state = core.get_current_state()
        assert isinstance(state, jax.Array)
        return state[0, 0]

    rate = jnp.asarray(-0.3, dtype=jnp.float32)
    value, derivative = jax.jit(jax.value_and_grad(final_state))(rate)
    expected = np.exp(-0.3)

    assert float(value) == pytest.approx(expected, rel=3e-5)
    assert float(derivative) == pytest.approx(expected, rel=5e-5)
