"""Cross-backend contracts for Array-API explicit solver paths."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore
from op_engine.core_solver import AdaptiveConfig, RunConfig
from op_engine.model_core import ModelCoreOptions


def _solve_explicit(
    *,
    use_jax: bool,
    method: str,
    adaptive: bool,
) -> tuple[Array, Array]:
    """Solve one decay problem in the requested runtime namespace.

    Returns:
        Final state and full stored history.
    """
    xp = pytest.importorskip("jax.numpy") if use_jax else np

    core = ModelCore(
        n_states=2,
        n_subgroups=1,
        time_grid=np.asarray([0.0, 0.2, 0.7, 1.0]),
        options=ModelCoreOptions(dtype=np.float32, store_history=True),
    )
    core.set_initial_state(xp.asarray([[1.0], [0.5]], dtype=xp.float32))

    def rhs(_time: float, state: Array) -> Array:
        namespace = cast("Any", state.__array_namespace__())
        return cast("Array", namespace.multiply(state, -0.25))

    config = RunConfig(
        method=method,
        adaptive=adaptive,
        adaptive_cfg=AdaptiveConfig(
            rtol=1e-4,
            atol=1e-6,
            dt_init=0.05,
        ),
    )
    CoreSolver(core).run(rhs, config=config)
    assert core.state_array is not None
    return core.get_current_state(), core.state_array


def test_array_protocol_is_exported_and_runtime_checkable() -> None:
    """The public protocol matches NumPy's Array-API surface."""
    assert isinstance(np.asarray([1.0]), Array)


def test_model_core_options_no_longer_store_backend_module() -> None:
    """Namespace selection comes from state values, not an ``xp`` option."""
    assert "xp" not in {field.name for field in fields(ModelCoreOptions)}


def test_model_core_preserves_jax_namespace_for_updates_and_history() -> None:
    """Immutable JAX state/history updates never fall back to NumPy."""
    jnp = pytest.importorskip("jax.numpy")
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 0.5, 1.0]),
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(jnp.asarray([[1.0]], dtype=jnp.float32))
    core.apply_deltas(jnp.asarray([[0.5]], dtype=jnp.float32))
    core.apply_next_state(jnp.asarray([[3.0]], dtype=jnp.float32))

    assert core.get_current_state().__array_namespace__() is jnp
    assert core.state_array is not None
    assert core.state_array.__array_namespace__() is jnp
    assert np.array_equal(np.asarray(core.state_array[:, 0, 0]), [1.0, 1.5, 3.0])


@pytest.mark.parametrize("method", ["euler", "heun"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_numpy_and_jax_explicit_paths_agree(method: str, *, adaptive: bool) -> None:
    """The same explicit solve stays native and agrees across namespaces."""
    jnp = pytest.importorskip("jax.numpy")
    numpy_state, numpy_history = _solve_explicit(
        use_jax=False,
        method=method,
        adaptive=adaptive,
    )
    jax_state, jax_history = _solve_explicit(
        use_jax=True,
        method=method,
        adaptive=adaptive,
    )

    assert numpy_state.__array_namespace__() is np
    assert numpy_history.__array_namespace__() is np
    assert jax_state.__array_namespace__() is jnp
    assert jax_history.__array_namespace__() is jnp
    assert np.allclose(np.asarray(jax_state), np.asarray(numpy_state), rtol=2e-6)
    assert np.allclose(np.asarray(jax_history), np.asarray(numpy_history), rtol=2e-6)


def test_jax_state_gets_clear_error_at_scipy_boundary() -> None:
    """Implicit/IMEX methods remain explicitly NumPy-only until issue #69."""
    jnp = pytest.importorskip("jax.numpy")
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 1.0]),
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(jnp.asarray([[1.0]]))
    identity = np.eye(1)
    solver = CoreSolver(core, operators=(identity, identity))

    with pytest.raises(TypeError, match="requires NumPy state arrays"):
        solver.run(
            lambda _time, state: state,
            config=RunConfig(method="imex-euler"),
        )
