"""Cross-backend contracts for Array-API solver paths."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore, implicit_solve
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def _central_difference(
    function: Callable[..., Any],
    values: tuple[float, ...],
    argnum: int,
    *,
    step: float = 2e-3,
) -> float:
    """Approximate one scalar derivative independently of autodiff.

    Returns:
        Centered finite-difference derivative.
    """
    plus = list(values)
    minus = list(values)
    plus[argnum] += step
    minus[argnum] -= step
    return float(function(*plus) - function(*minus)) / (2.0 * step)


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


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
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


def test_dense_implicit_solve_preserves_jax_namespace() -> None:
    """Dense operators fall back to the input namespace's linear solver."""
    jnp = pytest.importorskip("jax.numpy")
    left = np.asarray([[2.0, -0.5], [-0.5, 1.5]], dtype=np.float32)
    right = np.asarray([[1.0, 0.25], [0.0, 1.0]], dtype=np.float32)
    state = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    result = implicit_solve(left, right, state)

    assert result.__array_namespace__() is jnp
    expected = np.linalg.solve(left, right @ np.asarray(state))
    assert np.allclose(np.asarray(result), expected, rtol=2e-6)


def test_cupy_sparse_implicit_solve_when_available() -> None:
    """The optional CuPy adapter solves sparse systems without a host result."""
    cupy = pytest.importorskip("cupy")
    cupy_sparse = pytest.importorskip("cupyx.scipy.sparse")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy is installed but the CUDA runtime is unavailable")

    left_dense = np.asarray([[2.0, -0.5], [-0.5, 1.5]], dtype=np.float32)
    right_dense = np.eye(2, dtype=np.float32)
    left = cupy_sparse.csr_matrix(cupy.asarray(left_dense))
    right = cupy_sparse.csr_matrix(cupy.asarray(right_dense))
    state = cupy.asarray([1.0, 2.0], dtype=cupy.float32)

    result = implicit_solve(left, right, state)

    assert isinstance(result, cupy.ndarray)
    expected = np.linalg.solve(left_dense, np.asarray([1.0, 2.0], dtype=np.float32))
    assert np.allclose(cupy.asnumpy(result), expected, rtol=2e-5)


def _solve_implicit_method(
    *,
    use_jax: bool,
    method: str,
    adaptive: bool = False,
) -> Array:
    """Solve one linear problem with an implicit or IMEX method.

    Returns:
        Final state in the requested namespace.
    """
    xp = pytest.importorskip("jax.numpy") if use_jax else np
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 0.1, 0.2]),
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(xp.asarray([[1.0]], dtype=xp.float32))

    def rhs(_time: float, state: Array) -> Array:
        namespace = cast("Any", state.__array_namespace__())
        return cast("Array", namespace.multiply(state, -0.2))

    if method.startswith("imex-"):
        left = np.asarray([[1.04]], dtype=np.float32)
        right = np.asarray([[1.0]], dtype=np.float32)

        if adaptive:

            def operator_factory(
                dt: float,
                scale: float,
                _context: object,
            ) -> tuple[np.ndarray, np.ndarray]:
                effective_dt = dt * scale
                return (
                    np.asarray([[1.0 + 0.4 * effective_dt]], dtype=np.float32),
                    right,
                )

            operators = operator_factory
        else:
            operators = (left, right)
        solver = CoreSolver(core, operators=operators)
        config = RunConfig(
            method=method,
            adaptive=adaptive,
            adaptive_cfg=AdaptiveConfig(rtol=1e-4, atol=1e-6, dt_init=0.025),
        )
    else:
        solver = CoreSolver(core)

        def jacobian(_time: float, state: Array) -> Array:
            namespace = cast("Any", state.__array_namespace__())
            return cast(
                "Array",
                namespace.asarray([[-0.2]], dtype=state.dtype),
            )

        config = RunConfig(
            method=method,
            jacobian=jacobian,
            adaptive=adaptive,
            adaptive_cfg=AdaptiveConfig(rtol=1e-4, atol=1e-6, dt_init=0.025),
        )

    solver.run(rhs, config=config)
    return core.get_current_state()


def _solve_jax_adaptive_method(
    method: str,
    explicit_rate: Array,
    operator_rate: Array,
    initial: Array,
    *,
    schedule: AdaptiveStepSchedule | None = None,
) -> tuple[Array, AdaptiveStepSchedule]:
    """Run or replay one adaptive scalar problem with JAX arrays.

    Returns:
        Final scalar state and the accepted-step schedule.
    """
    jnp = pytest.importorskip("jax.numpy")
    times = np.asarray([0.0, 0.2, 0.5], dtype=np.float32)
    core = ModelCore(
        1,
        1,
        times,
        options=ModelCoreOptions(dtype=np.float32),
    )
    core.set_initial_state(jnp.reshape(initial, (1, 1)))

    def rhs(_time: float, state: Array) -> Array:
        return cast("Array", jnp.multiply(state, explicit_rate))

    config_kwargs: dict[str, Any] = {}
    if method.startswith("imex-"):

        def operators(
            dt: float,
            scale: float,
            _context: object,
        ) -> tuple[Array, Array]:
            identity = jnp.eye(1, dtype=jnp.float32)
            generator = jnp.reshape(operator_rate, (1, 1))
            left = jnp.subtract(
                identity,
                jnp.multiply(generator, dt * scale),
            )
            return cast("Array", left), cast("Array", identity)

        solver = CoreSolver(core, operators=operators)
    else:
        solver = CoreSolver(core)
        if method not in {"euler", "heun", "rk4", "dopri5"}:

            def jacobian(_time: float, _state: Array) -> Array:
                return cast("Array", jnp.reshape(explicit_rate, (1, 1)))

            config_kwargs["jacobian"] = jacobian

    config = RunConfig(
        method=method,
        adaptive=True,
        adaptive_cfg=AdaptiveConfig(rtol=1e-4, atol=1e-6, dt_init=0.05),
        **config_kwargs,
    )
    if schedule is None:
        solver.run(rhs, config=config)
    else:
        solver.replay_adaptive_schedule(rhs, schedule, config=config)

    recorded_schedule = solver.last_adaptive_schedule
    assert recorded_schedule is not None
    final_state = cast("Array", core.get_current_state()[0, 0])
    return final_state, recorded_schedule


@pytest.mark.parametrize(
    "method",
    [
        "imex-euler",
        "imex-heun-tr",
        "imex-trbdf2",
        "implicit-euler",
        "trapezoidal",
        "bdf2",
        "ros2",
    ],
)
def test_numpy_and_jax_implicit_methods_agree(method: str) -> None:
    """Every existing dense linearly implicit method preserves its namespace."""
    jnp = pytest.importorskip("jax.numpy")
    numpy_result = _solve_implicit_method(use_jax=False, method=method)
    jax_result = _solve_implicit_method(use_jax=True, method=method)

    assert numpy_result.__array_namespace__() is np
    assert jax_result.__array_namespace__() is jnp
    assert np.allclose(np.asarray(jax_result), np.asarray(numpy_result), rtol=2e-5)


@pytest.mark.parametrize("method", ["imex-euler", "implicit-euler"])
def test_numpy_and_jax_adaptive_implicit_methods_agree(method: str) -> None:
    """Adaptive implicit controllers stay in the active array namespace."""
    jnp = pytest.importorskip("jax.numpy")
    numpy_result = _solve_implicit_method(
        use_jax=False,
        method=method,
        adaptive=True,
    )
    jax_result = _solve_implicit_method(
        use_jax=True,
        method=method,
        adaptive=True,
    )

    assert numpy_result.__array_namespace__() is np
    assert jax_result.__array_namespace__() is jnp
    assert np.allclose(np.asarray(jax_result), np.asarray(numpy_result), rtol=2e-5)


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_fixed_step_explicit_methods_support_jax_jit_and_grad(method: str) -> None:
    """Portable explicit methods trace and differentiate without Diffrax."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.asarray([0.0, 0.2, 0.5, 1.0], dtype=np.float32)

    def final_state(rate: Array, initial: Array) -> Array:
        core = ModelCore(
            1,
            1,
            times,
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.reshape(initial, (1, 1)))

        def rhs(_time: float, state: Array) -> Array:
            return cast("Array", jnp.multiply(state, rate))

        CoreSolver(core).run(rhs, config=RunConfig(method=method))
        return cast("Array", core.get_current_state()[0, 0])

    compiled = jax.jit(jax.value_and_grad(final_state, argnums=(0, 1)))
    values = (-0.3, 1.2)
    value, gradients = compiled(*map(jnp.asarray, values))
    expected_gradients = tuple(
        _central_difference(final_state, values, argnum) for argnum in range(2)
    )

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.allclose(np.asarray(gradients), expected_gradients, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize(
    "method",
    ["imex-euler", "imex-heun-tr", "imex-trbdf2"],
)
def test_fixed_step_imex_methods_support_jax_jit_and_grad(method: str) -> None:
    """Portable IMEX methods preserve gradients through dense operators."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.asarray([0.0, 0.2, 0.5], dtype=np.float32)

    def final_state(
        explicit_rate: Array,
        operator_rate: Array,
        initial: Array,
    ) -> Array:
        core = ModelCore(
            1,
            1,
            times,
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.reshape(initial, (1, 1)))

        def rhs(_time: float, state: Array) -> Array:
            return cast("Array", jnp.multiply(state, explicit_rate))

        def operators(
            dt: float,
            scale: float,
            _context: object,
        ) -> tuple[Array, Array]:
            effective_dt = dt * scale
            identity = jnp.eye(1, dtype=jnp.float32)
            generator = jnp.reshape(operator_rate, (1, 1))
            left = jnp.subtract(identity, jnp.multiply(generator, effective_dt))
            return cast("Array", left), cast("Array", identity)

        CoreSolver(core, operators=operators).run(
            rhs,
            config=RunConfig(method=method),
        )
        return cast("Array", core.get_current_state()[0, 0])

    compiled = jax.jit(jax.value_and_grad(final_state, argnums=(0, 1, 2)))
    values = (-0.2, -0.4, 1.2)
    value, gradients = compiled(*map(jnp.asarray, values))
    expected_gradients = tuple(
        _central_difference(final_state, values, argnum) for argnum in range(3)
    )

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.allclose(np.asarray(gradients), expected_gradients, rtol=3e-2, atol=3e-3)


@pytest.mark.parametrize(
    "method",
    ["implicit-euler", "trapezoidal", "bdf2", "ros2"],
)
def test_fixed_step_dense_implicit_methods_support_jax_jit_and_grad(
    method: str,
) -> None:
    """Dense linearly implicit methods trace through Array-API linalg."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.asarray([0.0, 0.2, 0.4], dtype=np.float32)

    def final_state(rate: Array, initial: Array) -> Array:
        core = ModelCore(
            1,
            1,
            times,
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.reshape(initial, (1, 1)))

        def rhs(_time: float, state: Array) -> Array:
            return cast("Array", jnp.multiply(state, rate))

        def jacobian(_time: float, _state: Array) -> Array:
            return cast("Array", jnp.reshape(rate, (1, 1)))

        CoreSolver(core).run(
            rhs,
            config=RunConfig(method=method, jacobian=jacobian),
        )
        return cast("Array", core.get_current_state()[0, 0])

    compiled = jax.jit(jax.value_and_grad(final_state, argnums=(0, 1)))
    values = (-0.3, 1.2)
    value, gradients = compiled(*map(jnp.asarray, values))
    expected_gradients = tuple(
        _central_difference(final_state, values, argnum) for argnum in range(2)
    )

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.allclose(np.asarray(gradients), expected_gradients, rtol=3e-2, atol=3e-3)


@pytest.mark.parametrize(
    ("method", "argnums"),
    [
        ("euler", (0, 2)),
        ("heun", (0, 2)),
        ("rk4", (0, 2)),
        ("dopri5", (0, 2)),
        ("imex-euler", (0, 1, 2)),
        ("imex-heun-tr", (0, 1, 2)),
        ("imex-trbdf2", (0, 1, 2)),
        ("implicit-euler", (0, 2)),
        ("trapezoidal", (0, 2)),
        ("ros2", (0, 2)),
    ],
)
def test_live_adaptive_methods_support_eager_jax_grad(
    method: str,
    argnums: tuple[int, ...],
) -> None:
    """Live Python adaptivity preserves gradients through every step family."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def final_state(
        explicit_rate: Array,
        operator_rate: Array,
        initial: Array,
    ) -> Array:
        result, _schedule = _solve_jax_adaptive_method(
            method,
            explicit_rate,
            operator_rate,
            initial,
        )
        return result

    values = (-0.2, -0.4, 1.2)
    value, gradients = jax.value_and_grad(final_state, argnums=argnums)(
        *map(jnp.asarray, values)
    )
    expected_gradients = tuple(
        _central_difference(final_state, values, argnum) for argnum in argnums
    )

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.allclose(np.asarray(gradients), expected_gradients, rtol=4e-2, atol=4e-3)


@pytest.mark.parametrize(
    ("method", "argnums"),
    [
        ("euler", (0, 2)),
        ("heun", (0, 2)),
        ("rk4", (0, 2)),
        ("dopri5", (0, 2)),
        ("imex-euler", (0, 1, 2)),
        ("imex-heun-tr", (0, 1, 2)),
        ("imex-trbdf2", (0, 1, 2)),
        ("implicit-euler", (0, 2)),
        ("trapezoidal", (0, 2)),
        ("ros2", (0, 2)),
    ],
)
def test_adaptive_schedule_replay_supports_jax_jit_and_grad(
    method: str,
    argnums: tuple[int, ...],
) -> None:
    """Frozen accepted meshes compile and differentiate through native kernels."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    values = (-0.2, -0.4, 1.2)
    live_value, schedule = _solve_jax_adaptive_method(
        method,
        *map(jnp.asarray, values),
    )

    def replayed_final_state(
        explicit_rate: Array,
        operator_rate: Array,
        initial: Array,
    ) -> Array:
        result, _schedule = _solve_jax_adaptive_method(
            method,
            explicit_rate,
            operator_rate,
            initial,
            schedule=schedule,
        )
        return result

    replay_value = replayed_final_state(*map(jnp.asarray, values))
    compiled = jax.jit(jax.value_and_grad(replayed_final_state, argnums=argnums))
    value, gradients = compiled(*map(jnp.asarray, values))
    expected_gradients = tuple(
        _central_difference(replayed_final_state, values, argnum) for argnum in argnums
    )

    assert np.allclose(replay_value, live_value, rtol=2e-6, atol=2e-7)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.allclose(np.asarray(gradients), expected_gradients, rtol=4e-2, atol=4e-3)
