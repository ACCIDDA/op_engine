"""Public CoreSolver contracts for nonlinear Alexander SDIRK2 integration."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from op_engine import (
    Array,
    CoreSolver,
    DenseNewtonSolver,
    ModelCore,
    NewtonConfig,
    NonlinearConvergenceError,
    NonlinearIntegrationConvergenceError,
    NonlinearIntegrationDiagnostics,
    NonlinearMethodConfig,
)
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    RunConfig,
)
from op_engine.model_core import ModelCoreOptions


def _nonlinear_decay_config(
    *,
    adaptive: bool = False,
    newton: NewtonConfig | None = None,
    adaptive_config: AdaptiveConfig | None = None,
) -> RunConfig:
    """Return an SDIRK2 config for ``y'=-y**2``."""

    def jacobian(_time: float, state: Array) -> Array:
        xp = state.__array_namespace__()
        return cast(
            "Array",
            xp.reshape(xp.multiply(state, -2.0), (1, 1)),
        )

    return RunConfig(
        method="sdirk2",
        adaptive=adaptive,
        adaptive_cfg=adaptive_config or AdaptiveConfig(),
        nonlinear=NonlinearMethodConfig(
            rhs_jacobian=jacobian,
            solver=DenseNewtonSolver(
                newton or NewtonConfig(max_iterations=6, atol=1e-13)
            ),
        ),
    )


def _nonlinear_decay(_time: float, state: Array) -> Array:
    xp = state.__array_namespace__()
    return cast("Array", xp.negative(xp.multiply(state, state)))


def _integrate_nonlinear_decay(dt: float) -> float:
    """Integrate one scalar nonlinear decay problem through CoreSolver.

    Returns:
        Final scalar state at time one.
    """
    n_steps = round(1.0 / dt)
    times = np.linspace(0.0, 1.0, n_steps + 1)
    core = ModelCore(1, 1, times)
    core.set_initial_state(np.asarray([[1.0]]))

    diagnostics = CoreSolver(core).run(
        _nonlinear_decay,
        config=_nonlinear_decay_config(),
    )

    assert diagnostics is not None
    diagnostics.require_converged()
    return float(core.get_current_state()[0, 0])


def test_core_sdirk2_has_second_order_on_nonlinear_decay() -> None:
    """The selectable method retains the prototype's nonlinear formal order."""
    errors = [abs(_integrate_nonlinear_decay(dt) - 0.5) for dt in (0.1, 0.05, 0.025)]
    orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0) for index in range(2)
    ]

    assert min(orders) > 1.8


def test_core_sdirk2_damps_a_very_stiff_mode() -> None:
    """The public Alexander method remains L-stable on a large negative mode."""
    rate = -1000.0
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))

    diagnostics = CoreSolver(core).run(
        lambda _time, state: cast("Array", np.multiply(state, rate)),
        config=RunConfig(
            method="sdirk2",
            nonlinear=NonlinearMethodConfig(
                rhs_jacobian=lambda _time, _state: cast("Array", np.asarray([[rate]]))
            ),
        ),
    )

    assert diagnostics is not None
    assert bool(diagnostics.converged.item())
    assert abs(float(core.get_current_state()[0, 0])) < 0.01


def test_fixed_sdirk2_returns_flat_array_diagnostics() -> None:
    """Fixed steps expose stable attempted-step and flattened-stage arrays."""
    core = ModelCore(1, 1, np.asarray([0.0, 0.1, 0.2]))
    core.set_initial_state(np.asarray([[1.0]]))
    solver = CoreSolver(core)

    diagnostics = solver.run(
        _nonlinear_decay,
        config=_nonlinear_decay_config(),
    )

    assert diagnostics is solver.last_nonlinear_diagnostics
    assert diagnostics is not None
    assert diagnostics.require_converged() is diagnostics
    assert diagnostics.converged.__array_namespace__() is np
    assert np.array_equal(diagnostics.step_accepted, [True, True])
    assert np.array_equal(diagnostics.stages_per_step, [2, 2])
    assert diagnostics.stage_converged.shape == (4,)
    assert diagnostics.residual_norm.shape == (4,)
    assert np.array_equal(diagnostics.iterations, [6, 6, 6, 6])


def test_fixed_eager_nonconvergence_raises_without_advancing_state() -> None:
    """Fixed eager failure preserves state and raises the exact stage error."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[10.0]]))
    solver = CoreSolver(core)
    config = _nonlinear_decay_config(
        newton=NewtonConfig(max_iterations=1, rtol=0.0, atol=1e-14),
    )

    with pytest.raises(NonlinearConvergenceError):
        solver.run(_nonlinear_decay, config=config)

    assert core.current_step == 0
    assert np.array_equal(core.get_current_state(), [[10.0]])
    diagnostics = solver.last_nonlinear_diagnostics
    assert diagnostics is not None
    assert not bool(diagnostics.converged.item())
    assert np.array_equal(diagnostics.step_accepted, [False])


def test_live_adaptive_sdirk2_rejects_nonconvergence_then_recovers() -> None:
    """Ordinary Newton failure consumes rejection budget and reduces dt."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))
    solver = CoreSolver(core)
    config = _nonlinear_decay_config(
        adaptive=True,
        newton=NewtonConfig(max_iterations=2, rtol=0.0, atol=1e-6),
        adaptive_config=AdaptiveConfig(
            rtol=1.0,
            atol=1.0,
            dt_init=1.0,
            max_reject=8,
        ),
    )

    diagnostics = solver.run(_nonlinear_decay, config=config)

    assert diagnostics is not None
    assert bool(diagnostics.converged.item())
    step_converged = np.asarray(diagnostics.step_converged)
    step_accepted = np.asarray(diagnostics.step_accepted)
    assert np.any(np.logical_and(~step_converged, ~step_accepted))
    assert np.all(step_converged[step_accepted])
    assert np.all(np.asarray(diagnostics.stages_per_step) == 6)
    assert solver.last_adaptive_schedule is not None
    assert core.current_step == 1


def test_adaptive_nonconvergence_honors_max_reject_without_state_change() -> None:
    """Repeated Newton failure stops at max_reject and never commits a state."""
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[10.0]]))
    solver = CoreSolver(core)
    config = _nonlinear_decay_config(
        adaptive=True,
        newton=NewtonConfig(max_iterations=1, rtol=0.0, atol=1e-14),
        adaptive_config=AdaptiveConfig(dt_init=1.0, max_reject=2),
    )

    with pytest.raises(RuntimeError, match="Too many rejected steps"):
        solver.run(_nonlinear_decay, config=config)

    assert core.current_step == 0
    assert np.array_equal(core.get_current_state(), [[10.0]])
    diagnostics = solver.last_nonlinear_diagnostics
    assert diagnostics is not None
    assert diagnostics.step_accepted.shape == (2,)
    assert not np.any(np.asarray(diagnostics.step_accepted))


def test_adaptive_callback_exception_remains_fatal() -> None:
    """Model callback failures are not reclassified as timestep rejection."""

    class CallbackError(RuntimeError):
        """Sentinel model failure."""

    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))
    solver = CoreSolver(core)

    def broken_rhs(_time: float, _state: Array) -> Array:
        raise CallbackError

    with pytest.raises(CallbackError):
        solver.run(
            broken_rhs,
            config=_nonlinear_decay_config(adaptive=True),
        )

    assert core.current_step == 0


def test_adaptive_linear_algebra_exception_remains_fatal() -> None:
    """A singular Newton matrix is not reclassified as nonconvergence."""
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(np.asarray([[1.0]]))

    with pytest.raises(np.linalg.LinAlgError):
        CoreSolver(core).run(
            lambda _time, state: cast("Array", np.ones_like(state)),
            config=RunConfig(
                method="sdirk2",
                adaptive=True,
                nonlinear=NonlinearMethodConfig(
                    rhs_jacobian=lambda _time, _state: cast(
                        "Array", np.asarray([[1.0 / gamma]])
                    )
                ),
            ),
        )

    assert core.current_step == 0


def test_numpy_and_jax_fixed_sdirk2_agree() -> None:
    """The same nonlinear stages preserve and agree across array namespaces."""
    jnp = pytest.importorskip("jax.numpy")

    def solve(*, use_jax: bool) -> tuple[np.ndarray, NonlinearIntegrationDiagnostics]:
        xp = jnp if use_jax else np
        dtype = np.float32
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 0.1, 0.2]),
            options=ModelCoreOptions(dtype=dtype),
        )
        core.set_initial_state(xp.asarray([[1.0]], dtype=dtype))
        diagnostics = CoreSolver(core).run(
            _nonlinear_decay,
            config=_nonlinear_decay_config(
                newton=NewtonConfig(max_iterations=6, rtol=1e-6, atol=1e-7)
            ),
        )
        assert diagnostics is not None
        return np.asarray(core.get_current_state()), diagnostics

    numpy_state, numpy_diagnostics = solve(use_jax=False)
    jax_state, jax_diagnostics = solve(use_jax=True)

    assert numpy_diagnostics.converged.__array_namespace__() is np
    assert jax_diagnostics.converged.__array_namespace__() is jnp
    assert np.allclose(jax_state, numpy_state, rtol=2e-6)
    assert np.allclose(
        np.asarray(jax_diagnostics.residual_norm),
        np.asarray(numpy_diagnostics.residual_norm),
        rtol=1e-4,
        atol=1e-7,
    )


def test_multidimensional_sdirk2_uses_full_flattened_jacobian() -> None:
    """A rank-two state is solved as one four-unknown nonlinear system."""
    initial = np.asarray([[1.0, 0.8], [0.6, 0.4]])
    core = ModelCore(
        2,
        2,
        np.asarray([0.0, 0.05]),
        options=ModelCoreOptions(dtype=np.float64),
    )
    core.set_initial_state(initial)

    def rhs(_time: float, state: Array) -> Array:
        return cast("Array", np.negative(np.multiply(state, state)))

    def full_jacobian(_time: float, state: Array) -> Array:
        diagonal = np.reshape(np.multiply(state, -2.0), (4,))
        return cast("Array", np.diag(diagonal))

    diagnostics = CoreSolver(core).run(
        rhs,
        config=RunConfig(
            method="sdirk2",
            nonlinear=NonlinearMethodConfig(rhs_jacobian=full_jacobian),
        ),
    )

    assert diagnostics is not None
    diagnostics.require_converged()
    assert core.get_current_state().shape == (2, 2)
    assert np.all(np.asarray(core.get_current_state()) < initial)


def test_sdirk2_rejects_operator_axis_jacobian_shape() -> None:
    """The nonlinear Jacobian cannot silently use operator-axis semantics."""
    core = ModelCore(2, 2, np.asarray([0.0, 0.05]))
    core.set_initial_state(np.ones((2, 2)))

    with pytest.raises(ValueError, match=r"flattened system shape \(4, 4\)"):
        CoreSolver(core).run(
            lambda _time, state: cast("Array", np.negative(state)),
            config=RunConfig(
                method="sdirk2",
                nonlinear=NonlinearMethodConfig(
                    rhs_jacobian=lambda _time, _state: cast(
                        "Array", np.negative(np.eye(2))
                    )
                ),
            ),
        )


def test_sdirk2_requires_separate_nonlinear_configuration() -> None:
    """The existing operator-axis Jacobian is not overloaded for SDIRK2."""
    core = ModelCore(1, 1, np.asarray([0.0, 0.1]))
    core.set_initial_state(np.asarray([[1.0]]))

    with pytest.raises(ValueError, match="NonlinearMethodConfig"):
        CoreSolver(core).run(
            _nonlinear_decay,
            config=RunConfig(
                method="sdirk2",
                jacobian=lambda _time, _state: np.asarray([[-2.0]]),
            ),
        )


def test_jax_compiled_replay_returns_diagnostics_and_gradients() -> None:
    """Frozen SDIRK meshes return native diagnostics under jit and grad."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    schedule = AdaptiveStepSchedule(
        output_times=(0.0, 0.5),
        step_sizes=((0.25, 0.25),),
    )

    def replay(
        rate: Array,
        initial: Array,
    ) -> tuple[Array, NonlinearIntegrationDiagnostics]:
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 0.5]),
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.reshape(initial, (1, 1)))

        def rhs(_time: float, state: Array) -> Array:
            return cast("Array", jnp.multiply(state, rate))

        def jacobian(_time: float, _state: Array) -> Array:
            return cast("Array", jnp.reshape(rate, (1, 1)))

        diagnostics = CoreSolver(core).replay_adaptive_schedule(
            rhs,
            schedule,
            config=RunConfig(
                method="sdirk2",
                adaptive=True,
                nonlinear=NonlinearMethodConfig(
                    rhs_jacobian=jacobian,
                    solver=DenseNewtonSolver(
                        NewtonConfig(max_iterations=3, rtol=1e-6, atol=1e-7)
                    ),
                ),
            ),
        )
        assert diagnostics is not None
        return cast("Array", core.get_current_state()[0, 0]), diagnostics

    rate = jnp.asarray(-0.4, dtype=jnp.float32)
    initial = jnp.asarray(1.2, dtype=jnp.float32)
    value, diagnostics = jax.jit(replay)(rate, initial)
    differentiated_value, gradients = jax.jit(
        jax.value_and_grad(lambda r, y: replay(r, y)[0], argnums=(0, 1))
    )(rate, initial)

    epsilon = 1e-3
    finite_rate = (
        replay(rate + epsilon, initial)[0] - replay(rate - epsilon, initial)[0]
    ) / (2.0 * epsilon)
    finite_initial = (
        replay(rate, initial + epsilon)[0] - replay(rate, initial - epsilon)[0]
    ) / (2.0 * epsilon)

    assert diagnostics.converged.__array_namespace__() is jnp
    assert bool(diagnostics.converged.item())
    assert np.array_equal(np.asarray(diagnostics.stages_per_step), [6, 6])
    assert np.allclose(differentiated_value, value, rtol=1e-6)
    assert np.allclose(gradients[0], finite_rate, rtol=2e-3, atol=2e-4)
    assert np.allclose(gradients[1], finite_initial, rtol=2e-3, atol=2e-4)


def test_failed_compiled_replay_can_be_invalidated_after_execution() -> None:
    """A frozen mesh reports failure on-device and raises only when inspected."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    schedule = AdaptiveStepSchedule(
        output_times=(0.0, 1.0),
        step_sizes=((1.0,),),
    )

    def replay(initial: Array) -> NonlinearIntegrationDiagnostics:
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 1.0], dtype=np.float32),
            options=ModelCoreOptions(dtype=np.float32),
        )
        core.set_initial_state(jnp.reshape(initial, (1, 1)))

        def rhs(_time: float, state: Array) -> Array:
            return cast("Array", jnp.negative(jnp.multiply(state, state)))

        def jacobian(_time: float, state: Array) -> Array:
            return cast("Array", jnp.reshape(jnp.multiply(state, -2.0), (1, 1)))

        diagnostics = CoreSolver(core).replay_adaptive_schedule(
            rhs,
            schedule,
            config=RunConfig(
                method="sdirk2",
                adaptive=True,
                nonlinear=NonlinearMethodConfig(
                    rhs_jacobian=jacobian,
                    solver=DenseNewtonSolver(
                        NewtonConfig(max_iterations=1, rtol=0.0, atol=1e-14)
                    ),
                ),
            ),
        )
        assert diagnostics is not None
        return diagnostics

    diagnostics = jax.jit(replay)(jnp.asarray(10.0, dtype=jnp.float32))

    assert not bool(diagnostics.converged.item())
    with pytest.raises(NonlinearIntegrationConvergenceError):
        diagnostics.require_converged()
