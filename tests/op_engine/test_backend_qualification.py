"""Required CPU method contracts on a structured, axis-coupled linear model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from op_engine import (
    Array,
    CoreSolver,
    DenseNewtonSolver,
    ModelCore,
    NewtonConfig,
    NonlinearMethodConfig,
    array_namespace,
    implicit_solve,
)
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    OperatorSpecs,
    RHSFunction,
    RunConfig,
    StepIO,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Iterator

    from op_engine.matrix_ops import StageOperatorContext

BACKENDS = ("numpy", "jax", "torch")
METHODS = (
    "euler",
    "heun",
    "rk4",
    "dopri5",
    "imex-euler",
    "imex-heun-tr",
    "imex-trbdf2",
    "imex-ark3",
    "implicit-euler",
    "trapezoidal",
    "bdf2",
    "ros2",
    "sdirk2",
)
TIMES = np.linspace(0.0, 0.12, 7)
INITIAL = np.arange(1.0, 19.0).reshape(2, 3, 3) / 20 + 0.5
# Target-row/source-column reflecting transport; every column sums to zero.
TRANSPORT = np.asarray([[-0.3, 0.0, 0.0], [0.3, -0.3, 0.0], [0.0, 0.3, 0.0]])
GENERATOR = TRANSPORT - 0.2 * np.eye(3)


def _module(backend: str) -> Any:  # noqa: ANN401
    """Resolve the explicitly selected optional namespace.

    Returns:
        Numerical array module.
    """
    return (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )


@contextmanager
def _precision(backend: str) -> Iterator[None]:
    if backend == "jax":
        jax = pytest.importorskip("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        with enable():
            yield
    else:
        yield


def _host(value: Any) -> np.ndarray:  # noqa: ANN401
    """Extract eager output for independent comparisons.

    Returns:
        Host values.
    """
    return np.asarray(value.detach().numpy() if hasattr(value, "detach") else value)


def _reference(time: float, initial: np.ndarray = INITIAL) -> np.ndarray:
    """Solve each independent batch using the exact matrix exponential.

    Returns:
        Reference state retaining compartment/group/trait order.
    """
    propagator = expm(time * GENERATOR)
    return np.asarray([[propagator @ row for row in group] for group in initial])


def _solver_case(
    module: Any,  # noqa: ANN401
    dtype: str,
    method: str,
    *,
    initial: Array | None = None,
    adaptive: bool = False,
) -> tuple[CoreSolver, RHSFunction, RunConfig]:
    """Build native stage operators acting along the non-default trait axis.

    Returns:
        Solver, independent numerical RHS, and matching configuration.
    """
    core = ModelCore(
        2,
        3,
        TIMES,
        options=ModelCoreOptions(
            dtype=np.dtype(dtype),
            other_axes=(3,),
            axis_names=("state", "group", "trait"),
        ),
    )
    core.set_initial_state(
        module.asarray(INITIAL.astype(dtype)) if initial is None else initial
    )
    state = core.get_current_state()
    xp = array_namespace(state)
    generator = xp.asarray(GENERATOR, dtype=state.dtype)
    transport = xp.asarray(TRANSPORT, dtype=state.dtype)

    def rhs(_time: object, value: Array) -> Array:
        return cast(
            "Array",
            -0.2 * value
            if method.startswith("imex-")
            else xp.matmul(value, xp.matrix_transpose(generator)),
        )

    def jacobian(_time: object, _state: Array) -> Array:
        return cast("Array", generator)

    def full_jacobian(_time: object, _state: Array) -> Array:
        return cast(
            "Array", xp.asarray(np.kron(np.eye(6), GENERATOR), dtype=state.dtype)
        )

    def factory(dt: Any, scale: float, _context: object) -> tuple[Array, Array]:  # noqa: ANN401
        identity = xp.eye(3, dtype=state.dtype)
        return cast("Array", identity - dt * scale * transport), cast("Array", identity)

    def trapezoidal(dt: Any, scale: float, _context: object) -> tuple[Array, Array]:  # noqa: ANN401
        identity = xp.eye(3, dtype=state.dtype)
        half = 0.5 * dt * scale * transport
        return cast("Array", identity - half), cast("Array", identity + half)

    operators = OperatorSpecs()
    if method in {"imex-euler", "imex-ark3"}:
        operators = OperatorSpecs(default=factory)
    elif method == "imex-heun-tr":
        operators = OperatorSpecs(default=trapezoidal)
    elif method == "imex-trbdf2":
        operators = OperatorSpecs(tr=trapezoidal, bdf2=factory)
    config = RunConfig(
        method=method,
        adaptive=adaptive,
        adaptive_cfg=AdaptiveConfig(rtol=2e-5, atol=2e-7, dt_init=0.01),
        operators=operators,
        jacobian=jacobian,
        nonlinear=NonlinearMethodConfig(
            rhs_jacobian=full_jacobian,
            solver=DenseNewtonSolver(
                NewtonConfig(
                    max_iterations=3,
                    rtol=1e-5,
                    atol=3e-7 if dtype == "float32" else 1e-12,
                ),
            ),
        )
        if method == "sdirk2"
        else None,
    )
    return CoreSolver(core, operator_axis="trait"), rhs, config


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", METHODS)
def test_structured_fixed_methods(backend: str, dtype: str, method: str) -> None:
    """Every deterministic method preserves layout and matches the independent model."""
    with _precision(backend):
        solver, rhs, config = _solver_case(_module(backend), dtype, method)
        diagnostics = solver.run(rhs, config=config)
        if diagnostics is not None:
            diagnostics.require_converged()
        state = solver.core.get_current_state()
        assert state.shape == INITIAL.shape
        assert state.dtype == solver.core.get_state_at(0).dtype
        assert array_namespace(state) is array_namespace(solver.core.get_state_at(0))
        np.testing.assert_allclose(
            _host(state), _reference(TIMES[-1]), rtol=8e-4, atol=1e-6
        )
        assert solver.core.state_array is not None
        assert solver.core.state_array.shape == (len(TIMES), *INITIAL.shape)
        np.testing.assert_allclose(
            _host(state).sum(axis=2),
            INITIAL.sum(axis=2) * np.exp(-0.2 * TIMES[-1]),
            rtol=1e-4,
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", ["dopri5", "imex-heun-tr", "ros2", "sdirk2"])
def test_structured_live_and_portable_replay(
    backend: str, dtype: str, method: str
) -> None:
    """Native live stepping and a NumPy-recorded mesh produce the same trajectory."""
    nominal, rhs, config = _solver_case(np, dtype, method, adaptive=True)
    nominal.run(rhs, config=config)
    schedule = nominal.last_adaptive_schedule
    assert schedule is not None
    with _precision(backend):
        module = _module(backend)
        live, live_rhs, live_config = _solver_case(module, dtype, method, adaptive=True)
        live.run(live_rhs, config=live_config)
        replay, replay_rhs, replay_config = _solver_case(
            module, dtype, method, adaptive=True
        )
        replay.replay_adaptive_schedule(
            replay_rhs, schedule, config=replace(replay_config, replay_loop="auto")
        )
        for solver in [live, replay]:
            state = solver.core.get_current_state()
            assert state.shape == INITIAL.shape
            assert state.dtype == solver.core.get_state_at(0).dtype
            assert array_namespace(state) is array_namespace(
                solver.core.get_state_at(0)
            )
            np.testing.assert_allclose(
                _host(state), _reference(TIMES[-1]), rtol=1e-4, atol=2e-6
            )
        np.testing.assert_allclose(
            _host(replay.core.get_current_state()),
            nominal.core.get_current_state(),
            rtol=3e-6,
            atol=2e-6,
        )


def _heun_tr_decay_case(
    module: Any,  # noqa: ANN401
    explicit_rate: float,
    *,
    tolerance: float,
) -> tuple[CoreSolver, RHSFunction, RunConfig, list[tuple[float, float, str]]]:
    """Build y' = (-2 + explicit_rate)*y with an independent exponential solution.

    Returns:
        Solver, RHS, configuration and eager operator factory observations.
    """
    core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
    core.set_initial_state(module.asarray([[1.0]], dtype=module.float64))
    state = core.get_current_state()
    xp = array_namespace(state)
    events: list[tuple[float, float, str]] = []

    def rhs(_time: object, value: Array) -> Array:
        return cast("Array", xp.multiply(value, explicit_rate))

    def factory(
        dt: float, scale: float, context: StageOperatorContext
    ) -> tuple[Array, Array]:
        events.append((dt, context.t, context.stage))
        identity = xp.eye(1, dtype=state.dtype)
        # A=-2, so L=I-h*A/2 and R=I+h*A/2.
        return (
            cast("Array", xp.multiply(identity, 1.0 + dt * scale)),
            cast("Array", xp.multiply(identity, 1.0 - dt * scale)),
        )

    config = RunConfig(
        method="imex-heun-tr",
        adaptive=True,
        operators=OperatorSpecs(default=factory),
        adaptive_cfg=AdaptiveConfig(rtol=tolerance, atol=tolerance * 1e-3, dt_init=1.0),
    )
    return CoreSolver(core), rhs, config, events


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("explicit_rate", [0.0, -0.3])
def test_heun_tr_estimator_covers_both_partitions(
    backend: str, explicit_rate: float
) -> None:
    """The cubic estimate approaches the full-step error, including when F=0."""
    with _precision(backend):
        solver, rhs, config, events = _heun_tr_decay_case(
            _module(backend), explicit_rate, tolerance=1e-3
        )
        state = solver.core.get_current_state()
        plan = solver._resolve_run_plan(config)  # noqa: SLF001
        errors = []
        for dt in (0.05, 0.025):
            events.clear()
            if backend == "numpy":
                high = np.empty_like(state)
                error = np.empty_like(state)
                controller_order = solver._attempt_step(  # noqa: SLF001
                    rhs,
                    plan=plan,
                    step=StepIO(t=0.0, dt=dt, y=state, out=high, err_out=error),
                )
            else:
                high, error, controller_order = solver._attempt_array_implicit_step(  # noqa: SLF001
                    rhs, plan=plan, t=0.0, dt=dt, y=state
                )
            actual_error = abs(
                float(_host(high)[0, 0]) - np.exp((-2 + explicit_rate) * dt)
            )
            estimated_error = abs(float(_host(error)[0, 0]))
            assert estimated_error > 0
            assert estimated_error / actual_error == pytest.approx(1.0, rel=0.06)
            assert controller_order == 2
            assert high.dtype == state.dtype
            assert array_namespace(high) is array_namespace(state)
            assert error.dtype == state.dtype
            assert array_namespace(error) is array_namespace(state)
            assert events == [
                (dt, dt, "tr-predictor"),
                (dt, dt, "tr"),
                (dt / 2, dt / 2, "tr-predictor"),
                (dt / 2, dt / 2, "tr"),
                (dt / 2, dt, "tr-predictor"),
                (dt / 2, dt, "tr"),
            ]
            if explicit_rate == 0:
                assert float(_host(high)[0, 0]) == pytest.approx((1 - dt) / (1 + dt))
            errors.append(estimated_error)
        assert errors[0] / errors[1] == pytest.approx(8.0, rel=0.1)
        np.testing.assert_array_equal(_host(state), [[1.0]])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("explicit_rate", [0.0, -0.3])
def test_heun_tr_tolerance_refinement_and_replay(
    backend: str, explicit_rate: float
) -> None:
    """Tighter tolerance improves independent accuracy; replay skips refinement."""
    with _precision(backend):
        module = _module(backend)
        errors = []
        counts = []
        for tolerance in (1e-3, 1e-6):
            solver, rhs, config, _events = _heun_tr_decay_case(
                module, explicit_rate, tolerance=tolerance
            )
            solver.run(rhs, config=config)
            schedule = solver.last_adaptive_schedule
            assert schedule is not None
            assert schedule.provenance is not None
            discovery = dict(schedule.provenance.discovery_settings)
            assert discovery["error_estimator"] == "step-doubling: full-step error"
            assert discovery["error_estimator_partitions"] == "explicit,implicit"
            assert discovery["error_estimator_order"] == "3"
            counts.append(sum(map(len, schedule.step_sizes)))
            errors.append(
                abs(
                    float(_host(solver.core.get_current_state())[0, 0])
                    - np.exp(-2 + explicit_rate)
                )
            )
            replay, replay_rhs, replay_config, events = _heun_tr_decay_case(
                module, explicit_rate, tolerance=tolerance
            )
            replay.replay_adaptive_schedule(
                replay_rhs,
                schedule,
                config=replace(replay_config, replay_loop="unroll"),
            )
            assert len(events) == 2 * counts[-1]
            np.testing.assert_array_equal(
                _host(replay.core.get_current_state()),
                _host(solver.core.get_current_state()),
            )
        assert counts[1] > counts[0]
        assert errors[1] < 0.04 * errors[0]
        assert errors[1] < 5e-5


@pytest.mark.parametrize("backend", BACKENDS)
def test_heun_tr_fixed_static_operator_retains_original_step(backend: str) -> None:
    """Fixed static pairs retain one trapezoidal step, without half-step solves."""
    with _precision(backend):
        fixed, fixed_rhs, fixed_config, fixed_events = _heun_tr_decay_case(
            _module(backend), 0.0, tolerance=1e-3
        )
        fixed.run(fixed_rhs, config=replace(fixed_config, adaptive=False))
        assert fixed_events == [(1.0, 1.0, "tr-predictor"), (1.0, 1.0, "tr")]
        np.testing.assert_array_equal(_host(fixed.core.get_current_state()), [[0.0]])
        solver, rhs, config, events = _heun_tr_decay_case(
            _module(backend), 0.0, tolerance=1e-3
        )
        state = solver.core.get_current_state()
        xp = array_namespace(state)
        # One h=1 step for y'=-2y yields zero, whereas two half steps yield 1/9.
        solver.run(
            rhs,
            config=replace(
                config,
                adaptive=False,
                operators=OperatorSpecs(
                    default=(
                        cast("Array", 2 * xp.eye(1, dtype=state.dtype)),
                        cast("Array", xp.zeros((1, 1), dtype=state.dtype)),
                    )
                ),
            ),
        )
        np.testing.assert_array_equal(_host(solver.core.get_current_state()), [[0.0]])
        assert events == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsupported_bdf_adaptive_and_forced_scan(backend: str) -> None:
    """Method/controller/loop restrictions fail explicitly in each namespace."""
    with _precision(backend):
        solver, rhs, config = _solver_case(
            _module(backend), "float64", "bdf2", adaptive=True
        )
        with pytest.raises(ValueError, match=r"bdf2.*adaptive|adaptive.*bdf2"):
            solver.run(rhs, config=config)
        solver, rhs, config = _solver_case(
            _module(backend), "float64", "sdirk2", adaptive=True
        )
        schedule = AdaptiveStepSchedule(
            tuple(TIMES),
            tuple((float(end - start),) for start, end in pairwise(TIMES)),
        )
        with pytest.raises(ValueError, match="Scan replay requires"):
            solver.replay_adaptive_schedule(
                rhs, schedule, config=replace(config, replay_loop="scan")
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_foreign_scipy_sparse_solve_boundary(backend: str) -> None:
    """SciPy acceleration remains NumPy-only and cannot silently coerce native state."""
    with _precision(backend):
        module = _module(backend)
        state = module.asarray(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        left = csr_matrix([[2.0, 0.0], [0.0, 3.0]])
        right = csr_matrix(np.eye(2))
        if backend == "numpy":
            np.testing.assert_allclose(
                implicit_solve(left, right, state), [[0.5, 1.0], [1.0, 4 / 3]]
            )
        else:
            with pytest.raises(TypeError, match="require a compatible state"):
                implicit_solve(left, right, state)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("method", ["dopri5", "imex-ark3", "ros2", "sdirk2"])
def test_native_structured_initial_state_derivatives(backend: str, method: str) -> None:
    """Representative explicit, implicit, and nonlinear-stage AD preserves the model."""
    with _precision(backend):
        module = _module(backend)
        initial = module.asarray(INITIAL)

        def objective(gain: Any) -> Any:  # noqa: ANN401
            solver, rhs, config = _solver_case(
                module, "float64", method, initial=initial * gain
            )
            solver.run(rhs, config=config)
            state = solver.core.get_current_state()
            return module.sum(state * state)

        gain = module.asarray(np.asarray(0.9))
        if backend == "jax":
            jax = pytest.importorskip("jax")
            value, gradient = jax.jit(jax.value_and_grad(objective))(gain)
        else:
            value = objective(gain)
            gradient = module.func.jacrev(objective)(gain)
        assert value.dtype == gain.dtype
        assert gradient.dtype == gain.dtype
        exact = _reference(TIMES[-1])
        np.testing.assert_allclose(_host(value), 0.9**2 * np.sum(exact**2), rtol=1e-4)
        np.testing.assert_allclose(_host(gradient), 1.8 * np.sum(exact**2), rtol=1e-4)
