"""Full-state residual actions at the public SDIRK integration boundary."""

# ruff: noqa: PLC2701

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    CoreSolver,
    DenseNewtonSolver,
    NewtonConfig,
    NonlinearMethodConfig,
    NonlinearProblem,
    NonlinearSolveDiagnostics,
    NonlinearSolveResult,
    array_namespace,
)
from op_engine._sdirk import (
    SDIRK2_ALEXANDER,
    _make_stage_problem,
    evaluate_sdirk_step,
)
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _DiagonalJvpNewton:
    """Test-only diagonal solver: no matrices, basis probes or dense solves.

    The manufactured diagonal RHS makes ``J @ ones`` its diagonal. This is not
    a general solver; production Newton--Krylov is a separate increment.
    """

    def solve(  # noqa: PLR6301
        self, problem: NonlinearProblem, initial_guess: Array, /
    ) -> NonlinearSolveResult:
        assert problem.jacobian is None
        assert problem.jvp is not None
        xp = array_namespace(initial_guess)
        state = initial_guess
        initial_norm = xp.sqrt(xp.mean(problem.residual(state) ** 2))
        for _ in range(8):
            step = problem.residual(state) / problem.jvp(state, xp.ones_like(state))
            state = state - step  # noqa: PLR6104 - preserve the input and AD graph
        residual = problem.residual(state)
        norm = xp.sqrt(xp.mean(residual**2))
        return NonlinearSolveResult(
            state,
            residual,
            NonlinearSolveDiagnostics(
                xp.logical_and(xp.isfinite(norm), norm <= 2e-6),
                8,
                10,
                0,
                initial_norm,
                norm,
                xp.sqrt(xp.mean(step**2)),
            ),
        )


def _decay(_time: float, state: Array) -> Array:
    return cast("Array", -20 * state**2)


def _decay_jvp(_time: float, state: Array, direction: Array) -> Array:
    return cast("Array", -40 * state * direction)


def test_numpy_jvp_only_core_fixed_live_and_replay_avoid_dense_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual method takes a JVP-only path in all three step consumers."""
    times = np.linspace(0, 0.2, 41)
    initial = np.array([[0.8, 0.5], [0.4, 0.2]])

    def solve(
        *,
        action: bool,
        adaptive: bool = False,
        schedule: AdaptiveStepSchedule | None = None,
    ) -> tuple[np.ndarray, AdaptiveStepSchedule | None]:
        core = ModelCore(2, 2, times, options=ModelCoreOptions(dtype=np.float64))
        core.set_initial_state(initial)
        method = (
            NonlinearMethodConfig(
                rhs_jvp=_decay_jvp,
                solver=_DiagonalJvpNewton(),
            )
            if action
            else NonlinearMethodConfig(
                rhs_jacobian=lambda _t, state: np.diag(-40 * np.asarray(state).ravel()),
                solver=DenseNewtonSolver(NewtonConfig(max_iterations=8, atol=2e-6)),
            )
        )
        solver = CoreSolver(core)
        config = RunConfig(
            method="sdirk2",
            adaptive=adaptive,
            nonlinear=method,
            adaptive_cfg=AdaptiveConfig(rtol=1e-7, atol=1e-10),
        )
        diagnostics = (
            solver.run(_decay, config=config)
            if schedule is None
            else solver.replay_adaptive_schedule(_decay, schedule, config=config)
        )
        assert diagnostics is not None
        diagnostics.require_converged()
        if action:
            np.testing.assert_array_equal(diagnostics.jacobian_evaluations, 0)
        return np.asarray(core.get_current_state()), solver.last_adaptive_schedule

    fixed_reference, _ = solve(action=False)
    live_reference, schedule = solve(action=False, adaptive=True)
    assert schedule is not None
    replay_reference, _ = solve(action=False, adaptive=True, schedule=schedule)

    def forbid_identity(*_args: object, **_kwargs: object) -> None:
        pytest.fail("JVP-only SDIRK allocated a dense identity")

    monkeypatch.setattr(array_namespace(initial), "eye", forbid_identity)
    fixed, _ = solve(action=True)
    live, action_schedule = solve(action=True, adaptive=True)
    assert action_schedule is not None
    replay, _ = solve(action=True, adaptive=True, schedule=action_schedule)
    np.testing.assert_allclose(fixed, fixed_reference, rtol=1e-12)
    np.testing.assert_allclose(live, live_reference, rtol=1e-9)
    np.testing.assert_allclose(replay, replay_reference, rtol=1e-12)
    # Independent continuous solution for y'=-20y², with fixed-mesh error.
    exact = initial / (1 + 4 * initial)
    np.testing.assert_allclose(fixed, exact, rtol=8e-4)
    np.testing.assert_allclose(live, exact, rtol=1e-5)


def test_numpy_coupled_time_dependent_stage_action_matches_finite_difference() -> None:
    """Residual JVP includes cross-axis coupling, stage time and dt diagonal."""
    base = np.array([[0.4, 0.7], [0.2, 0.6]])
    state = np.array([[0.3, 0.8], [0.1, 0.5]])
    direction = np.array([[0.2, -0.1], [0.6, -0.4]])
    matrix = np.array(
        [[-80, 2, 0, 3], [1, -40, 4, 0], [5, 0, -20, 1], [0, 3, 2, -10]], dtype=float
    )
    stage_time = 0.37
    diagonal_step = 0.13

    def rhs(t: float, value: Array) -> Array:
        flat = np.asarray(value).ravel()
        return (matrix @ flat + t * flat**2).reshape(2, 2)

    def action(t: float, value: Array, vector: Array) -> Array:
        assert t == stage_time
        return (matrix @ np.asarray(vector).ravel()).reshape(
            2, 2
        ) + 2 * t * value * vector

    problem = _make_stage_problem(
        rhs,
        lambda t, value: matrix + np.diag(2 * t * np.asarray(value).ravel()),
        rhs_jvp=action,
        base=base,
        stage_t=stage_time,
        dt_diagonal=diagonal_step,
        identity=np.eye(4),
    )
    assert problem.jvp is not None
    assert problem.jacobian is not None
    actual = problem.jvp(state, direction)
    epsilon = 1e-5
    finite = (
        problem.residual(state + epsilon * direction)
        - problem.residual(state - epsilon * direction)
    ) / (2 * epsilon)
    np.testing.assert_allclose(actual, finite, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(problem.jacobian(state)) @ direction.ravel(),
        actual.ravel(),
        rtol=1e-14,
    )


@pytest.mark.parametrize("wrong", ["shape", "dtype", "namespace"])
@pytest.mark.parametrize("boundary", ["direction", "result"])
def test_jax_stage_jvp_rejects_malformed_boundary(wrong: str, boundary: str) -> None:
    """Wrong directions fail before user code; wrong results fail before adoption."""
    jnp = pytest.importorskip("jax.numpy")
    state = jnp.ones((2, 2), dtype=jnp.float32)
    bad = (
        jnp.ones((4,), dtype=jnp.float32)
        if wrong == "shape"
        else (
            jnp.ones((2, 2), dtype=jnp.int32)
            if wrong == "dtype"
            else np.ones((2, 2), dtype=np.float32)
        )
    )
    calls = []

    def action(_t: float, _state: Array, direction: Array) -> Array:
        calls.append(1)
        return bad if boundary == "result" else direction

    problem = _make_stage_problem(
        lambda _t, value: value,
        None,
        rhs_jvp=action,
        base=state,
        stage_t=0.3,
        dt_diagonal=0.2,
        identity=None,
    )
    assert problem.jvp is not None
    with pytest.raises((TypeError, ValueError), match=wrong):
        problem.jvp(state, bad if boundary == "direction" else state)
    assert len(calls) == (1 if boundary == "result" else 0)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_jvp_only_step_preserves_precision(backend: str, dtype: str) -> None:
    """Native stages preserve the caller namespace and precision without a Jacobian."""
    module = pytest.importorskip("jax.numpy" if backend == "jax" else backend)
    context = nullcontext()
    if backend == "jax":
        jax = pytest.importorskip("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        context = enable()
    with context:
        initial = module.asarray([[0.8, 0.5], [0.4, 0.2]], dtype=getattr(module, dtype))
        result = evaluate_sdirk_step(
            tableau=SDIRK2_ALEXANDER,
            nonlinear_solver=_DiagonalJvpNewton(),
            rhs=_decay,
            rhs_jvp=_decay_jvp,
            t=0,
            dt=0.01,
            y=initial,
        )
        assert result.state.dtype == initial.dtype
        assert array_namespace(result.state) is array_namespace(initial)
        assert bool(result.converged.item())
        expected = evaluate_sdirk_step(
            tableau=SDIRK2_ALEXANDER,
            nonlinear_solver=DenseNewtonSolver(NewtonConfig(max_iterations=8)),
            rhs=_decay,
            rhs_jacobian=lambda _t, state: np.diag(-40 * np.asarray(state).ravel()),
            t=0,
            dt=0.01,
            y=np.asarray([[0.8, 0.5], [0.4, 0.2]]),
        )
        actual = (
            result.state.detach().numpy()
            if backend == "torch"
            else np.asarray(result.state)
        )
        np.testing.assert_allclose(actual, expected.state, rtol=2e-6)


def _native_solve(rate: Array, initial: Array) -> Array:
    """Expose finite JVP-only iterations through a fixed public CoreSolver run.

    Returns:
        Final scalar state in the input namespace.
    """
    xp = array_namespace(initial)
    core = ModelCore(
        1, 1, np.linspace(0, 0.02, 9), options=ModelCoreOptions(dtype=np.float64)
    )
    core.set_initial_state(xp.reshape(initial, (1, 1)))
    diagnostics = CoreSolver(core).run(
        lambda _t, state: -rate * state**2,
        config=RunConfig(
            method="sdirk2",
            nonlinear=NonlinearMethodConfig(
                rhs_jvp=lambda _t, state, direction: -2 * rate * state * direction,
                solver=_DiagonalJvpNewton(),
            ),
        ),
    )
    assert diagnostics is not None
    return cast("Array", core.get_current_state()[0, 0])


def test_jax_jvp_only_core_jit_grad_and_vmap() -> None:
    """Finite-iteration derivatives compile and batch without traced host decisions."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        rate, initial = jnp.asarray(20.0), jnp.asarray(0.8)
        value, gradients = jax.jit(jax.value_and_grad(_native_solve, argnums=(0, 1)))(
            rate, initial
        )
        _check_gradient(_native_solve, rate, initial, value, gradients)
        batched = jax.jit(jax.vmap(_native_solve))(
            jnp.array([20.0, 10.0]), jnp.array([0.8, 0.5])
        )
        np.testing.assert_allclose(batched[0], value, rtol=1e-13)


def test_torch_jvp_only_core_grad_and_explicit_vmap_limit() -> None:
    """CPU functional AD works; existing eager host checks still exclude vmap."""
    torch = pytest.importorskip("torch")
    rate, initial = (
        torch.asarray(20.0, dtype=torch.float64),
        torch.asarray(0.8, dtype=torch.float64),
    )
    gradients, value = torch.func.grad_and_value(_native_solve, argnums=(0, 1))(
        rate, initial
    )
    _check_gradient(_native_solve, rate, initial, value, gradients)
    with pytest.raises(RuntimeError, match=r"vmap.*item"):
        torch.vmap(_native_solve)(
            torch.tensor([20.0, 10.0], dtype=torch.float64),
            torch.tensor([0.8, 0.5], dtype=torch.float64),
        )


def _check_gradient(
    solve: Callable[[Array, Array], Array],
    rate: Array,
    initial: Array,
    value: Array,
    gradients: tuple[Array, Array],
) -> None:
    """Compare finite-iteration AD with finite differences and a continuous oracle."""
    epsilon = 1e-5
    finite_rate = (solve(rate + epsilon, initial) - solve(rate - epsilon, initial)) / (
        2 * epsilon
    )
    finite_initial = (
        solve(rate, initial + epsilon) - solve(rate, initial - epsilon)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        gradients, (finite_rate, finite_initial), rtol=1e-7, atol=1e-10
    )
    denominator = 1 + 0.02 * rate * initial
    np.testing.assert_allclose(value, initial / denominator, rtol=3e-4)
    np.testing.assert_allclose(
        gradients, (-0.02 * initial**2 / denominator**2, 1 / denominator**2), rtol=2e-3
    )


def test_numpy_nonlinear_config_retains_dense_positional_arguments() -> None:
    """Existing two-position calls remain valid and derivative forms are explicit."""
    solver = DenseNewtonSolver()
    jacobian = lambda _t, _state: np.eye(1)  # noqa: E731
    config = NonlinearMethodConfig(jacobian, solver)
    assert config.rhs_jacobian is jacobian
    assert config.rhs_jvp is None
    assert config.solver is solver
    with pytest.raises(ValueError, match="at least one"):
        NonlinearMethodConfig()
    for field in ("rhs_jacobian", "rhs_jvp"):
        with pytest.raises(TypeError, match=f"{field} must be callable"):
            NonlinearMethodConfig(**{field: 3})  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match=r"DenseNewtonSolver requires NonlinearProblem.jacobian"
    ):
        evaluate_sdirk_step(
            tableau=SDIRK2_ALEXANDER,
            nonlinear_solver=solver,
            rhs=_decay,
            rhs_jvp=_decay_jvp,
            t=0,
            dt=0.1,
            y=np.ones((1, 1)),
        )
