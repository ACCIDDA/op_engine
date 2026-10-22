"""Independent nonlinear roots, failure semantics and full-state integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import root

from op_engine import (
    ActionOperator,
    Array,
    CoreSolver,
    DenseNewtonSolver,
    LinearSolveCandidate,
    LinearSolveOptions,
    LinearSolveRequest,
    ModelCore,
    NewtonConfig,
    NewtonKrylovConfig,
    NewtonKrylovSolver,
    NewtonKrylovTermination,
    NewtonLinearization,
    NonlinearConvergenceError,
    NonlinearIntegrationDiagnostics,
    NonlinearMethodConfig,
    NonlinearProblem,
    NonlinearStageContext,
    ScipyGmresSolver,
    array_namespace,
    require_converged,
)
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


def _coupled_root() -> NonlinearProblem:
    """Return a coupled root with a known independent analytic answer."""
    matrix = np.array([[3.0, -1.0, 0.2], [0.3, 4.0, -0.5], [-0.2, 0.4, 2.0]])
    target = np.array([0.4, -0.2, 0.7])
    forcing = matrix @ target + target**3

    def residual(state: Array) -> Array:
        value = cast("FloatArray", state)
        return (
            (matrix.astype(value.dtype) @ value.ravel()).reshape(value.shape)
            + value**3
            - forcing.astype(value.dtype).reshape(value.shape)
        )

    def jvp(state: Array, direction: Array) -> Array:
        value, vector = cast("FloatArray", state), cast("FloatArray", direction)
        return (matrix.astype(value.dtype) @ vector.ravel()).reshape(
            value.shape
        ) + 3 * value**2 * vector

    return NonlinearProblem(residual=residual, jvp=jvp)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 1)])
def test_coupled_root_matches_analytic_and_independent_dense(
    dtype: type[np.floating], shape: tuple[int, ...]
) -> None:
    """Coupled state axes are flattened once for the true full-system action."""
    problem = _coupled_root()
    guess = np.zeros(shape, dtype=dtype)
    before = guess.copy()
    tolerance = 5e-6 if dtype is np.float32 else 1e-11
    solver = NewtonKrylovSolver(NewtonKrylovConfig(rtol=0, atol=tolerance))
    result = require_converged(solver.solve(problem, guess))
    independent = root(
        lambda value: problem.residual(value.reshape(shape)).ravel(), guess.ravel()
    )
    np.testing.assert_allclose(result.value.ravel(), [0.4, -0.2, 0.7], atol=tolerance)
    np.testing.assert_allclose(result.value.ravel(), independent.x, atol=tolerance)
    np.testing.assert_array_equal(guess, before)
    assert not np.shares_memory(result.value, guess)
    assert result.value.dtype == guess.dtype
    assert result.diagnostics.jacobian_evaluations == 0
    assert result.diagnostics.jvp_evaluations > 0
    assert all(np.all(item.converged) for item in result.diagnostics.linear_solves)


def test_backtracking_globalizes_a_bad_initial_scalar_guess() -> None:
    """The first undamped Newton update increases the residual substantially."""
    problem = NonlinearProblem(
        residual=lambda value: cast("Array", value * value - 2),
        jvp=lambda value, vector: cast("Array", 2 * value * vector),
    )
    result = require_converged(
        NewtonKrylovSolver(
            NewtonKrylovConfig(max_iterations=20, rtol=0, atol=1e-12)
        ).solve(problem, np.array([0.01]))
    )
    np.testing.assert_allclose(result.value, np.sqrt(2), atol=1e-12)
    assert result.diagnostics.backtracks > 0
    assert result.diagnostics.residual_evaluations > result.diagnostics.iterations + 1


def _failed_inner(request: LinearSolveRequest) -> LinearSolveCandidate:
    """Return a failed inner candidate even if its exact linear residual is zero."""
    return LinearSolveCandidate(
        request.rhs, success=False, iterations=1, message="deliberate failure"
    )


def test_failed_inner_never_applies_candidate_and_remains_inspectable() -> None:
    """An unsuccessful callback cannot be accepted through the outer residual."""
    guess = np.array([3.0])
    result = NewtonKrylovSolver(linear_solver=_failed_inner).solve(
        NonlinearProblem(lambda value: value, jvp=lambda _state, vector: vector), guess
    )
    np.testing.assert_array_equal(result.value, guess)
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.termination == NewtonKrylovTermination.LINEAR_FAILURE
    assert not result.diagnostics.linear_solves[0].converged[0]
    with pytest.raises(NonlinearConvergenceError):
        require_converged(result)


def test_exhausted_globalization_keeps_last_accepted_state() -> None:
    """A wrong derivative cannot silently advance through a residual increase."""
    result = NewtonKrylovSolver(NewtonKrylovConfig(max_backtracks=3)).solve(
        NonlinearProblem(lambda value: value, jvp=lambda _state, vector: -vector),
        np.array([2.0]),
    )
    np.testing.assert_array_equal(result.value, [2.0])
    assert result.diagnostics.termination == NewtonKrylovTermination.LINE_SEARCH_FAILURE
    assert result.diagnostics.backtracks == 3
    assert result.diagnostics.residual_evaluations == 5


def test_initial_root_skips_all_linear_preparation() -> None:
    """An already converged candidate needs no action, solve or preconditioner."""
    result = NewtonKrylovSolver().solve(
        NonlinearProblem(lambda value: value, jvp=lambda _state, vector: vector),
        np.zeros((2, 3)),
    )
    assert result.diagnostics.termination == NewtonKrylovTermination.CONVERGED
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.jvp_evaluations == 0
    assert result.diagnostics.linear_solves == ()


@pytest.mark.parametrize(
    "case", ["residual_shape", "residual_dtype", "jvp_shape", "jvp_dtype"]
)
def test_callback_contract_errors_are_fatal(case: str) -> None:
    """Invalid callback arrays are configuration errors, not numerical failures."""
    residual = (
        (lambda value: value)
        if not case.startswith("residual")
        else (
            (lambda _value: np.zeros(3))
            if case.endswith("shape")
            else (lambda value: np.asarray(value, dtype=np.float32))
        )
    )
    jvp = (
        (lambda _state, vector: vector)
        if not case.startswith("jvp")
        else (
            (lambda _state, _vector: np.zeros(3))
            if case.endswith("shape")
            else (lambda _state, vector: np.asarray(vector, dtype=np.float32))
        )
    )
    with pytest.raises(
        ValueError if case.endswith("shape") else TypeError, match=r"shape|dtype"
    ):
        NewtonKrylovSolver().solve(NonlinearProblem(residual, jvp=jvp), np.ones(2))


def test_unique_readonly_linearization_contexts_and_preconditioning() -> None:
    """A fresh exact-action request binds every inverse preparation lifetime."""
    contexts: list[NewtonLinearization] = []
    problem = _coupled_root()
    marker = object()
    problem = NonlinearProblem(problem.residual, jvp=problem.jvp, context=marker)

    def factory(request: LinearSolveRequest) -> ActionOperator:
        context = request.context
        assert isinstance(context, NewtonLinearization)
        assert context.problem_context is marker
        assert not context.iterate.flags.writeable
        contexts.append(context)
        return ActionOperator(
            request.operator.shape,
            request.rhs.dtype,
            array_namespace(request.rhs),
            lambda vector: vector,
        )

    solver = NewtonKrylovSolver(preconditioner_factory=factory)
    first = require_converged(solver.solve(problem, np.zeros(3)))
    second = require_converged(solver.solve(problem, np.zeros(3)))
    assert len({id(context) for context in contexts}) == len(contexts)
    assert len(contexts) == first.diagnostics.iterations + second.diagnostics.iterations
    assert (
        contexts[0].iteration == contexts[first.diagnostics.iterations].iteration == 0
    )


def _trajectory(  # noqa: PLR0913 - explicit independent consumer controls
    *,
    eager: bool,
    adaptive: bool = False,
    schedule: AdaptiveStepSchedule | None = None,
    amplitude: float = 1.0,
    newton: NewtonKrylovConfig | None = None,
    dtype: type[np.floating] = np.float64,
    adaptive_options: AdaptiveConfig | None = None,
) -> tuple[FloatArray, NonlinearIntegrationDiagnostics, AdaptiveStepSchedule | None]:
    times = np.linspace(0.0, 0.1, 11)
    matrix = np.array([
        [-120.0, 20.0, 0.0, 0.0],
        [15.0, -100.0, 10.0, 0.0],
        [0.0, 12.0, -80.0, 10.0],
        [2.0, 0.0, 10.0, -50.0],
    ])
    matrix = matrix.astype(dtype)
    core = ModelCore(2, 2, times, options=ModelCoreOptions(dtype=dtype))
    core.set_initial_state(np.array([[0.8, 0.6], [0.4, 0.2]], dtype=dtype) * amplitude)
    stage_tolerance = 2e-6 if dtype is np.float32 else 1e-12

    def rhs(_time: float, state: Array) -> Array:
        value = cast("FloatArray", state)
        return (matrix @ value.ravel()).reshape(value.shape) - value**3

    def jvp(_time: float, state: Array, direction: Array) -> Array:
        value, vector = cast("FloatArray", state), cast("FloatArray", direction)
        return (matrix @ vector.ravel()).reshape(value.shape) - 3 * value**2 * vector

    def jacobian(_time: float, state: Array) -> Array:
        return matrix - np.diag(3 * cast("FloatArray", state).ravel() ** 2)

    config = RunConfig(
        method="sdirk2",
        adaptive=adaptive,
        adaptive_cfg=adaptive_options or AdaptiveConfig(rtol=1e-5, atol=1e-8),
        nonlinear=NonlinearMethodConfig(
            rhs_jvp=jvp if eager else None,
            rhs_jacobian=None if eager else jacobian,
            solver=NewtonKrylovSolver(
                newton or NewtonKrylovConfig(rtol=0, atol=stage_tolerance)
            )
            if eager
            else DenseNewtonSolver(
                NewtonConfig(max_iterations=8, rtol=0, atol=stage_tolerance)
            ),
        ),
    )
    solver = CoreSolver(core)
    diagnostics = (
        solver.run(rhs, config=config)
        if schedule is None
        else solver.replay_adaptive_schedule(rhs, schedule, config=config)
    )
    assert diagnostics is not None
    diagnostics.require_converged()
    return (
        cast("FloatArray", core.get_current_state()),
        diagnostics,
        solver.last_adaptive_schedule,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_state_fixed_live_replay_and_inner_diagnostic_propagation(
    dtype: type[np.floating],
) -> None:
    """General GMRES preserves cross-axis coupling through every SDIRK consumer."""
    for adaptive in (False, True):
        expected, _, _ = _trajectory(eager=False, adaptive=adaptive, dtype=dtype)
        actual, diagnostics, schedule = _trajectory(
            eager=True, adaptive=adaptive, dtype=dtype
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=5e-4 if dtype is np.float32 else 1e-8,
            atol=2e-6 if dtype is np.float32 else 1e-10,
        )
        assert diagnostics.work is not None
        assert np.all(diagnostics.work.converged)
        assert np.sum(diagnostics.work.solves_per_stage) == len(
            diagnostics.work.iterations
        )
        assert np.all(diagnostics.work.termination == NewtonKrylovTermination.CONVERGED)
        assert np.all(diagnostics.jacobian_evaluations == 0)
        if adaptive:
            assert schedule is not None
            replay, _, _ = _trajectory(
                eager=True, adaptive=True, schedule=schedule, dtype=dtype
            )
            np.testing.assert_allclose(
                replay,
                actual,
                rtol=1e-6 if dtype is np.float32 else 1e-12,
                atol=1e-8 if dtype is np.float32 else 1e-13,
            )


def test_trajectory_and_sensitivity_match_independent_references() -> None:
    """Continuous Radau and dense finite differences qualify values and sensitivity."""
    matrix = np.array([
        [-120.0, 20.0, 0.0, 0.0],
        [15.0, -100.0, 10.0, 0.0],
        [0.0, 12.0, -80.0, 10.0],
        [2.0, 0.0, 10.0, -50.0],
    ])
    initial = np.array([0.8, 0.6, 0.4, 0.2])

    def reference(amplitude: float) -> FloatArray:
        return solve_ivp(
            lambda _t, y: matrix @ y - y**3,
            (0.0, 0.1),
            initial * amplitude,
            method="Radau",
            rtol=1e-11,
            atol=1e-13,
        ).y[:, -1]

    accuracy = AdaptiveConfig(rtol=1e-7, atol=1e-11)
    actual, _, schedule = _trajectory(
        eager=True, adaptive=True, adaptive_options=accuracy
    )
    np.testing.assert_allclose(actual.ravel(), reference(1.0), rtol=2e-4, atol=1e-7)
    assert schedule is not None
    _, _, dense_schedule = _trajectory(
        eager=False, adaptive=True, adaptive_options=accuracy
    )
    assert dense_schedule is not None
    step = 1e-4
    derivatives = []
    for eager, mesh in ((True, schedule), (False, dense_schedule)):
        plus, _, _ = _trajectory(
            eager=eager,
            adaptive=True,
            schedule=mesh,
            amplitude=1 + step,
            adaptive_options=accuracy,
        )
        minus, _, _ = _trajectory(
            eager=eager,
            adaptive=True,
            schedule=mesh,
            amplitude=1 - step,
            adaptive_options=accuracy,
        )
        derivatives.append((plus.ravel() - minus.ravel()) / (2 * step))
    np.testing.assert_allclose(derivatives[0], derivatives[1], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        derivatives[0],
        (reference(1 + step) - reference(1 - step)) / (2 * step),
        rtol=2e-4,
        atol=1e-7,
    )


def test_changed_inner_controls_invalidate_recorded_schedule() -> None:
    """Frozen meshes retain the actual numerical solver settings in provenance."""
    _, _, schedule = _trajectory(eager=True, adaptive=True)
    assert schedule is not None
    changed = NewtonKrylovConfig(
        rtol=0, atol=1e-12, linear_options=LinearSolveOptions(rtol=1e-3)
    )
    with pytest.raises(ValueError, match="nonlinear_signature mismatch"):
        _trajectory(eager=True, adaptive=True, schedule=schedule, newton=changed)


def test_sdirk_preconditioner_observes_current_stage_and_scale() -> None:
    """Stage and retry metadata reaches the existing per-request context."""
    contexts: list[NonlinearStageContext] = []

    def factory(request: LinearSolveRequest) -> None:
        assert isinstance(request.context, NewtonLinearization)
        stage = request.context.problem_context
        assert isinstance(stage, NonlinearStageContext)
        contexts.append(stage)

    core = ModelCore(1, 1, np.array([0.0, 0.1, 0.3]))
    core.set_initial_state(np.ones((1, 1)))
    CoreSolver(core).run(
        lambda _time, state: -state,
        config=RunConfig(
            method="sdirk2",
            nonlinear=NonlinearMethodConfig(
                rhs_jvp=lambda _time, _state, vector: -vector,
                solver=NewtonKrylovSolver(preconditioner_factory=factory),
            ),
        ),
    )
    assert {stage.stage_index for stage in contexts} == {0, 1}
    assert {round(stage.step_size, 5) for stage in contexts} == {0.1, 0.2}
    assert all(0 < stage.diagonal_step < stage.step_size for stage in contexts)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_iterations": True},
        {"rtol": -1.0},
        {"atol": float("nan")},
        {"damping": 0.0},
        {"max_backtracks": -1},
        {"max_backtracks": True},
        {"backtrack_factor": 1.0},
        {"armijo": 0.0},
    ],
)
def test_invalid_static_controls_fail_before_solving(kwargs: dict[str, object]) -> None:
    """Bad outer/globalization controls cannot enter an attempted stage."""
    with pytest.raises(ValueError, match=r"finite|positive|nonnegative|non-negative"):
        NewtonKrylovConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("guess", [np.ones(2, dtype=int), np.ones(2, dtype=complex)])
def test_unsupported_state_types_are_never_coerced(guess: FloatArray) -> None:
    """An eager real solver rejects unsupported inputs explicitly."""
    with pytest.raises(TypeError, match="eager NumPy"):
        NewtonKrylovSolver().solve(_coupled_root(), guess)


def test_inner_exceptions_propagate_as_fatal_errors() -> None:
    """Arbitrary linear exceptions do not become adaptive numerical rejections."""

    def explode(_request: LinearSolveRequest) -> LinearSolveCandidate:
        msg = "injected linear failure"
        raise ArithmeticError(msg)

    with pytest.raises(ArithmeticError, match="injected linear failure"):
        NewtonKrylovSolver(linear_solver=explode).solve(_coupled_root(), np.zeros(3))


def test_fixed_failure_preserves_state_and_adaptive_failure_retries() -> None:
    """Failed inner candidates are retained as rejected attempts, never adopted."""

    def inner(request: LinearSolveRequest) -> LinearSolveCandidate:
        context = request.context
        assert isinstance(context, NewtonLinearization)
        assert isinstance(context.problem_context, NonlinearStageContext)
        if context.problem_context.step_size > 0.02:
            return LinearSolveCandidate(
                np.zeros_like(request.rhs), success=False, iterations=0
            )
        return ScipyGmresSolver()(request)

    config = RunConfig(
        method="sdirk2",
        nonlinear=NonlinearMethodConfig(
            rhs_jvp=lambda _time, _state, vector: -vector,
            solver=NewtonKrylovSolver(linear_solver=inner),
        ),
    )
    core = ModelCore(1, 1, np.array([0.0, 0.1]))
    core.set_initial_state(np.ones((1, 1)))
    solver = CoreSolver(core)
    with pytest.raises(NonlinearConvergenceError) as caught:
        solver.run(lambda _time, state: -state, config=config)
    np.testing.assert_array_equal(core.get_current_state(), [[1.0]])
    assert (
        caught.value.result.diagnostics.termination
        == NewtonKrylovTermination.LINEAR_FAILURE
    )

    from dataclasses import replace  # noqa: PLC0415 - keep the policy change local

    diagnostics = solver.run(
        lambda _time, state: -state, config=replace(config, adaptive=True)
    )
    assert diagnostics is not None
    assert diagnostics.work is not None
    diagnostics.require_converged()
    assert np.any(~diagnostics.step_accepted)
    assert np.any(~diagnostics.work.converged)
    assert np.any(
        diagnostics.work.termination == NewtonKrylovTermination.LINEAR_FAILURE
    )
    assert np.all(diagnostics.step_converged[diagnostics.step_accepted])
    np.testing.assert_allclose(core.get_current_state(), np.exp(-0.1), rtol=2e-6)


def test_large_action_only_problem_never_assembles_a_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coupled 4096-state nonlinear solve needs neither basis probing nor a matrix."""

    def action(state: Array, vector: Array) -> Array:
        value, direction = cast("FloatArray", state), cast("FloatArray", vector)
        result = (3 + 3 * value**2) * direction
        result[1:] -= direction[:-1]
        result[:-1] -= direction[1:]
        return result

    def residual(state: Array) -> Array:
        value = cast("FloatArray", state)
        result = 3 * value + value**3
        result[1:] -= value[:-1]
        result[:-1] -= value[1:]
        forcing = np.full_like(value, 2.0)
        forcing[[0, -1]] = 3.0
        return result - forcing

    def forbid(*_args: object, **_kwargs: object) -> None:
        pytest.fail("matrix-free nonlinear solve requested a dense matrix/solve")

    monkeypatch.setattr(np, "eye", forbid)
    monkeypatch.setattr(np, "diag", forbid)
    monkeypatch.setattr(np.linalg, "solve", forbid)
    result = require_converged(
        NewtonKrylovSolver().solve(
            NonlinearProblem(residual, jvp=action), np.full(4096, 1.2)
        )
    )
    np.testing.assert_allclose(result.value, 1.0, rtol=1e-8, atol=1e-10)
    assert result.diagnostics.jvp_evaluations < 100


def test_reused_residual_buffer_preserves_last_accepted_pair_on_line_failure() -> None:
    """Rejected trial evaluation cannot replace or alias retained residual data."""
    buffer = np.zeros(1)

    def residual(value: Array) -> Array:
        np.copyto(buffer, cast("FloatArray", value))
        return buffer

    result = NewtonKrylovSolver(NewtonKrylovConfig(max_backtracks=0)).solve(
        NonlinearProblem(residual, jvp=lambda _state, vector: -vector),
        np.array([2.0]),
    )
    assert result.diagnostics.termination == NewtonKrylovTermination.LINE_SEARCH_FAILURE
    np.testing.assert_array_equal(result.value, [2.0])
    np.testing.assert_array_equal(result.residual, [2.0])
    assert result.diagnostics.residual_norm == 2.0
    assert not np.shares_memory(result.residual, buffer)
    buffer.fill(100)
    np.testing.assert_array_equal(result.residual, [2.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_identity_jvp_can_return_its_borrowed_direction(
    dtype: type[np.floating],
) -> None:
    """SciPy's orthogonalization cannot overwrite its own basis through an alias."""
    result = require_converged(
        NewtonKrylovSolver().solve(
            NonlinearProblem(lambda value: value, jvp=lambda _value, vector: vector),
            np.array([2.0, 3.0], dtype=dtype),
        )
    )
    np.testing.assert_allclose(result.value, 0.0, atol=1e-6)
