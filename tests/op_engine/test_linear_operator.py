"""Independent action, residual, registry and implicit-stage contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from op_engine import (
    ActionOperator,
    CoreSolver,
    LinearSolveCandidate,
    LinearSolveError,
    LinearSolveRequest,
    ModelCore,
    SparseAdapter,
    array_namespace,
    implicit_solve,
    matrix_ops,
    register_sparse_adapter,
    require_linear_converged,
    solve_linear,
    unregister_sparse_adapter,
)
from op_engine.core_solver import OperatorSpecs, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Iterator


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


def _module(backend: str) -> Any:  # noqa: ANN401
    return (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )


def _host(value: Any) -> np.ndarray:  # noqa: ANN401
    return np.asarray(value.detach().numpy() if hasattr(value, "detach") else value)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_action_and_euclidean_transpose(backend: str, dtype: str) -> None:
    """Rectangular actions and their transpose agree with an independent matrix."""
    with _precision(backend):
        module = _module(backend)
        x = module.asarray(np.arange(6, dtype=dtype).reshape(3, 2))
        xp = array_namespace(x)

        def forward(v: Any) -> Any:  # noqa: ANN401
            return xp.stack((2 * v[0] + v[2], -v[1] + 3 * v[2]))

        def transpose(w: Any) -> Any:  # noqa: ANN401
            return xp.stack((2 * w[0], -w[1], w[0] + 3 * w[1]))

        action = ActionOperator((2, 3), x.dtype, xp, forward, transpose)
        matrix = np.asarray([[2, 0, 1], [0, -1, 3]], dtype=dtype)
        np.testing.assert_allclose(_host(action @ x), matrix @ _host(x))
        np.testing.assert_allclose(_host(action @ x[:, 0]), matrix @ _host(x[:, 0]))
        w = module.asarray(np.asarray([0.3, -0.4], dtype=dtype))
        np.testing.assert_allclose(_host(action.T @ w), matrix.T @ _host(w))
        assert (action @ x).dtype == x.dtype
        assert (action @ x[:, :0]).shape == (2, 0)
        assert array_namespace(action @ x) is xp
        np.testing.assert_allclose(_host(action.T.T @ x), matrix @ _host(x))


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_action_parameter_derivative(backend: str, dtype: str) -> None:
    """Native differentiation follows the action without a hidden dense matrix."""
    with _precision(backend):
        module = _module(backend)
        x = module.asarray(np.asarray([0.2, -0.4, 0.7], dtype=dtype))
        xp = array_namespace(x)
        coefficient = module.asarray(np.asarray(0.6, dtype=dtype))

        def objective(c: Any) -> Any:  # noqa: ANN401
            action = ActionOperator((3, 3), x.dtype, xp, lambda v: c * v)
            value = action @ x
            return xp.sum(value * value)

        if backend == "jax":
            jax = pytest.importorskip("jax")
            value, gradient = jax.jit(jax.value_and_grad(objective))(coefficient)
            batched = jax.jit(jax.vmap(objective))(xp.stack((coefficient, coefficient)))
        else:
            value = objective(coefficient)
            gradient = module.func.jacrev(objective)(coefficient)
            batched = module.func.vmap(objective)(xp.stack((coefficient, coefficient)))
        expected = float(np.dot(_host(x), _host(x)))
        np.testing.assert_allclose(_host(value), 0.36 * expected, rtol=2e-7)
        np.testing.assert_allclose(_host(gradient), 1.2 * expected, rtol=2e-7)
        np.testing.assert_allclose(_host(batched), [0.36 * expected] * 2, rtol=2e-7)


def _action(diagonal: np.ndarray) -> ActionOperator:
    return ActionOperator(
        (len(diagonal), len(diagonal)),
        diagonal.dtype,
        array_namespace(diagonal),
        lambda v: diagonal * v,
    )


def test_missing_transpose_and_unregistered_solve_refuse_numpy() -> None:
    """Missing capabilities never trigger numerical basis probing or coercion."""
    action = _action(np.asarray([2.0, 3.0]))
    with pytest.raises(NotImplementedError, match="transpose"):
        _ = action.T
    with pytest.raises(TypeError, match="explicitly supported"):
        implicit_solve(action, np.eye(2), np.ones(2))


@pytest.mark.parametrize(
    "bad", [np.ones(3, dtype=np.int64), np.ones((3, 1, 1)), np.ones(2)]
)
def test_action_rejects_invalid_input_numpy(bad: np.ndarray) -> None:
    """Action inputs cannot silently cast dtype or flatten dimensions."""
    with pytest.raises((TypeError, ValueError)):
        _ = _action(np.ones(3)) @ bad


@pytest.mark.parametrize(
    "bad", [np.ones(2), np.ones((3, 1)), np.ones(3, dtype=np.float32)]
)
def test_action_rejects_invalid_callback_result_numpy(bad: np.ndarray) -> None:
    """Each callback result retains the declared vector shape and dtype."""
    action = ActionOperator(
        (3, 3), np.dtype("float64"), array_namespace(bad), lambda _: bad
    )
    with pytest.raises((TypeError, ValueError)):
        _ = action @ np.ones(3)


@pytest.mark.parametrize("columns", [None, 1, 3])
def test_solve_linear_checks_original_residual_per_column_numpy(
    columns: int | None,
) -> None:
    """A false solver success report cannot hide one bad RHS column."""
    rhs = np.asarray([2.0, 6.0])
    if columns is not None:
        rhs = np.tile(rhs[:, None], (1, columns))
    request = LinearSolveRequest(_action(np.asarray([2.0, 3.0])), rhs)
    solution = rhs / (
        np.asarray([2.0, 3.0]) if columns is None else np.asarray([[2.0], [3.0]])
    )
    result = solve_linear(
        request, lambda _: LinearSolveCandidate(solution, success=True, iterations=1)
    )
    np.testing.assert_array_equal(
        result.diagnostics.converged, np.ones(columns or 1, dtype=bool)
    )
    np.testing.assert_allclose(require_linear_converged(result), solution)
    bad = solution.copy()
    if columns is None:
        bad[0] += 1.0
    else:
        bad[0, -1] += 1.0
    result = solve_linear(
        request, lambda _: LinearSolveCandidate(bad, success=True, iterations=1)
    )
    assert not result.diagnostics.converged[-1]
    assert np.all(result.diagnostics.converged[:-1])
    with pytest.raises(LinearSolveError) as failure:
        require_linear_converged(result)
    assert failure.value.result is result


def test_zero_rhs_and_solver_breakdown_numpy() -> None:
    """Zero RHS uses atol; solver breakdown remains a failure at zero residual."""
    request = LinearSolveRequest(_action(np.ones(2)), np.zeros(2))
    result = solve_linear(
        request,
        lambda _: LinearSolveCandidate(
            np.zeros(2), success=False, iterations=0, message="breakdown"
        ),
    )
    assert result.diagnostics.tolerance.tolist() == [request.options.atol]
    with pytest.raises(LinearSolveError, match="breakdown"):
        require_linear_converged(result)


def test_nonfinite_candidate_is_not_hidden_by_zero_operator_numpy() -> None:
    """A provider ignoring its input cannot turn an infinite candidate into success."""
    action = ActionOperator(
        (2, 2),
        np.dtype("float64"),
        array_namespace(np.ones(2)),
        np.zeros_like,
    )
    result = solve_linear(
        LinearSolveRequest(action, np.zeros(2)),
        lambda _: LinearSolveCandidate(np.full(2, np.inf), success=True, iterations=0),
    )
    assert not result.diagnostics.converged[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_extreme_finite_norm_is_stable_numpy(dtype: Any) -> None:  # noqa: ANN401
    """Finite large columns do not overflow the RHS tolerance norm."""
    magnitude = 1e30 if dtype is np.float32 else 1e300
    rhs = np.asarray([magnitude, 2 * magnitude], dtype=dtype)
    result = solve_linear(
        LinearSolveRequest(_action(np.ones(2, dtype=dtype)), rhs),
        lambda _: LinearSolveCandidate(rhs.copy(), success=True, iterations=0),
    )
    assert result.diagnostics.converged[0]
    assert np.isfinite(result.diagnostics.tolerance[0])


@pytest.mark.parametrize("iterations", [-1, True, 1.5, 101])
def test_invalid_iteration_diagnostics_refuse_numpy(iterations: Any) -> None:  # noqa: ANN401
    """Iteration metadata cannot silently exceed or misstate the requested budget."""
    with pytest.raises(ValueError, match="iterations"):
        solve_linear(
            LinearSolveRequest(_action(np.ones(2)), np.ones(2)),
            lambda _: LinearSolveCandidate(
                np.ones(2), success=True, iterations=iterations
            ),
        )


@pytest.fixture
def registered() -> Iterator[list[str]]:
    """Clean up test-only adapters.

    Yields:
        Registered identifiers to remove after the test.
    """
    names: list[str] = []
    try:
        yield names
    finally:
        for name in names:
            if name in vars(matrix_ops)["_SPARSE_ADAPTERS"]:
                unregister_sparse_adapter(name)


def _adapter(name: str, diagonal: np.ndarray) -> SparseAdapter:
    def solve(request: LinearSolveRequest) -> LinearSolveCandidate:
        divisor = diagonal if request.rhs.ndim == 1 else diagonal[:, None]
        return LinearSolveCandidate(request.rhs / divisor, success=True, iterations=1)

    return SparseAdapter(
        name,
        lambda op: isinstance(op, ActionOperator),
        linear_solver=solve,
        accepts_right=lambda op: isinstance(
            op, (ActionOperator, np.ndarray, csr_matrix)
        ),
    )


def test_invalid_primary_predicate_refuses_at_construction_numpy() -> None:
    """Malformed required matching metadata fails before modifying dispatch state."""
    with pytest.raises(TypeError, match="issparse"):
        SparseAdapter("invalid-predicate", None, linear_solver=lambda _: None)
    with pytest.raises(ValueError, match="ecosystem_id"):
        SparseAdapter(42, lambda _: False, linear_solver=lambda _: None)


def test_registered_mixed_pair_and_context_are_uncached_numpy(
    registered: list[str],
) -> None:
    """New solves receive current context/preconditioner without engine caching."""
    diagonal = np.asarray([2.0, 3.0])
    action = _action(diagonal)
    inverse = _action(1 / diagonal)
    seen: list[object] = []
    original = _adapter("test-context", diagonal)

    def solve(request: LinearSolveRequest) -> LinearSolveCandidate:
        seen.append(request.context)
        assert request.preconditioner is inverse
        return LinearSolveCandidate(
            request.preconditioner @ request.rhs, success=True, iterations=1
        )

    adapter = replace(
        original,
        linear_solver=solve,
        preconditioner=inverse,
        context=("mesh0", "coeff0", 0.1),
    )
    register_sparse_adapter(adapter)
    registered.append(adapter.ecosystem_id)
    rhs = np.asarray([[2.0, 4.0], [6.0, 9.0]])
    cache = vars(matrix_ops)["_IMPLICIT_SOLVER_CACHE"]
    before = dict(cache)
    np.testing.assert_allclose(implicit_solve(action, np.eye(2), rhs), [[1, 2], [2, 3]])
    np.testing.assert_allclose(implicit_solve(action, np.eye(2), rhs), [[1, 2], [2, 3]])
    register_sparse_adapter(
        replace(adapter, context=("mesh1", "coeff1", 0.2)), replace=True
    )
    implicit_solve(action, np.eye(2), rhs)
    assert seen == [("mesh0", "coeff0", 0.1)] * 2 + [("mesh1", "coeff1", 0.2)]
    assert dict(cache) == before


def test_registry_duplicate_overlap_and_replacement_numpy(
    registered: list[str],
) -> None:
    """Explicit replacement invalidates factors; overlapping pair matches refuse."""
    adapter = _adapter("test-unique", np.ones(2))
    register_sparse_adapter(adapter)
    registered.append(adapter.ecosystem_id)
    with pytest.raises(ValueError, match="already registered"):
        register_sparse_adapter(adapter)
    cache = vars(matrix_ops)["_IMPLICIT_SOLVER_CACHE"]
    cache[adapter.ecosystem_id, 1, 2] = object()
    register_sparse_adapter(adapter, replace=True)
    assert (adapter.ecosystem_id, 1, 2) not in cache
    other = replace(adapter, ecosystem_id="test-overlap")
    register_sparse_adapter(other)
    registered.append(other.ecosystem_id)
    with pytest.raises(ValueError, match="ambiguous"):
        implicit_solve(_action(np.ones(2)), np.eye(2), np.ones(2))
    cache[other.ecosystem_id, 1, 2] = object()
    unregister_sparse_adapter(other.ecosystem_id)
    assert (other.ecosystem_id, 1, 2) not in cache


@pytest.mark.parametrize(
    "method", ["imex-euler", "imex-heun-tr", "imex-trbdf2", "imex-ark3"]
)
def test_registered_action_reaches_implicit_stages_numpy(
    registered: list[str], method: str
) -> None:
    """Structured stage solves preserve Ly=Rx and match assembled independent solves."""
    diagonal = np.asarray([-0.3, -0.7, -1.1])
    options = ModelCoreOptions(other_axes=(3,), axis_names=("state", "group", "trait"))
    initial = np.arange(1.0, 13.0).reshape(2, 2, 3) / 10
    times = np.linspace(0, 0.2, 5)
    stage_diagonal = np.ones(3)

    def solve(request: LinearSolveRequest) -> LinearSolveCandidate:
        return LinearSolveCandidate(
            request.rhs / stage_diagonal[:, None], success=True, iterations=1
        )

    adapter = replace(_adapter("test-stages", diagonal), linear_solver=solve)
    register_sparse_adapter(adapter)
    registered.append(adapter.ecosystem_id)

    def run(*, actions: bool) -> np.ndarray:
        core = ModelCore(2, 2, times, options=options)
        core.set_initial_state(initial.copy())

        def factory(dt: float, scale: float, _context: object) -> tuple[Any, Any]:
            nonlocal stage_diagonal
            weight = 0.5 if method in {"imex-heun-tr", "imex-trbdf2"} else 1.0
            stage_diagonal = 1 - weight * dt * scale * diagonal
            right = 1 + weight * dt * scale * diagonal if weight == 0.5 else np.ones(3)
            return (
                (_action(stage_diagonal), _action(right))
                if actions
                else (np.diag(stage_diagonal), np.diag(right))
            )

        config = RunConfig(method=method, operators=OperatorSpecs(default=factory))
        CoreSolver(core, operator_axis="trait").run(
            lambda _t, y: -0.2 * y, config=config
        )
        return core.get_current_state()

    np.testing.assert_allclose(
        run(actions=True), run(actions=False), rtol=2e-14, atol=2e-14
    )


def test_failed_registered_solve_is_not_adopted_numpy(registered: list[str]) -> None:
    """A dishonest successful candidate is stopped by the original residual gate."""
    adapter = replace(
        _adapter("test-failure", np.ones(2)),
        linear_solver=lambda request: LinearSolveCandidate(
            np.zeros_like(request.rhs), success=True, iterations=0
        ),
    )
    register_sparse_adapter(adapter)
    registered.append(adapter.ecosystem_id)
    with pytest.raises(LinearSolveError):
        implicit_solve(_action(np.ones(2)), np.eye(2), np.ones(2))
    core = ModelCore(2, 1, [0.0, 0.1])
    initial = np.ones((2, 1))
    core.set_initial_state(initial)
    config = RunConfig(
        method="imex-euler",
        operators=OperatorSpecs(default=(_action(np.ones(2)), np.eye(2))),
    )
    with pytest.raises(LinearSolveError):
        CoreSolver(core).run(lambda _t, y: np.zeros_like(y), config=config)
    np.testing.assert_array_equal(core.get_current_state(), initial)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_inputs_do_not_enter_eager_solver(
    backend: str, registered: list[str]
) -> None:
    """Selecting the eager callback never silently transfers native inputs."""
    rhs = _module(backend).asarray(np.ones(2, dtype=np.float32))
    xp = array_namespace(rhs)
    action = ActionOperator((2, 2), rhs.dtype, xp, lambda v: v)
    adapter = _adapter("test-eager-only", np.ones(2, dtype=np.float32))
    register_sparse_adapter(adapter)
    registered.append(adapter.ecosystem_id)
    with pytest.raises(TypeError, match="eager NumPy"):
        implicit_solve(action, np.eye(2, dtype=np.float32), rhs)


def test_callback_precision_change_refuses_numpy() -> None:
    """A callback cannot silently promote a float32 solve to host float64."""
    request = LinearSolveRequest(
        _action(np.ones(2, dtype=np.float32)), np.ones(2, dtype=np.float32)
    )
    with pytest.raises(TypeError, match="NumPy namespace and dtype"):
        solve_linear(
            request,
            lambda _: LinearSolveCandidate(np.ones(2), success=True, iterations=1),
        )


def test_wrong_preconditioner_refuses_before_callback_numpy() -> None:
    """A mismatched inverse action cannot be applied to an unrelated stage."""
    request = LinearSolveRequest(
        _action(np.ones(2)), np.ones(2), preconditioner=_action(np.ones(3))
    )
    with pytest.raises(ValueError, match="preconditioner"):
        solve_linear(request, lambda _: pytest.fail("invalid request reached solver"))


@pytest.mark.parametrize(
    ("left_sparse", "right_sparse"), [(False, True), (True, False)]
)
def test_historical_mixed_dense_csr_fallback_numpy(
    *, left_sparse: bool, right_sparse: bool
) -> None:
    """Dense/CSR mixed pairs keep their existing assembled fallback behavior."""
    left = np.asarray([[2.0, 0.0], [0.0, 3.0]])
    right = np.asarray([[1.0, 0.1], [0.0, 1.0]])
    rhs = np.asarray([1.0, 2.0])
    actual = implicit_solve(
        csr_matrix(left) if left_sparse else left,
        csr_matrix(right) if right_sparse else right,
        rhs,
    )
    np.testing.assert_allclose(actual, np.linalg.solve(left, right @ rhs))
