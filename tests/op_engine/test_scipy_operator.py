"""Independent dense references for explicit eager banded and Krylov adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from op_engine import (
    ActionOperator,
    BandedOperator,
    CoreSolver,
    LinearSolveError,
    LinearSolveOptions,
    LinearSolveRequest,
    ModelCore,
    ScipyGmresSolver,
    array_namespace,
    implicit_solve,
    register_sparse_adapter,
    require_linear_converged,
    scipy_banded_adapter,
    scipy_gmres_adapter,
    scipy_operator,
    solve_linear,
    unregister_sparse_adapter,
)
from op_engine.core_solver import OperatorSpecs, RunConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


MATRIX = np.asarray(
    [
        [5, 2, -1, 0, 0],
        [1, 4, 2, -1, 0],
        [0, 1, 3, 2, -1],
        [0, 0, 1, 2, 2],
        [0, 0, 0, 1, 1],
    ],
    dtype=np.float64,
)
BANDS = np.asarray([
    [np.nan, np.nan, -1, -1, -1],
    [np.nan, 2, 2, 2, 2],
    [5, 4, 3, 2, 1],
    [1, 1, 1, 1, np.nan],
])


@pytest.fixture
def adapters() -> Iterator[None]:
    """Explicitly install both eager adapters and remove them after the test.

    Yields:
        No shared hidden solver state.
    """
    options = LinearSolveOptions(rtol=1e-5, atol=1e-7)
    register_sparse_adapter(scipy_banded_adapter(options=options))
    register_sparse_adapter(
        scipy_gmres_adapter(options=LinearSolveOptions(rtol=1e-11, atol=1e-13))
    )
    try:
        yield
    finally:
        unregister_sparse_adapter("scipy-banded")
        unregister_sparse_adapter("scipy-gmres")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("columns", [None, 1, 3, 0])
@pytest.mark.usefixtures("adapters")
def test_banded_application_transpose_and_solve_numpy(
    dtype: str, columns: int | None
) -> None:
    """Asymmetric packed bands agree with a separately specified full matrix."""
    operator = BandedOperator(BANDS.astype(dtype), lower=1, upper=2)
    rhs = np.arange(1, 6, dtype=dtype)
    if columns is not None:
        rhs = np.tile(rhs[:, None], (1, columns))
    matrix = MATRIX.astype(dtype)
    np.testing.assert_allclose(operator @ rhs, matrix @ rhs, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(operator.T @ rhs, matrix.T @ rhs, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(operator.T.T @ rhs, matrix @ rhs, rtol=2e-6, atol=2e-6)
    result = implicit_solve(operator, np.eye(5, dtype=dtype), rhs)
    assert result.dtype == rhs.dtype
    assert result.shape == rhs.shape
    np.testing.assert_allclose(
        result, np.linalg.solve(matrix, rhs), rtol=2e-5, atol=2e-5
    )


@pytest.mark.usefixtures("adapters")
def test_banded_storage_is_copied_and_new_operator_changes_solution_numpy() -> None:
    """Input mutation and a new coefficient/stage cannot reuse stale preparation."""
    original = np.asarray([[2.0, 3.0]])
    first = BandedOperator(original, lower=0, upper=0)
    original[:] = [[4.0, 5.0]]
    second = BandedOperator(original, lower=0, upper=0)
    rhs = np.asarray([8.0, 15.0])
    np.testing.assert_allclose(implicit_solve(first, np.eye(2), rhs), [4, 5])
    np.testing.assert_allclose(implicit_solve(second, np.eye(2), rhs), [2, 3])
    assert not first.bands.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.bands[0, 0] = 0


@pytest.mark.usefixtures("adapters")
def test_banded_singular_failure_and_preconditioner_refusal_numpy() -> None:
    """A singular direct solve reports failure rather than adopting its candidate."""
    operator = BandedOperator(np.zeros((1, 2)), lower=0, upper=0)
    with pytest.raises(LinearSolveError, match="singular"):
        implicit_solve(operator, np.eye(2), np.ones(2))
    inverse = ActionOperator(
        (2, 2), np.dtype("float64"), array_namespace(np.ones(2)), lambda v: v
    )
    adapter = scipy_banded_adapter()
    register_sparse_adapter(replace(adapter, preconditioner=inverse), replace=True)
    with pytest.raises(ValueError, match="preconditioner"):
        implicit_solve(operator, np.eye(2), np.ones(2))


def _weighted_diffusion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    widths = np.asarray([0.2, 0.4, 0.9, 1.4])
    incidence = np.asarray([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
    conductances = np.asarray([1.8, 0.7, 1.1])
    stiffness = -incidence.T @ np.diag(conductances) @ incidence
    generator = stiffness / widths[:, None]
    return widths, conductances, generator


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_matrix_free_weighted_diffusion_numpy(
    dtype: str, *, preconditioned: bool
) -> None:
    """A flux action solves a weighted-self-adjoint, Euclidean-nonsymmetric system."""
    widths, conductances, generator = _weighted_diffusion()
    np.testing.assert_allclose(
        widths[:, None] * generator, (widths[:, None] * generator).T
    )
    assert not np.allclose(generator, generator.T)
    widths, conductances = widths.astype(dtype), conductances.astype(dtype)
    h = 0.17

    def forward(v: np.ndarray) -> np.ndarray:
        flux = conductances * (v[1:] - v[:-1])
        net = np.concatenate((flux[:1], flux[1:] - flux[:-1], -flux[-1:]))
        return v - h * net / widths

    def transpose(v: np.ndarray) -> np.ndarray:
        weighted = v / widths
        flux = conductances * (weighted[1:] - weighted[:-1])
        net = np.concatenate((flux[:1], flux[1:] - flux[:-1], -flux[-1:]))
        return v - h * net

    reference = np.eye(4, dtype=dtype) - h * generator.astype(dtype)
    rhs = np.asarray([[1, 2], [0.3, -0.2], [0.8, 0.4], [1.1, 0.7]], dtype=dtype)
    xp = array_namespace(rhs)
    action = ActionOperator((4, 4), rhs.dtype, xp, forward, transpose)
    np.testing.assert_allclose(action.T @ rhs, reference.T @ rhs, rtol=2e-6, atol=2e-6)
    inverse = (
        ActionOperator((4, 4), rhs.dtype, xp, lambda v: v / np.diag(reference))
        if preconditioned
        else None
    )
    options = LinearSolveOptions(
        rtol=1e-5 if dtype == "float32" else 1e-11,
        atol=1e-7 if dtype == "float32" else 1e-13,
    )
    request = LinearSolveRequest(
        action, rhs, options, inverse, ("nonuniform-mesh-v1", h)
    )
    result = solve_linear(request, ScipyGmresSolver(restart=4))
    np.testing.assert_allclose(
        require_linear_converged(result),
        np.linalg.solve(reference, rhs),
        rtol=2e-5,
        atol=2e-6,
    )
    assert 0 < result.diagnostics.iterations <= 4


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("storage", ["input", "readonly", "scratch"])
def test_gmres_borrowed_action_and_inverse_outputs_numpy(
    dtype: str, storage: str
) -> None:
    """Identity actions may return aliases, read-only views or reused storage."""
    rhs = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 4.0], [5.0, 0.0]], dtype=dtype)
    before = rhs.copy()
    rhs.flags.writeable = False
    forward_buffer = np.zeros(4, dtype=dtype)
    inverse_buffer = np.zeros(4, dtype=dtype)

    def output(vector: np.ndarray, buffer: np.ndarray) -> np.ndarray:
        if storage == "input":
            return vector
        if storage == "readonly":
            result = vector.view()
            result.flags.writeable = False
            return result
        np.copyto(buffer, vector)
        return buffer

    xp = array_namespace(rhs)
    action = ActionOperator((4, 4), rhs.dtype, xp, lambda v: output(v, forward_buffer))
    inverse = ActionOperator((4, 4), rhs.dtype, xp, lambda v: output(v, inverse_buffer))
    result = solve_linear(
        LinearSolveRequest(
            action,
            rhs,
            LinearSolveOptions(rtol=1e-5, atol=1e-7),
            preconditioner=inverse,
        ),
        ScipyGmresSolver(),
    )
    np.testing.assert_allclose(
        require_linear_converged(result), rhs, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_array_equal(rhs, before)


def test_gmres_inner_iteration_budget_and_diagnostics_numpy() -> None:
    """A too-small inner budget fails explicitly while retaining the actual residual."""
    diagonal = np.asarray([1.0, 2.0, 4.0, 8.0])
    action = ActionOperator(
        (4, 4), diagonal.dtype, array_namespace(diagonal), lambda v: diagonal * v
    )
    request = LinearSolveRequest(
        action,
        np.ones((4, 2)),
        LinearSolveOptions(rtol=1e-14, atol=0, max_iterations=1),
    )
    result = solve_linear(request, ScipyGmresSolver(restart=4))
    assert result.diagnostics.iterations == 1
    assert result.diagnostics.residual_norm.shape == (2,)
    assert np.all(result.diagnostics.residual_norm > result.diagnostics.tolerance)
    assert "[1, 1]" in result.diagnostics.message
    with pytest.raises(LinearSolveError):
        require_linear_converged(result)


@pytest.mark.parametrize("kind", ["banded", "matrix-free"])
@pytest.mark.usefixtures("adapters")
def test_variable_stage_mesh_uses_current_operator_numpy(kind: str) -> None:
    """Real variable-dt stages compare with independent small direct recurrences."""
    diagonal = np.asarray([-0.3, -0.7, -1.1])
    times = np.asarray([0.0, 0.025, 0.09, 0.2])
    initial = np.asarray([[0.3, 0.2], [0.7, 0.8], [1.1, 1.2]])
    core = ModelCore(3, 2, times)
    core.set_initial_state(initial.copy())

    def factory(dt: float, scale: float, _context: object) -> tuple[object, np.ndarray]:
        bands = BandedOperator((1 - dt * scale * diagonal)[None, :], lower=0, upper=0)
        operator = (
            bands
            if kind == "banded"
            else ActionOperator(
                bands.shape, bands.dtype, array_namespace(initial), bands.__matmul__
            )
        )
        return operator, np.eye(3)

    CoreSolver(core).run(
        lambda _t, y: np.zeros_like(y),
        config=RunConfig(
            method="imex-euler",
            adaptive=False,
            operators=OperatorSpecs(default=factory),
        ),
    )
    expected = initial.copy()
    for dt in np.diff(times):
        # The existing NumPy IMEX Euler kernel adopts the two-half-step result
        # of its step-doubling pair, including fixed runs. Preserve that scheme.
        half_left = np.eye(3) - 0.5 * dt * np.diag(diagonal)
        expected = np.linalg.solve(half_left, np.linalg.solve(half_left, expected))
    np.testing.assert_allclose(core.get_current_state(), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.usefixtures("adapters")
def test_singleton_band_numpy() -> None:
    """The n=1 band storage edge remains a valid direct solve."""
    operator = BandedOperator(np.asarray([[2.0]]), lower=0, upper=0)
    np.testing.assert_allclose(
        implicit_solve(operator, np.eye(1), np.asarray([6.0])), [3]
    )


@pytest.mark.parametrize(
    "bands", [np.ones((2, 3)), np.ones((3, 3), dtype=np.int64), np.full((3, 3), np.inf)]
)
def test_invalid_band_storage_refuses_numpy(bands: np.ndarray) -> None:
    """Invalid dimensions, dtype or physical entries are refused before LAPACK."""
    with pytest.raises((TypeError, ValueError)):
        BandedOperator(bands, lower=1, upper=1)


def test_legacy_tolerance_keyword_with_real_solver_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tol-only API receives explicit tolerances and still performs real GMRES."""
    original = scipy_operator.gmres
    observed: list[float] = []

    def legacy(
        operator: object, rhs: np.ndarray, *, tol: float, **kwargs: object
    ) -> tuple[np.ndarray, int]:
        observed.append(tol)
        return original(operator, rhs, rtol=tol, **kwargs)

    monkeypatch.setattr(scipy_operator, "gmres", legacy)
    diagonal = np.asarray([1.0, 2.0, 4.0])
    action = ActionOperator(
        (3, 3), diagonal.dtype, array_namespace(diagonal), lambda v: diagonal * v
    )
    result = solve_linear(LinearSolveRequest(action, np.ones(3)), ScipyGmresSolver())
    np.testing.assert_allclose(require_linear_converged(result), 1 / diagonal)
    assert observed == [LinearSolveOptions().rtol]
