# tests/test_matrix_ops.py
"""Unit tests for op_engine.matrix_ops.

This module verifies:
- Laplacian construction for supported boundary conditions.
- Crank-Nicolson operator construction (dense/sparse autodispatch).
- Predictor-corrector construction for dense and sparse base matrices.
- IMEX/TR-BDF2 utilities:
    * identity + implicit-euler + trapezoidal stage operators
    * StageOperatorContext dataclass behavior
    * make_constant_base_builder wrapper
    * stage-operator factory behavior for:
        - constant base builders
        - time/state dependent base builders
- implicit_solve correctness for 1D and 2D RHS (dense and sparse paths).
- Group aggregation utilities (matrix-based and ID-based).
- Kronecker utilities (kron_prod, kron_sum) basic algebra and shape behavior.
- smooth behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity, issparse

from op_engine.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    StageOperatorContext,
    build_crank_nicolson_operator,
    build_identity_operator,
    build_implicit_euler_operators,
    build_laplacian_tridiag,
    build_predictor_corrector,
    build_trapezoidal_operators,
    encode_groups,
    grouped_count_ids,
    grouped_sum_ids,
    grouped_sum_ids_2d,
    implicit_solve,
    kron_prod,
    kron_sum,
    make_constant_base_builder,
    make_stage_operator_factory,
    matrix_grouped_count,
    matrix_grouped_sum,
    smooth,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


def _as_dense(mat: object) -> NDArray[np.floating]:
    """
    Convert a matrix-like to a dense ndarray.

    Args:
        mat: Input matrix-like object (ndarray or sparse matrix).

    Returns:
        Dense ndarray representation of the input.
    """
    if hasattr(mat, "toarray"):
        return np.asarray(mat.toarray())
    return np.asarray(mat)


# -------------------------------------------------------------------
# Laplacian
# -------------------------------------------------------------------


def test_laplacian_tridiag_neumann_structure_small() -> None:
    """Laplacian with Neumann BC has endpoint -1 on the main diagonal (scaled)."""
    n = 5
    dx = 1.0
    coeff = 1.0
    a = build_laplacian_tridiag(n, dx, coeff, bc="neumann")
    a_dense = a.toarray()

    assert np.allclose(a_dense, a_dense.T)

    main_diag = np.diag(a_dense)
    off1 = np.diag(a_dense, k=1)
    offm1 = np.diag(a_dense, k=-1)

    assert main_diag[0] == pytest.approx(-1.0)
    assert main_diag[-1] == pytest.approx(-1.0)
    assert np.allclose(main_diag[1:-1], -2.0)

    assert np.allclose(off1, 1.0)
    assert np.allclose(offm1, 1.0)


def test_laplacian_tridiag_absorbing_structure_small() -> None:
    """Laplacian with absorbing BC has endpoint -2 on the main diagonal (scaled)."""
    n = 5
    dx = 1.0
    coeff = 1.0
    a = build_laplacian_tridiag(n, dx, coeff, bc="absorbing")
    a_dense = a.toarray()

    main_diag = np.diag(a_dense)
    assert main_diag[0] == pytest.approx(-2.0)
    assert main_diag[-1] == pytest.approx(-2.0)
    assert np.allclose(main_diag[1:-1], -2.0)


def test_laplacian_unknown_bc_raises() -> None:
    """Unknown boundary conditions raise ValueError."""
    with pytest.raises(ValueError, match="Unknown bc"):
        _ = build_laplacian_tridiag(5, 1.0, 1.0, bc="nope")


# -------------------------------------------------------------------
# Crank-Nicolson operators
# -------------------------------------------------------------------


def test_crank_nicolson_operator_dense_matches_formula() -> None:
    """Dense CN operators match the closed-form I ± 0.5*dt*A construction."""
    geom = GridGeometry(n=10, dx=0.1)
    cfg = DiffusionConfig(coeff=0.5, bc="neumann")
    dt = 0.01

    left, right = build_crank_nicolson_operator(geom, cfg, dt)

    lap = build_laplacian_tridiag(
        geom.n, geom.dx, cfg.coeff, dtype=cfg.dtype, bc=cfg.bc
    )
    a = lap.toarray() * dt
    ident = np.eye(geom.n, dtype=np.dtype(cfg.dtype))

    left_expected = ident - 0.5 * a
    right_expected = ident + 0.5 * a

    assert not issparse(left)
    assert not issparse(right)
    assert np.allclose(np.asarray(left), left_expected)
    assert np.allclose(np.asarray(right), right_expected)


def test_crank_nicolson_operator_sparse_matches_formula_large() -> None:
    """Sparse CN operators match the same formula, and return CSR on large n."""
    geom = GridGeometry(n=351, dx=0.1)
    cfg = DiffusionConfig(coeff=0.25, bc="neumann")
    dt = 0.02

    left, right = build_crank_nicolson_operator(geom, cfg, dt)

    assert issparse(left)
    assert issparse(right)

    lap = build_laplacian_tridiag(
        geom.n, geom.dx, cfg.coeff, dtype=cfg.dtype, bc=cfg.bc
    )
    a = lap.toarray() * dt
    ident = np.eye(geom.n, dtype=np.dtype(cfg.dtype))

    left_expected = ident - 0.5 * a
    right_expected = ident + 0.5 * a

    assert np.allclose(_as_dense(left), left_expected)
    assert np.allclose(_as_dense(right), right_expected)


def test_crank_nicolson_absorbing_enforces_identity_rows() -> None:
    """Absorbing BC forces first/last rows to identity in CN operators."""
    geom = GridGeometry(n=20, dx=0.1)
    cfg = DiffusionConfig(coeff=0.3, bc="absorbing")
    dt = 0.05

    left, right = build_crank_nicolson_operator(geom, cfg, dt)

    left_d = _as_dense(left)
    right_d = _as_dense(right)

    eye = np.eye(geom.n)
    for mat in (left_d, right_d):
        assert np.allclose(mat[0, :], eye[0, :])
        assert np.allclose(mat[-1, :], eye[-1, :])


# -------------------------------------------------------------------
# Predictor-corrector
# -------------------------------------------------------------------


def test_build_predictor_corrector_dense_base_matrix() -> None:
    """Predictor-corrector from dense base uses predictor=I and I ± 0.5*A."""
    n = 8
    rng = np.random.default_rng(0)
    base = rng.standard_normal(size=(n, n))

    predictor, left, right = build_predictor_corrector(base)

    ident = np.eye(n, dtype=base.dtype)
    assert np.allclose(np.asarray(predictor), ident)
    assert np.allclose(np.asarray(left), ident - 0.5 * base)
    assert np.allclose(np.asarray(right), ident + 0.5 * base)


def test_build_predictor_corrector_sparse_base_matrix() -> None:
    """Predictor-corrector from CSR base returns operators consistent with dense."""
    n = 10
    lap = build_laplacian_tridiag(n, 0.1, 0.3)
    base = lap * 0.25  # CSR base

    predictor, left, right = build_predictor_corrector(base)

    base_d = base.toarray()
    ident = np.eye(n, dtype=base_d.dtype)
    assert np.allclose(_as_dense(predictor), ident)
    assert np.allclose(_as_dense(left), ident - 0.5 * base_d)
    assert np.allclose(_as_dense(right), ident + 0.5 * base_d)


# -------------------------------------------------------------------
# IMEX/TR-BDF2 support utilities (operators + factories)
# -------------------------------------------------------------------


def test_build_identity_operator_dense_small() -> None:
    """build_identity_operator returns dense for small n by default."""
    n = 10
    ident = build_identity_operator(n, dtype=np.float64)
    assert not issparse(ident)
    assert ident.shape == (n, n)
    assert np.allclose(np.asarray(ident), np.eye(n))


def test_build_identity_operator_sparse_large() -> None:
    """build_identity_operator returns CSR for large n by default."""
    n = 400
    ident = build_identity_operator(n, dtype=np.float64)
    assert ident.shape == (n, n)
    assert np.allclose(_as_dense(ident), np.eye(n))


def test_build_implicit_euler_operators_dense_matches_formula() -> None:
    """Implicit Euler operators satisfy L = I - s*A and R = I for dense A."""
    n = 6
    rng = np.random.default_rng(10)
    a = rng.standard_normal(size=(n, n))
    s = 0.25

    left, right = build_implicit_euler_operators(a, dt_scale=s)

    ident = np.eye(n, dtype=a.dtype)
    assert np.allclose(np.asarray(left), ident - s * a)
    assert np.allclose(np.asarray(right), ident)


def test_build_implicit_euler_operators_sparse_matches_formula() -> None:
    """Implicit Euler operators satisfy L = I - s*A and R = I for sparse A."""
    n = 12
    rng = np.random.default_rng(11)
    a_dense = rng.standard_normal(size=(n, n))
    a_sparse = csr_matrix(a_dense)
    s = 0.1

    left, right = build_implicit_euler_operators(a_sparse, dt_scale=s)

    assert issparse(left)
    assert issparse(right)

    ident = np.eye(n, dtype=a_dense.dtype)
    assert np.allclose(_as_dense(left), ident - s * a_dense)
    assert np.allclose(_as_dense(right), ident)


def test_build_trapezoidal_operators_dense_matches_formula() -> None:
    """Trapezoidal operators satisfy L = I - 0.5*s*A and R = I + 0.5*s*A (dense)."""
    n = 7
    rng = np.random.default_rng(12)
    a = rng.standard_normal(size=(n, n))
    s = 0.4

    left, right = build_trapezoidal_operators(a, dt_scale=s)

    ident = np.eye(n, dtype=a.dtype)
    assert np.allclose(np.asarray(left), ident - 0.5 * s * a)
    assert np.allclose(np.asarray(right), ident + 0.5 * s * a)


def test_build_trapezoidal_operators_sparse_matches_formula() -> None:
    """Trapezoidal operators satisfy L/R formula for sparse A."""
    n = 9
    rng = np.random.default_rng(13)
    a_dense = rng.standard_normal(size=(n, n))
    a_sparse = csr_matrix(a_dense)
    s = 0.33

    left, right = build_trapezoidal_operators(a_sparse, dt_scale=s)

    assert issparse(left)
    assert issparse(right)

    ident = np.eye(n, dtype=a_dense.dtype)
    assert np.allclose(_as_dense(left), ident - 0.5 * s * a_dense)
    assert np.allclose(_as_dense(right), ident + 0.5 * s * a_dense)


def test_stage_operator_context_fields_roundtrip() -> None:
    """StageOperatorContext carries (t, y, stage) faithfully."""
    rng = np.random.default_rng(99)
    y = rng.standard_normal(size=5).astype(float)
    ctx = StageOperatorContext(t=1.25, y=y, stage="tr")
    assert ctx.t == pytest.approx(1.25)
    assert ctx.stage == "tr"
    assert np.allclose(ctx.y, y)


def test_make_constant_base_builder_returns_same_operator() -> None:
    """make_constant_base_builder returns the same object (dense) each time."""
    a = np.eye(4, dtype=float)
    builder = make_constant_base_builder(a)
    ctx = StageOperatorContext(t=0.0, y=np.zeros(4), stage=None)
    out1 = builder(ctx)
    out2 = builder(ctx)
    # We expect identity by value; by object is also true in current implementation.
    assert np.allclose(np.asarray(out1), a)
    assert np.allclose(np.asarray(out2), a)


def test_stage_op_factory_implicit_euler_constant_builder_matches_direct_builder() -> (
    None
):
    """Factory(implicit-euler) same as build_implicit_euler_operators for constant."""
    n = 8
    rng = np.random.default_rng(20)
    a = rng.standard_normal(size=(n, n))
    builder = make_constant_base_builder(a)
    factory = make_stage_operator_factory(builder, scheme="implicit-euler")

    dt = 0.5
    scale = 0.3
    ctx = StageOperatorContext(t=0.0, y=np.zeros(n), stage="tr")

    left_f, right_f = factory(dt, scale, ctx)
    left_d, right_d = build_implicit_euler_operators(a, dt_scale=dt * scale)

    assert np.allclose(_as_dense(left_f), np.asarray(left_d))
    assert np.allclose(_as_dense(right_f), np.asarray(right_d))


def test_stage_op_factory_traz_constant_builder_matches_direct_builder() -> None:
    """Factory(trapezoidal) same as build_trapezoidal_operators for constant base."""
    n = 8
    rng = np.random.default_rng(21)
    a = rng.standard_normal(size=(n, n))
    builder = make_constant_base_builder(a)
    factory = make_stage_operator_factory(builder, scheme="trapezoidal")

    dt = 0.25
    scale = 0.8
    ctx = StageOperatorContext(t=2.0, y=np.ones(n), stage="bdf2")

    left_f, right_f = factory(dt, scale, ctx)
    left_d, right_d = build_trapezoidal_operators(a, dt_scale=dt * scale)

    assert np.allclose(_as_dense(left_f), np.asarray(left_d))
    assert np.allclose(_as_dense(right_f), np.asarray(right_d))


def test_stage_operator_factory_time_state_dependent_builder_is_used() -> None:
    """Factory passes ctx through to base_builder; builder can depend on t and y."""
    n = 5

    # Base operator depends on t and sum(y) so we can assert it changes.
    def base_builder(ctx: StageOperatorContext) -> NDArray[np.floating]:
        s = float(np.sum(ctx.y))
        return (ctx.t + s) * np.eye(n, dtype=float)

    factory = make_stage_operator_factory(base_builder, scheme="implicit-euler")

    dt = 0.1
    scale = 2.0

    ctx1 = StageOperatorContext(t=1.0, y=np.zeros(n), stage="tr")
    left1, right1 = factory(dt, scale, ctx1)

    ctx2 = StageOperatorContext(t=2.0, y=np.ones(n), stage="tr")
    left2, right2 = factory(dt, scale, ctx2)

    # For implicit-euler: L = I - (dt*scale) * A, with A = (t + sum(y))*I
    ident = np.eye(n, dtype=float)
    a1 = (ctx1.t + float(np.sum(ctx1.y))) * np.eye(n, dtype=float)
    a2 = (ctx2.t + float(np.sum(ctx2.y))) * np.eye(n, dtype=float)

    assert np.allclose(_as_dense(right1), ident)
    assert np.allclose(_as_dense(right2), ident)
    assert np.allclose(_as_dense(left1), ident - (dt * scale) * a1)
    assert np.allclose(_as_dense(left2), ident - (dt * scale) * a2)

    # and they should not be identical
    assert not np.allclose(_as_dense(left1), _as_dense(left2))


def test_stage_operator_factory_unknown_scheme_raises() -> None:
    """Unknown scheme in stage-operator factory raises ValueError."""
    n = 4
    a = np.eye(n)

    builder = make_constant_base_builder(a)
    with pytest.raises(ValueError, match="Unknown scheme"):
        _ = make_stage_operator_factory(builder, scheme="nope")


def test_build_stage_operators_nonfinite_scale_raises() -> None:
    """Stage operator builders reject non-finite dt_scale."""
    n = 4
    a = np.eye(n)
    with pytest.raises(ValueError, match="scale must be a finite float"):
        _ = build_implicit_euler_operators(a, dt_scale=float("nan"))
    with pytest.raises(ValueError, match="scale must be a finite float"):
        _ = build_trapezoidal_operators(a, dt_scale=float("inf"))


def test_stage_operator_factory_rejects_bad_base_builder_return_type() -> None:
    """If base_builder returns an invalid type, factory should raise TypeError."""
    n = 3

    def bad_builder(ctx: StageOperatorContext) -> object:  # noqa: ARG001
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]  # list, not ndarray/csr

    factory = make_stage_operator_factory(bad_builder, scheme="implicit-euler")  # type: ignore[arg-type]
    ctx = StageOperatorContext(t=0.0, y=np.zeros(n), stage=None)
    with pytest.raises(TypeError, match="base_builder must return"):
        _ = factory(0.1, 1.0, ctx)


# -------------------------------------------------------------------
# implicit_solve correctness
# -------------------------------------------------------------------


def test_implicit_solve_dense_matches_direct() -> None:
    """implicit_solve (dense) matches np.linalg.solve on the equivalent system."""
    geom = GridGeometry(n=12, dx=0.1)
    cfg = DiffusionConfig(coeff=0.2, bc="neumann")
    dt = 0.05
    left, right = build_crank_nicolson_operator(geom, cfg, dt)

    x = np.linspace(0.0, 1.0, geom.n, dtype=float)
    rhs = np.asarray(right) @ x
    y_direct = np.linalg.solve(np.asarray(left), rhs)

    y = implicit_solve(left, right, x)
    assert np.allclose(y, y_direct, atol=1e-10, rtol=1e-10)


def test_implicit_solve_sparse_matches_dense() -> None:
    """implicit_solve (sparse) matches dense result on the same operators."""
    geom_small = GridGeometry(n=40, dx=0.05)
    cfg = DiffusionConfig(coeff=0.1, bc="neumann")
    dt = 0.02

    lap = build_laplacian_tridiag(
        geom_small.n, geom_small.dx, cfg.coeff, dtype=cfg.dtype
    )
    a = lap.toarray() * dt
    ident = np.eye(geom_small.n, dtype=np.dtype(cfg.dtype))
    left_dense = ident - 0.5 * a
    right_dense = ident + 0.5 * a

    lap_csr = lap.tocsr()
    left_sparse = (csr_matrix(ident) - 0.5 * (lap_csr * dt)).tocsr()
    right_sparse = (csr_matrix(ident) + 0.5 * (lap_csr * dt)).tocsr()

    rng = np.random.default_rng(7)
    x = rng.standard_normal(size=geom_small.n)

    y_dense = implicit_solve(left_dense, right_dense, x)
    y_sparse = implicit_solve(left_sparse, right_sparse, x)

    assert np.allclose(y_sparse, y_dense, atol=1e-10, rtol=1e-10)


def test_implicit_solve_2d_rhs_matches_columnwise() -> None:
    """implicit_solve supports 2D RHS and matches column-wise solves (dense path)."""
    geom = GridGeometry(n=10, dx=0.1)
    cfg = DiffusionConfig(coeff=0.2, bc="neumann")
    dt = 0.03
    left, right = build_crank_nicolson_operator(geom, cfg, dt)

    left_a = np.asarray(left)
    right_a = np.asarray(right)

    rng = np.random.default_rng(6)
    x = rng.standard_normal(size=(geom.n, 3))

    y_col = np.column_stack([
        np.linalg.solve(left_a, right_a @ x[:, k]) for k in range(x.shape[1])
    ])
    y = implicit_solve(left, right, x)

    assert np.allclose(y, y_col, atol=1e-10, rtol=1e-10)


def test_implicit_solve_identity_operators_returns_x() -> None:
    """implicit_solve with L=I and R=I returns x unchanged (1D and 2D)."""
    n = 11
    ident = identity(n, format="csr", dtype=np.float64)
    rng = np.random.default_rng(123)

    x1 = rng.standard_normal(size=n)
    y1 = implicit_solve(ident, ident, x1)
    assert np.allclose(y1, x1)

    x2 = rng.standard_normal(size=(n, 4))
    y2 = implicit_solve(ident, ident, x2)
    assert np.allclose(y2, x2)


# -------------------------------------------------------------------
# Group ops & IDs equivalence
# -------------------------------------------------------------------


def test_grouped_sum_ids_matches_matrix_grouped_sum_dense_encoding() -> None:
    """grouped_sum_ids matches matrix_grouped_sum for 1D values (dense encoding)."""
    rng = np.random.default_rng(0)
    n_rep = 100
    n_groups = 5

    group_ids = rng.integers(0, n_groups, size=n_rep)
    values = rng.normal(size=n_rep)

    group_matrix = encode_groups(group_ids, n_groups, prefer_sparse=False)
    sum_matrix = matrix_grouped_sum(group_matrix, values)
    sum_ids = grouped_sum_ids(values, group_ids, n_groups)

    assert np.allclose(sum_ids, sum_matrix)


def test_grouped_sum_ids_2d_matches_matrix_grouped_sum_dense_encoding() -> None:
    """grouped_sum_ids_2d matches matrix_grouped_sum for 2D values (dense encoding)."""
    rng = np.random.default_rng(1)
    n_rep = 50
    n_groups = 4
    k = 3

    group_ids = rng.integers(0, n_groups, size=n_rep)
    values = rng.normal(size=(n_rep, k))

    group_matrix = encode_groups(group_ids, n_groups, prefer_sparse=False)
    sum_matrix = matrix_grouped_sum(group_matrix, values)
    sum_ids = grouped_sum_ids_2d(values, group_ids, n_groups)

    assert np.allclose(sum_ids, sum_matrix)


def test_grouped_count_ids_matches_matrix_grouped_count_dense_encoding() -> None:
    """grouped_count_ids matches matrix_grouped_count (dense encoding)."""
    rng = np.random.default_rng(2)
    n_rep = 100
    n_groups = 7

    group_ids = rng.integers(0, n_groups, size=n_rep)
    group_matrix = encode_groups(group_ids, n_groups, prefer_sparse=False)

    count_matrix = matrix_grouped_count(group_matrix)
    count_ids = grouped_count_ids(group_ids, n_groups)

    assert np.allclose(count_ids, count_matrix)


def test_encode_groups_prefer_sparse_roundtrip_type() -> None:
    """encode_groups honors prefer_sparse and returns the expected representation."""
    group_ids = np.array([0, 1, 0, 2, 1], dtype=int)

    dense = encode_groups(group_ids, n_groups=3, prefer_sparse=False)
    assert not issparse(dense)

    sparse = encode_groups(group_ids, n_groups=3, prefer_sparse=True)
    assert issparse(sparse)


# -------------------------------------------------------------------
# Kronecker utilities
# -------------------------------------------------------------------


def test_kron_prod_shapes_and_dense_value() -> None:
    """kron_prod returns correct shape and matches np.kron for dense inputs."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.5, 0.0], [0.0, 2.0]])

    out = kron_prod(a, b)
    assert out.shape == (4, 4)
    assert np.allclose(np.asarray(out), np.kron(a, b))


def test_kron_sum_two_terms_matches_definition_dense() -> None:
    """kron_sum([A, B]) matches A ⊗ I + I ⊗ B for dense inputs."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 0.0], [0.0, 6.0]])

    out = kron_sum([a, b])
    ident = np.eye(2)

    expected = np.kron(a, ident) + np.kron(ident, b)
    assert out.shape == (4, 4)
    assert np.allclose(np.asarray(out), expected)


def test_kron_sum_empty_raises() -> None:
    """kron_sum requires at least one operator."""
    with pytest.raises(ValueError, match="at least one operator"):
        _ = kron_sum([])


# -------------------------------------------------------------------
# smooth behavior
# -------------------------------------------------------------------


def test_smooth_alpha_zero_identity() -> None:
    """smooth(alpha=0) returns the input unchanged."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 5, 6))

    y = smooth(x, alpha=0.0)
    assert np.allclose(y, x)


def test_smooth_alpha_one_full_mean() -> None:
    """smooth(alpha=1) returns the mean along the last axis."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 5, 6))

    y = smooth(x, alpha=1.0)
    mean_last = x.mean(axis=-1, keepdims=True)

    assert np.allclose(y, mean_last)


def test_smooth_out_argument_used() -> None:
    """Smooth uses and returns the provided out array."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 5, 6))
    out = np.empty_like(x)

    y = smooth(x, alpha=0.5, out=out)

    assert y is out
    mean_last = x.mean(axis=-1, keepdims=True)
    expected = 0.5 * x + 0.5 * mean_last
    assert np.allclose(y, expected)
