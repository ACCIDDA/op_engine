# tests/test_matrix_ops.py
"""Unit tests for op_engine.matrix_ops.

This module verifies:
- Laplacian construction for supported boundary conditions.
- Crank-Nicolson operator construction (dense/sparse autodispatch).
- Predictor-corrector construction for dense and sparse base matrices.
- implicit_solve correctness for 1D and 2D RHS (dense and sparse paths).
- Group aggregation utilities (matrix-based and ID-based).
- Kronecker utilities (kron_prod, kron_sum) basic algebra and shape behavior.
- smooth behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from op_engine.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_operator,
    build_laplacian_tridiag,
    build_predictor_corrector,
    encode_groups,
    grouped_count_ids,
    grouped_sum_ids,
    grouped_sum_ids_2d,
    implicit_solve,
    kron_prod,
    kron_sum,
    matrix_grouped_count,
    matrix_grouped_sum,
    smooth,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.floating]


def _as_dense(mat: object) -> NDArray[np.floating]:
    """Convert a matrix-like to a dense ndarray.

    Args:
        mat: Dense ndarray or a scipy sparse matrix.

    Returns:
        Dense ndarray form of mat.
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

    # Symmetric
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
        geom.n,
        geom.dx,
        cfg.coeff,
        dtype=cfg.dtype,
        bc=cfg.bc)
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
    # Use a size large enough to trigger sparse autodispatch (threshold is internal).
    geom = GridGeometry(n=351, dx=0.1)
    cfg = DiffusionConfig(coeff=0.25, bc="neumann")
    dt = 0.02

    left, right = build_crank_nicolson_operator(geom, cfg, dt)

    assert issparse(left)
    assert issparse(right)

    lap = build_laplacian_tridiag(
        geom.n,
        geom.dx,
        cfg.coeff,
        dtype=cfg.dtype,
        bc=cfg.bc)
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

    for mat in (left_d, right_d):
        assert np.allclose(mat[0, :], np.eye(geom.n)[0, :])
        assert np.allclose(mat[-1, :], np.eye(geom.n)[-1, :])


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

    # Force dense and sparse via manual construction:
    # - Dense: small n via CN formula on dense Laplacian.
    lap = build_laplacian_tridiag(
        geom_small.n,
        geom_small.dx,
        cfg.coeff,
        dtype=cfg.dtype)
    a = lap.toarray() * dt
    ident = np.eye(geom_small.n, dtype=np.dtype(cfg.dtype))
    left_dense = ident - 0.5 * a
    right_dense = ident + 0.5 * a

    # - Sparse: use large-n dispatch, then slice down to compare on same n is messy.
    # Instead, directly build sparse CN operators from CSR Laplacian here:
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

    y_col = np.column_stack(
        [np.linalg.solve(left_a, right_a @ x[:, k]) for k in range(x.shape[1])]
    )
    y = implicit_solve(left, right, x)

    assert np.allclose(y, y_col, atol=1e-10, rtol=1e-10)


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
