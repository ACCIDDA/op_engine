"""op_engine multiphysics ODE/PDE engine package."""

from __future__ import annotations

from .core_solver import CoreSolver, RHSFunction, ReactionRHSFunction
from .matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    Operator,
    build_crank_nicolson_operator,
    build_laplacian_tridiag,
    build_predictor_corrector,
    clear_implicit_solver_cache,
    encode_groups,
    grouped_count_ids,
    grouped_sum_ids,
    grouped_sum_ids_2d,
    implicit_solve,
    kron_prod,
    kron_sum,
    matrix_grouped_count,
    matrix_grouped_sum,
    matrix_masked_sum,
    smooth,
)
from .model_core import ModelCore

__all__ = [
    "CoreSolver",
    "DiffusionConfig",
    "GridGeometry",
    "ModelCore",
    "Operator",
    "RHSFunction",
    "ReactionRHSFunction",
    "build_crank_nicolson_operator",
    "build_laplacian_tridiag",
    "build_predictor_corrector",
    "clear_implicit_solver_cache",
    "encode_groups",
    "grouped_count_ids",
    "grouped_sum_ids",
    "grouped_sum_ids_2d",
    "implicit_solve",
    "kron_prod",
    "kron_sum",
    "matrix_grouped_count",
    "matrix_grouped_sum",
    "matrix_masked_sum",
    "smooth",
]

__version__ = "0.1.0"

