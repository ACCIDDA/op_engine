"""op_engine multiphysics ODE/PDE engine package."""

from __future__ import annotations

from importlib.metadata import version as _metadata_version

from ._typing import Array
from .core_solver import CoreSolver
from .matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    Operator,
    build_advection_matrix,
    build_crank_nicolson_operator,
    build_diffusion_matrix,
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
from .nonlinear_solver import (
    DenseNewtonSolver,
    JacobianVectorProduct,
    NewtonConfig,
    NonlinearConvergenceError,
    NonlinearJacobian,
    NonlinearProblem,
    NonlinearResidual,
    NonlinearSolveDiagnostics,
    NonlinearSolver,
    NonlinearSolveResult,
    require_converged,
)
from .stochastic_solver import (
    NumpyPoissonSampler,
    PoissonSampler,
    TauLeapingConfig,
    TauLeapingSolver,
)

__all__ = [
    "Array",
    "CoreSolver",
    "DenseNewtonSolver",
    "DiffusionConfig",
    "GridGeometry",
    "JacobianVectorProduct",
    "ModelCore",
    "NewtonConfig",
    "NonlinearConvergenceError",
    "NonlinearJacobian",
    "NonlinearProblem",
    "NonlinearResidual",
    "NonlinearSolveDiagnostics",
    "NonlinearSolveResult",
    "NonlinearSolver",
    "NumpyPoissonSampler",
    "Operator",
    "PoissonSampler",
    "TauLeapingConfig",
    "TauLeapingSolver",
    "build_advection_matrix",
    "build_crank_nicolson_operator",
    "build_diffusion_matrix",
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
    "require_converged",
    "smooth",
]

__version__ = _metadata_version("op_engine")
