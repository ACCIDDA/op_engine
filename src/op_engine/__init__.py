"""op_engine multiphysics ODE/PDE engine package."""

from __future__ import annotations

from importlib.metadata import version as _metadata_version

from ._array import array_namespace as array_namespace
from ._typing import Array as Array
from ._typing import Scalar as Scalar
from .adaptive_tau import AdaptiveTauLeapingConfig, AdaptiveTauLeapingSolver
from .core_solver import (
    CoreSolver,
    NonlinearIntegrationConvergenceError,
    NonlinearIntegrationDiagnostics,
    NonlinearMethodConfig,
)
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
from .spatial_grid import generate_adaptive_grid, generate_adaptive_grid_from_data
from .stochastic_solver import (
    DirectSSAConfig,
    DirectSSASolver,
    NumpyPoissonSampler,
    NumpySSASampler,
    PoissonSampler,
    SSASample,
    SSASampler,
    TauLeapingConfig,
    TauLeapingSolver,
)
from .thinning_ssa import (
    NumpyThinningSampler,
    RateBoundFunction,
    ThinningSample,
    ThinningSampler,
    ThinningSSAConfig,
    ThinningSSASolver,
    TotalRateBound,
)

__all__ = [
    "AdaptiveTauLeapingConfig",
    "AdaptiveTauLeapingSolver",
    "Array",
    "CoreSolver",
    "DenseNewtonSolver",
    "DiffusionConfig",
    "DirectSSAConfig",
    "DirectSSASolver",
    "GridGeometry",
    "JacobianVectorProduct",
    "ModelCore",
    "NewtonConfig",
    "NonlinearConvergenceError",
    "NonlinearIntegrationConvergenceError",
    "NonlinearIntegrationDiagnostics",
    "NonlinearJacobian",
    "NonlinearMethodConfig",
    "NonlinearProblem",
    "NonlinearResidual",
    "NonlinearSolveDiagnostics",
    "NonlinearSolveResult",
    "NonlinearSolver",
    "NumpyPoissonSampler",
    "NumpySSASampler",
    "NumpyThinningSampler",
    "Operator",
    "PoissonSampler",
    "RateBoundFunction",
    "SSASample",
    "SSASampler",
    "Scalar",
    "TauLeapingConfig",
    "TauLeapingSolver",
    "ThinningSSAConfig",
    "ThinningSSASolver",
    "ThinningSample",
    "ThinningSampler",
    "TotalRateBound",
    "array_namespace",
    "build_advection_matrix",
    "build_crank_nicolson_operator",
    "build_diffusion_matrix",
    "build_laplacian_tridiag",
    "build_predictor_corrector",
    "clear_implicit_solver_cache",
    "encode_groups",
    "generate_adaptive_grid",
    "generate_adaptive_grid_from_data",
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
