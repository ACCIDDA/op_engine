"""Advance an externally assembled 1D finite-element diffusion system.

This example deliberately keeps mesh and element assembly outside ``op_engine``.
It assembles the P1 semi-discrete system

    M du/dt = K u,

eliminates homogeneous Dirichlet boundary degrees of freedom, and maps the
mass and stiffness matrices to the existing ``(L, R)`` stage-operator contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, diags

from op_engine.core_solver import CoreSolver, OperatorSpecs, RunConfig
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from op_engine.matrix_ops import StageOperatorContext

_STATE_ARRAY_NONE_ERROR = "state_array is None despite store_history=True"


@dataclass(frozen=True, slots=True)
class FemDiffusionResult:
    """Stored solution and assembled operators for the example run."""

    nodes: NDArray[np.float64]
    times: NDArray[np.float64]
    states: NDArray[np.float64]
    mass: csr_matrix
    stiffness: csr_matrix


def assemble_p1_diffusion(
    nodes: NDArray[np.float64],
    diffusivity: float,
) -> tuple[csr_matrix, csr_matrix]:
    """Assemble P1 mass and diffusion matrices on an arbitrary 1D mesh.

    Homogeneous Dirichlet values at the two endpoints are eliminated. The
    returned stiffness matrix includes the diffusion sign, so the interior
    system is ``M du/dt = K u`` with negative-definite ``K``.

    Args:
        nodes: Strictly increasing mesh nodes, including both endpoints.
        diffusivity: Non-negative diffusion coefficient.

    Returns:
        Interior consistent-mass and signed-stiffness CSR matrices.

    Raises:
        ValueError: If the mesh or diffusivity is invalid.
    """
    mesh = np.asarray(nodes, dtype=np.float64)
    if mesh.ndim != 1 or mesh.size < 3:
        msg = "nodes must be a one-dimensional array with at least three entries"
        raise ValueError(msg)
    widths = np.diff(mesh)
    if not np.all(np.isfinite(mesh)) or np.any(widths <= 0.0):
        msg = "nodes must be finite and strictly increasing"
        raise ValueError(msg)
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        msg = "diffusivity must be finite and non-negative"
        raise ValueError(msg)

    mass_diagonal = (widths[:-1] + widths[1:]) / 3.0
    mass_off_diagonal = widths[1:-1] / 6.0
    stiffness_diagonal = -diffusivity * (
        np.reciprocal(widths[:-1]) + np.reciprocal(widths[1:])
    )
    stiffness_off_diagonal = diffusivity * np.reciprocal(widths[1:-1])

    mass = diags(
        (mass_off_diagonal, mass_diagonal, mass_off_diagonal),
        offsets=(-1, 0, 1),
        format="csr",
    )
    stiffness = diags(
        (stiffness_off_diagonal, stiffness_diagonal, stiffness_off_diagonal),
        offsets=(-1, 0, 1),
        format="csr",
    )
    return mass, stiffness


def make_mass_trapezoidal_factory(
    mass: csr_matrix,
    stiffness: csr_matrix,
) -> Callable[[float, float, StageOperatorContext], tuple[csr_matrix, csr_matrix]]:
    """Map ``M du/dt = K u`` to trapezoidal stage operators.

    The closure reuses matrix objects for repeated stage sizes. This matters
    because op_engine's sparse solve cache keys factorizations by operator
    identity.

    Returns:
        A stage-operator factory returning ``(M-hK/2, M+hK/2)``.
    """
    cache: dict[float, tuple[csr_matrix, csr_matrix]] = {}

    def factory(
        dt: float,
        scale: float,
        ctx: StageOperatorContext,
    ) -> tuple[csr_matrix, csr_matrix]:
        del ctx
        stage_size = float(dt) * float(scale)
        operators = cache.get(stage_size)
        if operators is None:
            half_step = 0.5 * stage_size
            operators = (
                (mass - half_step * stiffness).tocsr(),
                (mass + half_step * stiffness).tocsr(),
            )
            cache[stage_size] = operators
        return operators

    return factory


def run_fem_diffusion(
    *,
    nodes: NDArray[np.float64],
    times: NDArray[np.float64],
    diffusivity: float = 0.1,
) -> FemDiffusionResult:
    """Run the P1 heat-equation example from a sine initial condition.

    Args:
        nodes: Mesh nodes on ``[0, 1]``, including endpoints.
        times: Strictly increasing output and fixed-step times.
        diffusivity: Non-negative diffusion coefficient.

    Returns:
        Stored interior states and the externally assembled matrices.

    Raises:
        RuntimeError: If the configured history is unexpectedly unavailable.
    """
    mesh = np.asarray(nodes, dtype=np.float64)
    output_times = np.asarray(times, dtype=np.float64)
    mass, stiffness = assemble_p1_diffusion(mesh, diffusivity)
    interior = mesh[1:-1]

    options = ModelCoreOptions(store_history=True, dtype=np.float64)
    core = ModelCore(
        n_states=interior.size,
        n_subgroups=1,
        time_grid=output_times,
        options=options,
    )
    core.set_initial_state(np.sin(np.pi * interior)[:, np.newaxis])

    def zero_explicit_rhs(
        _time: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.zeros_like(state)

    factory = make_mass_trapezoidal_factory(mass, stiffness)
    config = RunConfig(
        method="imex-heun-tr",
        adaptive=False,
        operators=OperatorSpecs(default=factory),
    )
    CoreSolver(core, operator_axis="state").run(
        zero_explicit_rhs,  # type: ignore[arg-type]
        config=config,
    )
    if core.state_array is None:
        raise RuntimeError(_STATE_ARRAY_NONE_ERROR)

    return FemDiffusionResult(
        nodes=mesh,
        times=output_times,
        states=core.state_array,
        mass=mass,
        stiffness=stiffness,
    )
