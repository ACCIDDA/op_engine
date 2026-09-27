"""Trait-structured chemostat PDE with a non-uniform finite-volume grid.

The semi-discrete model is

    dn/dt = n * g(theta, S) - dilution * n + diffusivity * d2n/dtheta2
    dS/dt = dilution * (resource_in - S)
             - integral(n * g(theta, S) * uptake(theta), theta)

Diffusion is the linear implicit partition. Growth, dilution, and resource
coupling are evaluated explicitly. A high-accuracy SciPy RK45 solve of the
same semi-discrete equations provides a reference trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from op_engine.core_solver import CoreSolver, OperatorSpecs, RunConfig
from op_engine.matrix_ops import (
    build_diffusion_matrix,
    make_constant_base_builder,
    make_stage_operator_factory,
)
from op_engine.model_core import ModelCore, ModelCoreOptions
from op_engine.spatial_grid import generate_adaptive_grid

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_STATE_HISTORY_ERROR = "state history is unavailable despite store_history=True"
_REFERENCE_SOLVE_ERROR = "RK45 reference failed: {message}"


@dataclass(frozen=True, slots=True)
class ChemostatParameters:
    """Parameters for the trait-structured chemostat.

    Rates use days as the time unit and the trait coordinate is dimensionless.
    """

    dilution: float = 0.18
    resource_in: float = 2.0
    max_growth: float = 0.8
    half_saturation: float = 0.35
    optimal_trait: float = 0.2
    trait_width: float = 0.65
    diffusivity: float = 0.006
    yield_coefficient: float = 0.8
    uptake_trait_slope: float = 0.12
    initial_biomass: float = 0.3
    initial_resource: float = 1.4
    initial_trait_mean: float = -0.55
    initial_trait_width: float = 0.32

    def __post_init__(self) -> None:
        """Validate the physical parameter domain.

        Raises:
            ValueError: If a rate, scale, or initial quantity is invalid.
        """
        nonnegative = {
            "dilution": self.dilution,
            "diffusivity": self.diffusivity,
            "initial_biomass": self.initial_biomass,
            "initial_resource": self.initial_resource,
        }
        positive = {
            "resource_in": self.resource_in,
            "max_growth": self.max_growth,
            "half_saturation": self.half_saturation,
            "trait_width": self.trait_width,
            "yield_coefficient": self.yield_coefficient,
            "initial_trait_width": self.initial_trait_width,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                msg = f"{name} must be finite and non-negative"
                raise ValueError(msg)
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                msg = f"{name} must be finite and positive"
                raise ValueError(msg)
        if not np.isfinite(self.optimal_trait) or not np.isfinite(
            self.initial_trait_mean
        ):
            msg = "trait locations must be finite"
            raise ValueError(msg)
        if not np.isfinite(self.uptake_trait_slope):
            msg = "uptake_trait_slope must be finite"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TraitObservables:
    """Resource and first five trait-distribution observables over time."""

    biomass: NDArray[np.float64]
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]
    skewness: NDArray[np.float64]
    kurtosis: NDArray[np.float64]
    resource: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChemostatResult:
    """Grid, trajectories, and observables from one comparison run."""

    trait_grid: NDArray[np.float64]
    cell_weights: NDArray[np.float64]
    times: NDArray[np.float64]
    imex_states: NDArray[np.float64]
    rk45_states: NDArray[np.float64]
    imex_observables: TraitObservables
    rk45_observables: TraitObservables


def cell_volume_weights(
    centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the finite-volume weights implied by non-uniform diffusion.

    Boundary faces are inferred one half-spacing outside the endpoint centers,
    matching :func:`op_engine.matrix_ops.build_diffusion_matrix`.

    Args:
        centers: Strictly increasing finite cell centers.

    Returns:
        Positive cell-volume weights with the same shape as ``centers``.

    Raises:
        ValueError: If the center coordinates are invalid.
    """
    grid = np.asarray(centers, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        msg = "centers must be one-dimensional with at least two entries"
        raise ValueError(msg)
    spacing = np.diff(grid)
    if not np.isfinite(grid).all() or not np.all(spacing > 0.0):
        msg = "centers must be finite and strictly increasing"
        raise ValueError(msg)
    weights = np.empty_like(grid)
    weights[0] = spacing[0]
    weights[-1] = spacing[-1]
    if grid.size > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights


def build_trait_grid(
    n_cells: int = 41,
    *,
    domain: tuple[float, float] = (-2.0, 2.0),
    parameters: ChemostatParameters | None = None,
) -> NDArray[np.float64]:
    """Generate a curvature-weighted grid around the initial trait profile.

    Args:
        n_cells: Number of finite-volume centers, including the domain ends.
        domain: Increasing trait-coordinate bounds.
        parameters: Optional model parameters controlling the initial profile.

    Returns:
        A static non-uniform trait grid.
    """
    params = parameters or ChemostatParameters()

    def profile(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled = (theta - params.initial_trait_mean) / params.initial_trait_width
        return np.asarray(np.exp(-0.5 * scaled * scaled), dtype=np.float64)

    span = float(domain[1] - domain[0])
    return generate_adaptive_grid(
        profile,
        domain,
        n_cells,
        epsilon=0.025,
        sampling_points=max(2049, 32 * n_cells + 1),
        smoothing_window=9,
        minimum_spacing=span / (50.0 * n_cells),
    )


def trait_growth_rate(
    trait: NDArray[np.float64],
    resource: float,
    parameters: ChemostatParameters,
) -> NDArray[np.float64]:
    """Evaluate Monod growth modulated by a Gaussian trait optimum.

    Returns:
        Growth rate at every trait center.
    """
    resource_value = float(resource)
    monod = resource_value / (parameters.half_saturation + resource_value)
    scaled = (trait - parameters.optimal_trait) / parameters.trait_width
    fitness = np.exp(-0.5 * scaled * scaled)
    return np.asarray(parameters.max_growth * monod * fitness, dtype=np.float64)


def uptake_weight(
    trait: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> NDArray[np.float64]:
    """Return positive trait-dependent resource uptake per unit growth.

    Returns:
        Uptake weights at every trait center.
    """
    return np.asarray(
        np.exp(parameters.uptake_trait_slope * trait) / parameters.yield_coefficient,
        dtype=np.float64,
    )


def initial_state(
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> NDArray[np.float64]:
    """Construct a normalized Gaussian density plus scalar resource.

    Returns:
        Flat state ``[n(theta_0), ..., n(theta_N), resource]``.
    """
    scaled = (trait - parameters.initial_trait_mean) / parameters.initial_trait_width
    density = np.exp(-0.5 * scaled * scaled)
    normalizer = float(np.dot(weights, density))
    density *= parameters.initial_biomass / normalizer
    return np.concatenate((density, np.asarray([parameters.initial_resource])))


def reaction_tendency(
    state: NDArray[np.float64],
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> NDArray[np.float64]:
    """Evaluate the explicit growth, dilution, and resource partition.

    Returns:
        Flat explicit derivative with the same shape as ``state``.

    Raises:
        ValueError: If the flat state shape does not match the trait grid.
    """
    values = np.asarray(state, dtype=np.float64)
    n_cells = trait.size
    if values.shape != (n_cells + 1,):
        msg = f"state must have shape {(n_cells + 1,)}; got {values.shape}"
        raise ValueError(msg)
    density = values[:n_cells]
    resource = float(values[-1])
    growth = trait_growth_rate(trait, resource, parameters)
    density_growth = density * growth
    density_tendency = density_growth - parameters.dilution * density
    consumption = float(
        np.dot(weights, density_growth * uptake_weight(trait, parameters))
    )
    resource_tendency = (
        parameters.dilution * (parameters.resource_in - resource) - consumption
    )
    return np.concatenate((density_tendency, np.asarray([resource_tendency])))


def diffusion_operator(
    trait: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> NDArray[np.float64]:
    """Embed no-flux trait diffusion in the population/resource state.

    Returns:
        Dense operator with a zero resource row and column.
    """
    n_cells = trait.size
    trait_operator = np.asarray(
        build_diffusion_matrix(
            n_cells,
            None,
            parameters.diffusivity,
            grid=trait,
            bc="neumann",
        ),
        dtype=np.float64,
    )
    operator = np.zeros((n_cells + 1, n_cells + 1), dtype=np.float64)
    operator[:n_cells, :n_cells] = trait_operator
    return operator


def make_explicit_rhs(
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
    """Build the tensor-shaped explicit RHS consumed by ``CoreSolver``.

    Returns:
        Callable preserving the solver's ``(state, subgroup)`` shape.
    """

    def rhs(
        _time: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        flat = np.asarray(state, dtype=np.float64).reshape(-1)
        return reaction_tendency(flat, trait, weights, parameters)[:, np.newaxis]

    return rhs


def make_full_rhs(
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
    """Build the full semi-discrete RHS used by the RK45 reference.

    Returns:
        Flat SciPy-compatible RHS callable.
    """
    operator = diffusion_operator(trait, parameters)

    def rhs(
        _time: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return reaction_tendency(state, trait, weights, parameters) + operator @ state

    return rhs


def run_imex(
    times: NDArray[np.float64],
    state0: NDArray[np.float64],
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameters: ChemostatParameters,
) -> NDArray[np.float64]:
    """Integrate the split system with fixed-step IMEX Heun--trapezoidal.

    Returns:
        State trajectory with shape ``(times, trait cells + resource)``.

    Raises:
        RuntimeError: If full history storage is unexpectedly unavailable.
    """
    options = ModelCoreOptions(
        axis_names=("state", "subgroup"),
        store_history=True,
        dtype=np.float64,
    )
    core = ModelCore(
        n_states=state0.size,
        n_subgroups=1,
        time_grid=times,
        options=options,
    )
    core.set_initial_state(state0[:, np.newaxis])
    base_builder = make_constant_base_builder(diffusion_operator(trait, parameters))
    factory = make_stage_operator_factory(base_builder, scheme="trapezoidal")
    config = RunConfig(
        method="imex-heun-tr",
        adaptive=False,
        strict=True,
        operators=OperatorSpecs(default=factory),
    )
    CoreSolver(core, operator_axis="state").run(
        make_explicit_rhs(trait, weights, parameters),
        config=config,
    )
    if core.state_array is None:
        raise RuntimeError(_STATE_HISTORY_ERROR)
    return np.asarray(core.state_array[:, :, 0], dtype=np.float64)


def run_rk45_reference(  # noqa: PLR0913
    times: NDArray[np.float64],
    state0: NDArray[np.float64],
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
    parameters: ChemostatParameters,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> NDArray[np.float64]:
    """Integrate the same semi-discrete model with SciPy RK45.

    Returns:
        Reference state trajectory at ``times``.

    Raises:
        RuntimeError: If SciPy reports an integration failure.
    """
    solution = solve_ivp(
        make_full_rhs(trait, weights, parameters),
        (float(times[0]), float(times[-1])),
        state0,
        method="RK45",
        t_eval=times,
        rtol=rtol,
        atol=atol,
        max_step=float(np.max(np.diff(times))),
    )
    if not solution.success:
        raise RuntimeError(_REFERENCE_SOLVE_ERROR.format(message=solution.message))
    return np.asarray(solution.y.T, dtype=np.float64)


def compute_observables(
    states: NDArray[np.float64],
    trait: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> TraitObservables:
    """Compute biomass, mean, variance, skewness, and kurtosis.

    Kurtosis is the standardized fourth central moment (not excess kurtosis).

    Returns:
        Vector-valued observables for every state row.

    Raises:
        ValueError: If state shape or total biomass is invalid.
    """
    trajectory = np.asarray(states, dtype=np.float64)
    expected_columns = trait.size + 1
    if trajectory.ndim != 2 or trajectory.shape[1] != expected_columns:
        msg = (
            f"states must have shape (time, {expected_columns}); got {trajectory.shape}"
        )
        raise ValueError(msg)
    density_mass = trajectory[:, : trait.size] * weights[np.newaxis, :]
    biomass = np.sum(density_mass, axis=1)
    if not np.all(np.isfinite(biomass)) or np.any(biomass <= 0.0):
        msg = "trait observables require finite positive total biomass"
        raise ValueError(msg)
    mean = np.sum(density_mass * trait[np.newaxis, :], axis=1) / biomass
    centered = trait[np.newaxis, :] - mean[:, np.newaxis]
    variance = np.sum(density_mass * centered**2, axis=1) / biomass
    scale = np.sqrt(np.maximum(variance, np.finfo(np.float64).tiny))
    skewness = np.sum(density_mass * centered**3, axis=1) / biomass / scale**3
    kurtosis = np.sum(density_mass * centered**4, axis=1) / biomass / scale**4
    return TraitObservables(
        biomass=np.asarray(biomass),
        mean=np.asarray(mean),
        variance=np.asarray(variance),
        skewness=np.asarray(skewness),
        kurtosis=np.asarray(kurtosis),
        resource=np.asarray(trajectory[:, -1]),
    )


def run_chemostat(
    *,
    n_cells: int = 41,
    times: NDArray[np.float64] | None = None,
    parameters: ChemostatParameters | None = None,
) -> ChemostatResult:
    """Run the canonical IMEX/RK45 chemostat comparison.

    Args:
        n_cells: Number of non-uniform finite-volume trait centers.
        times: Output and fixed-step grid. Defaults to 0--6 days in 0.02-day
            steps.
        parameters: Optional model parameters.

    Returns:
        Both trajectories and their structured observables.

    Raises:
        ValueError: If the supplied time grid is invalid.
    """
    params = parameters or ChemostatParameters()
    output_times = (
        np.linspace(0.0, 6.0, 301, dtype=np.float64)
        if times is None
        else np.asarray(times, dtype=np.float64)
    )
    if (
        output_times.ndim != 1
        or output_times.size < 2
        or not np.isfinite(output_times).all()
        or not np.all(np.diff(output_times) > 0.0)
    ):
        msg = "times must be a finite, strictly increasing one-dimensional grid"
        raise ValueError(msg)
    trait = build_trait_grid(n_cells, parameters=params)
    weights = cell_volume_weights(trait)
    state0 = initial_state(trait, weights, params)
    imex_states = run_imex(output_times, state0, trait, weights, params)
    rk45_states = run_rk45_reference(
        output_times,
        state0,
        trait,
        weights,
        params,
    )
    return ChemostatResult(
        trait_grid=trait,
        cell_weights=weights,
        times=output_times,
        imex_states=imex_states,
        rk45_states=rk45_states,
        imex_observables=compute_observables(imex_states, trait, weights),
        rk45_observables=compute_observables(rk45_states, trait, weights),
    )


def save_summary_figure(
    result: ChemostatResult,
    output_path: Path,
) -> None:
    """Save density, resource/biomass, and moment comparison panels."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)

    axes[0].plot(
        result.trait_grid,
        result.imex_states[0, :-1],
        color="0.55",
        label="initial",
    )
    axes[0].plot(
        result.trait_grid,
        result.imex_states[-1, :-1],
        label="IMEX final",
    )
    axes[0].plot(
        result.trait_grid,
        result.rk45_states[-1, :-1],
        linestyle="--",
        label="RK45 final",
    )
    axes[0].set(xlabel="trait θ", ylabel="population density n", title="Trait density")
    axes[0].legend()

    axes[1].plot(
        result.times,
        result.imex_observables.biomass,
        label="biomass (IMEX)",
    )
    axes[1].plot(
        result.times,
        result.rk45_observables.biomass,
        linestyle="--",
        label="biomass (RK45)",
    )
    axes[1].plot(
        result.times,
        result.imex_observables.resource,
        label="resource (IMEX)",
    )
    axes[1].plot(
        result.times,
        result.rk45_observables.resource,
        linestyle="--",
        label="resource (RK45)",
    )
    axes[1].set(xlabel="time (days)", ylabel="amount", title="Chemostat totals")
    axes[1].legend(fontsize="small")

    axes[2].plot(
        result.times,
        result.imex_observables.mean,
        label="mean trait (IMEX)",
    )
    axes[2].plot(
        result.times,
        result.rk45_observables.mean,
        linestyle="--",
        label="mean trait (RK45)",
    )
    axes[2].fill_between(
        result.times,
        result.imex_observables.mean - np.sqrt(result.imex_observables.variance),
        result.imex_observables.mean + np.sqrt(result.imex_observables.variance),
        alpha=0.2,
        label="IMEX ± SD",
    )
    axes[2].set(xlabel="time (days)", ylabel="trait", title="Trait moments")
    axes[2].legend(fontsize="small")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run the canonical comparison and write its summary figure."""
    result = run_chemostat()
    output = Path("examples/output/chemostat_pde/chemostat_summary.png")
    save_summary_figure(result, output)
    state_error = float(np.max(np.abs(result.imex_states - result.rk45_states)))
    biomass_error = float(
        np.max(
            np.abs(result.imex_observables.biomass - result.rk45_observables.biomass)
        )
    )
    print(f"wrote {output}")  # noqa: T201
    print(f"maximum state error: {state_error:.3e}")  # noqa: T201
    print(f"maximum biomass error: {biomass_error:.3e}")  # noqa: T201


if __name__ == "__main__":
    main()
