"""Generic hierarchical nonlinear two-pool model with independent references.

Each group carries one or more local parameter/initial-condition cells. Shared
positive gain/recovery rates combine with group-local log rate offsets. The
coupled variant uses the group mean of the first pool in its forward flux.
Each cell conserves its two-pool total. This synthetic workload is numerical
inference evidence, not a calibrated model of a particular scientific system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.integrate import solve_ivp

from op_engine import Array, CoreSolver, ModelCore, array_namespace
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class HierarchicalProblem:
    """Static layout and regime for a reproducible hierarchical ODE workload."""

    groups: int = 2
    local_dimension: int = 1
    coupled: bool = False
    stiff: bool = False

    def __post_init__(self) -> None:
        """Validate positive fixture dimensions.

        Raises:
            ValueError: If a dimension is not a positive integer.
        """
        for name in ("groups", "local_dimension"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                msg = f"{name} must be a positive integer"
                raise ValueError(msg)

    @property
    def cells(self) -> int:
        """Number of local rate/initial-condition pairs."""
        return self.groups * self.local_dimension

    @property
    def parameter_count(self) -> int:
        """Two shared log rates plus local rate offsets and initial values."""
        return 2 + 2 * self.cells

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Pool/group/local-cell axis order."""
        return (2, self.groups, self.local_dimension)

    @property
    def model_id(self) -> str:
        """Structural identity excluding fitted parameter values."""
        return (
            f"hierarchical-two-pool:v1:groups={self.groups}:"
            f"local={self.local_dimension}:coupled={self.coupled}"
        )


def nominal_parameters(problem: HierarchicalProblem) -> np.ndarray:
    """Return deterministic concrete shared/local reference parameters.

    Returns:
        Parameter vector in shared gain/recovery, local rates, local IC order.
    """
    return np.concatenate((
        [math.log(0.4), math.log(50.0 if problem.stiff else 0.15)],
        np.linspace(-0.2, 0.25, problem.cells),
        np.linspace(-0.3, 0.4, problem.cells),
    ))


def initial_state(problem: HierarchicalProblem, parameters: Array) -> Array:
    """Construct differentiable group-local initial values with unit totals.

    Returns:
        Native state tensor with one positive two-pool pair per local cell.
    """
    xp = array_namespace(parameters)
    first = 0.55 + 0.1 * xp.tanh(
        xp.reshape(
            parameters[2 + problem.cells :], (problem.groups, problem.local_dimension)
        )
    )
    return cast("Array", xp.stack((first, 1 - first)))


def rhs(
    problem: HierarchicalProblem, parameters: Array
) -> Callable[[object, Array], Array]:
    """Bind native nonlinear equations without host parameter conversion.

    Returns:
        Array-API RHS with shared/global and group-local dynamic inputs.
    """
    xp = array_namespace(parameters)
    gain, recovery = xp.exp(parameters[:2])
    local = xp.exp(
        xp.reshape(
            parameters[2 : 2 + problem.cells], (problem.groups, problem.local_dimension)
        )
    )

    def evaluate(_time: object, state: Array) -> Array:
        first, second = state
        driver = first + (0.15 * xp.mean(first, axis=0) if problem.coupled else 0.0)
        forward = gain * local * first * driver / (1 + first)
        net = forward - recovery * second
        return cast("Array", xp.stack((-net, net)))

    return evaluate


def reference_initial(
    problem: HierarchicalProblem, parameters: np.ndarray
) -> np.ndarray:
    """Evaluate initial cells independently using scalar standard-library math.

    Returns:
        Concrete reference state.
    """
    result = np.empty(problem.state_shape)
    for group in range(problem.groups):
        for cell in range(problem.local_dimension):
            index = group * problem.local_dimension + cell
            first = 0.55 + 0.1 * math.tanh(float(parameters[2 + problem.cells + index]))
            result[:, group, cell] = [first, 1 - first]
    return result


def reference_rhs(
    problem: HierarchicalProblem, parameters: np.ndarray, state: np.ndarray
) -> np.ndarray:
    """Independent explicit cell equations, with no generated/native RHS calls.

    Returns:
        Concrete two-pool derivatives.
    """
    result = np.empty_like(state)
    gain, recovery = (math.exp(float(value)) for value in parameters[:2])
    for group in range(problem.groups):
        for cell in range(problem.local_dimension):
            first, second = (float(value) for value in state[:, group, cell])
            mean_first = 0.0
            if problem.coupled:
                mean_first = (
                    sum(float(state[0, g, cell]) for g in range(problem.groups))
                    / problem.groups
                )
            local = math.exp(
                float(parameters[2 + group * problem.local_dimension + cell])
            )
            forward = gain * local * first * (first + 0.15 * mean_first) / (1 + first)
            result[:, group, cell] = [
                recovery * second - forward,
                forward - recovery * second,
            ]
    return result


def reference_trajectory(
    problem: HierarchicalProblem, parameters: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Compute a tight-tolerance independent SciPy trajectory.

    Returns:
        Reference output states including the initial point.

    Raises:
        RuntimeError: If reference integration fails.
    """
    initial = reference_initial(problem, parameters)
    solution = solve_ivp(
        lambda _time, flat: reference_rhs(
            problem, parameters, flat.reshape(problem.state_shape)
        ).ravel(),
        (float(times[0]), float(times[-1])),
        initial.ravel(),
        t_eval=times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T.reshape((len(times), *problem.state_shape))


def record_schedule(
    problem: HierarchicalProblem, parameters: np.ndarray, times: np.ndarray
) -> tuple[AdaptiveStepSchedule, RunConfig]:
    """Discover a nominal accepted mesh at an eager boundary.

    Returns:
        Accepted schedule and its structural/reference-labeled configuration.

    Raises:
        RuntimeError: If discovery fails to produce a mesh.
    """
    core = ModelCore(
        2,
        problem.groups,
        times,
        options=ModelCoreOptions(
            other_axes=(problem.local_dimension,), axis_names=("pool", "group", "local")
        ),
    )
    core.set_initial_state(initial_state(problem, parameters))
    config = RunConfig(
        method="dopri5",
        adaptive=True,
        adaptive_cfg=AdaptiveConfig(rtol=2e-8, atol=2e-10, dt_init=0.02),
        replay_model_id=problem.model_id,
        replay_reference_regime="nominal_parameters()"
        + (":stiff" if problem.stiff else ":nonstiff"),
    )
    solver = CoreSolver(core)
    solver.run(rhs(problem, parameters), config=config)
    schedule = solver.last_adaptive_schedule
    if schedule is None:
        msg = "Adaptive discovery did not produce a schedule"
        raise RuntimeError(msg)
    return schedule, config

