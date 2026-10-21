"""Bounded forward sensitivities of smooth deterministic fixed-mesh ODEs.

The augmented equations use the existing explicit integrator. They approximate
continuous sensitivities, including caller-supplied initial-condition effects.
No compiler, statistical noise model or Fisher definition is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from ._array import array_namespace
from .core_solver import CoreSolver, RunConfig
from .model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from ._typing import Array, Scalar
    from .observations import ObservationEvaluator

__all__ = [
    "DerivativeArtifact",
    "DerivativeMatrices",
    "SensitivityInitial",
    "SensitivityOptions",
    "SensitivitySolution",
    "SensitivitySystem",
    "solve_sensitivities",
]


class _Identity(Protocol):
    @property
    def digest(self) -> str: ...


class DerivativeMatrices(Protocol):
    """Four native matrices in declared physical state/parameter coordinates."""

    @property
    def state(self) -> object:
        """Return df/dy with shape (N,N)."""
        ...

    @property
    def parameters(self) -> object:
        """Return df/dtheta with shape (N,Q)."""
        ...

    @property
    def observable_state(self) -> object:
        """Return dg/dy with shape (M,N)."""
        ...

    @property
    def observable_parameters(self) -> object:
        """Return dg/dtheta with shape (M,Q)."""
        ...


class DerivativeArtifact(Protocol):
    """Structural compiler boundary, with no runtime compiler dependency.

    Matrices have shapes (N,N), (N,Q), (M,N), (M,Q). The producer declares
    supported operations and must represent the same smooth RHS/measurements.
    """

    @property
    def model(self) -> _Identity:
        """Identify the derivative producer's source model."""
        ...

    @property
    def state_names(self) -> tuple[str, ...]:
        """Give the ordered expanded state coordinates."""
        ...

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Give RHS and measurement parameter coordinates."""
        ...

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Give the ordered scalar measurement coordinates."""
        ...

    def evaluate(
        self, t: object, y: object, **parameters: object
    ) -> DerivativeMatrices:
        """Evaluate native derivatives without changing coordinates or graphs."""
        ...


@dataclass(frozen=True, slots=True)
class SensitivitySystem:
    """Bind one RHS to derivatives and explicitly named initial-only parameters.

    ``rhs`` accepts every derivative-artifact parameter, including measurement-
    only parameters. Initial-only names are appended to that ordered layout and
    excluded from RHS/derivative calls; their effects enter through the required
    initial derivative matrix. The caller is responsible for binding the same
    RHS as the derivative producer and for declaring a smooth deterministic ODE.
    """

    rhs: ObservationEvaluator
    derivatives: DerivativeArtifact
    initial_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous coordinate layouts before numerical work.

        Raises:
            ValueError: Names are empty, duplicate or overlap.
            TypeError: The RHS is not callable.
        """
        if not callable(self.rhs):
            msg = "Sensitivity RHS must be callable"
            raise TypeError(msg)
        if not isinstance(self.initial_parameters, tuple):
            msg = "Initial-only parameter names must be a tuple"
            raise TypeError(msg)
        groups = (
            self.derivatives.state_names,
            self.parameter_names,
            self.derivatives.observable_names,
        )
        if (
            not self.derivatives.state_names
            or not self.parameter_names
            or any(
                not isinstance(names, tuple)
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)
                for names in groups
            )
        ):
            msg = "Sensitivity layouts require nonempty unique state/parameter names"
            raise ValueError(msg)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return derivative parameters followed by initial-only parameters."""
        return self.derivatives.parameter_names + self.initial_parameters


@dataclass(frozen=True, slots=True)
class SensitivityInitial:
    """Native state (N,) and its required parameter derivative (N,P).

    Pass an explicit zero matrix for parameter-independent initial states.
    This avoids silently dropping initial-condition derivatives.
    """

    state: Array
    parameter_derivatives: Array


@dataclass(frozen=True, slots=True)
class SensitivityOptions:
    """Bound fixed explicit integration and retained numerical output.

    Only the final output is retained by default. History adds requested output
    times, not internal stages. Byte limits cover returned numeric arrays, not
    temporary work, AD graphs, compiler memory or the whole process.
    """

    solver: RunConfig = field(default_factory=lambda: RunConfig(method="rk4"))
    store_history: bool = False
    max_augmented_states: int = 16_384
    max_output_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject unsupported numerical paths instead of silently downshifting.

        Raises:
            ValueError: Solver mode or storage bounds are unsupported.
            TypeError: History is not boolean.
        """
        cfg = self.solver
        if not isinstance(cfg, RunConfig):
            msg = "Sensitivity solver options require a RunConfig"
            raise TypeError(msg)
        if (
            cfg.adaptive
            or not cfg.strict
            or cfg.method not in {"euler", "heun", "rk4", "dopri5"}
            or cfg.forcing_breakpoints
            or any(
                value is not None
                for value in (
                    cfg.state_checks,
                    cfg.nonlinear,
                    cfg.jacobian,
                    cfg.operators.default,
                    cfg.operators.tr,
                    cfg.operators.bdf2,
                )
            )
        ):
            msg = "Sensitivities require a smooth fixed explicit ODE without operators"
            raise ValueError(msg)
        if type(self.store_history) is not bool:
            msg = "Sensitivity store_history must be boolean"
            raise TypeError(msg)
        if any(
            type(bound) is not int or bound < 1
            for bound in (self.max_augmented_states, self.max_output_bytes)
        ):
            msg = "Sensitivity storage bounds must be positive integers"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SensitivitySolution:
    """Native output states and parameter derivatives with static provenance.

    Array shapes are (T,N), (T,N,P) and (T,M,P). Measurement sensitivities are
    ``dg/dy @ S + dg/dtheta`` in the same parameter order. They do not include
    observation noise, prior curvature or nuisance elimination. ``times``
    contains only retained outputs; ``mesh`` is the requested solver mesh.
    """

    source_model: str
    state_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    times: tuple[float, ...]
    mesh: tuple[float, ...]
    states: Array
    parameter_derivatives: Array
    observable_derivatives: Array
    augmented_dimension: int
    options: SensitivityOptions


def _checked(reference: Array, value: object, shape: tuple[int, ...]) -> Array:
    if (
        getattr(value, "shape", None) != shape
        or getattr(value, "dtype", None) != reference.dtype
        or array_namespace(value) is not array_namespace(reference)
    ):
        msg = "Sensitivity array shape, namespace or dtype differs from its layout"
        raise ValueError(msg)
    return cast("Array", value)


def _matrices(
    system: SensitivitySystem, t: object, y: Array, parameters: Mapping[str, object]
) -> tuple[Array, Array, Array, Array]:
    artifact = system.derivatives
    matrices = artifact.evaluate(t, y, **parameters)
    n, q, m = (
        len(artifact.state_names),
        len(artifact.parameter_names),
        len(artifact.observable_names),
    )
    values = (
        _checked(y, matrices.state, (n, n)),
        _checked(y, matrices.parameters, (n, q)),
        _checked(y, matrices.observable_state, (m, n)),
        _checked(y, matrices.observable_parameters, (m, q)),
    )
    if not system.initial_parameters:
        return values
    xp = array_namespace(y)
    extra = len(system.initial_parameters)
    zero = xp.zeros_like(xp.sum(y, keepdims=True))
    return (
        values[0],
        cast(
            "Array",
            xp.concat(
                (values[1], xp.broadcast_to(zero, (n, extra))),
                axis=1,
            ),
        ),
        values[2],
        cast(
            "Array",
            xp.concat(
                (values[3], xp.broadcast_to(zero, (m, extra))),
                axis=1,
            ),
        ),
    )


def _augmented_rhs(
    system: SensitivitySystem, parameters: Mapping[str, object]
) -> Callable[[Scalar, Array], Array]:
    n, p = len(system.derivatives.state_names), len(system.parameter_names)

    def evaluate(t: Scalar, augmented: Array) -> Array:
        xp = array_namespace(augmented)
        flat = xp.reshape(augmented, (-1,))
        y = cast("Array", flat[:n])
        sensitivities = xp.reshape(flat[n:], (n, p))
        values = _matrices(system, t, y, parameters)
        rhs = _checked(y, system.rhs(t, y, **parameters), (n,))
        derivative = values[0] @ sensitivities + values[1]
        return cast(
            "Array",
            xp.reshape(xp.concat((rhs, xp.reshape(derivative, (-1,)))), (-1, 1)),
        )

    return evaluate


def _validate_inputs(
    system: SensitivitySystem,
    parameters: Mapping[str, object],
    initial: SensitivityInitial,
) -> dict[str, object]:
    if set(parameters) != set(system.parameter_names):
        msg = "Sensitivity parameter bindings must match the complete ordered layout"
        raise ValueError(msg)
    y = initial.state
    xp = array_namespace(y)
    if not xp.isdtype(y.dtype, "real floating"):
        msg = "Sensitivity state requires a native real floating array"
        raise TypeError(msg)
    _checked(y, y, (len(system.derivatives.state_names),))
    _checked(
        y,
        initial.parameter_derivatives,
        (len(system.derivatives.state_names), len(system.parameter_names)),
    )
    for value in parameters.values():
        if hasattr(value, "shape"):
            _checked(y, value, ())
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            msg = "Sensitivity parameters must be real scalar bindings"
            raise TypeError(msg)
    return {name: parameters[name] for name in system.derivatives.parameter_names}


def solve_sensitivities(
    system: SensitivitySystem,
    parameters: Mapping[str, object],
    initial: SensitivityInitial,
    time_grid: NDArray[np.floating],
    *,
    options: SensitivityOptions | None = None,
) -> SensitivitySolution:
    """Integrate ``S' = (df/dy) S + df/dtheta`` with existing explicit kernels.

    The compiler is optional: any correctly ordered derivative artifact can
    satisfy the protocol. Callbacks/mesh are static under JIT; native scalar
    bindings and both initial arrays retain their graphs. Batch independent
    systems through native vmap. Forcing jumps, events, history dependence,
    stochastic paths and operator solvers need separately qualified methods.

    Returns:
        Native retained outputs and measured sensitivities in physical coordinates.

    Raises:
        ValueError: Layout, mesh or declared storage limits disagree.
    """
    opts = options or SensitivityOptions()
    bindings = _validate_inputs(system, parameters, initial)
    xp = array_namespace(initial.state)
    n, p, m = (
        len(system.derivatives.state_names),
        len(system.parameter_names),
        len(system.derivatives.observable_names),
    )
    grid = np.asarray(time_grid, dtype=np.float64)
    if (
        grid.ndim != 1
        or not grid.size
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        msg = "Sensitivity mesh must be finite and strictly increasing"
        raise ValueError(msg)
    count = grid.size if opts.store_history else 1
    dimension = n * (1 + p)
    if (
        dimension > opts.max_augmented_states
        or count * (n + n * p + m * p) * (xp.finfo(initial.state.dtype).bits // 8)
        > opts.max_output_bytes
    ):
        msg = "Sensitivity solve exceeds its augmented/output storage limit"
        raise ValueError(msg)
    core = ModelCore(
        dimension,
        1,
        grid,
        options=ModelCoreOptions(
            dtype=np.dtype(str(initial.state.dtype).removeprefix("torch.")),
            store_history=opts.store_history,
        ),
    )
    core.set_initial_state(
        xp.reshape(
            xp.concat((
                initial.state,
                xp.reshape(initial.parameter_derivatives, (-1,)),
            )),
            (-1, 1),
        )
    )
    CoreSolver(core).run(_augmented_rhs(system, bindings), config=opts.solver)
    return _solution(system, bindings, core, opts)


def _solution(
    system: SensitivitySystem,
    bindings: Mapping[str, object],
    core: ModelCore,
    opts: SensitivityOptions,
) -> SensitivitySolution:
    xp = array_namespace(core.get_current_state())
    n, p = len(system.derivatives.state_names), len(system.parameter_names)
    dimension = n * (p + 1)
    count = core.n_timesteps if opts.store_history else 1
    history = core.state_array if opts.store_history else core.get_current_state()
    augmented = xp.reshape(history, (count, dimension))
    states = augmented[:, :n]
    derivatives = xp.reshape(augmented[:, n:], (count, n, p))
    times = (
        tuple(float(t) for t in core.time_grid)
        if opts.store_history
        else (float(core.time_grid[-1]),)
    )
    measured = []
    for index, t in enumerate(times):
        matrices = _matrices(system, t, states[index], bindings)
        measured.append(matrices[2] @ derivatives[index] + matrices[3])
    return SensitivitySolution(
        system.derivatives.model.digest,
        system.derivatives.state_names,
        system.parameter_names,
        system.derivatives.observable_names,
        times,
        tuple(float(t) for t in core.time_grid),
        cast("Array", states),
        cast("Array", derivatives),
        cast("Array", xp.stack(measured)),
        dimension,
        opts,
    )
