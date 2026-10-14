"""Native named observations at stored output times, without dense interpolation."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast

import numpy as np

from ._array import array_namespace

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._typing import Array
    from .model_core import ModelCore


class ObservationEvaluator(Protocol):
    """Evaluate one native output from an instantaneous state and parameters."""

    def __call__(self, t: object, y: object, **params: object) -> object:
        """Return one native output with the declared shape and state dtype."""
        ...


class ObservationMeasure(NamedTuple):
    """Describe weights already applied by an evaluator, without applying them again."""

    kind: str
    weights: tuple[float, ...]
    units: str | None = None


class ObservationMetadata(NamedTuple):
    """Static scientific output metadata, separate from solver state layout.

    Attributes:
        output_shape: Shape after evaluator reductions, excluding time.
        coordinates: Ordered output axis names and coordinate labels.
        units: Declared output units; no dimensional verification is implied.
        state_dependencies: Direct state base dependencies.
        parameter_shapes: Direct parameters and complete input shapes.
        measure_requirements: Named measures already applied by the evaluator.
        time_dependent: Whether the expression explicitly references time.
        description: Optional scientific interpretation.
    """

    output_shape: tuple[int, ...] = ()
    coordinates: tuple[tuple[str, tuple[str, ...]], ...] = ()
    units: str | None = None
    state_dependencies: tuple[str, ...] = ()
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    measure_requirements: tuple[tuple[str, ObservationMeasure], ...] = ()
    time_dependent: bool = False
    description: str | None = None


class ObservationSpec(NamedTuple):
    """A named evaluator and its immutable output description.

    Custom evaluators receive each state in the trajectory's natural shape.
    Use :func:`observation_from_compiled` to flatten each state for an op_system
    compiled observable, preserving that compiler's state ordering.
    """

    name: str
    eval_fn: ObservationEvaluator
    metadata: ObservationMetadata = ObservationMetadata()


class ObservationSeries(NamedTuple):
    """Native ``(n_output_times, *output_shape)`` values with their specification."""

    spec: ObservationSpec
    values: Array


class ObservationDiagnostics(NamedTuple):
    """Describe evaluation coverage without inventing a numerical error estimate.

    Accuracy inherits both the supplied trajectory and evaluator. No method has
    a qualified continuous extension through this interface yet. Spatial
    evaluation is owned by the callback; the compiler adapter selects declared
    nodes and applies declared measures without interpolation.
    """

    evaluation_count: int
    temporal_interpolation: Literal["none"] = "none"
    dense_output_order: None = None
    error_estimate: None = None


class ObservationResult(NamedTuple):
    """Static times, read-only named series mapping, and evaluation diagnostics."""

    times: tuple[float, ...]
    series: Mapping[str, ObservationSeries]
    diagnostics: ObservationDiagnostics


class _DefinitionLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def units(self) -> str | None: ...

    @property
    def description(self) -> str | None: ...


class _MeasureLike(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def weights(self) -> tuple[float, ...]: ...

    @property
    def units(self) -> str | None: ...


class CompiledObservableLike(Protocol):
    """Read-only structural compiler surface; core imports no compiler or provider."""

    @property
    def definition(self) -> _DefinitionLike:
        """Scientific name, units and description."""
        ...

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Output dimensions excluding time."""
        ...

    @property
    def coordinates(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Output axes and their coordinate labels."""
        ...

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """Direct state base dependencies."""
        ...

    @property
    def parameter_shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Direct parameters and complete native buffer shapes."""
        ...

    @property
    def measure_requirements(self) -> tuple[tuple[str, _MeasureLike], ...]:
        """Physical measures already applied by the evaluator."""
        ...

    @property
    def time_dependent(self) -> bool:
        """Whether the expression explicitly references time."""
        ...

    @property
    def eval_fn(self) -> ObservationEvaluator:
        """Native flat-state evaluator."""
        ...


class _Indexable(Protocol):
    def __getitem__(self, index: int) -> Array: ...


def observation_from_compiled(observable: CompiledObservableLike) -> ObservationSpec:
    """Adapt a compiled observable structurally, without a runtime compiler dependency.

    The input trajectory must retain the compiler's flattened state order. The
    adapter reshapes each state to a vector and delegates all spatial reductions
    and coordinate selection to the compiler, applying no additional weights.

    Args:
        observable: Artifact exposing op_system's compiled-observable surface.

    Returns:
        Native evaluator and detached scientific metadata.
    """
    evaluate = observable.eval_fn

    def eval_fn(t: object, y: object, **params: object) -> object:
        xp = array_namespace(y)
        return evaluate(t, xp.reshape(y, (-1,)), **params)

    return ObservationSpec(
        observable.definition.name,
        eval_fn,
        ObservationMetadata(
            tuple(observable.output_shape),
            tuple((axis, tuple(labels)) for axis, labels in observable.coordinates),
            observable.definition.units,
            tuple(observable.state_dependencies),
            tuple((name, tuple(shape)) for name, shape in observable.parameter_shapes),
            tuple(
                (
                    axis,
                    ObservationMeasure(
                        measure.kind, tuple(measure.weights), measure.units
                    ),
                )
                for axis, measure in observable.measure_requirements
            ),
            observable.time_dependent,
            observable.definition.description,
        ),
    )


def _validate_specs(observations: Sequence[ObservationSpec]) -> None:
    names: set[str] = set()
    for spec in observations:
        if (
            not isinstance(spec.name, str)
            or not spec.name.strip()
            or spec.name in names
        ):
            msg = "observation names must be nonempty and unique"
            raise ValueError(msg)
        names.add(spec.name)
        if not callable(spec.eval_fn):
            msg = "observation evaluator must be callable"
            raise TypeError(msg)
        metadata = spec.metadata
        if not isinstance(metadata.output_shape, tuple) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in metadata.output_shape
        ):
            msg = "observation output_shape must be a tuple of nonnegative integers"
            raise ValueError(msg)
        if metadata.coordinates:
            axes = tuple(axis for axis, _ in metadata.coordinates)
            if (
                len(axes) != len(metadata.output_shape)
                or len(set(axes)) != len(axes)
                or any(not isinstance(axis, str) or not axis for axis in axes)
                or tuple(len(labels) for _, labels in metadata.coordinates)
                != metadata.output_shape
            ):
                msg = (
                    "observation coordinates must match output_shape "
                    "in unique axis order"
                )
                raise ValueError(msg)


def observe_solution(
    times: Sequence[float] | np.ndarray,
    trajectory: Array,
    observations: Sequence[ObservationSpec],
    *,
    params: Mapping[str, object] | None = None,
) -> ObservationResult:
    """Evaluate native named outputs at every supplied stored output time.

    Time is a static, finite, strictly increasing host grid. Each callback must
    return an array in the trajectory namespace and dtype with its declared
    shape. Native values are stacked without conversion or gradient detachment.
    Inputs and returned arrays should be treated as immutable.

    Args:
        times: Stored output times, independent of internal solver step times.
        trajectory: Native state history with a leading output-time dimension.
        observations: Unique named evaluators with static output metadata.
        params: Parameters passed to each evaluator, including observation-only ones.

    Returns:
        Native series and evaluation coverage. No interpolation/error estimate.

    Raises:
        ValueError: If grids, trajectory shape, metadata or output shapes disagree.
        TypeError: If states/outputs violate the real floating native-array contract.
    """
    grid = np.asarray(times)
    if (
        grid.ndim != 1
        or grid.size == 0
        or grid.dtype.kind not in "fiu"
        or not np.all(np.isfinite(grid))
        or not np.all(grid[1:] > grid[:-1])
    ):
        msg = "observation times must be a nonempty finite strictly increasing 1D grid"
        raise ValueError(msg)
    if not trajectory.shape or trajectory.shape[0] != grid.size:
        msg = "trajectory must have one leading state row per observation time"
        raise ValueError(msg)
    xp = array_namespace(trajectory)
    if not xp.isdtype(trajectory.dtype, "real floating"):
        msg = "observation trajectory must have a real floating dtype"
        raise TypeError(msg)
    _validate_specs(observations)
    static_times = tuple(float(time) for time in grid)
    states = cast("_Indexable", trajectory)
    series = {}
    for spec in observations:
        values = []
        for index, time in enumerate(static_times):
            value = spec.eval_fn(
                time, states[index], **({} if params is None else params)
            )
            if (
                array_namespace(value) is not xp
                or getattr(value, "dtype", None) != trajectory.dtype
            ):
                msg = (
                    f"observation {spec.name!r} must preserve trajectory "
                    "namespace and dtype"
                )
                raise TypeError(msg)
            if getattr(value, "shape", None) != spec.metadata.output_shape:
                msg = (
                    f"observation {spec.name!r} must have shape "
                    f"{spec.metadata.output_shape}"
                )
                raise ValueError(msg)
            values.append(value)
        series[spec.name] = ObservationSeries(
            spec, cast("Array", xp.stack(values, axis=0))
        )
    return ObservationResult(
        static_times,
        MappingProxyType(series),
        ObservationDiagnostics(len(observations) * len(static_times)),
    )


def observe_model(
    core: ModelCore,
    observations: Sequence[ObservationSpec],
    *,
    params: Mapping[str, object] | None = None,
) -> ObservationResult:
    """Evaluate a completed ModelCore's stored history using its output grid.

    Args:
        core: Completed state manager configured to store its full trajectory.
        observations: Named evaluators and their output metadata.
        params: Optional observation parameters.

    Returns:
        Native named series at the model's stored output times.

    Raises:
        ValueError: If history is unavailable or the model has not completed.
    """
    if not core.store_history or core.state_array is None:
        msg = "observations require stored history; streaming is not available"
        raise ValueError(msg)
    if core.current_step != core.n_timesteps - 1:
        msg = "observations require a completed model trajectory"
        raise ValueError(msg)
    return observe_solution(
        core.time_grid, core.state_array, observations, params=params
    )
