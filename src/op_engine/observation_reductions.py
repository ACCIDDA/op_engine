"""Single-pass native endpoint and sampled-time observation summaries."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np

from ._array import array_namespace
from .observations import ObservationDiagnostics, observe_solution

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ._typing import Array
    from .observations import ObservationResult, ObservationSpec


class ObservationSummaryValue(NamedTuple):
    """Native summary arrays with the original scientific output specification.

    Attributes:
        spec: Instantaneous output shape, coordinates, units and interpretation.
        initial: Output at the first supplied sample.
        final: Output at the last supplied sample.
        change: Final minus initial; appropriate for a cumulative counter.
        integral: Composite-trapezoid time integral over the supplied samples.
            Its units are output units multiplied by the caller's time unit.
    """

    spec: ObservationSpec
    initial: Array
    final: Array
    change: Array
    integral: Array


class ObservationSummary(NamedTuple):
    """A sampled window and bounded-size named reductions, without a trajectory.

    The window is exactly the first/last supplied time. There is no off-grid
    clipping, extrapolation or qualified method-specific dense output. The
    diagnostics report callback count without an independent error estimate.
    """

    window: tuple[float, float]
    sample_count: int
    values: Mapping[str, ObservationSummaryValue]
    diagnostics: ObservationDiagnostics
    quadrature: Literal["trapezoid"] = "trapezoid"


class _StateLayout(NamedTuple):
    shape: tuple[int, ...]
    dtype: object
    namespace: object


def _static_time(time: float) -> float:
    """Validate one static real finite time.

    Returns:
        The time as a host float.

    Raises:
        ValueError: If the sample time is not a finite real scalar.
    """
    value = np.asarray(time)
    if value.ndim != 0 or value.dtype.kind not in "fiu" or not np.isfinite(value):
        msg = "summary sample times must be finite real static scalars"
        raise ValueError(msg)
    return float(value)


def _sample(
    time: float,
    state: Array,
    observations: Sequence[ObservationSpec],
    params: Mapping[str, object] | None,
) -> ObservationResult:
    """Evaluate and snapshot one sample through the existing observation contract.

    Returns:
        One-time native series, whose stacked arrays snapshot reused producer buffers.
    """
    xp = array_namespace(state)
    return observe_solution(
        [time],
        cast("Array", xp.expand_dims(state, axis=0)),
        observations,
        params=params,
    )


def _initial_values(result: ObservationResult) -> dict[str, ObservationSummaryValue]:
    """Initialize a zero-duration summary without retaining its state.

    Returns:
        Named native endpoint snapshots and zero changes/integrals.
    """
    values = {}
    for name, series in result.series.items():
        xp = array_namespace(series.values)
        value = cast("Array", xp.squeeze(series.values, axis=0))
        zero = cast("Array", xp.zeros_like(value))
        values[name] = ObservationSummaryValue(series.spec, value, value, zero, zero)
    return values


def _advance_values(
    previous: Mapping[str, ObservationSummaryValue],
    current: ObservationResult,
    duration: float,
) -> dict[str, ObservationSummaryValue]:
    """Accumulate adjacent samples without stacking their history.

    Returns:
        Replaced endpoint snapshots, changes and running trapezoid integrals.
    """
    values = {}
    for name, series in current.series.items():
        xp = array_namespace(series.values)
        value = xp.squeeze(series.values, axis=0)
        old = previous[name]
        increment = xp.multiply(xp.add(old.final, value), duration / 2)
        values[name] = ObservationSummaryValue(
            series.spec,
            old.initial,
            cast("Array", value),
            cast("Array", xp.subtract(value, old.initial)),
            cast("Array", xp.add(old.integral, increment)),
        )
    return values


def reduce_observations(
    samples: Iterable[tuple[float, Array]],
    observations: Sequence[ObservationSpec],
    *,
    params: Mapping[str, object] | None = None,
) -> ObservationSummary:
    """Reduce a finite native sample stream in one pass without storing a trajectory.

    Samples define the exact window endpoints. Native initial/final values,
    changes and composite-trapezoid integrals preserve evaluator shapes and
    gradients. A single sample has zero duration, change and integral. Callback
    outputs are snapshotted so a producer may reuse its numerical state buffer.

    The reducer retains a constant number of output arrays and one current
    state reference. Producer storage and reverse-mode computation graphs are
    separate: no constant-memory gradient guarantee is implied. There is no
    persistent checkpoint or partial-result recovery; an interrupted call must
    restart from a producer-owned replay/checkpoint. Callbacks must be pure.

    Args:
        samples: Finite, nonempty iterable of static times and native states.
            Times must increase strictly; state shape, namespace and real floating
            dtype stay fixed. Only the supplied times define quadrature samples.
        observations: Unique named evaluators and scientific output metadata.
        params: Observation parameters passed to every callback.

    Returns:
        Bounded-size native reductions with window, sample count and diagnostics.
        Integrals inherit trajectory and sampled-quadrature error; no dense-output
        order, event reconstruction or separate error estimate is asserted.

    Raises:
        ValueError: If the stream is empty, intervals are invalid or shapes change.
        TypeError: If states or outputs violate the native namespace/dtype contract.
    """
    selection = tuple(observations)
    iterator = iter(samples)
    try:
        time, state = next(iterator)
    except StopIteration as error:
        msg = "observation summary requires at least one sample"
        raise ValueError(msg) from error
    first = _sample(_static_time(time), state, selection, params)
    layout = _StateLayout(state.shape, state.dtype, array_namespace(state))
    values = _initial_values(first)
    last_time = first.times[0]
    count = 1
    for sample_time, state in iterator:
        time = _static_time(sample_time)
        duration = time - last_time
        if not np.isfinite(duration) or duration <= 0:
            msg = "summary sample times must increase with finite intervals"
            raise ValueError(msg)
        if state.shape != layout.shape:
            msg = "summary samples must preserve state shape"
            raise ValueError(msg)
        if (
            state.dtype != layout.dtype
            or array_namespace(state) is not layout.namespace
        ):
            msg = "summary samples must preserve state namespace and dtype"
            raise TypeError(msg)
        current = _sample(time, state, selection, params)
        values = _advance_values(values, current, duration)
        last_time = time
        count += 1
    return ObservationSummary(
        (first.times[0], last_time),
        count,
        MappingProxyType(values),
        ObservationDiagnostics(count * len(selection)),
    )
