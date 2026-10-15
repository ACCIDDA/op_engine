"""Static forcing boundaries shared by integration methods."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from ._array import array_namespace

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._typing import Array, Scalar

StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class _ForcingSchedule:
    """Validated, immutable times at which external forcing may change.

    The schedule is independent of observation times. Global schedules may
    include times outside a particular solve's interval.
    """

    breakpoints: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot and validate a one-dimensional sequence of real times.

        Raises:
            ValueError: If times are non-finite, unordered, or not real scalars.
        """
        message = "forcing_breakpoints must be a 1D sequence of finite real times"
        try:
            points = tuple(self.breakpoints)
        except TypeError as error:
            raise ValueError(message) from error
        if any(
            not isinstance(point, Real)
            or isinstance(point, bool)
            or not math.isfinite(point)
            for point in points
        ):
            raise ValueError(message)
        normalized = tuple(float(point) for point in points)
        if any(right <= left for left, right in pairwise(normalized)):
            msg = "forcing_breakpoints must be strictly increasing"
            raise ValueError(msg)
        object.__setattr__(self, "breakpoints", normalized)

    def next_after(self, time: float) -> float:
        """Find the next boundary, treating forcing as right-continuous.

        Returns:
            The first breakpoint strictly after ``time``, or infinity.
        """
        index = bisect_right(self.breakpoints, time)
        return self.breakpoints[index] if index < len(self.breakpoints) else math.inf

    def segments(self, start: float, end: float) -> tuple[float, ...]:
        """Return exact segment endpoints, including the interval endpoints.

        Returns:
            Increasing integration segment coordinates.
        """
        return (start, *(p for p in self.breakpoints if start < p < end), end)

    def next_after_array(self, time: Scalar, reference: Array) -> Array:
        """Find the next boundary without extracting a traced time scalar.

        Returns:
            Native scalar with the first later boundary, or infinity.
        """
        xp = array_namespace(reference)
        points = xp.asarray((*self.breakpoints, math.inf), dtype=reference.dtype)
        index = xp.searchsorted(
            points, xp.asarray(time, dtype=reference.dtype), side="right"
        )
        return cast("Array", cast("Any", points)[index])

    def rhs_for_step(
        self,
        rhs: Callable[[Scalar, StateT], StateT],
        time: Scalar,
        reference: Array,
    ) -> Callable[[Scalar, StateT], StateT]:
        """Keep a step's RHS evaluations on the incoming side of its next jump.

        The driver must first clip the step at that jump. Construct the left
        neighbor in the state/time precision: nudging a Python float and then
        converting to float32 can round straight back onto the discontinuity.

        Returns:
            One-sided RHS callback for a step already clipped by its driver.

        Raises:
            TypeError: If the time precision is unsupported.
            ValueError: If declared coordinates collapse or overflow in that precision.
        """
        if not self.breakpoints:
            return rhs
        dtype_name = str(reference.dtype).removeprefix("torch.")
        if dtype_name not in {"float32", "float64"}:
            msg = (
                "Deterministic forcing requires float32 or float64 state/time precision"
            )
            raise TypeError(msg)
        with np.errstate(over="ignore"):
            native_points = np.asarray(self.breakpoints, dtype=dtype_name)
        if not np.all(np.isfinite(native_points)) or np.any(
            native_points[1:] <= native_points[:-1]
        ):
            msg = (
                "forcing_breakpoints must remain finite and distinct "
                "in state/time precision"
            )
            raise ValueError(msg)
        xp = array_namespace(reference)
        boundary = self.next_after_array(time, reference)
        left = xp.nextafter(boundary, xp.asarray(-math.inf, dtype=reference.dtype))
        start = xp.asarray(time, dtype=reference.dtype)
        points = xp.asarray(self.breakpoints, dtype=reference.dtype)
        at_jump = xp.any(xp.equal(points, start))
        # A float32 coordinate can lie just below the producer's float64
        # coordinate. Its right neighbor is safely on the outgoing side in
        # either precision, without changing the integrated time or duration.
        right = xp.nextafter(start, xp.asarray(math.inf, dtype=reference.dtype))
        lower = xp.where(at_jump, right, start)

        def incoming(stage_time: Scalar, state: StateT) -> StateT:
            return rhs(
                cast(
                    "Array",
                    xp.minimum(
                        xp.maximum(
                            xp.asarray(stage_time, dtype=reference.dtype), lower
                        ),
                        left,
                    ),
                ),
                state,
            )

        return incoming

    def step_times(self, start: float, steps: Sequence[float]) -> tuple[float, ...]:
        """Reconstruct replay starts, rejecting steps across a declared jump.

        Snap roundoff at a declared endpoint; never insert or split a frozen
        accepted step, since that would change the replayed numerical map.

        Returns:
            Start time of each unchanged accepted step.

        Raises:
            ValueError: If a frozen step crosses a forcing boundary.
        """
        times: list[float] = []
        time = start
        roundoff = 0.0
        for dt in steps:
            times.append(time)
            # Preserve the low part of the accumulated time. Plain repeated
            # addition can miss a valid jump after thousands of small steps,
            # particularly when the solve begins at a large time offset.
            end = math.fsum((time, roundoff, dt)) if self.breakpoints else time + dt
            roundoff = (
                math.fsum((time, roundoff, dt, -end)) if self.breakpoints else 0.0
            )
            boundary = self.next_after(time)
            if math.isfinite(boundary) and math.isclose(
                end,
                boundary,
                rel_tol=0.0,
                abs_tol=8 * max(math.ulp(time), math.ulp(end), math.ulp(boundary)),
            ):
                end = boundary
                roundoff = 0.0
            if end > boundary:
                msg = (
                    "Adaptive schedule crosses a forcing breakpoint; "
                    "discover a fresh mesh"
                )
                raise ValueError(msg)
            time = end
        return tuple(times)

    def validate_mesh(
        self, output_times: Sequence[float], step_sizes: Sequence[Sequence[float]]
    ) -> None:
        """Require every frozen interval to respect the declared boundaries."""
        for start, steps in zip(output_times[:-1], step_sizes, strict=True):
            self.step_times(start, steps)
