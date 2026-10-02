"""Static forcing boundaries shared by stochastic integration methods."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real


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
