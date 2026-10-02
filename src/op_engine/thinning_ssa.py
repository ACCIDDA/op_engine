"""Exact bounded thinning for time-dependent reaction propensities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np

from ._forcing import _ForcingSchedule

if TYPE_CHECKING:
    from ._typing import Array


def _nonnegative_real(value: float, *, name: str) -> float:
    """Validate a finite non-negative real control.

    Returns:
        The normalized scalar.

    Raises:
        ValueError: If the control is not a finite non-negative real number.
    """
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        msg = f"{name} must be a finite non-negative real scalar"
        raise ValueError(msg)
    return float(value)


@dataclass(frozen=True, slots=True)
class TotalRateBound:
    """Upper bound on all reaction/batch rates until an exclusive endpoint.

    Attributes:
        rate: Finite non-negative total-rate bound at the unchanged state.
        valid_until: Finite exclusive endpoint, checked against the current
            time and forcing/solve limit when the callback returns it.
    """

    rate: float
    valid_until: float

    def __post_init__(self) -> None:
        """Normalize and validate the scalar controls.

        Raises:
            ValueError: If either control is invalid.
        """
        object.__setattr__(
            self, "rate", _nonnegative_real(self.rate, name="bound rate")
        )
        if (
            not isinstance(self.valid_until, Real)
            or isinstance(self.valid_until, bool)
            or not math.isfinite(self.valid_until)
        ):
            msg = "valid_until must be a finite real scalar"
            raise ValueError(msg)
        object.__setattr__(self, "valid_until", float(self.valid_until))


class RateBoundFunction(Protocol):
    """Bound total propensity at a fixed state on ``[t, valid_until)``.

    The callback must return ``t < valid_until <= limit``. ``limit`` is the
    next forcing change or solve endpoint, independent of observation times.
    """

    def __call__(self, t: float, state: Array, limit: float, /) -> TotalRateBound:
        """Return a certified bound and its exclusive expiry time."""


class ThinningSample(NamedTuple):
    """Independent exponential wait and uniform in the active array namespace.

    Both values are real floating scalar arrays. ``waiting_time`` is finite
    and positive; ``uniform`` is finite and belongs to ``[0, 1)``.
    """

    waiting_time: Array
    uniform: Array


class ThinningSampler(Protocol):
    """Sample candidates with globally increasing zero-based draw indices."""

    def __call__(self, bound_rate: Array, draw_index: int, /) -> ThinningSample:
        """Draw a wait at ``bound_rate`` and an independent uniform."""


@dataclass(frozen=True, slots=True)
class ThinningSSAConfig:
    """Controls for exact bounded-thinning SSA.

    Attributes:
        max_candidates: Maximum candidate draws across the entire run,
            including rejected and expired candidates.
        forcing_breakpoints: Strictly increasing finite forcing changes.
            Forcing and bound expiry take precedence over tied candidates.
    """

    max_candidates: int = 1_000_000
    forcing_breakpoints: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the candidate guard and snapshot the forcing schedule.

        Raises:
            ValueError: If either control is invalid.
        """
        if (
            not isinstance(self.max_candidates, Integral)
            or isinstance(self.max_candidates, bool)
            or self.max_candidates < 1
        ):
            msg = "max_candidates must be a positive integer"
            raise ValueError(msg)
        schedule = _ForcingSchedule(self.forcing_breakpoints)
        object.__setattr__(self, "forcing_breakpoints", schedule.breakpoints)


class NumpyThinningSampler:
    """Seeded stateful thinning sampler for NumPy reaction networks."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a NumPy random generator with an optional seed."""
        self._rng = np.random.default_rng(seed)

    def __call__(self, bound_rate: Array, draw_index: int, /) -> ThinningSample:
        """Draw a candidate wait and uniform in the input dtype.

        Returns:
            Scalar NumPy sampling arrays.

        Raises:
            TypeError: If the bound rate is not a NumPy array.
        """
        del draw_index
        if not isinstance(bound_rate, np.ndarray):
            msg = "NumpyThinningSampler requires a NumPy bound-rate array"
            raise TypeError(msg)
        wait = np.asarray(
            self._rng.exponential(scale=1.0 / float(bound_rate.item())),
            dtype=bound_rate.dtype,
        )
        uniform = np.asarray(self._rng.random(), dtype=bound_rate.dtype)
        return ThinningSample(cast("Array", wait), cast("Array", uniform))
