"""Known-time, native-array state updates for explicit integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import TYPE_CHECKING

from ._array import array_namespace
from ._forcing import _ForcingSchedule

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._typing import Array, Scalar


@dataclass(frozen=True, slots=True)
class StateImpulse:
    """Replace the state at one fixed time with ``update(time, pre_state)``.

    The pure callback returns the post-state in the same shape, dtype and array
    namespace. It must not mutate its input. Parameters may be captured as native
    arrays; event times are static, not differentiable inputs. ``name`` is an
    optional descriptive label, not a semantic identity for the callback.

    Examples:
        >>> import numpy as np
        >>> event = StateImpulse(0.5, lambda time, state: state + 2.0)
        >>> event.update(event.time, np.array([1.0, 3.0])).tolist()
        [3.0, 5.0]
    """

    time: float
    update: Callable[[Scalar, Array], Array]
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate static event metadata without evaluating the callback.

        Raises:
            ValueError: Time is not a finite real scalar or name is empty.
            TypeError: The update is not callable or name is not a string.
        """
        msg = "Impulse time must be a finite real scalar"
        if not isinstance(self.time, Real) or isinstance(self.time, bool):
            raise ValueError(msg)
        try:
            time = float(self.time)
        except (OverflowError, ValueError) as error:
            raise ValueError(msg) from error
        if not math.isfinite(time):
            raise ValueError(msg)
        object.__setattr__(self, "time", time)
        if not callable(self.update):
            msg = "Impulse update must be callable"
            raise TypeError(msg)
        if self.name is not None:
            if not isinstance(self.name, str):
                msg = "Impulse name must be a string or None"
                raise TypeError(msg)
            if not self.name.strip():
                msg = "Impulse name cannot be empty"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _ImpulseSchedule:
    """Validate events and apply only an exact declared endpoint."""

    events: tuple[StateImpulse, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot records and require strictly increasing event times.

        Raises:
            TypeError: An event is not a StateImpulse.
            ValueError: Times are not strictly increasing.
        """
        events = tuple(self.events)
        if any(not isinstance(event, StateImpulse) for event in events):
            msg = "impulses must contain StateImpulse records"
            raise TypeError(msg)
        if any(b.time <= a.time for a, b in pairwise(events)):
            msg = "Impulse times must be strictly increasing"
            raise ValueError(msg)
        object.__setattr__(self, "events", events)

    @property
    def times(self) -> tuple[float, ...]:
        """Return the frozen event coordinates."""
        return tuple(event.time for event in self.events)

    def boundaries(self, forcing: tuple[float, ...]) -> _ForcingSchedule:
        """Return combined boundaries without changing RHS forcing metadata."""
        return _ForcingSchedule(tuple(sorted(set(forcing) | set(self.times))))

    def apply(self, time: float, state: Array) -> tuple[Array, bool]:
        """Apply a matching event once; callers own advancing the time cursor.

        Returns:
            Post-state and whether the first-stage cache must be invalidated.

        Raises:
            TypeError: The callback changes array namespace or dtype.
            ValueError: The callback changes state shape.
        """
        for event in self.events:
            if time == event.time:
                result = event.update(time, state)
                if array_namespace(result) is not array_namespace(state):
                    msg = "Impulse update must preserve state array namespace"
                    raise TypeError(msg)
                if result.dtype != state.dtype:
                    msg = "Impulse update must preserve state dtype"
                    raise TypeError(msg)
                if result.shape != state.shape:
                    msg = "Impulse update must preserve state shape"
                    raise ValueError(msg)
                return result, True
        return state, False
