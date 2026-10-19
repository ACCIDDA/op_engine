"""Opt-in checks of states at requested output times.

These diagnostics do not inspect internal RK stages, rejected attempts or
unobserved substeps, and do not project or clamp the numerical solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from ._array import array_namespace

if TYPE_CHECKING:
    from ._typing import Array


def _indices(values: object, *, name: str, allow_empty: bool) -> tuple[int, ...]:
    """Snapshot unique nonnegative component selectors.

    Returns:
        Validated component indices.

    Raises:
        TypeError: If selectors are not an integer sequence.
        ValueError: If selectors are negative, repeated or disallowed empty.
    """
    if not isinstance(values, (tuple, list)) or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        msg = f"{name} must be a sequence of integer component indices"
        raise TypeError(msg)
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result) or len(set(result)) != len(result):
        msg = f"{name} must contain unique nonnegative indices"
        raise ValueError(msg)
    if not result and not allow_empty:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    return result


@dataclass(frozen=True, slots=True)
class OutputStateCheckConfig:
    """Components and tolerances for checks at requested output times.

    Attributes:
        nonnegative_indices: State components expected to be nonnegative;
            ``None`` selects every component. Selectors address the state axis.
        monotone_indices: Components expected to be nondecreasing, such as
            cumulative counters. Empty disables increment checks.
        negative_tolerance: A value below minus this absolute tolerance flags
            an output. This changes the diagnostic threshold, not the state.
        monotone_tolerance: An increment below minus this absolute tolerance
            flags an output interval. It is not scaled by interval duration.
    """

    nonnegative_indices: tuple[int, ...] | None = None
    monotone_indices: tuple[int, ...] = ()
    negative_tolerance: float = 0.0
    monotone_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Detach selectors and validate finite nonnegative tolerances.

        Raises:
            TypeError: If a selector or tolerance has the wrong type.
            ValueError: If a selector or tolerance is invalid.
        """
        if self.nonnegative_indices is not None:
            object.__setattr__(
                self,
                "nonnegative_indices",
                _indices(
                    self.nonnegative_indices,
                    name="nonnegative_indices",
                    allow_empty=False,
                ),
            )
        object.__setattr__(
            self,
            "monotone_indices",
            _indices(self.monotone_indices, name="monotone_indices", allow_empty=True),
        )
        for name in ("negative_tolerance", "monotone_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                msg = f"{name} must be a finite nonnegative number"
                raise TypeError(msg)
            normalized = float(value)
            if not np.isfinite(normalized) or normalized < 0:
                msg = f"{name} must be a finite nonnegative number"
                raise ValueError(msg)
            object.__setattr__(self, name, normalized)


class OutputStateDiagnostics(NamedTuple):
    """Per-subgroup/field-cell checks over the requested output grid.

    Arrays retain every state axis except the state-component axis. Counts
    count outputs or intervals with at least one flagged component, not the
    number of components. Values and flags stay in the input namespace.

    Attributes:
        n_outputs: Number of inspected outputs, including the initial state.
        complete: Whether every requested output was inspected.
        min_state: Minimum selected nonnegative-state value across outputs.
            NaNs propagate; inspect nonfinite counts too.
        n_negative_outputs: Outputs with any selected state below tolerance.
        n_nonfinite_outputs: Outputs with any nonfinite state component,
            including components outside ``nonnegative_indices``.
        min_increment: Minimum increment of the selected monotone components,
            or ``None`` if disabled. Positive infinity when no interval exists.
        n_decreasing_intervals: Output intervals with any selected decreasing
            component, or ``None`` if monotonicity checks are disabled.
    """

    n_outputs: int
    complete: bool
    min_state: Array
    n_negative_outputs: Array
    n_nonfinite_outputs: Array
    min_increment: Array | None
    n_decreasing_intervals: Array | None

    @property
    def valid(self) -> Array:
        """Return native flags for complete, finite, unflagged output records."""
        xp = array_namespace(self.min_state)
        valid = xp.logical_and(
            xp.equal(self.n_negative_outputs, 0), xp.equal(self.n_nonfinite_outputs, 0)
        )
        if self.n_decreasing_intervals is not None:
            valid = xp.logical_and(valid, xp.equal(self.n_decreasing_intervals, 0))
        return cast("Array", xp.logical_and(valid, xp.asarray(self.complete)))


def _validate_components(config: OutputStateCheckConfig, n_states: int) -> None:
    """Reject selectors outside the state-component axis.

    Raises:
        ValueError: If there are no components or a selector is out of bounds.
    """
    selected = (config.nonnegative_indices or ()) + config.monotone_indices
    if n_states < 1 or any(index >= n_states for index in selected):
        msg = "Output-state diagnostic component index exceeds the state axis"
        raise ValueError(msg)


def diagnose_trajectory(
    trajectory: Array,
    config: OutputStateCheckConfig | None = None,
) -> OutputStateDiagnostics:
    """Check an output trajectory without modifying it or extracting host scalars.

    Args:
        trajectory: Array shaped ``(output, state, *subgroups_or_field_axes)``.
        config: Component selectors and absolute tolerances.

    Returns:
        Diagnostics for the complete supplied trajectory. ``valid`` describes
        these recorded outputs, not intermediate-step positivity or accuracy.

    Raises:
        TypeError: If config has the wrong type.
        ValueError: If the trajectory shape or selectors are invalid.

    Examples:
        >>> import numpy as np
        >>> path = np.array([[[1.0, 1.0]], [[0.9, -1.0]], [[0.81, 1.0]]])
        >>> result = diagnose_trajectory(path)
        >>> result.n_negative_outputs.tolist()
        [0, 1]
        >>> result.valid.tolist()
        [True, False]
    """
    cfg = OutputStateCheckConfig() if config is None else config
    if not isinstance(cfg, OutputStateCheckConfig):
        msg = "config must be an OutputStateCheckConfig"
        raise TypeError(msg)
    if len(trajectory.shape) < 2 or trajectory.shape[0] < 1:
        msg = "Trajectory must have nonempty output and state axes"
        raise ValueError(msg)
    _validate_components(cfg, trajectory.shape[1])
    xp = array_namespace(trajectory)
    if not xp.isdtype(trajectory.dtype, "real floating"):
        msg = "Output-state diagnostics require real floating states"
        raise TypeError(msg)
    values = xp.asarray(trajectory)
    selected = (
        values
        if cfg.nonnegative_indices is None
        else values[:, xp.asarray(cfg.nonnegative_indices, dtype=xp.int32)]
    )
    minimum = xp.min(xp.min(selected, axis=1), axis=0)
    negatives = xp.sum(
        xp.astype(xp.any(selected < -cfg.negative_tolerance, axis=1), xp.int32), axis=0
    )
    nonfinite = xp.sum(
        xp.astype(xp.any(~xp.isfinite(values), axis=1), xp.int32), axis=0
    )
    min_increment = decreasing = None
    if cfg.monotone_indices:
        monotone = values[:, xp.asarray(cfg.monotone_indices, dtype=xp.int32)]
        if trajectory.shape[0] == 1:
            min_increment = xp.full_like(minimum, float("inf"))
            decreasing = xp.zeros_like(negatives)
        else:
            increments = monotone[1:] - monotone[:-1]
            min_increment = xp.min(xp.min(increments, axis=1), axis=0)
            decreasing = xp.sum(
                xp.astype(
                    xp.any(increments < -cfg.monotone_tolerance, axis=1), xp.int32
                ),
                axis=0,
            )
    return OutputStateDiagnostics(
        n_outputs=int(trajectory.shape[0]),
        complete=True,
        min_state=minimum,
        n_negative_outputs=negatives,
        n_nonfinite_outputs=nonfinite,
        min_increment=min_increment,
        n_decreasing_intervals=decreasing,
    )


class _OutputStateMonitor:
    """Collect output checks without requiring retained state history."""

    def __init__(
        self, config: OutputStateCheckConfig, initial: Array, n_outputs: int
    ) -> None:
        """Initialize component validation and the initial-state record."""
        _validate_components(config, initial.shape[0])
        self.config = config
        self.expected_outputs = n_outputs
        self.result = diagnose_trajectory(
            array_namespace(initial).expand_dims(initial, axis=0), config
        )
        self.result = self.result._replace(complete=n_outputs == 1)
        self._previous_monotone = self._monotone(initial)

    def _monotone(self, state: Array) -> Array | None:
        """Copy selected components so later buffer writes cannot change the reference.

        Returns:
            Selected components, or ``None`` when disabled.
        """
        xp = array_namespace(state)
        return (
            xp.asarray(state)[xp.asarray(self.config.monotone_indices, dtype=xp.int32)]
            if self.config.monotone_indices
            else None
        )

    def observe(self, state: Array) -> None:
        """Inspect one requested output after a successful core update."""
        xp = array_namespace(state)
        one = diagnose_trajectory(xp.expand_dims(state, axis=0), self.config)
        old = self.result
        current = self._monotone(state)
        min_increment, decreasing = old.min_increment, old.n_decreasing_intervals
        if (
            current is not None
            and self._previous_monotone is not None
            and min_increment is not None
            and decreasing is not None
        ):
            increments = xp.subtract(current, self._previous_monotone)
            min_increment = xp.minimum(min_increment, xp.min(increments, axis=0))
            decreasing = xp.add(
                decreasing,
                xp.astype(
                    xp.any(increments < -self.config.monotone_tolerance, axis=0),
                    xp.int32,
                ),
            )
        count = old.n_outputs + 1
        self.result = OutputStateDiagnostics(
            count,
            count == self.expected_outputs,
            xp.minimum(old.min_state, one.min_state),
            xp.add(old.n_negative_outputs, one.n_negative_outputs),
            xp.add(old.n_nonfinite_outputs, one.n_nonfinite_outputs),
            min_increment,
            decreasing,
        )
        self._previous_monotone = current

    def replace_trajectory(self, trajectory: Array) -> None:
        """Adopt whole-grid checks without unrolling diagnostics under JIT."""
        self.result = diagnose_trajectory(trajectory, self.config)
        self.result = self.result._replace(
            complete=self.result.n_outputs == self.expected_outputs
        )


__all__ = ["OutputStateCheckConfig", "OutputStateDiagnostics", "diagnose_trajectory"]
