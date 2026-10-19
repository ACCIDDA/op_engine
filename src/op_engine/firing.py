"""Optional host-side records of accepted stochastic reaction firings."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class FiringRecorder:
    """Record selected channels in output intervals, without adding state rows.

    Counts are NumPy int64 diagnostics, even when the evolving state uses another
    namespace. Intervals are ``(t[i], t[i+1]]``. A forcing change wins a tie with
    a candidate event, so that discarded event is not counted. Reusing a recorder
    starts a fresh record. Recording never consumes random numbers.
    """

    def __init__(self, channels: tuple[int, ...] | None = None) -> None:
        """Select local reaction indices; None records every channel.

        Raises:
            ValueError: Channel selections contain invalid or duplicate indices.
        """
        if channels is not None and (
            any(
                isinstance(i, bool) or not isinstance(i, Integral) or i < 0
                for i in channels
            )
            or len(set(channels)) != len(channels)
        ):
            msg = "channels must contain distinct non-negative integer indices"
            raise ValueError(msg)
        self._selection = None if channels is None else tuple(int(i) for i in channels)
        self.channel_indices: tuple[int, ...] = ()
        self._rows: dict[int, int] = {}
        self._counts: NDArray[np.int64] | None = None
        self._reaction_shape: tuple[int, ...] = ()
        self._reaction_axis = 0

    @property
    def counts(self) -> NDArray[np.int64]:
        """Return read-only (interval, selected channel, batch...) counts.

        Raises:
            RuntimeError: The recorder has not been attached to a run.
        """
        if self._counts is None:
            msg = "FiringRecorder has not started a run"
            raise RuntimeError(msg)
        view = self._counts.view()
        view.setflags(write=False)
        return view

    @property
    def cumulative(self) -> NDArray[np.int64]:
        """Return cumulative counts at output times, including an initial zero."""
        counts = self.counts
        zero = np.zeros((1, *counts.shape[1:]), dtype=np.int64)
        return np.concatenate((zero, np.cumsum(counts, axis=0)), axis=0)

    def start(
        self, n_intervals: int, reaction_shape: tuple[int, ...], reaction_axis: int
    ) -> None:
        """Allocate a fresh record for a solver's validated reaction layout.

        Raises:
            ValueError: A selected channel is outside the reaction layout.
        """
        n_channels = reaction_shape[reaction_axis]
        selection = (
            self._selection if self._selection is not None else tuple(range(n_channels))
        )
        if any(i >= n_channels for i in selection):
            msg = "Selected firing channel is outside the reaction layout"
            raise ValueError(msg)
        self.channel_indices = selection
        self._rows = {channel: row for row, channel in enumerate(selection)}
        batch_shape = tuple(
            n for i, n in enumerate(reaction_shape) if i != reaction_axis
        )
        self._counts = np.zeros(
            (n_intervals, len(selection), *batch_shape), dtype=np.int64
        )
        self._reaction_shape = reaction_shape
        self._reaction_axis = reaction_axis

    def record(self, interval: int, flat_event_index: int) -> None:
        """Count one accepted event in its channel and batch cell.

        Raises:
            RuntimeError: Recording has not started or an interval count overflows.
        """
        if self._counts is None:
            msg = "FiringRecorder has not started a run"
            raise RuntimeError(msg)
        coordinate = np.unravel_index(flat_event_index, self._reaction_shape)
        row = self._rows.get(int(coordinate[self._reaction_axis]))
        if row is None:
            return
        batch = tuple(
            int(v) for i, v in enumerate(coordinate) if i != self._reaction_axis
        )
        index = (interval, row, *batch)
        if self._counts[index] == np.iinfo(np.int64).max:
            msg = "Accepted firing count exceeds int64 capacity"
            raise RuntimeError(msg)
        self._counts[index] += 1
