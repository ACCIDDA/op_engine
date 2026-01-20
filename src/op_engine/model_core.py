# op_engine/src/op_engine/model_core.py
"""Core class for managing the numerical state of a model.

This module provides a lightweight state container and time-grid manager for
time-evolving models. It is designed to support:

- Non-uniform time grids via per-step dt accessors.
- Multi-axis state tensors (e.g., state x age x space x traits).
- Optional full history storage for post-analysis.
- A minimal backend hook to ease future NumPy->(JAX/CuPy) integration.

The core intentionally does not build RHS functions or construct operators; it
only manages state, time, and shape/axis metadata in a solver-friendly manner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import DTypeLike


# Error / message constants -------------------------------------------------

_TIMEGRID_1D_ERROR = "time_grid must be a 1D array"
_TIMEGRID_MIN_POINTS_ERROR = "time_grid must contain at least one time point"
_TIMEGRID_MONOTONE_ERROR = "time_grid must be strictly increasing"

_AXIS_NAMES_LEN_ERROR = "axis_names length {actual} doesn't match state rank {expected}"
_AXIS_UNKNOWN_ERROR = "Unknown axis: {axis}"
_AXIS_INDEX_OOB_ERROR = "Axis index out of bounds: {axis}"

_INITIAL_STATE_SHAPE_ERROR = "Initial state shape {actual} mismatch vs. {expected}"
_DELTAS_SHAPE_ERROR = "Deltas shape {actual} does not match expected {expected}"
_NEXT_STATE_SHAPE_ERROR = "Next state shape {actual} does not match expected {expected}"

_HISTORY_NOT_STORED_ERROR = (
    "Full history is not stored (store_history=False); get_state_at is unavailable."
)
_STEP_OOB_ERROR = "Step out of bounds"
_FINAL_TIMESTEP_ERROR = "Simulation has already reached final timestep"
_DT_INDEX_OOB_ERROR = "dt index out of bounds: {idx}"
_TIME_INDEX_OOB_ERROR = "time index out of bounds: {idx}"


# Typing helpers ------------------------------------------------------------

FloatArray = npt.NDArray[np.floating[Any]]


@dataclass(slots=True)
class ModelCoreOptions:
    """Optional configuration for ModelCore.

    This object groups non-essential constructor parameters to keep the
    ModelCore initializer compact and stable while allowing future
    extensions without breaking the public API.

    Attributes:
        other_axes: Additional axis sizes appended after the default
            (state, subgroup) axes. Example: (n_space,) or
            (n_space, n_trait).
        axis_names: Optional names for each axis in the state tensor.
            Length must equal the total state rank.
        axis_coords: Optional coordinate arrays keyed by axis name.
            This is metadata only and enables non-uniform grids and
            FV/FDM operator builders.
        store_history: Whether to store the full time history.
        dtype: Floating-point dtype for internal arrays.
        xp: Array backend module (default NumPy). This is a forward-
            compatibility hook for GPU backends.
    """

    other_axes: tuple[int, ...] = ()
    axis_names: tuple[str, ...] | None = None
    axis_coords: Mapping[str, np.ndarray] | None = None
    store_history: bool = True
    dtype: DTypeLike = np.float64
    xp: object = np


class ArrayBackend(Protocol):
    """Minimal array backend interface for ModelCore numerical storage."""

    def asarray(
        self,
        x: object,
        dtype: object | None = None,
    ) -> np.ndarray:
        """Convert input to an array of the backend type."""
        ...

    def zeros(
        self,
        shape: tuple[int, ...],
        dtype: object | None = None,
    ) -> np.ndarray:
        """Return a new array of given shape filled with zeros."""
        ...


class ModelCore:
    """Core state and time manager for time-evolving models."""

    def __init__(
        self,
        n_states: int,
        n_subgroups: int,
        time_grid: np.ndarray,
        *,
        options: ModelCoreOptions | None = None,
    ) -> None:
        """
        Initialize ModelCore.

        Args:
            n_states: Number of state variables.
            n_subgroups: Number of subgroups (e.g., age groups).
            time_grid: 1D array of times, shape (n_timesteps,).
            options: Optional ModelCoreOptions for additional configuration.

        Raises:
            ValueError: if time_grid is invalid or any shapes mismatch.
        """
        opts = options or ModelCoreOptions()

        # Forward-compat hook; stored for later wiring. For now ModelCore remains
        # NumPy-first; higher layers can decide how to allocate/convert.
        self.xp = opts.xp

        self.dtype = np.dtype(opts.dtype)

        self.time_grid = np.asarray(time_grid, dtype=self.dtype)
        if self.time_grid.ndim != 1:
            raise ValueError(_TIMEGRID_1D_ERROR)

        self.n_timesteps = int(self.time_grid.size)
        if self.n_timesteps < 1:
            raise ValueError(_TIMEGRID_MIN_POINTS_ERROR)

        if self.n_timesteps > 1:
            dt_arr = np.diff(self.time_grid)
            if np.any(dt_arr <= 0):
                raise ValueError(_TIMEGRID_MONOTONE_ERROR)
            self.dt_grid = np.asarray(dt_arr, dtype=self.dtype)
            self.dt = float(self.dt_grid.mean())
        else:
            self.dt_grid = np.asarray([], dtype=self.dtype)
            self.dt = 0.0

        self.n_states = int(n_states)
        self.n_subgroups = int(n_subgroups)
        self._other_axes = tuple(int(x) for x in opts.other_axes)

        self.state_shape = (self.n_states, self.n_subgroups, *self._other_axes)
        self.store_history = bool(opts.store_history)

        if opts.axis_names is None:
            base = ["state", "subgroup"]
            extra = [f"axis{i}" for i in range(2, len(self.state_shape))]
            self.axis_names = tuple(base + extra)
        else:
            if len(opts.axis_names) != len(self.state_shape):
                raise ValueError(
                    _AXIS_NAMES_LEN_ERROR.format(
                        actual=len(opts.axis_names),
                        expected=len(self.state_shape),
                    )
                )
            self.axis_names = tuple(opts.axis_names)

        self.axis_coords: dict[str, np.ndarray] = {}
        if opts.axis_coords is not None:
            self.axis_coords = {
                str(k): np.asarray(v, dtype=self.dtype)
                for k, v in opts.axis_coords.items()
            }

        self.current_step = 0

        # Per-timestep working state (contiguous).
        self.current_state = np.zeros(self.state_shape, dtype=self.dtype)

        # Optional full history: (n_timesteps, *state_shape)
        self.state_array: FloatArray | None
        if self.store_history:
            self.state_array = cast(
                "FloatArray",
                np.zeros((self.n_timesteps, *self.state_shape), dtype=self.dtype),
            )
        else:
            self.state_array = None

    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------

    @property
    def state_ndim(self) -> int:
        """
        Number of state tensor dimensions (rank).

        Args:
            None.

        Returns:
            State tensor rank as an integer.
        """
        return len(self.state_shape)

    def axis_index(self, axis: str | int) -> int:
        """
        Resolve an axis name or integer into an axis index.

        Args:
            axis: Axis name or index.

        Raises:
            IndexError: if axis index is out of bounds.
            ValueError: if axis name is unknown.

        Returns:
            Axis index as an integer.
        """
        if isinstance(axis, int):
            if not (0 <= axis < self.state_ndim):
                raise IndexError(_AXIS_INDEX_OOB_ERROR.format(axis=axis))
            return axis

        try:
            return self.axis_names.index(axis)
        except ValueError as exc:
            raise ValueError(_AXIS_UNKNOWN_ERROR.format(axis=axis)) from exc

    def get_axis_coords(self, axis: str | int) -> np.ndarray | None:
        """
        Return coordinate array for a given axis, or None if not set.

        Args:
            axis: Axis name or index.

        Returns:
            Coordinate array for the axis, or None if not set.
        """
        idx = self.axis_index(axis)
        name = self.axis_names[idx]
        return self.axis_coords.get(name)

    def validate_state_shape(self, arr: np.ndarray, *, msg: str | None = None) -> None:
        """
        Validate that arr has state_shape.

        This is intentionally small and solver-friendly; higher layers can reuse it
        to avoid duplicating shape checks for intermediate/stage states.

        Args:
            arr: Array to validate.
            msg: Optional custom error message prefix.

        Raises:
            ValueError: if arr does not have shape state_shape.
        """
        arr_shape = np.asarray(arr).shape
        if arr_shape != self.state_shape:
            raise ValueError(
                (msg or _NEXT_STATE_SHAPE_ERROR).format(
                    actual=arr_shape, expected=self.state_shape
                )
            )

    def reshape_for_axis_solve(
        self,
        x: np.ndarray,
        axis: str | int,
    ) -> tuple[np.ndarray, tuple[int, ...], int]:
        """Reshape a state-like tensor into 2D for an axis-local operator solve.

        Args:
            x: State-like tensor of shape state_shape.
            axis: Axis name or index along which to reshape.

        Returns:
            (x2d, original_shape, axis_index)

        Raises:
            ValueError: if x does not have shape state_shape.

        Contract:
            - x2d has shape (axis_len, batch)
            - batch is the product of all non-axis dimensions
            - unreshape_from_axis_solve(inverse) reconstructs exactly.
        """
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.shape != self.state_shape:
            raise ValueError(
                _NEXT_STATE_SHAPE_ERROR.format(
                    actual=x_arr.shape, expected=self.state_shape
                )
            )

        original_shape = x_arr.shape
        axis_idx = self.axis_index(axis)
        axis_len = int(original_shape[axis_idx])

        moved = np.moveaxis(x_arr, axis_idx, 0)
        x2d = moved.reshape(axis_len, -1)
        return x2d, original_shape, axis_idx

    def unreshape_from_axis_solve(
        self,
        x2d: np.ndarray,
        original_shape: tuple[int, ...],
        axis: str | int,
    ) -> np.ndarray:
        """
        Inverse of reshape_for_axis_solve.

        Args:
            x2d: 2D array of shape (axis_len, batch).
            original_shape: Original full shape before reshape.
            axis: Axis name or index along which the reshape was done.

        Returns:
            Reconstructed array of shape original_shape.
        """
        axis_idx = self.axis_index(axis)
        axis_len = int(original_shape[axis_idx])

        # Rebuild shape with axis leading, then move it back.
        trailing = tuple(d for i, d in enumerate(original_shape) if i != axis_idx)
        arr = np.asarray(x2d, dtype=self.dtype).reshape((axis_len, *trailing))
        return np.moveaxis(arr, 0, axis_idx)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    @property
    def current_time(self) -> float:
        """
        Current simulation time t = time_grid[current_step].

        Args:
            None.

        Returns:
            Current time as a float.
        """
        return float(self.time_grid[self.current_step])

    def get_time_at(self, step_idx: int) -> float:
        """
        Return time at a given step index.

        Args:
            step_idx: Timestep index in [0, n_timesteps).

        Raises:
            IndexError: if step_idx is out of bounds.

        Returns:
            Time as a float.
        """
        if not (0 <= step_idx < self.n_timesteps):
            raise IndexError(_TIME_INDEX_OOB_ERROR.format(idx=step_idx))
        return float(self.time_grid[step_idx])

    def get_dt(self, step_idx: int) -> float:
        """
        Return dt for the step [t_step_idx, t_step_idx+1].

        Args:
            step_idx: Timestep index in [0, n_timesteps - 1].

        Raises:
            IndexError: if step_idx is out of bounds.

        Returns:
            dt as a float.
        """
        if self.n_timesteps <= 1:
            return 0.0
        if not (0 <= step_idx < self.n_timesteps - 1):
            raise IndexError(_DT_INDEX_OOB_ERROR.format(idx=step_idx))
        return float(self.dt_grid[step_idx])

    # ------------------------------------------------------------------
    # Initialization / accessors
    # ------------------------------------------------------------------

    def set_initial_state(self, initial_state: np.ndarray) -> None:
        """
        Set the initial state at time_grid[0].

        Args:
            initial_state: Initial state, shape state_shape.

        Raises:
            ValueError: if initial_state has incorrect shape.
        """
        initial_state_arr = np.asarray(initial_state, dtype=self.dtype)
        if initial_state_arr.shape != self.state_shape:
            raise ValueError(
                _INITIAL_STATE_SHAPE_ERROR.format(
                    actual=initial_state_arr.shape,
                    expected=self.state_shape,
                )
            )

        np.copyto(self.current_state, initial_state_arr)

        if self.store_history and self.state_array is not None:
            self.state_array[0] = self.current_state

        self.current_step = 0

    def get_current_state(self) -> np.ndarray:
        """
        Retrun the current state.

        Args:
            None.

        Returns:
            Current state, shape state_shape.
        """
        return self.current_state

    def get_state_at(self, step: int) -> FloatArray:
        """
        Return the state at a given timestep from history.

        Args:
            step: Timestep index in [0, n_timesteps).

        Returns:
            State at the given timestep, shape state_shape.

        Raises:
            RuntimeError: if history is not stored.
            IndexError: if step is out of bounds.
        """
        if not self.store_history or self.state_array is None:
            raise RuntimeError(_HISTORY_NOT_STORED_ERROR)

        if not (0 <= step < self.n_timesteps):
            raise IndexError(_STEP_OOB_ERROR)

        # NumPy typing stubs often type ndarray.__getitem__ as Any under mypy,
        # which triggers --strict [no-any-return] without an explicit cast.
        return cast("FloatArray", self.state_array[step])

    # ------------------------------------------------------------------
    # Stepping / updates
    # ------------------------------------------------------------------

    def _check_can_advance(self) -> None:
        if self.current_step >= self.n_timesteps - 1:
            raise RuntimeError(_FINAL_TIMESTEP_ERROR)

    def apply_deltas(self, deltas: np.ndarray) -> None:
        """
        Apply state deltas, advancing the timestep.

        Supports alternate solver implementations (e.g., splitting updates,
        additive increments) without forcing allocation of y_next.

        Args:
            deltas: State deltas to apply, shape state_shape.

        Raises:
            ValueError: if deltas has incorrect shape.
        """
        deltas_arr = np.asarray(deltas, dtype=self.dtype)
        if deltas_arr.shape != self.state_shape:
            raise ValueError(
                _DELTAS_SHAPE_ERROR.format(
                    actual=deltas_arr.shape, expected=self.state_shape
                )
            )

        self._check_can_advance()

        self.current_state += deltas_arr

        self.current_step += 1
        if self.store_history and self.state_array is not None:
            self.state_array[self.current_step] = self.current_state

    def apply_next_state(self, next_state: np.ndarray) -> None:
        """
        Set the next state directly, advancing the timestep.

        Args:
            next_state: State at the next timestep, shape state_shape.

        Raises:
            ValueError: if next_state has incorrect shape.
        """
        next_state_arr = np.asarray(next_state, dtype=self.dtype)
        if next_state_arr.shape != self.state_shape:
            raise ValueError(
                _NEXT_STATE_SHAPE_ERROR.format(
                    actual=next_state_arr.shape, expected=self.state_shape
                )
            )

        self._check_can_advance()

        np.copyto(self.current_state, next_state_arr)

        self.current_step += 1
        if self.store_history and self.state_array is not None:
            self.state_array[self.current_step] = self.current_state

    def advance_timestep(self, next_state: np.ndarray) -> None:
        """Alias for apply_next_state, for solver-friendly naming."""
        self.apply_next_state(next_state)
