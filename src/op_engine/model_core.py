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
from typing import TYPE_CHECKING, Protocol

import numpy as np

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
    """Minimal array backend interface for ModelCore numerical storage.

    This protocol defines the minimal subset of the NumPy array API required
    by ModelCore for allocating and converting arrays. It exists to enable
    future drop-in backends (for example JAX or CuPy) without changing the
    public ModelCore interface.

    Any backend implementing this protocol must provide NumPy-compatible
    semantics for array creation and conversion. Additional functionality
    (linear algebra, sparse operators, etc.) is intentionally excluded and
    handled at higher layers.

    Notes:
        This is intentionally minimal. It is not intended to be a full Array
        API standard abstraction. The goal is to allow simple backend
        substitution while preserving performance and avoiding excessive
        indirection in the core engine.
    """

    def asarray(self, x: object, dtype: object | None = None) -> np.ndarray:
        """Convert input to an array on the backend.

        Args:
            x: Input object to convert.
            dtype: Optional dtype specification.

        Returns:
            Backend array representation of the input.
        """

    def zeros(
        self,
        shape: tuple[int, ...],
        dtype: object | None = None,
    ) -> np.ndarray:
        """Allocate a zero-initialized array.

        Args:
            shape: Shape of the output array.
            dtype: Optional dtype specification.

        Returns:
            Zero-initialized backend array with the requested shape and dtype.
        """



class ModelCore:
    """Core class for managing the numerical state of a model.

    Design goals:
        - CPU-friendly layout: per-timestep state is contiguous in memory.
        - Full state history by default, but can be disabled for very large runs.
        - Support non-uniform time grids via dt_grid and get_dt(step).
        - Support multi-axis state tensors via state_shape and axis metadata.
        - Keep a minimal backend hook (xp) for future GPU-facing backends.

    Attributes:
        time_grid: Array of simulation times of shape (n_timesteps,).
        dt_grid: Per-step timestep widths of shape (n_timesteps - 1,).
        dt: Convenience mean timestep (float). Do not use for stepping on
            non-uniform grids; prefer get_dt(step).
        n_timesteps: Number of time points in the simulation.
        current_step: Index of the current timestep (0-based).
        state_shape: Shape of a single model state tensor (multi-axis).
        axis_names: Optional names for axes in state_shape.
        axis_coords: Optional coordinate arrays keyed by axis name.
        current_state: State tensor at the current timestep, shape state_shape.
        state_array: Full state history, shape (n_timesteps, *state_shape), or
            None if store_history=False.
        store_history: Whether the full time history is being stored.
        dtype: Floating-point dtype used for all internal arrays.
        xp: Array API module (NumPy for now). This is a forward-compat hook.
    """

    def __init__(
        self,
        n_states: int,
        n_subgroups: int,
        time_grid: np.ndarray,
        *,
        options: ModelCoreOptions | None = None,
    ) -> None:
        """Initialize the ModelCore with simulation parameters.

        Args:
            n_states: Size of the leading "state" axis (e.g., compartments).
            n_subgroups: Size of the second axis (e.g., age groups). This is a
                convenience axis; additional axes can be provided via
                ModelCoreOptions.other_axes.
            time_grid: Strictly increasing array of simulation time points.
            options: Optional configuration bundle. If not provided, defaults are
                applied via ModelCoreOptions().

        Raises:
            ValueError: If time_grid is not 1D, has no time points, or is not
                strictly increasing.
            ValueError: If axis_names is provided and does not match state rank.
        """
        opts = options or ModelCoreOptions()

        # Forward-compat hook; kept as attribute for later wiring.
        self.xp = opts.xp

        self.time_grid = np.asarray(time_grid, dtype=opts.dtype)
        if self.time_grid.ndim != 1:
            msg = _TIMEGRID_1D_ERROR
            raise ValueError(msg)

        self.n_timesteps = int(self.time_grid.size)
        if self.n_timesteps < 1:
            msg = _TIMEGRID_MIN_POINTS_ERROR
            raise ValueError(msg)

        if self.n_timesteps > 1:
            dt_arr = np.diff(self.time_grid)
            if np.any(dt_arr <= 0):
                msg = _TIMEGRID_MONOTONE_ERROR
                raise ValueError(msg)
            self.dt_grid = np.asarray(dt_arr, dtype=opts.dtype)
            self.dt = float(self.dt_grid.mean())
        else:
            self.dt_grid = np.asarray([], dtype=opts.dtype)
            self.dt = 0.0

        self.n_states = int(n_states)
        self.n_subgroups = int(n_subgroups)
        self._other_axes = tuple(int(x) for x in opts.other_axes)

        self.state_shape = (self.n_states, self.n_subgroups, *self._other_axes)
        self.store_history = bool(opts.store_history)
        self.dtype = np.dtype(opts.dtype)

        if opts.axis_names is None:
            # Stable defaults; keeps user code readable without requiring config.
            base = ["state", "subgroup"]
            extra = [f"axis{i}" for i in range(2, len(self.state_shape))]
            self.axis_names = tuple(base + extra)
        else:
            if len(opts.axis_names) != len(self.state_shape):
                msg = _AXIS_NAMES_LEN_ERROR.format(
                    actual=len(opts.axis_names),
                    expected=len(self.state_shape),
                )
                raise ValueError(msg)
            self.axis_names = tuple(opts.axis_names)

        # Store axis coordinates as metadata keyed by axis name.
        self.axis_coords: dict[str, np.ndarray] = {}
        if opts.axis_coords is not None:
            self.axis_coords = {
                str(k): np.asarray(
                    v,
                    dtype=self.dtype,
                )
                for k, v in opts.axis_coords.items()
            }


        self.current_step = 0

        # Per-timestep working state (contiguous)
        self.current_state = np.zeros(self.state_shape, dtype=self.dtype)

        # Optional full history: (n_timesteps, *state_shape)
        if self.store_history:
            self.state_array: np.ndarray | None = np.zeros(
                (self.n_timesteps, *self.state_shape),
                dtype=self.dtype,
            )
        else:
            self.state_array = None


    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------

    def axis_index(self, axis: str | int) -> int:
        """Resolve an axis name or integer into an axis index.

        Args:
            axis: Axis name (if axis_names are set) or integer axis index.

        Returns:
            Integer axis index in [0, state_ndim).

        Raises:
            ValueError: If axis name is unknown.
            IndexError: If axis index is out of bounds.
        """
        if isinstance(axis, int):
            if not (0 <= axis < len(self.state_shape)):
                msg = _AXIS_INDEX_OOB_ERROR.format(axis=axis)
                raise IndexError(msg)
            return axis

        try:
            return self.axis_names.index(axis)
        except ValueError as exc:
            msg = _AXIS_UNKNOWN_ERROR.format(axis=axis)
            raise ValueError(msg) from exc

    def get_axis_coords(self, axis: str | int) -> np.ndarray | None:
        """Return coordinate metadata for an axis, if provided.

        Args:
            axis: Axis name or integer axis index.

        Returns:
            Coordinate array if available, otherwise None.
        """
        idx = self.axis_index(axis)
        name = self.axis_names[idx]
        return self.axis_coords.get(name)

    def reshape_for_axis_solve(
        self,
        x: np.ndarray,
        axis: str | int,
    ) -> tuple[np.ndarray, tuple[int, ...], int]:
        """Reshape a state-like tensor into 2D for an axis-local operator solve.

        This is intended for applying an operator along a single axis while
        batching over all other axes. The output is shaped:

            (axis_len, batch)

        where batch = product(other axis lengths).

        Args:
            x: Input array with shape state_shape.
            axis: Axis along which an operator will act.

        Returns:
            A tuple (x2d, original_shape, axis_len), where:
                x2d: 2D view/copy shaped (axis_len, batch).
                original_shape: The original shape of x.
                axis_len: Length of the operator axis.

        Raises:
            ValueError: If x does not have shape state_shape.
        """
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.shape != self.state_shape:
            msg = _NEXT_STATE_SHAPE_ERROR.format(
                actual=x_arr.shape,
                expected=self.state_shape)
            raise ValueError(msg)

        original_shape = x_arr.shape
        axis_idx = self.axis_index(axis)
        axis_len = original_shape[axis_idx]

        moved = np.moveaxis(x_arr, axis_idx, 0)
        x2d = moved.reshape(axis_len, -1)
        return x2d, original_shape, axis_len

    def unreshape_from_axis_solve(
        self,
        x2d: np.ndarray,
        original_shape: tuple[int, ...],
        axis: str | int,
    ) -> np.ndarray:
        """Inverse of reshape_for_axis_solve.

        Args:
            x2d: 2D array of shape (axis_len, batch).
            original_shape: Original tensor shape prior to reshaping.
            axis: Axis along which the operator acted.

        Returns:
            Reshaped tensor with shape original_shape.
        """
        axis_idx = self.axis_index(axis)
        axis_len = original_shape[axis_idx]

        arr = np.asarray(x2d, dtype=self.dtype).reshape(
            (axis_len, *[d for i, d in enumerate(original_shape) if i != axis_idx]),
        )
        return np.moveaxis(arr, 0, axis_idx)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def get_dt(self, step_idx: int) -> float:
        """Return dt for the step [t_step_idx, t_step_idx+1].

        Args:
            step_idx: Step index in [0, n_timesteps - 2].

        Returns:
            dt for the requested step.

        Raises:
            IndexError: If step_idx is outside [0, n_timesteps - 2].
        """
        if self.n_timesteps <= 1:
            return 0.0
        if not (0 <= step_idx < self.n_timesteps - 1):
            msg = _DT_INDEX_OOB_ERROR.format(idx=step_idx)
            raise IndexError(msg)
        return float(self.dt_grid[step_idx])

    # ------------------------------------------------------------------
    # Initialization / accessors
    # ------------------------------------------------------------------

    def set_initial_state(self, initial_state: np.ndarray) -> None:
        """Set the state at t = time_grid[0].

        Args:
            initial_state: Array of shape state_shape containing the initial
                model values.

        Raises:
            ValueError: If initial_state does not have shape state_shape.
        """
        initial_state_arr = np.asarray(initial_state, dtype=self.dtype)
        if initial_state_arr.shape != self.state_shape:
            msg = _INITIAL_STATE_SHAPE_ERROR.format(
                actual=initial_state_arr.shape,
                expected=self.state_shape,
            )
            raise ValueError(msg)

        np.copyto(self.current_state, initial_state_arr)

        if self.store_history and self.state_array is not None:
            self.state_array[0] = self.current_state

        self.current_step = 0

    def get_current_state(self) -> np.ndarray:
        """Return the current state.

        Returns:
            A view of the current state array with shape state_shape.
        """
        return self.current_state

    def get_state_at(self, step: int) -> np.ndarray:
        """Return the state at a given timestep.

        Requires that store_history=True.

        Args:
            step: Timestep index in the range [0, n_timesteps).

        Returns:
            State at the requested timestep, shape state_shape.

        Raises:
            RuntimeError: If store_history is False (no history stored).
            IndexError: If step is outside [0, n_timesteps).
        """
        if not self.store_history or self.state_array is None:
            msg = _HISTORY_NOT_STORED_ERROR
            raise RuntimeError(msg)

        if not (0 <= step < self.n_timesteps):
            msg = _STEP_OOB_ERROR
            raise IndexError(msg)

        return self.state_array[step]

    # ------------------------------------------------------------------
    # Stepping / updates
    # ------------------------------------------------------------------

    def _check_can_advance(self) -> None:
        if self.current_step >= self.n_timesteps - 1:
            msg = _FINAL_TIMESTEP_ERROR
            raise RuntimeError(msg)

    def apply_deltas(self, deltas: np.ndarray) -> None:
        """Advance one timestep by applying additive deltas.

        Args:
            deltas: Array of shape state_shape representing state changes to
                apply over the next dt.

        Raises:
            ValueError: If deltas does not have shape state_shape.
        """
        deltas_arr = np.asarray(deltas, dtype=self.dtype)
        if deltas_arr.shape != self.state_shape:
            msg = _DELTAS_SHAPE_ERROR.format(
                actual=deltas_arr.shape,
                expected=self.state_shape,
            )
            raise ValueError(msg)

        self._check_can_advance()

        self.current_state += deltas_arr

        self.current_step += 1
        if self.store_history and self.state_array is not None:
            self.state_array[self.current_step] = self.current_state

    def apply_next_state(self, next_state: np.ndarray) -> None:
        """Advance one timestep by directly specifying the next state.

        Args:
            next_state: Array of shape state_shape representing the state at
                the next timestep.

        Raises:
            ValueError: If next_state does not have shape state_shape.
        """
        next_state_arr = np.asarray(next_state, dtype=self.dtype)
        if next_state_arr.shape != self.state_shape:
            msg = _NEXT_STATE_SHAPE_ERROR.format(
                actual=next_state_arr.shape,
                expected=self.state_shape,
            )
            raise ValueError(msg)

        self._check_can_advance()

        np.copyto(self.current_state, next_state_arr)

        self.current_step += 1
        if self.store_history and self.state_array is not None:
            self.state_array[self.current_step] = self.current_state

    def advance_timestep(self, next_state: np.ndarray) -> None:
        """Alias for apply_next_state, for solver-friendly naming.

        Args:
            next_state: State at the next timestep, shape state_shape.
        """
        self.apply_next_state(next_state)

