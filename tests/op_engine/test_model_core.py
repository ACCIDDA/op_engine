# tests/test_model_core.py
"""Unit tests for the ModelCore class.

This module tests the core functionality of ModelCore including:
- Time grid validation and per-step dt handling (including non-uniform grids)
- Multi-axis state tensor initialization and shape enforcement
- Axis metadata helpers (axis_index, axis_coords)
- Reshape/unreshape helpers for axis-local operator solves
- State management with and without history tracking
- State updates via deltas and direct assignment
- Bounds checking for timestep advancement
- New convenience APIs: current_time, get_time_at, state_ndim, validate_state_shape
"""

from __future__ import annotations

import numpy as np
import pytest

from op_engine.model_core import ModelCore, ModelCoreOptions

# -------------------------------------------------------------------
# Time grid / dt
# -------------------------------------------------------------------


def test_model_core_timegrid_uniform_dt_grid_ok() -> None:
    """Initialization with a uniform time grid and dt_grid correctness."""
    time_grid = np.linspace(0.0, 10.0, 11)  # dt = 1.0
    core = ModelCore(n_states=3, n_subgroups=2, time_grid=time_grid)

    assert core.n_states == 3
    assert core.n_subgroups == 2
    assert core.n_timesteps == 11

    assert core.dt_grid.shape == (10,)
    assert np.allclose(core.dt_grid, 1.0)
    assert np.isclose(core.dt, 1.0)

    assert core.state_shape == (3, 2)
    assert core.current_state.shape == core.state_shape

    # Time helpers
    assert np.isclose(core.current_time, float(time_grid[0]))
    assert np.isclose(core.get_time_at(0), float(time_grid[0]))
    assert np.isclose(core.get_time_at(10), float(time_grid[10]))


def test_model_core_timegrid_nonuniform_dt_grid_ok() -> None:
    """Initialization with a non-uniform time grid and get_dt access."""
    time_grid = np.array([0.0, 0.1, 0.4, 1.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid)

    assert core.n_timesteps == 4
    assert core.dt_grid.shape == (3,)
    assert np.allclose(core.dt_grid, np.array([0.1, 0.3, 0.6]))
    assert np.isclose(core.get_dt(0), 0.1)
    assert np.isclose(core.get_dt(1), 0.3)
    assert np.isclose(core.get_dt(2), 0.6)

    assert np.isclose(core.get_time_at(2), 0.4)


def test_model_core_timegrid_nonmonotone_raises() -> None:
    """Non-monotonic time grids are rejected."""
    time_grid = np.array([0.0, 1.0, 0.5], dtype=float)
    with pytest.raises(ValueError, match="strictly increasing"):
        ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)


def test_model_core_timegrid_repeated_time_raises() -> None:
    """Repeated time points are rejected."""
    time_grid = np.array([0.0, 1.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="strictly increasing"):
        ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)


def test_model_core_single_step_dt_grid_empty_and_get_dt_is_zero() -> None:
    """Single-step time grids yield empty dt_grid and get_dt returns 0."""
    time_grid = np.array([0.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)

    assert core.n_timesteps == 1
    assert core.dt == 0.0
    assert core.dt_grid.shape == (0,)
    assert np.isclose(core.get_dt(0), 0.0)

    assert np.isclose(core.current_time, 0.0)
    assert np.isclose(core.get_time_at(0), 0.0)


def test_get_dt_out_of_bounds_raises() -> None:
    """get_dt raises IndexError for invalid indices (multi-step grid)."""
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    core = ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid)

    with pytest.raises(IndexError, match="dt index out of bounds"):
        _ = core.get_dt(-1)

    with pytest.raises(IndexError, match="dt index out of bounds"):
        _ = core.get_dt(2)  # valid are 0..n_timesteps-2 == 1


def test_get_time_at_out_of_bounds_raises() -> None:
    """get_time_at raises IndexError for invalid indices."""
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    core = ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid)

    with pytest.raises(IndexError, match="time index out of bounds"):
        _ = core.get_time_at(-1)

    with pytest.raises(IndexError, match="time index out of bounds"):
        _ = core.get_time_at(99)


# -------------------------------------------------------------------
# Axis metadata
# -------------------------------------------------------------------


def test_axis_names_default_and_axis_index_and_state_ndim() -> None:
    """Default axis_names and axis_index resolution + state_ndim."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(4,))
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)

    assert core.state_shape == (2, 3, 4)
    assert core.state_ndim == 3
    assert core.axis_names == ("state", "subgroup", "axis2")

    assert core.axis_index("state") == 0
    assert core.axis_index("subgroup") == 1
    assert core.axis_index("axis2") == 2
    assert core.axis_index(2) == 2

    with pytest.raises(ValueError, match="Unknown axis"):
        _ = core.axis_index("not_an_axis")

    with pytest.raises(IndexError, match="Axis index out of bounds"):
        _ = core.axis_index(99)


def test_axis_names_length_mismatch_raises() -> None:
    """axis_names length mismatch raises ValueError."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(4,), axis_names=("state", "subgroup"))
    with pytest.raises(ValueError, match="axis_names length"):
        ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)


def test_axis_coords_metadata_roundtrip() -> None:
    """Axis coordinate metadata can be retrieved by name/index."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    axis_coords = {
        "subgroup": np.array([10.0, 20.0, 30.0]),
        "axis2": np.array([0.0, 0.5, 1.0, 2.0]),
    }
    opts = ModelCoreOptions(other_axes=(4,), axis_coords=axis_coords)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)

    coords_subgroup = core.get_axis_coords("subgroup")
    assert coords_subgroup is not None
    assert np.allclose(coords_subgroup, axis_coords["subgroup"])

    coords_axis2 = core.get_axis_coords(2)
    assert coords_axis2 is not None
    assert np.allclose(coords_axis2, axis_coords["axis2"])

    assert core.get_axis_coords("state") is None


# -------------------------------------------------------------------
# State shapes, history, initialization
# -------------------------------------------------------------------


def test_state_shape_multi_axis_and_history_on() -> None:
    """State/history shapes for a multi-axis state tensor."""
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(4, 5), store_history=True)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)

    assert core.state_shape == (2, 3, 4, 5)
    assert core.current_state.shape == (2, 3, 4, 5)

    assert core.state_array is not None
    assert core.state_array.shape == (3, 2, 3, 4, 5)


def test_set_initial_state_and_history_on() -> None:
    """Setting initial state with history enabled stores step 0."""
    time_grid = np.array([0.0, 1.0, 2.0], dtype=float)
    opts = ModelCoreOptions(store_history=True)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)

    init = np.arange(6, dtype=float).reshape(2, 3)
    core.set_initial_state(init)

    assert core.current_step == 0
    assert np.array_equal(core.get_current_state(), init)

    assert core.state_array is not None
    assert core.state_array.shape == (3, 2, 3)
    assert np.array_equal(core.state_array[0], init)


def test_set_initial_state_and_history_off() -> None:
    """Setting initial state with history disabled does not allocate history."""
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    opts = ModelCoreOptions(store_history=False)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)

    init = np.arange(6, dtype=float).reshape(2, 3)
    core.set_initial_state(init)

    assert core.current_step == 0
    assert np.array_equal(core.get_current_state(), init)
    assert core.state_array is None

    with pytest.raises(RuntimeError, match="store_history=False"):
        _ = core.get_state_at(0)


def test_set_initial_state_wrong_shape_raises() -> None:
    """Setting an initial state with the wrong shape raises an error."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid)

    bad_init = np.zeros((3, 2), dtype=float)
    with pytest.raises(ValueError, match="Initial state shape"):
        core.set_initial_state(bad_init)


# -------------------------------------------------------------------
# validate_state_shape helper
# -------------------------------------------------------------------


def test_validate_state_shape_accepts_and_rejects() -> None:
    """validate_state_shape accepts correct shape and rejects incorrect shape."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid)

    ok = np.zeros(core.state_shape, dtype=float)
    core.validate_state_shape(ok)  # should not raise

    bad = np.zeros((2, 2), dtype=float)
    with pytest.raises(ValueError, match="Next state shape"):
        core.validate_state_shape(bad)

    with pytest.raises(ValueError, match="custom"):
        core.validate_state_shape(bad, msg="custom {actual} {expected}")


# -------------------------------------------------------------------
# State updates: deltas / next state / bounds
# -------------------------------------------------------------------


def test_apply_deltas_updates_state_and_history_multi_axis() -> None:
    """Applying deltas updates state and history for multi-axis tensors."""
    time_grid = np.array([0.0, 1.0, 2.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(3,), store_history=True)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid, options=opts)

    init = np.zeros(core.state_shape, dtype=float)
    core.set_initial_state(init)

    delta1 = np.ones(core.state_shape, dtype=float)
    delta2 = 2.0 * np.ones(core.state_shape, dtype=float)

    core.apply_deltas(delta1)
    assert core.current_step == 1
    assert np.array_equal(core.get_current_state(), init + delta1)
    assert np.array_equal(core.get_state_at(1), init + delta1)
    assert np.isclose(core.current_time, float(time_grid[1]))

    core.apply_deltas(delta2)
    assert core.current_step == 2
    expected = init + delta1 + delta2
    assert np.array_equal(core.get_current_state(), expected)
    assert np.array_equal(core.get_state_at(2), expected)
    assert np.isclose(core.current_time, float(time_grid[2]))


def test_apply_deltas_wrong_shape_raises() -> None:
    """apply_deltas rejects wrong-shaped inputs."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)
    core.set_initial_state(np.zeros(core.state_shape, dtype=float))

    bad_delta = np.zeros((2, 3), dtype=float)
    with pytest.raises(ValueError, match="Deltas shape"):
        core.apply_deltas(bad_delta)


def test_apply_next_state_overwrites_state() -> None:
    """apply_next_state overwrites the current state correctly and stores history."""
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    opts = ModelCoreOptions(store_history=True)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid, options=opts)

    init = np.zeros(core.state_shape, dtype=float)
    core.set_initial_state(init)

    next_state = np.ones(core.state_shape, dtype=float)
    core.apply_next_state(next_state)

    assert core.current_step == 1
    assert np.array_equal(core.get_current_state(), next_state)
    assert np.array_equal(core.get_state_at(1), next_state)
    assert np.isclose(core.current_time, float(time_grid[1]))


def test_apply_next_state_wrong_shape_raises() -> None:
    """apply_next_state rejects wrong-shaped inputs."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)
    core.set_initial_state(np.zeros(core.state_shape, dtype=float))

    bad_next = np.zeros((2, 3), dtype=float)
    with pytest.raises(ValueError, match="Next state shape"):
        core.apply_next_state(bad_next)


def test_get_state_at_step_oob_raises() -> None:
    """get_state_at raises IndexError when step is out of bounds."""
    time_grid = np.array([0.0, 1.0, 2.0], dtype=float)
    opts = ModelCoreOptions(store_history=True)
    core = ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid, options=opts)
    core.set_initial_state(np.zeros(core.state_shape, dtype=float))

    with pytest.raises(IndexError, match="Step out of bounds"):
        _ = core.get_state_at(-1)

    with pytest.raises(IndexError, match="Step out of bounds"):
        _ = core.get_state_at(999)


def test_cannot_advance_past_last_step() -> None:
    """Advancing at the final timestep raises an error."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid)

    core.set_initial_state(np.array([[0.0]], dtype=float))
    core.apply_next_state(np.array([[1.0]], dtype=float))

    assert core.current_step == 1
    with pytest.raises(RuntimeError, match="final timestep"):
        core.apply_next_state(np.array([[2.0]], dtype=float))


# -------------------------------------------------------------------
# Reshape / unreshape helpers
# -------------------------------------------------------------------


def test_reshape_and_unreshape_for_axis_solve_roundtrip() -> None:
    """reshape_for_axis_solve and unreshape_from_axis_solve roundtrip."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(4,))
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, options=opts)

    init = np.arange(np.prod(core.state_shape), dtype=float).reshape(core.state_shape)
    core.set_initial_state(init)
    x = core.get_current_state()

    x2d, original_shape, axis_idx = core.reshape_for_axis_solve(x, "subgroup")
    assert original_shape == core.state_shape
    assert axis_idx == 1

    # axis_len = 3, batch = 2*4 = 8
    assert x2d.shape == (core.state_shape[1], core.state_shape[0] * core.state_shape[2])

    x_roundtrip = core.unreshape_from_axis_solve(x2d, original_shape, "subgroup")
    assert x_roundtrip.shape == core.state_shape
    assert np.array_equal(x_roundtrip, x)


def test_reshape_for_axis_solve_wrong_shape_raises() -> None:
    """reshape_for_axis_solve rejects arrays not matching state_shape."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(3,))
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid, options=opts)

    bad = np.zeros((2, 2), dtype=float)  # missing axis
    with pytest.raises(ValueError, match="Next state shape"):
        _ = core.reshape_for_axis_solve(bad, "subgroup")


def test_unreshape_from_axis_solve_respects_axis_position() -> None:
    """unreshape_from_axis_solve reconstructs the correct layout for a non-lead axis."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    opts = ModelCoreOptions(other_axes=(2,))  # shape (state, subgroup, axis2)
    core = ModelCore(n_states=3, n_subgroups=4, time_grid=time_grid, options=opts)

    x = np.arange(np.prod(core.state_shape), dtype=float).reshape(core.state_shape)
    core.set_initial_state(x)

    # Solve along axis2 (index 2): expect x2d shape (2, 3*4)
    x2d, original_shape, axis_idx = core.reshape_for_axis_solve(x, "axis2")
    assert axis_idx == 2
    assert x2d.shape == (2, 3 * 4)

    x_back = core.unreshape_from_axis_solve(x2d, original_shape, "axis2")
    assert np.array_equal(x_back, x)


# -------------------------------------------------------------------
# Views / dtype
# -------------------------------------------------------------------


def test_get_current_state_is_view() -> None:
    """get_current_state returns a view into the current state."""
    time_grid = np.array([0.0, 1.0], dtype=float)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)

    init = np.zeros(core.state_shape, dtype=float)
    core.set_initial_state(init)

    state_view = core.get_current_state()
    state_view[0, 0] = 42.0

    assert core.current_state[0, 0] == 42.0


def test_dtype_propagation() -> None:
    """Dtype is applied to internal arrays and dt_grid."""
    time_grid = np.array([0.0, 0.25, 1.0], dtype=float)
    opts = ModelCoreOptions(dtype=np.float32)
    core = ModelCore(n_states=1, n_subgroups=2, time_grid=time_grid, options=opts)

    assert core.current_state.dtype == np.float32
    assert core.dt_grid.dtype == np.float32
    if core.state_array is not None:
        assert core.state_array.dtype == np.float32
