"""Single-event NumPy layout, ownership and exact stochastic replay checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from op_engine import Array, ModelCore
from op_engine.model_core import ModelCoreOptions
from op_engine.stochastic_solver import (
    _ReactionNetwork,  # noqa: PLC2701 - portable numerical reference
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_numpy_event_changes_only_selected_cell_and_owns_output(
    dtype: str, axis: int
) -> None:
    """A prescribed channel updates one batch cell along any reaction axis."""
    core = ModelCore(
        3, 4, np.array([0.0, 1.0]), options=ModelCoreOptions(other_axes=(2,))
    )
    shape = core.state_shape
    species = shape[axis]
    stoichiometry = np.zeros((species, 3), dtype=np.float64)
    stoichiometry[0] = [-1, 1, -2]
    stoichiometry[-1] = [1, -1, 2]
    network = _ReactionNetwork(core, cast("Array", stoichiometry), axis)
    # Read-only, noncontiguous native input must stay unchanged.
    storage = np.full((*shape, 2), 20, dtype=dtype)
    state = storage[..., 0]
    state.flags.writeable = False
    cell = [1, 2, 1]
    cell[axis] = 2
    event = int(np.ravel_multi_index(tuple(cell), network.reaction_shape))
    result = np.asarray(network.apply_event(event, cast("Array", state)))
    expected = np.full(shape, 20, dtype=dtype)
    cell[axis] = 0
    expected[tuple(cell)] = 18
    cell[axis] = species - 1
    expected[tuple(cell)] = 22
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(storage, 20)
    np.testing.assert_array_equal(stoichiometry[0], [-1, 1, -2])
    assert result.dtype == state.dtype
    assert not np.shares_memory(result, state)
    assert result.flags.writeable


def _dense_event(network: _ReactionNetwork, event: int, state: Array) -> Array:
    """Original dense portable operation, retained as a replay reference.

    Returns:
        State updated through a full one-hot firing array.
    """
    firings = np.zeros(network.reaction_shape, dtype=np.asarray(state).dtype)
    firings.ravel()[event] = 1
    return network.apply_firings(cast("Array", firings), state)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", ["direct", "thinning", "adaptive"])
def test_numpy_event_preserves_paths_counts_and_streams(
    dtype: str, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forcing/output boundaries, accepted counts and post-run RNG match exactly."""
    from op_engine import (  # noqa: PLC0415
        AdaptiveTauLeapingConfig,
        AdaptiveTauLeapingSolver,
        DirectSSAConfig,
        DirectSSASolver,
        FiringRecorder,
        ThinningSSAConfig,
        ThinningSSASolver,
    )

    def run() -> tuple[object, object, object]:
        core = ModelCore(
            2,
            3,
            np.array([0, 0.5, 1, 2, 3]),
            options=ModelCoreOptions(dtype=getattr(np, dtype)),
        )
        core.set_initial_state(cast("Array", np.full((2, 3), 5, dtype=dtype)))
        stoichiometry = np.array([[-1, 1], [1, -1]], dtype=dtype)
        rng = np.random.default_rng(184)
        counts = FiringRecorder()
        solver: Any
        controls: dict[str, Any]
        if method == "direct":
            solver = DirectSSASolver(core, cast("Array", stoichiometry))
            controls = {"config": DirectSSAConfig(forcing_breakpoints=(0.75,))}
        elif method == "thinning":
            solver = ThinningSSASolver(core, cast("Array", stoichiometry))
            controls = {
                "config": ThinningSSAConfig(forcing_breakpoints=(0.75,)),
                "rate_bound": 30.0,
            }
        else:
            solver = AdaptiveTauLeapingSolver(
                core,
                cast("Array", stoichiometry),
                cast("Array", np.eye(2, dtype=dtype)),
            )
            controls = {"config": AdaptiveTauLeapingConfig(forcing_breakpoints=(0.75,))}

        def hazards(time: float, state: Array) -> Array:
            return cast("Array", np.asarray(state) * (0.7 if time < 0.75 else 0.2))

        # Only direct SSA exposes a firing recorder; all three expose RNG state.
        if method == "direct":
            controls["recorder"] = counts
        solver.run(hazards, rng=rng, **controls)
        return (
            np.asarray(core.state_array),
            counts.counts.copy() if method == "direct" else None,
            rng.bit_generator.state,
        )

    current = run()
    reference: Callable[[_ReactionNetwork, int, Array], Array] = _dense_event
    monkeypatch.setattr(_ReactionNetwork, "apply_event", reference)
    previous = run()
    np.testing.assert_array_equal(current[0], previous[0])
    if method == "direct":
        np.testing.assert_array_equal(current[1], previous[1])
    assert current[2] == previous[2]
