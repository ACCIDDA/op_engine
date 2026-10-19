"""Real compiler/engine integration for independent stochastic banks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_engine import from_compiled_replicates

compiler = pytest.importorskip("op_system")
if not hasattr(compiler.CompiledReaction, "resolved_propensity_fn"):
    pytest.skip(
        "Independent reaction contract requires op_system 0.8.0+",
        allow_module_level=True,
    )


def _spec(*, rep_first: bool = False, timed: bool = False) -> dict[str, Any]:
    axes = "rep,age" if rep_first else "age,rep"
    rate = f"beta[time,{axes}]" if timed else f"beta[{axes}]"
    return {
        "kind": "transitions",
        "factorize_axes": ["rep"],
        "time_interpolation": "previous",
        "axes": [
            {"name": "rep", "coords": ["a", "b"]},
            {"name": "age", "coords": ["young", "old", "elder"]},
            {"name": "time", "type": "continuous", "coords": [0.0, 2.0, 5.0]},
        ],
        "state": [f"S[{axes}]", f"I[{axes}]"],
        "transitions": [
            {"name": "infect", "from": f"S[{axes}]", "to": f"I[{axes}]", "rate": rate},
            {
                "name": "recover",
                "from": f"I[{axes}]",
                "to": f"S[{axes}]",
                "rate": "gamma",
            },
        ],
    }


@pytest.mark.parametrize("rep_first", [False, True])
def test_local_stoichiometry_and_full_state_order(*, rep_first: bool) -> None:
    """Removing the replicate dimension preserves state/channel ordering and drift."""
    spec = _spec(rep_first=rep_first)
    compiled = compiler.compile_spec(spec)
    shape = (2, 3) if rep_first else (3, 2)
    beta = np.arange(6.0).reshape(shape) * 0.01 + 0.1
    network = from_compiled_replicates(
        compiled, {"beta": beta, "gamma": 0.2}, batch_axis="rep"
    )
    assert network.stoichiometry.shape == (6, 6)
    flat = np.arange(12.0, dtype=float) + 10.0
    bank = network.pack_state(flat)
    restored = network.flatten_history(np.stack((bank, bank + 1)))
    np.testing.assert_array_equal(restored, np.stack((flat, flat + 1)))
    drift = network.stoichiometry @ network.propensity(np.zeros(2), bank)
    np.testing.assert_allclose(
        network.flatten_history(drift[None])[0],
        compiled.eval_fn(0.0, flat, beta=beta, gamma=0.2),
    )


@pytest.mark.parametrize("rep_first", [False, True])
def test_time_gather_is_diagonal_and_preserves_local_parameter_axes(
    *, rep_first: bool
) -> None:
    """Different clocks select one table row per replicate, not a Cartesian bank."""
    compiled = compiler.compile_spec(_spec(rep_first=rep_first, timed=True))
    shape = (2, 3) if rep_first else (3, 2)
    table = np.stack([np.full(shape, rate) for rate in [0.1, 0.3, 0.7]])
    network = from_compiled_replicates(
        compiled, {"beta": table, "gamma": 0.2}, batch_axis="rep"
    )
    bank = np.vstack((np.full((3, 2), 10.0), np.zeros((3, 2))))
    hazards = network.propensity(np.array([1.0, 5.0]), bank)
    np.testing.assert_allclose(hazards[:3], [[1.0, 7.0]] * 3)
    assert network.forcing_breakpoints == (2.0, 5.0)
    table[...] = 0.5
    np.testing.assert_allclose(network.propensity(np.array([1.0, 5.0]), bank)[:3], 5.0)


def test_uncertified_or_continuous_models_fail() -> None:
    """A batch-shaped network does not establish independence or frozen hazards."""
    spec = _spec()
    spec.pop("factorize_axes")
    with pytest.raises(ValueError, match="certified"):
        from_compiled_replicates(compiler.compile_spec(spec), {}, batch_axis="rep")
    spec = _spec(timed=True)
    spec["time_interpolation"] = "linear"
    with pytest.raises(ValueError, match="previous"):
        from_compiled_replicates(compiler.compile_spec(spec), {}, batch_axis="rep")
    spec = _spec()
    spec["transitions"][0]["rate"] = "beta * (1+t)"
    with pytest.raises(ValueError, match="time dependence"):
        from_compiled_replicates(compiler.compile_spec(spec), {}, batch_axis="rep")
