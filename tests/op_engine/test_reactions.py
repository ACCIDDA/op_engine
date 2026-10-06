"""Reaction-network compilation from op_system-shaped artifacts (#191).

The compiler reads artifacts structurally, so these tests describe an
age-structured SIR with small dataclasses carrying op_system's
``CompiledReaction`` fields. Tests against real op_system artifacts live in
the flepimop2 provider suite, which depends on op_system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from op_engine import (
    CompiledReactionNetwork,
    compile_reaction_network,
    from_compiled_rhs,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Reactant:
    """Molecular reactant fields of ``op_system.CompiledReactant``."""

    state_base: str
    state_axes: tuple[str, ...]
    full_axes: tuple[str, ...]
    pinned: tuple[tuple[str, int], ...] = ()
    order: int = 1


@dataclass(frozen=True)
class Reaction:
    """The ``op_system.CompiledReaction`` fields the compiler reads."""

    name: str
    from_base: str | None
    to_base: str
    propensity_fn: Callable[..., object]
    from_axes: tuple[str, ...] = ("age",)
    full_axes: tuple[str, ...] = ("age",)
    to_axes: tuple[str, ...] = ("age",)
    sum_axes: tuple[str, ...] = ()
    pinned: tuple[tuple[str, int], ...] = ()
    from_pinned: tuple[tuple[str, int], ...] = ()
    reactants: tuple[Reactant, ...] = field(default_factory=tuple)
    reactants_complete: bool = True


def _infect(_t: float, y: dict[str, Any], *, beta: float, **_: object) -> object:
    return beta * y["S"] * y["I"]


def _recover(_t: float, y: dict[str, Any], *, gamma: float, **_: object) -> object:
    return gamma * y["I"]


AGE = ("age",)
REACTIONS = (
    Reaction(
        "infect",
        "S",
        "I",
        _infect,
        reactants=(Reactant("S", AGE, AGE), Reactant("I", AGE, AGE)),
    ),
    Reaction("recover", "I", "R", _recover, reactants=(Reactant("I", AGE, AGE),)),
)
SHAPES = {"S": (2,), "I": (2,), "R": (2,)}
PARAMS = {"beta": 0.01, "gamma": 0.1}


def _network(**overrides: Any) -> CompiledReactionNetwork:  # noqa: ANN401
    kwargs: dict[str, Any] = {
        "template_shapes": SHAPES,
        "axis_sizes": {"age": 2},
        "params": PARAMS,
    }
    kwargs.update(overrides)
    reactions = kwargs.pop("reactions", REACTIONS)
    return compile_reaction_network(reactions, **kwargs)


def test_channels_follow_source_cells_and_template_order() -> None:
    """One channel per source cell; rows follow the template layout."""
    network = _network()
    assert network.n_state == 6
    assert network.n_channels == 4
    assert network.channel_reactions == ("infect", "infect", "recover", "recover")
    assert network.channel_names == (
        "infect[age=0]",
        "infect[age=1]",
        "recover[age=0]",
        "recover[age=1]",
    )
    np.testing.assert_array_equal(
        network.stoichiometry,
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


def test_reactant_orders_include_catalysts() -> None:
    """The catalytic ``I`` appears in reactant orders but not net change."""
    network = _network()
    np.testing.assert_array_equal(
        network.reactant_stoichiometry[:, :2],
        [[1, 0], [0, 1], [1, 0], [0, 1], [0, 0], [0, 0]],
    )
    assert network.reactants_complete is True


def test_mean_drift_is_stoichiometry_times_propensity() -> None:
    """The deterministic drift of the network is the SIR right-hand side."""
    network = _network()
    s, i, r = np.array([90.0, 80.0]), np.array([10.0, 20.0]), np.array([0.0, 5.0])
    state = np.concatenate([s, i, r])
    infection, recovery = 0.01 * s * i, 0.1 * i
    np.testing.assert_allclose(
        network.mean_drift(0.0, state),
        np.concatenate([-infection, infection - recovery, recovery]),
    )
    np.testing.assert_allclose(
        network.propensity(0.0, state[:, None]),
        np.concatenate([infection, recovery])[:, None],
    )


def test_state_size_defaults_to_the_template_layout() -> None:
    """``n_state`` is optional, and a disagreeing value is rejected."""
    assert _network(n_state=6).n_state == 6
    with pytest.raises(ValueError, match="describes 6 state cells"):
        _network(n_state=5)


def test_reaction_names_select_a_partition() -> None:
    """Selected reactions keep their channels; unknown names fail."""
    network = _network(reaction_names=("recover",))
    assert network.channel_reactions == ("recover", "recover")
    with pytest.raises(ValueError, match="Unknown stochastic reaction names: missing"):
        _network(reaction_names=("missing",))


def test_incomplete_reactions_are_named() -> None:
    """Reactions without complete reactant metadata are listed."""
    reactions = (REACTIONS[0], replace(REACTIONS[1], reactants_complete=False))
    network = _network(reactions=reactions)
    assert network.reactants_complete is False
    assert network.incomplete_reactions == ("recover",)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        pytest.param({"reactions": {}}, TypeError, "sequence", id="reactions"),
        pytest.param(
            {"template_shapes": [(2,)]}, TypeError, "template_shapes", id="shapes"
        ),
        pytest.param(
            {"axis_sizes": {"age": 0}}, TypeError, "axis_sizes", id="axis-size"
        ),
        pytest.param({"axis_sizes": {}}, ValueError, "unknown axis", id="axis-missing"),
    ],
)
def test_malformed_inputs_fail_visibly(
    overrides: dict[str, object], error: type[Exception], match: str
) -> None:
    """Layout errors name the offending input."""
    with pytest.raises(error, match=match):
        _network(**overrides)


def _compiled(**overrides: Any) -> SimpleNamespace:  # noqa: ANN401
    """Stand in for ``op_system.CompiledRhs``.

    Returns:
        An object with the attributes ``from_compiled_rhs`` reads.
    """
    fields: dict[str, Any] = {
        "reactions": REACTIONS,
        "template_shapes": SHAPES,
        "meta": {"axes": [{"name": "age", "coords": ["child", "adult"]}]},
        "reaction_gaps": (),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_from_compiled_rhs_reads_layout_and_axes() -> None:
    """Axis sizes come from the axis metadata's coordinates or size."""
    network = from_compiled_rhs(_compiled(), PARAMS)
    assert network.n_channels == 4
    sized = _compiled(meta={"axes": [{"name": "age", "size": 2}]})
    assert from_compiled_rhs(sized, PARAMS).n_channels == 4


def test_from_compiled_rhs_requires_a_vectorized_layout() -> None:
    """Without ``template_shapes`` the propensities cannot be indexed."""
    with pytest.raises(TypeError, match="vectorized state layout"):
        from_compiled_rhs(_compiled(template_shapes=None), PARAMS)


def test_from_compiled_rhs_rejects_reaction_gaps_unless_allowed() -> None:
    """Dynamics without a reaction would silently drop out of the network."""
    gap = SimpleNamespace(name="mix", origin="transitions[2]", reason="unnamed")
    compiled = _compiled(reaction_gaps=(gap,))
    with pytest.raises(ValueError, match=r"mix \(unnamed\)"):
        from_compiled_rhs(compiled, PARAMS)
    assert from_compiled_rhs(compiled, PARAMS, allow_gaps=True).n_channels == 4
    partition = from_compiled_rhs(compiled, PARAMS, reaction_names=("infect",))
    assert partition.channel_reactions == ("infect", "infect")
