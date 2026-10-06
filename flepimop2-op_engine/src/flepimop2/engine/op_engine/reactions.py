# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Compile a flepimop2 op_system system's reactions for stochastic solvers.

The compiler itself lives in :mod:`op_engine.reactions`, which reads op_system
artifacts structurally and has no flepimop2 dependency. This module adapts a
flepimop2 system's options to it and explains incomplete reactant metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from op_engine.reactions import (
    CompiledReactionNetwork,
    ReactionArtifact,
    ReactionReactantArtifact,
)
from op_engine.reactions import (
    compile_reaction_network as _compile_core_network,
)


class ReactionSystem(Protocol):
    """System option lookup needed by the reaction compiler."""

    def option(self, key: str, default: object = None) -> object:
        """Return a compiled system option."""


def compile_reaction_network(
    system: ReactionSystem,
    params: Mapping[str, object],
    *,
    n_state: int,
    reaction_names: Sequence[str] | None = None,
) -> CompiledReactionNetwork:
    """Compile a system's ``reactions`` option into flat channels.

    Args:
        system: System exposing the ``reactions``, ``template_shapes``, and
            ``axis_sizes`` options.
        params: Raw bound parameter values.
        n_state: Number of cells in the provider's flat state.
        reaction_names: Optional parent reaction names to select for a hybrid
            jump partition. ``None`` selects every artifact.

    Returns:
        Validated flat reaction network; see
        :func:`op_engine.reactions.compile_reaction_network`.

    Raises:
        TypeError: If a required system option is missing or malformed.
    """
    reactions = system.option("reactions", None)
    if not isinstance(reactions, tuple | list):
        msg = "system option 'reactions' must be a sequence."
        raise TypeError(msg)
    template_shapes = system.option("template_shapes", None)
    if not isinstance(template_shapes, Mapping):
        msg = (
            "Stochastic execution requires system option 'template_shapes' from "
            "typed op_system artifacts."
        )
        raise TypeError(msg)
    axis_sizes = system.option("axis_sizes", None)
    if not isinstance(axis_sizes, Mapping):
        msg = "system option 'axis_sizes' must be a mapping."
        raise TypeError(msg)
    return _compile_core_network(
        reactions,
        template_shapes=template_shapes,
        axis_sizes=axis_sizes,
        params=params,
        n_state=n_state,
        reaction_names=reaction_names,
    )


def incomplete_reactants_message(names: Sequence[str]) -> str:
    """Explain how to complete reactant metadata for adaptive tau-leaping.

    Returns:
        A message naming the reactions and the op_system remedies.
    """
    return (
        "Adaptive tau-leaping requires complete molecular reactant metadata, "
        f"but these reactions lack it: {', '.join(names)}. Declare "
        "'reactants: auto' (op_system 0.7.0+) to infer them from the rate, or "
        "an explicit 'reactants' list, on ordinary op_system transitions; on "
        "chain: entries (entry.catalysts and catalysts) and coord_shift "
        "entries, declare 'catalysts' (or 'catalysts: auto'), and the "
        "consumed source is added for you."
    )


__all__ = [
    "CompiledReactionNetwork",
    "ReactionArtifact",
    "ReactionReactantArtifact",
    "ReactionSystem",
    "compile_reaction_network",
    "incomplete_reactants_message",
]
