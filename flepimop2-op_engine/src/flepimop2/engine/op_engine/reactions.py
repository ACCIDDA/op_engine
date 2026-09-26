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

"""Compile typed op_system reaction artifacts for stochastic solvers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from flepimop2.typing import Array


class ReactionArtifact(Protocol):
    """Structural surface published by ``op_system.CompiledReaction``."""

    name: str
    from_base: str | None
    from_axes: tuple[str, ...]
    full_axes: tuple[str, ...]
    to_base: str
    to_axes: tuple[str, ...]
    sum_axes: tuple[str, ...]
    pinned: tuple[tuple[str, int], ...]
    from_pinned: tuple[tuple[str, int], ...]
    reactants: tuple[ReactionReactantArtifact, ...]
    reactants_complete: bool
    propensity_fn: Callable[..., object]


class ReactionReactantArtifact(Protocol):
    """Structural molecular-reactant surface published by op_system."""

    state_base: str
    state_axes: tuple[str, ...]
    full_axes: tuple[str, ...]
    pinned: tuple[tuple[str, int], ...]
    order: int


class ReactionSystem(Protocol):
    """System option lookup needed by the reaction compiler."""

    def option(self, key: str, default: object = None) -> object:
        """Return a compiled system option."""


class _IndexableArray(Protocol):
    """Internal slicing surface kept out of flepimop2's Array protocol."""

    def __getitem__(self, key: slice) -> Array:
        """Return an array slice."""


@dataclass(frozen=True, slots=True)
class _StateBlock:
    """One state template's location in the flat provider state."""

    base: str
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class CompiledReactionNetwork:
    """Flat stochastic network compiled from typed reaction artifacts.

    ``channel_reactions`` retains the parent reaction name for every expanded
    source-cell channel. ``channel_names`` adds integer coordinates and is
    intended for diagnostics only.
    """

    stoichiometry: np.ndarray
    reactant_stoichiometry: np.ndarray
    channel_reactions: tuple[str, ...]
    channel_names: tuple[str, ...]
    _blocks: tuple[_StateBlock, ...]
    _reactions: tuple[ReactionArtifact, ...]
    _event_shapes: tuple[tuple[int, ...], ...]
    _params: Mapping[str, object]
    reactants_complete: bool = False

    @property
    def n_state(self) -> int:
        """Return the number of flattened state cells."""
        return int(self.stoichiometry.shape[0])

    @property
    def n_channels(self) -> int:
        """Return the number of expanded reaction channels."""
        return int(self.stoichiometry.shape[1])

    def propensity(self, time: float, state: Array) -> Array:
        """Evaluate every reaction and concatenate its source-cell rates.

        The stochastic core stores provider state as ``(n_state, 1)``. A flat
        state is accepted as well so the same callback can construct hybrid
        mean drift. Returned shape mirrors the input convention.

        Returns:
            Propensities in the evolving state's array namespace.
        """
        xp = _namespace_of(state)
        batched = state.shape == (self.n_state, 1)
        if not batched and state.shape != (self.n_state,):
            msg = (
                f"Reaction propensity received state shape {state.shape}; expected "
                f"{(self.n_state,)} or {(self.n_state, 1)}."
            )
            raise ValueError(msg)
        flat_state = cast("Array", xp.reshape(state, (self.n_state,)))
        indexable_state = cast("_IndexableArray", flat_state)
        state_dict = {
            block.base: cast(
                "Array",
                xp.reshape(
                    indexable_state[block.offset : block.offset + block.size],
                    block.shape,
                ),
            )
            for block in self._blocks
        }

        values: list[Array] = []
        for reaction, expected_shape in zip(
            self._reactions,
            self._event_shapes,
            strict=True,
        ):
            raw = reaction.propensity_fn(time, state_dict, **self._params)
            raw_namespace = getattr(raw, "__array_namespace__", None)
            if raw_namespace is not None and raw_namespace() is not xp:
                msg = f"Reaction {reaction.name!r} changed the state array namespace."
                raise TypeError(msg)
            value = cast("Array", xp.asarray(raw, dtype=state.dtype))
            if value.shape != expected_shape:
                msg = (
                    f"Reaction {reaction.name!r} returned propensity shape "
                    f"{value.shape}; expected {expected_shape}."
                )
                raise ValueError(msg)
            values.append(cast("Array", xp.reshape(value, (-1,))))

        combined = cast("Array", xp.concat(tuple(values), axis=0))
        if batched:
            return cast("Array", xp.reshape(combined, (self.n_channels, 1)))
        return combined

    def mean_drift(self, time: float, state: Array) -> Array:
        """Return deterministic mean drift from the compiled channels.

        Returns:
            ``stoichiometry @ propensity`` in the state's namespace.
        """
        xp = _namespace_of(state)
        propensity = self.propensity(time, state)
        stoichiometry = cast(
            "Array",
            xp.asarray(self.stoichiometry, dtype=state.dtype),
        )
        return cast("Array", xp.matmul(stoichiometry, propensity))


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return an Array-API namespace advertised by ``value``."""
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        msg = (
            "Reaction-network arrays must implement __array_namespace__(); "
            f"got {type(value).__name__}."
        )
        raise TypeError(msg)
    return namespace()


def _require_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    """Validate one typed artifact's axis-name tuple.

    Returns:
        Normalized axis names.
    """
    if not isinstance(value, tuple | list) or not all(
        isinstance(item, str) for item in value
    ):
        msg = f"Reaction field {field!r} must be a sequence of axis names."
        raise TypeError(msg)
    result = tuple(value)
    if len(result) != len(set(result)):
        msg = f"Reaction field {field!r} contains duplicate axes."
        raise ValueError(msg)
    return result


def _require_pins(value: object, *, field: str) -> dict[str, int]:
    """Validate fixed-axis coordinate metadata.

    Returns:
        Mapping from axis name to integer coordinate.
    """
    if not isinstance(value, tuple | list):
        msg = f"Reaction field {field!r} must be a sequence of axis/index pairs."
        raise TypeError(msg)
    pins: dict[str, int] = {}
    for item in value:
        if (
            not isinstance(item, tuple | list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
        ):
            msg = f"Reaction field {field!r} contains an invalid axis/index pair."
            raise TypeError(msg)
        axis, index = item
        if axis in pins:
            msg = f"Reaction field {field!r} pins axis {axis!r} more than once."
            raise ValueError(msg)
        pins[axis] = index
    return pins


def _state_blocks(
    template_shapes: Mapping[object, object],
    *,
    n_state: int,
) -> tuple[_StateBlock, ...]:
    """Build and validate the flat state-template layout.

    Returns:
        State blocks in op_system's declared template order.
    """
    blocks: list[_StateBlock] = []
    offset = 0
    for base_obj, shape_obj in template_shapes.items():
        if not isinstance(base_obj, str):
            msg = "system option 'template_shapes' keys must be strings."
            raise TypeError(msg)
        if not isinstance(shape_obj, tuple | list) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in shape_obj
        ):
            msg = f"Template shape for {base_obj!r} must contain positive integers."
            raise TypeError(msg)
        shape = tuple(shape_obj)
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        blocks.append(_StateBlock(base_obj, shape, offset, size))
        offset += size
    if offset != n_state:
        msg = (
            "system option 'template_shapes' describes "
            f"{offset} state cells, but the provider state contains {n_state}."
        )
        raise ValueError(msg)
    return tuple(blocks)


def _reaction_axes(
    reactions: Sequence[ReactionArtifact],
    blocks: Mapping[str, _StateBlock],
    axis_sizes: Mapping[str, int],
) -> dict[str, tuple[str, ...]]:
    """Infer and validate each participating template's declared axes.

    Returns:
        State base to full ordered axes.
    """
    result: dict[str, tuple[str, ...]] = {}

    def register(base: str, axes_value: object, *, reaction_name: str) -> None:
        axes = _require_string_tuple(axes_value, field="full_axes")
        if base not in blocks:
            msg = f"Reaction {reaction_name!r} references unknown state {base!r}."
            raise ValueError(msg)
        previous = result.setdefault(base, axes)
        if previous != axes:
            msg = (
                f"Reaction {reaction_name!r} gives state {base!r} axes {axes}, "
                f"inconsistent with {previous}."
            )
            raise ValueError(msg)
        try:
            expected_shape = tuple(axis_sizes[axis] for axis in axes)
        except KeyError as error:
            msg = (
                f"Reaction {reaction_name!r} references unknown axis {error.args[0]!r}."
            )
            raise ValueError(msg) from error
        if blocks[base].shape != expected_shape:
            msg = (
                f"State {base!r} shape {blocks[base].shape} does not match "
                f"reaction axes {axes} with shape {expected_shape}."
            )
            raise ValueError(msg)

    for reaction in reactions:
        for base in (reaction.from_base, reaction.to_base):
            if base is None:
                continue
            register(base, reaction.full_axes, reaction_name=reaction.name)
        raw_reactants = getattr(reaction, "reactants", ())
        if not isinstance(raw_reactants, tuple | list):
            msg = f"Reaction {reaction.name!r} reactants must be a sequence."
            raise TypeError(msg)
        for reactant in raw_reactants:
            base = getattr(reactant, "state_base", None)
            if not isinstance(base, str) or not base:
                msg = f"Reaction {reaction.name!r} has an invalid reactant state base."
                raise TypeError(msg)
            register(
                base,
                getattr(reactant, "full_axes", None),
                reaction_name=reaction.name,
            )
    return result


def _expanded_reactants(  # noqa: PLR0913
    reaction: ReactionArtifact,
    *,
    varying: Mapping[str, int],
    from_axes: tuple[str, ...],
    base_axes: Mapping[str, tuple[str, ...]],
    blocks: Mapping[str, _StateBlock],
    n_state: int,
) -> np.ndarray:
    """Expand one reaction channel's molecular reactant-order column.

    Returns:
        Integer molecular orders aligned to the flat provider state.
    """
    raw_reactants = getattr(reaction, "reactants", None)
    if raw_reactants is None:
        # Compatibility with op_system artifacts predating reactant metadata.
        result = np.zeros(n_state, dtype=np.int64)
        if reaction.from_base is not None:
            source_coordinates = {
                **varying,
                **_require_pins(reaction.from_pinned, field="from_pinned"),
            }
            source = _flat_cell(
                blocks[reaction.from_base],
                base_axes[reaction.from_base],
                source_coordinates,
            )
            result[source] = 1
        return result
    if not isinstance(raw_reactants, tuple | list):
        msg = f"Reaction {reaction.name!r} reactants must be a sequence."
        raise TypeError(msg)

    result = np.zeros(n_state, dtype=np.int64)
    for reactant in raw_reactants:
        base = getattr(reactant, "state_base", None)
        if not isinstance(base, str) or base not in blocks:
            msg = f"Reaction {reaction.name!r} has an unknown reactant state."
            raise ValueError(msg)
        state_axes = _require_string_tuple(
            getattr(reactant, "state_axes", None),
            field="reactant.state_axes",
        )
        full_axes = _require_string_tuple(
            getattr(reactant, "full_axes", None),
            field="reactant.full_axes",
        )
        if full_axes != base_axes[base]:
            msg = (
                f"Reaction {reaction.name!r} reactant {base!r} has inconsistent "
                "full_axes metadata."
            )
            raise ValueError(msg)
        if not set(state_axes).issubset(from_axes):
            msg = (
                f"Reaction {reaction.name!r} reactant {base!r} has axes outside "
                "the expanded reaction channels."
            )
            raise ValueError(msg)
        pins = _require_pins(
            getattr(reactant, "pinned", None),
            field="reactant.pinned",
        )
        if set(state_axes) | set(pins) != set(full_axes):
            msg = (
                f"Reaction {reaction.name!r} reactant {base!r} has incomplete "
                "axis metadata."
            )
            raise ValueError(msg)
        if set(state_axes) & set(pins):
            msg = (
                f"Reaction {reaction.name!r} reactant {base!r} both varies and "
                "pins an axis."
            )
            raise ValueError(msg)
        order = getattr(reactant, "order", None)
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            msg = (
                f"Reaction {reaction.name!r} reactant {base!r} order must be a "
                "positive integer."
            )
            raise TypeError(msg)
        coordinates = {axis: varying[axis] for axis in state_axes} | pins
        cell = _flat_cell(blocks[base], full_axes, coordinates)
        result[cell] += order
    return result


def _flat_cell(
    block: _StateBlock,
    axes: tuple[str, ...],
    coordinates: Mapping[str, int],
) -> int:
    """Map named cell coordinates into the provider's flat state.

    Returns:
        Flat state index.
    """
    index = tuple(coordinates[axis] for axis in axes)
    for coordinate, size in zip(index, block.shape, strict=True):
        if not 0 <= coordinate < size:
            msg = f"Coordinate {coordinate} is outside state {block.base!r}."
            raise ValueError(msg)
    if not block.shape:
        return block.offset
    return block.offset + int(np.ravel_multi_index(index, block.shape))


def compile_reaction_network(
    system: ReactionSystem,
    params: Mapping[str, object],
    *,
    n_state: int,
    reaction_names: Sequence[str] | None = None,
) -> CompiledReactionNetwork:
    """Compile public op_system reaction options into flat channels.

    Each source-cell propensity becomes one channel. Consequently, a collapsed
    axis is represented by distinct columns that share one destination row;
    summing simultaneous firings is then exactly the stoichiometric matrix
    multiplication performed by the core stochastic solvers.

    Args:
        system: System exposing typed reaction and layout options.
        params: Raw bound parameter values.
        n_state: Number of cells in the provider's flat state.
        reaction_names: Optional parent reaction names to select for a hybrid
            jump partition. ``None`` selects every artifact.

    Returns:
        Validated flat reaction network.
    """
    raw_reactions = system.option("reactions", None)
    if not isinstance(raw_reactions, tuple | list):
        msg = "system option 'reactions' must be a sequence."
        raise TypeError(msg)
    reactions = tuple(cast("ReactionArtifact", item) for item in raw_reactions)
    if reaction_names is not None:
        requested = tuple(reaction_names)
        if len(requested) != len(set(requested)):
            msg = "stochastic_reactions must not contain duplicates."
            raise ValueError(msg)
        available = {reaction.name for reaction in reactions}
        missing = sorted(set(requested) - available)
        if missing:
            msg = f"Unknown stochastic reaction names: {', '.join(missing)}."
            raise ValueError(msg)
        selected = set(requested)
        reactions = tuple(
            reaction for reaction in reactions if reaction.name in selected
        )
    if not reactions:
        msg = "No typed reactions are available for stochastic execution."
        raise ValueError(msg)

    template_shapes = system.option("template_shapes", None)
    if not isinstance(template_shapes, Mapping):
        msg = (
            "Stochastic execution requires system option 'template_shapes' from "
            "typed op_system artifacts."
        )
        raise TypeError(msg)
    blocks_tuple = _state_blocks(template_shapes, n_state=n_state)
    blocks = {block.base: block for block in blocks_tuple}

    raw_axis_sizes = system.option("axis_sizes", None)
    if not isinstance(raw_axis_sizes, Mapping):
        msg = "system option 'axis_sizes' must be a mapping."
        raise TypeError(msg)
    axis_sizes: dict[str, int] = {}
    for axis, size in raw_axis_sizes.items():
        if (
            not isinstance(axis, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
        ):
            msg = "system option 'axis_sizes' contains an invalid entry."
            raise TypeError(msg)
        axis_sizes[axis] = size
    base_axes = _reaction_axes(reactions, blocks, axis_sizes)

    columns: list[np.ndarray] = []
    reactant_columns: list[np.ndarray] = []
    channel_reactions: list[str] = []
    channel_names: list[str] = []
    event_shapes: list[tuple[int, ...]] = []
    reactants_complete = True

    for reaction in reactions:
        complete = getattr(reaction, "reactants_complete", False)
        if not isinstance(complete, bool):
            msg = f"Reaction {reaction.name!r} reactants_complete must be boolean."
            raise TypeError(msg)
        if complete and getattr(reaction, "reactants", None) is None:
            msg = (
                f"Reaction {reaction.name!r} claims complete reactants without "
                "publishing reactant metadata."
            )
            raise ValueError(msg)
        reactants_complete = reactants_complete and complete
        from_axes = _require_string_tuple(reaction.from_axes, field="from_axes")
        to_axes = _require_string_tuple(reaction.to_axes, field="to_axes")
        sum_axes = _require_string_tuple(reaction.sum_axes, field="sum_axes")
        full_axes = base_axes[reaction.to_base]
        if not set(from_axes).issubset(full_axes):
            msg = f"Reaction {reaction.name!r} has axes outside full_axes."
            raise ValueError(msg)
        if not set(to_axes).issubset(full_axes):
            msg = f"Reaction {reaction.name!r} has destination axes outside full_axes."
            raise ValueError(msg)
        if set(sum_axes) != set(from_axes) - set(to_axes):
            msg = f"Reaction {reaction.name!r} has inconsistent sum_axes metadata."
            raise ValueError(msg)

        pinned = _require_pins(reaction.pinned, field="pinned")
        from_pinned = _require_pins(reaction.from_pinned, field="from_pinned")
        if set(from_axes) | set(from_pinned) != set(full_axes):
            msg = f"Reaction {reaction.name!r} has incomplete source-axis metadata."
            raise ValueError(msg)
        if set(to_axes) | set(pinned) != set(full_axes):
            msg = f"Reaction {reaction.name!r} has incomplete destination metadata."
            raise ValueError(msg)

        event_shape = tuple(axis_sizes[axis] for axis in from_axes)
        event_shapes.append(event_shape)
        coordinates = np.ndindex(event_shape) if event_shape else iter(((),))
        for coordinate in coordinates:
            varying = dict(zip(from_axes, coordinate, strict=True))
            column = np.zeros(n_state, dtype=np.int64)
            if reaction.from_base is not None:
                source_coordinates = {**varying, **from_pinned}
                source = _flat_cell(
                    blocks[reaction.from_base],
                    base_axes[reaction.from_base],
                    source_coordinates,
                )
                column[source] -= 1

            reactants = _expanded_reactants(
                reaction,
                varying=varying,
                from_axes=from_axes,
                base_axes=base_axes,
                blocks=blocks,
                n_state=n_state,
            )

            destination_coordinates = {**varying, **pinned}
            destination = _flat_cell(
                blocks[reaction.to_base],
                base_axes[reaction.to_base],
                destination_coordinates,
            )
            column[destination] += 1
            columns.append(column)
            reactant_columns.append(reactants)
            channel_reactions.append(reaction.name)
            coordinate_text = ",".join(
                f"{axis}={index}"
                for axis, index in zip(from_axes, coordinate, strict=True)
            )
            channel_names.append(
                reaction.name
                if not coordinate_text
                else f"{reaction.name}[{coordinate_text}]"
            )

    return CompiledReactionNetwork(
        stoichiometry=np.stack(columns, axis=1),
        reactant_stoichiometry=np.stack(reactant_columns, axis=1),
        channel_reactions=tuple(channel_reactions),
        channel_names=tuple(channel_names),
        _blocks=blocks_tuple,
        _reactions=reactions,
        _event_shapes=tuple(event_shapes),
        _params=dict(params),
        reactants_complete=reactants_complete,
    )


__all__ = [
    "CompiledReactionNetwork",
    "ReactionArtifact",
    "ReactionReactantArtifact",
    "compile_reaction_network",
]
