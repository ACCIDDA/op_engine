"""Compile op_system reaction artifacts into a flat stochastic network.

``op_system`` publishes one typed :class:`ReactionArtifact` per named
transition (``CompiledRhs.reactions``). :func:`compile_reaction_network`
expands each artifact into one channel per source cell and builds the
stoichiometry, reactant orders, and propensity callback that
:class:`~op_engine.DirectSSASolver`, :class:`~op_engine.TauLeapingSolver`,
and :class:`~op_engine.AdaptiveTauLeapingSolver` consume.
:func:`from_compiled_rhs` does the same from a compiled ``op_system`` RHS.

The module reads artifacts structurally and imports neither ``op_system``
nor ``flepimop2``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from ._typing import Array


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


class CompiledRhsLike(Protocol):
    """Structural surface of ``op_system.CompiledRhs`` read here."""

    reactions: Sequence[ReactionArtifact]
    template_shapes: Mapping[str, tuple[int, ...]] | None
    meta: Mapping[str, Any]
    reaction_gaps: Sequence[object]


class _IndexableArray(Protocol):
    """Internal slicing surface kept out of the public Array protocol."""

    def __getitem__(self, key: slice) -> Array:
        """Return an array slice."""


@dataclass(frozen=True, slots=True)
class _StateBlock:
    """One state template's location in the flat state."""

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
    #: True when every selected reaction's state dependence is fully
    #: described: by complete molecular reactants, or by dependencies and a
    #: propensity order. Adaptive tau-leaping requires it.
    reactants_complete: bool = False
    #: Names of selected reactions whose state dependence is not fully
    #: described.
    incomplete_reactions: tuple[str, ...] = ()
    #: ``(n_state, n_channels)`` 0/1 incidence of the cells each channel's
    #: propensity reads, for channels described by dependencies; ``None``
    #: when no channel is. Pass to ``AdaptiveTauLeapingSolver``.
    dependency_incidence: np.ndarray | None = None
    #: ``(n_channels,)`` elasticity bound of each dependency-described
    #: channel, zero for reactant-described ones; ``None`` when no channel
    #: is described by dependencies.
    propensity_orders: np.ndarray | None = None

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

        The stochastic core stores the state as ``(n_state, 1)``. A flat
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
    """Return the Array-API namespace ``value`` advertises.

    Unlike :func:`op_engine.array_namespace`, this does not wrap NumPy in a
    compatibility namespace: the propensity check compares a result's
    advertised namespace with the state's by identity.

    Raises:
        TypeError: If ``value`` does not advertise a namespace.
    """
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
            msg = "'template_shapes' keys must be strings."
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
            f"'template_shapes' describes {offset} state cells, but the state "
            f"contains {n_state}."
        )
        raise ValueError(msg)
    return tuple(blocks)


def _target_axes(reaction: ReactionArtifact) -> object:
    """Return the target template's axis order.

    ``full_axes`` is the source template's order. op_system #250 publishes
    ``to_full_axes`` because an axis-less source and a templated target (or
    the reverse) do not share it; older artifacts imply the shared order.

    Returns:
        ``to_full_axes`` when published, otherwise ``full_axes``.
    """
    to_full_axes = getattr(reaction, "to_full_axes", None)
    return reaction.full_axes if to_full_axes is None else to_full_axes


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
        if reaction.from_base is not None:
            register(
                reaction.from_base, reaction.full_axes, reaction_name=reaction.name
            )
        register(
            reaction.to_base,
            _target_axes(reaction),
            reaction_name=reaction.name,
        )
        for label, field in (("reactant", "reactants"), ("dependency", "dependencies")):
            raw_entries = getattr(reaction, field, ())
            if not isinstance(raw_entries, tuple | list):
                msg = f"Reaction {reaction.name!r} {field} must be a sequence."
                raise TypeError(msg)
            for entry in raw_entries:
                base = getattr(entry, "state_base", None)
                if not isinstance(base, str) or not base:
                    msg = (
                        f"Reaction {reaction.name!r} has an invalid {label} state base."
                    )
                    raise TypeError(msg)
                register(
                    base,
                    getattr(entry, "full_axes", None),
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
        Integer molecular orders aligned to the flat state.
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
    return _expanded_entries(
        reaction,
        raw_reactants,
        label="reactant",
        varying=varying,
        from_axes=from_axes,
        base_axes=base_axes,
        blocks=blocks,
        n_state=n_state,
    )


def _expanded_entries(  # noqa: PLR0913
    reaction: ReactionArtifact,
    entries: Sequence[object],
    *,
    label: str,
    varying: Mapping[str, int],
    from_axes: tuple[str, ...],
    base_axes: Mapping[str, tuple[str, ...]],
    blocks: Mapping[str, _StateBlock],
    n_state: int,
) -> np.ndarray:
    """Expand reactant-shaped entries into one channel's flat-state column.

    ``reactants`` and ``dependencies`` share this layout: each entry varies
    along some channel axes and pins the rest of its state's axes.

    Returns:
        Each entry's ``order`` summed into its cell.
    """
    result = np.zeros(n_state, dtype=np.int64)
    for entry in entries:
        base = getattr(entry, "state_base", None)
        if not isinstance(base, str) or base not in blocks:
            msg = f"Reaction {reaction.name!r} has an unknown {label} state."
            raise ValueError(msg)
        state_axes = _require_string_tuple(
            getattr(entry, "state_axes", None),
            field=f"{label}.state_axes",
        )
        full_axes = _require_string_tuple(
            getattr(entry, "full_axes", None),
            field=f"{label}.full_axes",
        )
        if full_axes != base_axes[base]:
            msg = (
                f"Reaction {reaction.name!r} {label} {base!r} has inconsistent "
                "full_axes metadata."
            )
            raise ValueError(msg)
        if not set(state_axes).issubset(from_axes):
            msg = (
                f"Reaction {reaction.name!r} {label} {base!r} has axes outside "
                "the expanded reaction channels."
            )
            raise ValueError(msg)
        pins = _require_pins(
            getattr(entry, "pinned", None),
            field=f"{label}.pinned",
        )
        if set(state_axes) | set(pins) != set(full_axes):
            msg = (
                f"Reaction {reaction.name!r} {label} {base!r} has incomplete "
                "axis metadata."
            )
            raise ValueError(msg)
        if set(state_axes) & set(pins):
            msg = (
                f"Reaction {reaction.name!r} {label} {base!r} both varies and "
                "pins an axis."
            )
            raise ValueError(msg)
        order = getattr(entry, "order", None)
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            msg = (
                f"Reaction {reaction.name!r} {label} {base!r} order must be a "
                "positive integer."
            )
            raise TypeError(msg)
        coordinates = {axis: varying[axis] for axis in state_axes} | pins
        cell = _flat_cell(blocks[base], full_axes, coordinates)
        result[cell] += order
    return result


def _dependency_order(reaction: ReactionArtifact) -> int:
    """Return the elasticity order of a reaction described by dependencies.

    op_system 0.7.0+ publishes ``dependencies``, ``propensity_order``, and
    ``dependencies_complete`` for ``reactants: auto`` rates that are not
    mass action. Older artifacts publish none of them.

    Returns:
        The positive ``propensity_order`` when ``dependencies_complete`` is
        true, otherwise zero.
    """
    complete = getattr(reaction, "dependencies_complete", False)
    if not isinstance(complete, bool):
        msg = f"Reaction {reaction.name!r} dependencies_complete must be boolean."
        raise TypeError(msg)
    if not complete:
        return 0
    order = getattr(reaction, "propensity_order", None)
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        msg = (
            f"Reaction {reaction.name!r} claims complete dependencies without a "
            "positive propensity_order."
        )
        raise TypeError(msg)
    return order


def _flat_cell(
    block: _StateBlock,
    axes: tuple[str, ...],
    coordinates: Mapping[str, int],
) -> int:
    """Map named cell coordinates into the flat state.

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


def _reaction_axis_metadata(
    reaction: ReactionArtifact,
    *,
    from_axes: tuple[str, ...],
    source_axes: tuple[str, ...],
    target_axes: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], tuple[str, ...]]:
    """Validate one reaction's channel, destination, offset, and routed axes.

    Every destination axis is copied from the channel (``to_axes``), fixed
    (``pinned``), shifted from the channel coordinate (``offsets``, from
    op_system's axis-wide ``coord_shift``), or read from a trailing channel
    dimension (``routed_axes``, from routing and fan-out transitions).
    Artifacts that predate ``offsets`` or ``routed_axes`` publish none.
    Channel and donor axes are checked against the source template (the
    destination template for a source-only reaction), and destination axes
    against the target template.

    Returns:
        ``(pinned, from_pinned, offsets, routed_axes)``.

    Raises:
        ValueError: If the axis metadata is inconsistent.
    """
    to_axes = _require_string_tuple(reaction.to_axes, field="to_axes")
    sum_axes = _require_string_tuple(reaction.sum_axes, field="sum_axes")
    pinned = _require_pins(reaction.pinned, field="pinned")
    from_pinned = _require_pins(reaction.from_pinned, field="from_pinned")
    offsets = _require_pins(getattr(reaction, "offsets", ()), field="offsets")
    routed = _require_string_tuple(
        getattr(reaction, "routed_axes", ()), field="routed_axes"
    )
    if not set(from_axes).issubset(source_axes):
        msg = f"Reaction {reaction.name!r} has axes outside full_axes."
        raise ValueError(msg)
    if not set(to_axes).issubset(target_axes):
        msg = f"Reaction {reaction.name!r} has destination axes outside full_axes."
        raise ValueError(msg)
    _validate_moved_axes(
        reaction,
        from_axes=from_axes,
        target_axes=target_axes,
        fixed=set(to_axes) | set(pinned),
        offsets=offsets,
        routed=routed,
    )
    if set(sum_axes) != set(from_axes) - set(to_axes) - set(offsets):
        msg = f"Reaction {reaction.name!r} has inconsistent sum_axes metadata."
        raise ValueError(msg)
    # Source-only channels use destination wildcard axes and have no
    # donor pins. Validate donor coverage only when a donor exists.
    if reaction.from_base is not None and (
        set(from_axes) | set(from_pinned) != set(source_axes)
    ):
        msg = f"Reaction {reaction.name!r} has incomplete source-axis metadata."
        raise ValueError(msg)
    if set(to_axes) | set(pinned) | set(offsets) | set(routed) != set(target_axes):
        msg = f"Reaction {reaction.name!r} has incomplete destination metadata."
        raise ValueError(msg)
    return pinned, from_pinned, offsets, routed


def _validate_moved_axes(  # noqa: PLR0913
    reaction: ReactionArtifact,
    *,
    from_axes: tuple[str, ...],
    target_axes: tuple[str, ...],
    fixed: set[str],
    offsets: Mapping[str, int],
    routed: tuple[str, ...],
) -> None:
    """Check offset and routed destination axes against the other fields.

    Raises:
        ValueError: If an offset does not shift a channel axis by a nonzero
            step, or a routed axis is not a free target axis.
    """
    if (
        not set(offsets).issubset(from_axes)
        or set(offsets) & fixed
        or 0 in offsets.values()
    ):
        msg = (
            f"Reaction {reaction.name!r} offsets must shift channel axes that "
            "are neither copied nor pinned, by a nonzero step."
        )
        raise ValueError(msg)
    if not set(routed).issubset(target_axes) or set(routed) & (fixed | set(offsets)):
        msg = (
            f"Reaction {reaction.name!r} routed axes must be target axes that "
            "are neither copied, pinned, nor offset."
        )
        raise ValueError(msg)


def _destination_cell(
    block: _StateBlock,
    axes: tuple[str, ...],
    coordinates: Mapping[str, int],
    *,
    offsets: Mapping[str, int],
) -> int | None:
    """Locate a firing's destination, shifting each offset axis by its step.

    Returns:
        Flat state index, or ``None`` when a shifted coordinate leaves its
        axis: the firing then removes the donor unit without a deposit.
    """
    shifted = dict(coordinates)
    for axis, step in offsets.items():
        shifted[axis] += step
        if not 0 <= shifted[axis] < block.shape[axes.index(axis)]:
            return None
    return _flat_cell(block, axes, shifted)


def compile_reaction_network(  # noqa: PLR0913
    reactions: Sequence[ReactionArtifact],
    *,
    template_shapes: Mapping[str, Sequence[int]],
    axis_sizes: Mapping[str, int],
    params: Mapping[str, object],
    n_state: int | None = None,
    reaction_names: Sequence[str] | None = None,
) -> CompiledReactionNetwork:
    """Compile op_system reaction artifacts into flat channels.

    Each source-cell propensity becomes one channel. Consequently, a collapsed
    axis is represented by distinct columns that share one destination row;
    summing simultaneous firings is then exactly the stoichiometric matrix
    multiplication performed by the core stochastic solvers. An offset axis
    moves each channel to its shifted coordinate; when that leaves the axis,
    the column only removes the donor.

    Args:
        reactions: Typed reaction artifacts, such as ``CompiledRhs.reactions``.
        template_shapes: Shape of each state template, in the flat state's
            order, such as ``CompiledRhs.template_shapes``.
        axis_sizes: Number of coordinates on each axis.
        params: Parameter values passed to every propensity.
        n_state: Number of cells in the flat state. Defaults to the total
            that ``template_shapes`` describes; when given, the two must
            agree.
        reaction_names: Optional parent reaction names to select for a hybrid
            jump partition. ``None`` selects every artifact.

    Returns:
        Validated flat reaction network.

    Raises:
        TypeError: If an argument or artifact field has the wrong type.
        ValueError: If the artifacts are inconsistent with each other or
            with the state layout.
    """
    if not isinstance(reactions, tuple | list):
        msg = "'reactions' must be a sequence of reaction artifacts."
        raise TypeError(msg)
    selected = tuple(cast("ReactionArtifact", item) for item in reactions)
    if reaction_names is not None:
        requested = tuple(reaction_names)
        if len(requested) != len(set(requested)):
            msg = "stochastic_reactions must not contain duplicates."
            raise ValueError(msg)
        available = {reaction.name for reaction in selected}
        missing = sorted(set(requested) - available)
        if missing:
            msg = f"Unknown stochastic reaction names: {', '.join(missing)}."
            raise ValueError(msg)
        wanted = set(requested)
        selected = tuple(reaction for reaction in selected if reaction.name in wanted)
    if not selected:
        msg = "No typed reactions are available for stochastic execution."
        raise ValueError(msg)
    reactions = selected

    if not isinstance(template_shapes, Mapping):
        msg = "'template_shapes' must map state template names to shapes."
        raise TypeError(msg)
    if n_state is None:
        n_state = sum(
            int(np.prod(tuple(shape), dtype=np.int64)) if shape else 1
            for shape in template_shapes.values()
            if isinstance(shape, tuple | list)
        )
    blocks_tuple = _state_blocks(
        cast("Mapping[object, object]", template_shapes), n_state=n_state
    )
    blocks = {block.base: block for block in blocks_tuple}

    if not isinstance(axis_sizes, Mapping):
        msg = "'axis_sizes' must be a mapping."
        raise TypeError(msg)
    raw_axis_sizes = axis_sizes
    axis_sizes = {}
    for axis, size in raw_axis_sizes.items():
        if (
            not isinstance(axis, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
        ):
            msg = "'axis_sizes' contains an invalid entry."
            raise TypeError(msg)
        axis_sizes[axis] = size
    base_axes = _reaction_axes(reactions, blocks, axis_sizes)

    columns: list[np.ndarray] = []
    reactant_columns: list[np.ndarray] = []
    dependency_columns: list[np.ndarray] = []
    channel_orders: list[int] = []
    channel_reactions: list[str] = []
    channel_names: list[str] = []
    event_shapes: list[tuple[int, ...]] = []
    reactants_complete = True
    incomplete: list[str] = []

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
        # Complete reactants take precedence; otherwise dependencies and a
        # propensity order can describe the reaction instead.
        order = 0 if complete else _dependency_order(reaction)
        covered = complete or order > 0
        reactants_complete = reactants_complete and covered
        if not covered:
            incomplete.append(reaction.name)
        from_axes = _require_string_tuple(reaction.from_axes, field="from_axes")
        target_axes = base_axes[reaction.to_base]
        pinned, from_pinned, offsets, routed = _reaction_axis_metadata(
            reaction,
            from_axes=from_axes,
            source_axes=(
                target_axes
                if reaction.from_base is None
                else base_axes[reaction.from_base]
            ),
            target_axes=target_axes,
        )

        # A routed target coordinate is a trailing channel dimension.
        event_shape = tuple(axis_sizes[axis] for axis in from_axes + routed)
        event_shapes.append(event_shape)
        coordinates = np.ndindex(event_shape) if event_shape else iter(((),))
        for coordinate in coordinates:
            varying = dict(zip(from_axes, coordinate[: len(from_axes)], strict=True))
            routed_to = dict(zip(routed, coordinate[len(from_axes) :], strict=True))
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

            read = np.zeros(n_state, dtype=np.int64)
            if order > 0:
                read = _expanded_entries(
                    reaction,
                    getattr(reaction, "dependencies", ()),
                    label="dependency",
                    varying=varying,
                    from_axes=from_axes,
                    base_axes=base_axes,
                    blocks=blocks,
                    n_state=n_state,
                )
                # A consumed species is read too, whether or not listed.
                read = ((read > 0) | (reactants > 0)).astype(np.int64)

            destination = _destination_cell(
                blocks[reaction.to_base],
                base_axes[reaction.to_base],
                {**varying, **pinned, **routed_to},
                offsets=offsets,
            )
            if destination is not None:
                column[destination] += 1
            columns.append(column)
            reactant_columns.append(reactants)
            dependency_columns.append(read)
            channel_orders.append(order)
            channel_reactions.append(reaction.name)
            coordinate_text = ",".join(
                [f"{axis}={index}" for axis, index in varying.items()]
                + [f"to:{axis}={index}" for axis, index in routed_to.items()]
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
        incomplete_reactions=tuple(incomplete),
        dependency_incidence=(
            np.stack(dependency_columns, axis=1) if any(channel_orders) else None
        ),
        propensity_orders=(
            np.asarray(channel_orders, dtype=np.int64) if any(channel_orders) else None
        ),
    )


def compiled_rhs_axis_sizes(compiled: CompiledRhsLike) -> dict[str, int]:
    """Return each axis's coordinate count from a compiled RHS's metadata.

    Returns:
        Mapping from axis name to size, from ``compiled.meta["axes"]``.

    Raises:
        TypeError: If an axis entry has no name.
    """
    sizes: dict[str, int] = {}
    for axis in compiled.meta.get("axes", ()) or ():
        name = axis.get("name") if isinstance(axis, Mapping) else None
        if not isinstance(name, str):
            msg = "compiled.meta['axes'] entries must name their axis."
            raise TypeError(msg)
        size = axis.get("size") or len(axis.get("coords") or ())
        sizes[name] = int(size)
    return sizes


def from_compiled_rhs(
    compiled: CompiledRhsLike,
    params: Mapping[str, object],
    *,
    reaction_names: Sequence[str] | None = None,
    allow_gaps: bool = False,
) -> CompiledReactionNetwork:
    """Compile the reaction network of a compiled ``op_system`` RHS.

    The flat state follows ``compiled.template_shapes``: each template's
    cells in C order, templates in declaration order. That is also the
    layout of ``compiled.eval_fn``'s state vector when the RHS is
    vectorized, so ``network.mean_drift(t, y)`` should equal
    ``compiled.eval_fn(t, y, **params)`` for every ``y``. Check that
    identity once for a new model: a mismatch means some dynamics have no
    reaction.

    Args:
        compiled: A compiled RHS, such as ``op_system.compile_spec(spec)``.
        params: Parameter values passed to every propensity.
        reaction_names: Optional reaction names to compile, for a hybrid
            jump partition. ``None`` compiles every reaction.
        allow_gaps: Compile even when ``compiled.reaction_gaps`` lists
            dynamics without a reaction. Selecting ``reaction_names`` also
            skips the check, since a partition is partial by design.

    Returns:
        Validated flat reaction network.

    Raises:
        TypeError: If the RHS has no vectorized state layout.
        ValueError: If the RHS has reaction gaps and neither ``allow_gaps``
            nor ``reaction_names`` is given.
    """
    template_shapes = compiled.template_shapes
    if template_shapes is None:
        msg = (
            "The compiled RHS has no vectorized state layout "
            "(template_shapes is None), so its reactions cannot be indexed."
        )
        raise TypeError(msg)
    gaps = tuple(compiled.reaction_gaps)
    if gaps and not allow_gaps and reaction_names is None:
        described = ", ".join(
            f"{getattr(gap, 'name', None) or getattr(gap, 'origin', '?')} "
            f"({getattr(gap, 'reason', 'unknown')})"
            for gap in gaps
        )
        msg = (
            "The compiled RHS has dynamics without a reaction artifact: "
            f"{described}. A network of its reactions alone would drop them; "
            "pass allow_gaps=True to compile it anyway."
        )
        raise ValueError(msg)
    return compile_reaction_network(
        tuple(compiled.reactions),
        template_shapes=template_shapes,
        axis_sizes=compiled_rhs_axis_sizes(compiled),
        params=params,
        reaction_names=reaction_names,
    )


__all__ = [
    "CompiledReactionNetwork",
    "CompiledRhsLike",
    "ReactionArtifact",
    "ReactionReactantArtifact",
    "compile_reaction_network",
    "compiled_rhs_axis_sizes",
    "from_compiled_rhs",
]
