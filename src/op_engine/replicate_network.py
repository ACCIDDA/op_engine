"""Structural op_system adapter preserving a local stochastic network per bank.

No compiler/framework import is required. Time tables are resolved once per
callback using diagonal (time, replicate) gathers rather than a Cartesian batch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from .reactions import compile_reaction_network, compiled_rhs_axis_sizes

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .batched_ssa import FloatBank
    from .reactions import CompiledRhsLike


class _IndependentReaction(Protocol):
    name: str
    from_axes: tuple[str, ...]
    resolved_propensity_fn: Callable[..., object]


@dataclass(frozen=True, slots=True)
class ReplicateReactionNetwork:
    """One local stoichiometry plus a vectorized, independent propensity bank.

    Local state/channel ordering follows the existing flat adapter after removing
    the declared replicate axis. Full input parameter arrays retain declared axis
    order. Parameters are read on every call; changing their values is allowed
    between runs and no interpolation result is cached.
    """

    stoichiometry: np.ndarray
    channel_names: tuple[str, ...]
    channel_reactions: tuple[str, ...]
    n_replicates: int
    forcing_breakpoints: tuple[float, ...]
    _batch_axis: str
    _states: tuple[tuple[str, tuple[int, ...], int], ...]
    _reactions: tuple[_IndependentReaction, ...]
    _params: Mapping[str, object]
    _time_axis: str
    _time_coordinates: np.ndarray
    _table_axes: Mapping[str, tuple[str, ...]]
    _axis_sizes: Mapping[str, int]

    def pack_state(self, flat_state: FloatBank) -> FloatBank:
        """Convert compiler flat state order into (local species, replicate).

        Returns:
            A local state bank preserving each template's declared axis order.

        Raises:
            ValueError: The flat state has an incompatible size.
        """
        expected = self.stoichiometry.shape[0] * self.n_replicates
        if flat_state.shape != (expected,):
            msg = f"Expected flat compiler state {(expected,)}"
            raise ValueError(msg)
        pieces = []
        offset = 0
        for _name, shape, position in self._states:
            size = int(np.prod(shape))
            values = flat_state[offset : offset + size].reshape(shape)
            pieces.append(
                np.moveaxis(values, position, -1).reshape((-1, self.n_replicates))
            )
            offset += size
        return np.concatenate(pieces, axis=0)

    def flatten_history(self, history: FloatBank) -> FloatBank:
        """Restore compiler flat cell order from a local bank history.

        Returns:
            History shaped (output time, full compiler state).

        Raises:
            ValueError: The bank history has an incompatible layout.
        """
        if history.ndim != 3 or history.shape[1:] != (
            self.stoichiometry.shape[0],
            self.n_replicates,
        ):
            msg = "Expected history (output time, local species, replicate)"
            raise ValueError(msg)
        pieces = []
        offset = 0
        for _name, shape, position in self._states:
            local_shape = tuple(n for i, n in enumerate(shape) if i != position)
            size = int(np.prod(local_shape))
            values = history[:, offset : offset + size].reshape((
                len(history),
                *local_shape,
                self.n_replicates,
            ))
            pieces.append(
                np.moveaxis(values, -1, position + 1).reshape((len(history), -1))
            )
            offset += size
        return np.concatenate(pieces, axis=1)

    def propensity(self, times: FloatBank, state: FloatBank) -> FloatBank:
        """Evaluate one hazard bank with every table resolved at local clocks.

        Returns:
            NumPy propensities shaped (local channel, replicate).

        Raises:
            ValueError: State, clocks, parameter tables or reaction outputs have
                incompatible shapes.
        """
        expected = (self.stoichiometry.shape[0], self.n_replicates)
        if state.shape != expected or times.shape != (self.n_replicates,):
            msg = f"Expected bank state {expected} and one clock per replicate"
            raise ValueError(msg)
        states = {}
        offset = 0
        for name, shape, position in self._states:
            local_shape = tuple(n for i, n in enumerate(shape) if i != position)
            size = int(np.prod(local_shape))
            state_values = state[offset : offset + size].reshape((
                *local_shape,
                self.n_replicates,
            ))
            states[name] = np.moveaxis(state_values, -1, position)
            offset += size
        params = self._resolve_tables(times)
        values = []
        for reaction in self._reactions:
            fn = reaction.resolved_propensity_fn
            raw = np.asarray(fn(times, states, **params))
            axes = reaction.from_axes + tuple(getattr(reaction, "routed_axes", ()))
            expected_shape = tuple(self._axis_sizes[a] for a in axes)
            if raw.shape != expected_shape:
                msg = (
                    f"Reaction {reaction.name!r} expected propensity {expected_shape}, "
                    f"received {raw.shape}"
                )
                raise ValueError(msg)
            values.append(
                np.moveaxis(raw, axes.index(self._batch_axis), -1).reshape((
                    -1,
                    self.n_replicates,
                ))
            )
        return np.concatenate(values, axis=0)

    def _resolve_tables(self, times: FloatBank) -> dict[str, object]:
        """Resolve each consumed previous table once for the current clock bank.

        Returns:
            Parameters with tables replaced by current per-replicate values.

        Raises:
            ValueError: A table does not match its declared axes.
        """
        params = dict(self._params)
        for name, axes in self._table_axes.items():
            table = np.asarray(params[name])
            if table.shape != tuple(self._axis_sizes[a] for a in axes):
                msg = f"Time parameter {name!r} must have declared axes {axes}"
                raise ValueError(msg)
            time_position = axes.index(self._time_axis)
            front = np.moveaxis(table, time_position, 0)
            reduced_axes = tuple(a for a in axes if a != self._time_axis)
            indices = np.clip(
                np.searchsorted(self._time_coordinates, times, side="right") - 1,
                0,
                len(self._time_coordinates) - 1,
            )
            batch_position = reduced_axes.index(self._batch_axis)
            front = np.moveaxis(front, batch_position + 1, -1)
            gather = indices.reshape((1,) * (front.ndim - 1) + (self.n_replicates,))
            value = np.take_along_axis(front, gather, axis=0)[0]
            params[name] = np.moveaxis(value, -1, batch_position)
        return params


def from_compiled_replicates(  # noqa: C901
    compiled: CompiledRhsLike,
    params: Mapping[str, object],
    *,
    batch_axis: str,
) -> ReplicateReactionNetwork:
    """Build a local network from compiler-certified independent reactions.

    Requires the independent-reaction contract from op_system 0.8.0+. Every
    evolving state must carry the batch axis. Operators, gaps, explicit-time
    hazards and nonconstant linear tables are rejected. Time-varying parameters
    must explicitly carry the batch axis to represent asynchronous current values.

    Returns:
        Local network for BatchedDirectSSASolver.

    Raises:
        ValueError: Independence, completeness or time assumptions are unmet.
        TypeError: Required compiler artifacts are unavailable.
    """
    if compiled.reaction_gaps or compiled.meta.get("operators"):
        msg = (
            "Independent SSA requires complete reactions and no unrepresented operators"
        )
        raise ValueError(msg)
    shapes = compiled.template_shapes
    if not shapes or not compiled.reactions:
        msg = "Independent SSA requires compiled reaction and template metadata"
        raise TypeError(msg)
    blocks = tuple(getattr(compiled, "block_axes", ()))
    block = next((b for b in blocks if b.name == batch_axis), None)
    if block is None or set(block.state_axis_pos) != set(shapes):
        msg = f"Every state must carry certified factorize axis {batch_axis!r}"
        raise ValueError(msg)
    sizes = compiled_rhs_axis_sizes(compiled)
    time_axis = str(compiled.meta.get("time_axis", "time"))
    coordinates = np.asarray(compiled.meta.get("time_coordinates", ()), dtype=float)
    local_shapes = {
        name: tuple(n for i, n in enumerate(shape) if i != block.state_axis_pos[name])
        for name, shape in shapes.items()
    }
    local_reactions = []
    tables: dict[str, tuple[str, ...]] = {}
    for reaction in compiled.reactions:
        if batch_axis not in getattr(reaction, "independent_axes", ()):
            msg = (
                f"Reaction {reaction.name!r} is not certified independent "
                f"on {batch_axis!r}"
            )
            raise ValueError(msg)
        if getattr(reaction, "uses_time", True):
            msg = (
                f"Reaction {reaction.name!r} has explicit or "
                "unqualified time dependence"
            )
            raise ValueError(msg)
        if not callable(getattr(reaction, "resolved_propensity_fn", None)):
            msg = (
                "Independent SSA requires op_system's resolved "
                "propensity contract (0.8.0+)"
            )
            raise TypeError(msg)
        tables.update(dict(getattr(reaction, "time_varying_params", ())))
        fields: dict[str, Any] = {
            name: tuple(a for a in getattr(reaction, name) if a != batch_axis)
            for name in ("from_axes", "full_axes", "to_axes", "sum_axes", "routed_axes")
        }
        destination_axes = getattr(reaction, "to_full_axes", None)
        if destination_axes is not None:
            fields["to_full_axes"] = tuple(
                a for a in destination_axes if a != batch_axis
            )
        fields.update(
            reactants=(),
            reactants_complete=False,
            dependencies=(),
            dependencies_complete=False,
            propensity_order=None,
        )
        local_reactions.append(replace(cast("Any", reaction), **fields))
    if tables and (
        coordinates.size == 0
        or (
            coordinates.size > 1
            and compiled.meta.get("time_interpolation") != "previous"
        )
    ):
        msg = (
            "Independent exact SSA requires previous interpolation "
            "or a constant one-knot table"
        )
        raise ValueError(msg)
    for name, axes in tables.items():
        if batch_axis not in axes:
            msg = f"Time parameter {name!r} must declare replicate axis {batch_axis!r}"
            raise ValueError(msg)
    local = compile_reaction_network(
        tuple(local_reactions),
        template_shapes=local_shapes,
        axis_sizes={a: n for a, n in sizes.items() if a != batch_axis},
        params={},
    )
    return ReplicateReactionNetwork(
        local.stoichiometry,
        local.channel_names,
        local.channel_reactions,
        int(sizes[batch_axis]),
        tuple(float(t) for t in coordinates[1:]) if tables else (),
        batch_axis,
        tuple(
            (name, tuple(shape), block.state_axis_pos[name])
            for name, shape in shapes.items()
        ),
        tuple(cast("_IndependentReaction", r) for r in compiled.reactions),
        dict(params),
        time_axis,
        coordinates,
        tables,
        sizes,
    )
