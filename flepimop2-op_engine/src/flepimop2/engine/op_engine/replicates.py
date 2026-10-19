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

"""Preserve provider state ordering while executing independent-clock banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from op_engine import (
    BatchedDirectSSASolver,
    BatchedSSAConfig,
    FiringRecorder,
    ModelCore,
    from_compiled_replicates,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from flepimop2.system.abc import SystemABC
    from flepimop2.typing import Array

    from op_engine.batched_ssa import BankStreams
    from op_engine.reactions import ReactionArtifact

    from .config import OpEngineEngineConfig


@dataclass
class _ReplicateRhs:
    reactions: Sequence[ReactionArtifact]
    template_shapes: Mapping[str, tuple[int, ...]] | None
    meta: Mapping[str, Any]
    reaction_gaps: Sequence[object]
    block_axes: tuple[Any, ...]


def run_replicate_bank(  # noqa: PLR0913
    system: SystemABC,
    times: np.ndarray,
    y0: Array,
    params: Mapping[str, object],
    *,
    config: OpEngineEngineConfig,
    kwargs: Mapping[str, object],
) -> Array:
    """Execute the provider's independent exact-SSA path.

    Returns:
        Flat provider state history in the compiler's original cell order.

    Raises:
        TypeError: The state or firing recorder has an unsupported type.
        ValueError: A flattened sampler is supplied for independent streams.
    """
    if not isinstance(y0, np.ndarray):
        msg = "replicate-direct-ssa currently requires NumPy state"
        raise TypeError(msg)
    if "ssa_sampler" in kwargs or "poisson_sampler" in kwargs:
        msg = "replicate-direct-ssa uses replicate_rng streams, not flattened samplers"
        raise ValueError(msg)
    sizes = cast("Mapping[str, int]", system.option("axis_sizes", {}))
    meta = {
        "axes": [
            {"name": name, "coords": tuple(range(size))} for name, size in sizes.items()
        ],
        "operators": system.option("operators", ()),
        "time_axis": system.option("time_axis", "time"),
        "time_interpolation": system.option("time_interpolation", "linear"),
        "time_coordinates": system.option("time_coordinates", ()),
    }
    rhs = _ReplicateRhs(
        cast("Sequence[ReactionArtifact]", system.option("reactions", ())),
        cast(
            "Mapping[str, tuple[int, ...]] | None",
            system.option("template_shapes", None),
        ),
        meta,
        cast("Sequence[object]", system.option("reaction_gaps", ())),
        cast("tuple[Any, ...]", system.option("block_axes", ())),
    )
    network = from_compiled_replicates(
        rhs, params, batch_axis=cast("str", config.replicate_axis)
    )
    initial = network.pack_state(y0)
    core = ModelCore(initial.shape[0], initial.shape[1], times)
    core.set_initial_state(cast("Array", initial))
    recorder = kwargs.get("firing_recorder")
    if recorder is not None and not isinstance(recorder, FiringRecorder):
        msg = "firing_recorder must be a FiringRecorder"
        raise TypeError(msg)
    streams = cast(
        "BankStreams",
        kwargs.get("replicate_rng", np.random.SeedSequence(config.random_seed)),
    )
    BatchedDirectSSASolver(core, network.stoichiometry).run(
        network.propensity,
        rng=streams,
        replicate_ids=cast("Sequence[int] | None", kwargs.get("replicate_ids")),
        config=BatchedSSAConfig(
            max_events=config.ssa_max_events,
            forcing_breakpoints=tuple(
                sorted({*config.forcing_breakpoints, *network.forcing_breakpoints})
            ),
            replicate_breakpoints=config.replicate_breakpoints,
        ),
        recorder=recorder,
    )
    return cast("Array", network.flatten_history(np.asarray(core.state_array)))
