"""Independent-clock exact SSA for NumPy simulation banks.

State and propensities are (local species/channel, replicate). The callback
receives one clock per replicate. Conditional on supplied parameters/forcing,
replicates must be independent. This eager CPU method is not differentiable.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from ._forcing import _ForcingSchedule
from .stochastic_solver import DirectSSAConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .firing import FiringRecorder
    from .model_core import ModelCore

FloatBank: TypeAlias = "NDArray[np.floating[Any]]"
BatchPropensity: TypeAlias = Callable[[FloatBank, FloatBank], FloatBank]
BankStreams: TypeAlias = (
    np.random.SeedSequence | Sequence[np.random.Generator | np.random.SeedSequence]
)


@dataclass(frozen=True, slots=True)
class BatchedSSAConfig:
    """Exact event limits and shared/per-replicate forcing declarations.

    ``max_events`` applies to each replicate in each output interval. Local
    schedules follow the input replicate order and are unioned with the shared
    schedule. Output times do not change the forcing or resample pending events.
    """

    max_events: int = 1_000_000
    forcing_breakpoints: tuple[float, ...] = ()
    replicate_breakpoints: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate and snapshot all declarations before executing callbacks."""
        shared = DirectSSAConfig(self.max_events, self.forcing_breakpoints)
        object.__setattr__(self, "forcing_breakpoints", shared.forcing_breakpoints)
        local = tuple(
            _ForcingSchedule(row).breakpoints for row in self.replicate_breakpoints
        )
        object.__setattr__(self, "replicate_breakpoints", local)


def _replicate_streams(
    rng: BankStreams, ids: tuple[int, ...]
) -> list[np.random.Generator]:
    """Lower indexed SeedSequence addresses or explicitly owned native streams.

    Returns:
        One mutable generator per replicate, independent of bank order/length.

    Raises:
        ValueError: Stream counts disagree or the same mutable generator is reused.
        TypeError: An unsupported random source is supplied.
    """
    if isinstance(rng, np.random.SeedSequence):
        sources: Sequence[np.random.Generator | np.random.SeedSequence] = [
            np.random.SeedSequence(
                rng.entropy,
                spawn_key=(*rng.spawn_key, 0x4F504231, i & 0xFFFFFFFF, i >> 32),
                pool_size=rng.pool_size,
            )
            for i in ids
        ]
    elif isinstance(rng, (tuple, list)):
        sources = rng
    else:
        msg = (
            "Batch rng must be a SeedSequence or one "
            "Generator/SeedSequence per replicate"
        )
        raise TypeError(msg)
    if len(sources) != len(ids):
        msg = "Batch rng requires one stream per replicate"
        raise ValueError(msg)
    generators = []
    for source in sources:
        if isinstance(source, np.random.Generator):
            generators.append(source)
        elif isinstance(source, np.random.SeedSequence):
            generators.append(np.random.Generator(np.random.PCG64DXSM(source)))
        else:
            msg = "Each replicate requires a NumPy Generator or SeedSequence"
            raise TypeError(msg)
    fingerprints = [
        json.dumps(
            g.bit_generator.state, sort_keys=True, default=lambda value: value.tolist()
        )
        for g in generators
    ]
    if len(set(fingerprints)) != len(fingerprints):
        msg = "Replicates require distinct random stream states, not duplicate seeds"
        raise ValueError(msg)
    return generators


class BatchedDirectSSASolver:
    """Advance independent replicates with local clocks and one local network.

    A callback taking ``(times, state)`` returns ``(channel, replicate)`` hazards.
    It must use each replicate's own time, preserve the state, and remain constant
    between jumps and declared forcing boundaries. Generic callbacks assert that
    independence themselves; the compiled-model adapter checks it structurally.

    Examples:
        >>> import numpy as np
        >>> from op_engine import FiringRecorder, ModelCore
        >>> core = ModelCore(2, 3, np.array([0.0, 1.0, 2.0]))
        >>> core.set_initial_state(np.array([[10.0] * 3, [0.0] * 3]))
        >>> counts = FiringRecorder((0,))
        >>> solver = BatchedDirectSSASolver(core, np.array([[-1.0], [1.0]]))
        >>> def hazards(times, state):
        ...     return 0.5 * state[:1]
        >>> solver.run(hazards, rng=np.random.SeedSequence(7), recorder=counts)
        >>> core.state_array.shape, counts.counts.shape
        ((3, 2, 3), (2, 1, 3))
        >>> bool(np.all(core.state_array.sum(axis=1) == 10.0))
        True
    """

    def __init__(self, core: ModelCore, stoichiometry: FloatBank) -> None:
        """Validate a NumPy local stoichiometric matrix against the bank state.

        Raises:
            ValueError: The local network or core layout is invalid.
            TypeError: State or stoichiometry is not a NumPy array.
        """
        state = core.get_current_state()
        if not isinstance(state, np.ndarray) or not isinstance(
            stoichiometry, np.ndarray
        ):
            msg = "Batched direct SSA currently requires NumPy state and stoichiometry"
            raise TypeError(msg)
        if (
            state.ndim != 2
            or stoichiometry.ndim != 2
            or stoichiometry.shape[0] != state.shape[0]
            or not state.shape[1]
            or not stoichiometry.shape[1]
        ):
            msg = (
                "Expected state (local species, replicate) and "
                "stoichiometry (local species, channel)"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(stoichiometry)) or np.any(
            stoichiometry != np.round(stoichiometry)
        ):
            msg = "Stoichiometry must contain finite integers"
            raise ValueError(msg)
        self.core = core
        self.stoichiometry = stoichiometry.copy()
        self.sweeps = 0
        self.events = 0

    def run(  # noqa: C901, PLR0912, PLR0914, PLR0915 - one independent-clock event loop
        self,
        propensity_func: BatchPropensity,
        *,
        rng: BankStreams,
        replicate_ids: Sequence[int] | None = None,
        config: BatchedSSAConfig | None = None,
        recorder: FiringRecorder | None = None,
    ) -> None:
        """Store bank trajectories and optional counts at the core's output times.

        A root SeedSequence is framed by stable unsigned 64-bit replicate IDs
        using scheme OPB1 and native PCG64DXSM. Explicit generators advance in
        place. Replay is bank/order/shard invariant for identical inputs and
        schedules; changing an actual forcing declaration can change paths.

        Raises:
            ValueError: IDs, state, schedules or callback hazards are invalid.
            RuntimeError: A population becomes invalid, time stagnates or a
                per-replicate interval event limit is exceeded.
        """
        state = np.asarray(self.core.get_current_state()).copy()
        _n_species, n_replicates = state.shape
        n_channels = self.stoichiometry.shape[1]
        ids = (
            tuple(range(n_replicates))
            if replicate_ids is None
            else tuple(replicate_ids)
        )
        if (
            len(ids) != n_replicates
            or len(set(ids)) != len(ids)
            or any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < 2**64
                for i in ids
            )
        ):
            msg = (
                "replicate_ids must be distinct unsigned 64-bit integers, "
                "one per replicate"
            )
            raise ValueError(msg)
        streams = _replicate_streams(rng, tuple(int(i) for i in ids))
        cfg = config or BatchedSSAConfig()
        if cfg.replicate_breakpoints and len(cfg.replicate_breakpoints) != n_replicates:
            msg = "replicate_breakpoints requires one schedule per replicate"
            raise ValueError(msg)
        if (
            np.any(~np.isfinite(state))
            or np.any(state < 0)
            or np.any(state != np.round(state))
        ):
            msg = "Initial counts must be finite non-negative integers"
            raise ValueError(msg)
        local = cfg.replicate_breakpoints or ((),) * n_replicates
        schedules = [
            np.asarray(sorted({*cfg.forcing_breakpoints, *row}), dtype=float)
            for row in local
        ]
        times = np.full(n_replicates, self.core.get_time_at(0), dtype=float)
        next_forcing = np.array([
            row[np.searchsorted(row, times[i], side="right")]
            if np.searchsorted(row, times[i], side="right") < len(row)
            else np.inf
            for i, row in enumerate(schedules)
        ])
        pending = np.full(n_replicates, np.nan)
        channels = np.full(n_replicates, -1, dtype=np.int64)
        n_intervals = int(self.core.n_timesteps) - 1
        if recorder is not None:
            recorder.start(n_intervals, (n_channels, n_replicates), 0)
        self.sweeps = self.events = 0
        for interval in range(n_intervals):
            target = self.core.get_time_at(interval + 1)
            interval_events = np.zeros(n_replicates, dtype=np.int64)
            while np.any(times < target):
                self.sweeps += 1
                active = times < target
                need = active & np.isnan(pending)
                if np.any(need):
                    hazards = propensity_func(times.copy(), state)
                    if not isinstance(hazards, np.ndarray) or hazards.shape != (
                        n_channels,
                        n_replicates,
                    ):
                        msg = (
                            "Batch propensity must return NumPy shape "
                            f"{(n_channels, n_replicates)}"
                        )
                        raise ValueError(msg)
                    if np.any(~np.isfinite(hazards)) or np.any(hazards < 0):
                        msg = "Reaction propensities must be finite and non-negative"
                        raise ValueError(msg)
                    cumulative = np.cumsum(hazards, axis=0)
                    total = cumulative[-1]
                    if not np.all(np.isfinite(total)):
                        msg = "Total reaction propensities must remain finite"
                        raise ValueError(msg)
                    pending[need] = np.inf
                    for column in np.flatnonzero(need & (total > 0)):
                        wait = streams[column].exponential() / total[column]
                        event_time = times[column] + wait
                        if not np.isfinite(event_time) or event_time <= times[column]:
                            msg = "Direct SSA event time failed to advance"
                            raise RuntimeError(msg)
                        threshold = min(
                            streams[column].random() * total[column],
                            np.nextafter(total[column], 0.0),
                        )
                        channels[column] = np.searchsorted(
                            cumulative[:, column], threshold, side="right"
                        )
                        pending[column] = event_time
                force = active & (next_forcing <= target) & (next_forcing <= pending)
                fire = active & ~force & (pending <= target)
                columns = np.flatnonzero(fire)
                if np.any(interval_events[columns] >= cfg.max_events):
                    msg = (
                        "Exceeded max_events for a replicate while "
                        "advancing to an output time"
                    )
                    raise RuntimeError(msg)
                if columns.size:
                    proposed = (
                        state[:, columns] + self.stoichiometry[:, channels[columns]]
                    )
                    if np.any(~np.isfinite(proposed)) or np.any(proposed < 0):
                        msg = "Direct SSA produced a negative or non-finite population"
                        raise RuntimeError(msg)
                    state[:, columns] = proposed
                    times[columns] = pending[columns]
                    pending[columns] = np.nan
                    interval_events[columns] += 1
                    self.events += int(columns.size)
                    if recorder is not None:
                        for column in columns:
                            recorder.record(
                                interval,
                                int(channels[column]) * n_replicates + int(column),
                            )
                times[force] = next_forcing[force]
                pending[force] = np.nan
                for column in np.flatnonzero(force):
                    row = schedules[column]
                    position = np.searchsorted(row, times[column], side="right")
                    next_forcing[column] = (
                        row[position] if position < len(row) else np.inf
                    )
                times[active & ~force & ~fire] = target
            self.core.advance_timestep(state.copy())
