"""Exact bounded thinning for time-dependent reaction propensities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import numpy as np

from ._array import array_namespace as _namespace_of
from ._forcing import _ForcingSchedule
from .stochastic_solver import _ReactionNetwork

if TYPE_CHECKING:
    from typing import Any

    from ._typing import Array
    from .model_core import ModelCore
    from .stochastic_solver import PropensityFunction


def _nonnegative_real(value: float, *, name: str) -> float:
    """Validate a finite non-negative real control.

    Returns:
        The normalized scalar.

    Raises:
        ValueError: If the control is not a finite non-negative real number.
    """
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        msg = f"{name} must be a finite non-negative real scalar"
        raise ValueError(msg)
    return float(value)


@dataclass(frozen=True, slots=True)
class TotalRateBound:
    """Upper bound on all reaction/batch rates until an exclusive endpoint.

    Attributes:
        rate: Finite non-negative total-rate bound at the unchanged state.
        valid_until: Finite exclusive endpoint, checked against the current
            time and forcing/solve limit when the callback returns it.
    """

    rate: float
    valid_until: float

    def __post_init__(self) -> None:
        """Normalize and validate the scalar controls.

        Raises:
            ValueError: If either control is invalid.
        """
        object.__setattr__(
            self, "rate", _nonnegative_real(self.rate, name="bound rate")
        )
        if (
            not isinstance(self.valid_until, Real)
            or isinstance(self.valid_until, bool)
            or not math.isfinite(self.valid_until)
        ):
            msg = "valid_until must be a finite real scalar"
            raise ValueError(msg)
        object.__setattr__(self, "valid_until", float(self.valid_until))


class RateBoundFunction(Protocol):
    """Bound total propensity at a fixed state on ``[t, valid_until)``.

    The callback must return ``t < valid_until <= limit``. ``limit`` is the
    next forcing change or solve endpoint, independent of observation times.
    """

    def __call__(self, t: float, state: Array, limit: float, /) -> TotalRateBound:
        """Return a certified bound and its exclusive expiry time."""


class ThinningSample(NamedTuple):
    """Independent exponential wait and uniform in the active array namespace.

    Both values are real floating scalar arrays. ``waiting_time`` is finite
    and positive; ``uniform`` is finite and belongs to ``[0, 1)``.
    """

    waiting_time: Array
    uniform: Array


class ThinningSampler(Protocol):
    """Sample candidates with globally increasing zero-based draw indices."""

    def __call__(self, bound_rate: Array, draw_index: int, /) -> ThinningSample:
        """Draw a wait at ``bound_rate`` and an independent uniform."""


@dataclass(frozen=True, slots=True)
class ThinningSSAConfig:
    """Controls for exact bounded-thinning SSA.

    Attributes:
        max_candidates: Maximum candidate draws across the entire run,
            including rejected and expired candidates.
        forcing_breakpoints: Strictly increasing finite forcing changes.
            Forcing and bound expiry take precedence over tied candidates.
    """

    max_candidates: int = 1_000_000
    forcing_breakpoints: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the candidate guard and snapshot the forcing schedule.

        Raises:
            ValueError: If either control is invalid.
        """
        if (
            not isinstance(self.max_candidates, Integral)
            or isinstance(self.max_candidates, bool)
            or self.max_candidates < 1
        ):
            msg = "max_candidates must be a positive integer"
            raise ValueError(msg)
        schedule = _ForcingSchedule(self.forcing_breakpoints)
        object.__setattr__(self, "forcing_breakpoints", schedule.breakpoints)


class NumpyThinningSampler:
    """Seeded stateful thinning sampler for NumPy reaction networks."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a NumPy random generator with an optional seed."""
        self._rng = np.random.default_rng(seed)

    def __call__(self, bound_rate: Array, draw_index: int, /) -> ThinningSample:
        """Draw a candidate wait and uniform in the input dtype.

        Returns:
            Scalar NumPy sampling arrays.

        Raises:
            TypeError: If the bound rate is not a NumPy array.
        """
        del draw_index
        if not isinstance(bound_rate, np.ndarray):
            msg = "NumpyThinningSampler requires a NumPy bound-rate array"
            raise TypeError(msg)
        wait = np.asarray(
            self._rng.exponential(scale=1.0 / float(bound_rate.item())),
            dtype=bound_rate.dtype,
        )
        uniform = np.asarray(self._rng.random(dtype=bound_rate.dtype))
        return ThinningSample(cast("Array", wait), cast("Array", uniform))


def _validate_sample(sample: ThinningSample, xp: Any) -> tuple[float, float]:  # noqa: ANN401
    """Extract validated candidate randomness.

    Returns:
        Waiting time and uniform as eager scalars.

    Raises:
        TypeError: If the sample type or array namespace is invalid.
        ValueError: If sampled values violate the sampling contract.
    """
    if not isinstance(sample, ThinningSample):
        msg = "thinning_sampler must return a ThinningSample"
        raise TypeError(msg)
    for value, name in (
        (sample.waiting_time, "waiting_time"),
        (sample.uniform, "uniform"),
    ):
        if _namespace_of(value) is not xp:
            msg = "thinning_sampler must preserve the state array namespace"
            raise TypeError(msg)
        if value.shape != () or not xp.isdtype(value.dtype, "real floating"):
            msg = f"Thinning {name} must be a real floating scalar array"
            raise ValueError(msg)
    wait, uniform = float(sample.waiting_time.item()), float(sample.uniform.item())
    if not math.isfinite(wait) or wait <= 0:
        msg = "Thinning waiting_time must be finite and positive"
        raise ValueError(msg)
    if not math.isfinite(uniform) or not 0 <= uniform < 1:
        msg = "Thinning uniform must be finite and in [0, 1)"
        raise ValueError(msg)
    return wait, uniform


def _resolve_bound(
    source: float | RateBoundFunction,
    t: float,
    state: Array,
    limit: float,
) -> TotalRateBound:
    """Resolve a bound against the current clock and array dtype.

    Returns:
        A finite, representable bound on a strictly advancing interval.

    Raises:
        TypeError: If the callback does not return a TotalRateBound.
        ValueError: If the endpoint or rate is invalid.
    """
    bound = (
        source(t, state, limit) if callable(source) else TotalRateBound(source, limit)
    )
    if not isinstance(bound, TotalRateBound):
        msg = "rate_bound callback must return a TotalRateBound"
        raise TypeError(msg)
    if not t < bound.valid_until <= limit:
        msg = f"valid_until must satisfy {t} < valid_until <= {limit}"
        raise ValueError(msg)
    xp = _namespace_of(state)
    rate = float(xp.asarray(bound.rate, dtype=state.dtype).item())
    if not math.isfinite(rate) or (bound.rate > 0 and rate == 0):
        msg = "bound rate must be representable in the state dtype"
        raise ValueError(msg)
    return TotalRateBound(rate, bound.valid_until)


def _candidate(
    t: float,
    rate: Array,
    sampler: ThinningSampler,
    draw_index: int,
) -> tuple[float, float]:
    """Draw one candidate that advances the clock.

    Returns:
        Absolute candidate time and uniform.

    Raises:
        RuntimeError: If adding the sampled wait cannot advance finitely.
    """
    wait, uniform = _validate_sample(sampler(rate, draw_index), _namespace_of(rate))
    candidate_time = t + wait
    if not math.isfinite(candidate_time) or candidate_time <= t:
        msg = "Thinning SSA candidate time failed to advance finitely"
        raise RuntimeError(msg)
    return candidate_time, uniform


class ThinningSSASolver:
    """Exact thinning SSA with user-certified total-rate bounds.

    Reaction and batch channels share one candidate clock. Rates may vary
    smoothly while the state is unchanged; a valid bound must cover their
    total over the entire declared interval. Observation times only observe
    the trajectory and never refresh bounds or discard candidates.
    """

    def __init__(
        self,
        core: ModelCore,
        stoichiometry: Array,
        *,
        reaction_axis: str | int = "state",
    ) -> None:
        """Initialize the shared validated reaction network.

        Args:
            core: Model state and observation-time container.
            stoichiometry: Integer-valued species-by-reaction matrix.
            reaction_axis: State axis changed by reaction firings.
        """
        self.core = core
        self._network = _ReactionNetwork(core, stoichiometry, reaction_axis)

    @property
    def n_reactions(self) -> int:
        """Return the number of reaction channels."""
        return self._network.n_reactions

    def _rates(
        self,
        propensity_func: PropensityFunction,
        t: float,
        state: Array,
        bound: TotalRateBound,
    ) -> tuple[Array, float]:
        """Evaluate candidate-time rates and check the total-rate bound.

        Returns:
            Flattened cumulative channel rates and their total.

        Raises:
            ValueError: If propensities are invalid or exceed the bound.
        """
        rates = self._network.evaluate_propensity(propensity_func, t=t, state=state)
        xp = _namespace_of(state)
        cumulative = cast("Array", xp.cumulative_sum(xp.reshape(rates, (-1,))))
        total = float(xp.max(cumulative).item())
        if not math.isfinite(total) or total > bound.rate:
            msg = f"Total propensity {total} exceeds bound rate {bound.rate} at t={t}"
            raise ValueError(msg)
        return cumulative, total

    def _accept(
        self,
        propensity_func: PropensityFunction,
        t: float,
        uniform: float,
        state: Array,
        bound: TotalRateBound,
    ) -> tuple[Array, bool]:
        """Reject or apply one candidate using rates at its time.

        Returns:
            State and whether the candidate was accepted.

        Raises:
            RuntimeError: If an accepted event produces invalid populations.
        """
        cumulative, total = self._rates(propensity_func, t, state, bound)
        xp = _namespace_of(state)
        threshold = uniform * bound.rate
        if threshold >= total:
            return state, False
        event = int(xp.sum(xp.less_equal(cumulative, threshold)).item())
        proposed = self._network.apply_event(event, state)
        try:
            self._network.validate_finite_nonnegative(
                proposed,
                message="Thinning SSA reaction produced invalid populations",
            )
        except ValueError as error:
            msg = "Thinning SSA reaction produced invalid populations"
            raise RuntimeError(msg) from error
        return proposed, True

    def _initial_state(self, rate_bound: float | RateBoundFunction) -> Array:
        """Validate initial populations and controls even for a singleton grid.

        Returns:
            The initial state in its active array namespace.

        Raises:
            TypeError: If the state dtype is not real floating.
            ValueError: If populations, bound, or solve times are invalid.
        """
        state = self.core.get_current_state()
        if not _namespace_of(state).isdtype(state.dtype, "real floating"):
            msg = "Thinning SSA requires a real floating state dtype"
            raise TypeError(msg)
        self._network.validate_finite_nonnegative(
            state, message="Initial state is invalid"
        )
        if not callable(rate_bound):
            _nonnegative_real(rate_bound, name="rate_bound")
        if not np.all(np.isfinite(self.core.time_grid)):
            msg = "Thinning SSA time_grid must be finite"
            raise ValueError(msg)
        return state

    def run(
        self,
        propensity_func: PropensityFunction,
        thinning_sampler: ThinningSampler,
        *,
        rate_bound: float | RateBoundFunction,
        config: ThinningSSAConfig | None = None,
    ) -> None:
        """Advance an exact trajectory with bounded candidate thinning.

        Args:
            propensity_func: Time-dependent batched reaction propensities.
            thinning_sampler: Namespace-preserving candidate sampler.
            rate_bound: Constant bound or callback returning a certified bound
                on ``[t, valid_until)`` at the supplied unchanged state.
            config: Candidate guard and forcing schedule.

        Raises:
            RuntimeError: If the candidate limit is exceeded, time cannot
                advance, or a reaction produces invalid populations.
        """
        cfg = config or ThinningSSAConfig()
        forcing = _ForcingSchedule(cfg.forcing_breakpoints)
        state = self._initial_state(rate_bound)
        xp = _namespace_of(state)
        times = np.asarray(self.core.time_grid, dtype=float)
        t, end = float(times[0]), float(times[-1])
        bound: TotalRateBound | None = None
        pending: tuple[float, float] | None = None
        draw_index = 0
        for target in times[1:]:
            while t < target:
                if bound is None:
                    bound = _resolve_bound(
                        rate_bound, t, state, min(forcing.next_after(t), end)
                    )
                    self._rates(propensity_func, t, state, bound)
                if pending is None:
                    if bound.rate == 0:
                        pending = (bound.valid_until, 0.0)
                    else:
                        if draw_index >= cfg.max_candidates:
                            msg = "Exceeded max_candidates during thinning SSA run"
                            raise RuntimeError(msg)
                        pending = _candidate(
                            t,
                            cast("Array", xp.asarray(bound.rate, dtype=state.dtype)),
                            thinning_sampler,
                            draw_index,
                        )
                        draw_index += 1
                if bound.valid_until <= target and bound.valid_until <= pending[0]:
                    t = bound.valid_until
                    bound, pending = None, None
                    continue
                if pending[0] > target:
                    break
                t, uniform = pending
                pending = None
                state, accepted = self._accept(
                    propensity_func, t, uniform, state, bound
                )
                if accepted:
                    bound = None
            self.core.advance_timestep(state)
