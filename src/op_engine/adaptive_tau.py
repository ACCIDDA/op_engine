"""Bounded adaptive tau-leaping for stochastic reaction networks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .stochastic_solver import (
    PoissonSampler,
    PropensityFunction,
    SSASampler,
    _array_any,
    _draw_ssa_event,
    _namespace_of,
    _ReactionNetwork,
)

if TYPE_CHECKING:
    from ._typing import Array
    from .model_core import ModelCore


_ADAPTIVE_NEGATIVE_STATE = (
    "Adaptive tau-leaping could not produce a non-negative population within "
    "max_retries"
)
_EXACT_NEGATIVE_STATE = (
    "An exact critical reaction produced a negative population; propensities "
    "must be zero when insufficient reactants are available"
)
_MAX_STEPS = "Exceeded max_steps while advancing to an output time"
_STEP_UNDERFLOW = "Adaptive tau-leaping step size underflowed"


@dataclass(slots=True, frozen=True)
class AdaptiveTauLeapingConfig:
    """Controls for bounded adaptive tau-leaping.

    Attributes:
        leap_tolerance: Relative leap-condition tolerance.
        critical_threshold: A reaction is critical when fewer than this many
            firings would exhaust one of its reactants. Zero disables critical
            classification.
        exact_fallback_multiplier: Use exact SSA when the proposed leap is less
            than this multiple of the mean time to the next event. Zero
            disables this fallback.
        max_steps: Maximum accepted leaps or exact events per output interval.
        max_retries: Maximum post-leap rejections before failing.
    """

    leap_tolerance: float = 0.03
    critical_threshold: int = 10
    exact_fallback_multiplier: float = 10.0
    max_steps: int = 1_000_000
    max_retries: int = 20

    def __post_init__(self) -> None:
        """Validate adaptive controls.

        Raises:
            ValueError: If a control is outside its valid range.
        """
        if (
            not isinstance(self.leap_tolerance, Real)
            or isinstance(self.leap_tolerance, bool)
            or not np.isfinite(self.leap_tolerance)
            or not 0.0 < self.leap_tolerance < 1.0
        ):
            msg = "leap_tolerance must be finite and strictly between zero and one"
            raise ValueError(msg)
        if (
            not isinstance(self.critical_threshold, Integral)
            or isinstance(self.critical_threshold, bool)
            or self.critical_threshold < 0
        ):
            msg = "critical_threshold must be a non-negative integer"
            raise ValueError(msg)
        if (
            not isinstance(self.exact_fallback_multiplier, Real)
            or isinstance(self.exact_fallback_multiplier, bool)
            or not np.isfinite(self.exact_fallback_multiplier)
            or self.exact_fallback_multiplier < 0.0
        ):
            msg = "exact_fallback_multiplier must be finite and non-negative"
            raise ValueError(msg)
        if (
            not isinstance(self.max_steps, Integral)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            msg = "max_steps must be a positive integer"
            raise ValueError(msg)
        if (
            not isinstance(self.max_retries, Integral)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            msg = "max_retries must be a non-negative integer"
            raise ValueError(msg)


class _ReactantStructure:
    """Validated reactant stoichiometry and leap-selection metadata."""

    def __init__(self, reactants: Array, network: _ReactionNetwork) -> None:
        if reactants.shape != (network.n_species, network.n_reactions):
            msg = (
                f"reactant_stoichiometry shape {reactants.shape} does not match "
                f"{(network.n_species, network.n_reactions)}"
            )
            raise ValueError(msg)
        xp = _namespace_of(reactants)
        invalid = xp.logical_or(
            xp.logical_not(xp.isfinite(reactants)),
            xp.logical_or(
                xp.less(reactants, 0),
                xp.not_equal(reactants, xp.round(reactants)),
            ),
        )
        if _array_any(cast("Array", invalid)):
            msg = "reactant_stoichiometry must contain finite non-negative integers"
            raise ValueError(msg)

        indexed = cast("Any", reactants)
        values = tuple(
            tuple(
                int(indexed[species, reaction].item())
                for reaction in range(network.n_reactions)
            )
            for species in range(network.n_species)
        )
        net_indexed = cast("Any", network.stoichiometry)
        if any(
            values[species][reaction] + int(net_indexed[species, reaction].item()) < 0
            for species in range(network.n_species)
            for reaction in range(network.n_reactions)
        ):
            msg = (
                "reactant_stoichiometry is incompatible with net stoichiometry; "
                "implied product counts must be non-negative"
            )
            raise ValueError(msg)
        reaction_orders = tuple(
            sum(values[species][reaction] for species in range(network.n_species))
            for reaction in range(network.n_reactions)
        )
        if any(order > 3 for order in reaction_orders):
            msg = (
                "Cao-Gillespie-Petzold tau selection supports reaction orders "
                "through three"
            )
            raise ValueError(msg)

        highest_orders: list[int] = []
        highest_multiplicities: list[int] = []
        for species in range(network.n_species):
            participating = [
                reaction
                for reaction in range(network.n_reactions)
                if values[species][reaction] > 0
            ]
            highest = max(
                (reaction_orders[reaction] for reaction in participating),
                default=0,
            )
            multiplicity = max(
                (
                    values[species][reaction]
                    for reaction in participating
                    if reaction_orders[reaction] == highest
                ),
                default=0,
            )
            highest_orders.append(highest)
            highest_multiplicities.append(multiplicity)

        self.reactants = reactants
        self.values = values
        self.incidence = tuple(
            tuple(float(value > 0) for value in row) for row in values
        )
        self.highest_orders = tuple(highest_orders)
        self.highest_multiplicities = tuple(highest_multiplicities)

    @staticmethod
    def _axis_order(axis: int, rank: int) -> tuple[int, ...]:
        """Move one axis to the front.

        Returns:
            Permutation with the requested axis first.
        """
        return (axis, *(index for index in range(rank) if index != axis))

    def critical_mask(
        self,
        state: Array,
        propensity: Array,
        network: _ReactionNetwork,
        *,
        threshold: int,
    ) -> Array:
        """Identify reaction/batch events close to exhausting a reactant.

        Returns:
            Boolean array with the batched reaction shape.
        """
        xp = _namespace_of(state)
        state_order = self._axis_order(network.reaction_axis, len(state.shape))
        state_front = cast("Array", xp.permute_dims(state, state_order))
        batch_shape = state_front.shape[1:]
        reactants = cast(
            "Array",
            xp.asarray(self.reactants, dtype=state.dtype),
        )
        reactants = cast(
            "Array",
            xp.reshape(
                reactants,
                (network.n_species, network.n_reactions, *(1 for _ in batch_shape)),
            ),
        )
        state_expanded = cast("Array", xp.expand_dims(state_front, axis=1))
        consumed = xp.greater(reactants, 0)
        safe_reactants = xp.where(consumed, reactants, xp.ones_like(reactants))
        firing_limits = xp.floor(xp.divide(state_expanded, safe_reactants))
        unlimited = xp.full(firing_limits.shape, np.inf, dtype=state.dtype)
        firing_limits = xp.where(consumed, firing_limits, unlimited)
        limit_front = xp.min(firing_limits, axis=0)

        reaction_order = self._axis_order(
            network.reaction_axis,
            len(propensity.shape),
        )
        propensity_front = xp.permute_dims(propensity, reaction_order)
        critical_front = xp.logical_and(
            xp.greater(propensity_front, 0),
            xp.less(limit_front, threshold),
        )
        inverse_order = tuple(
            reaction_order.index(axis) for axis in range(len(reaction_order))
        )
        return cast("Array", xp.permute_dims(critical_front, inverse_order))

    def _species_scaling(self, state_front: Array) -> Array:
        """Compute the Cao-Gillespie-Petzold species scaling factors.

        Returns:
            Scaling array with the species-first state shape.
        """
        xp = _namespace_of(state_front)
        one = xp.asarray(1.0, dtype=state_front.dtype)
        rows: list[Array] = []
        for species, (order, multiplicity) in enumerate(
            zip(
                self.highest_orders,
                self.highest_multiplicities,
                strict=True,
            )
        ):
            population = cast("Any", state_front)[species]
            if order <= 1:
                scaling = xp.ones_like(population)
            elif order == 2 and multiplicity == 2:
                denominator = xp.maximum(xp.subtract(population, one), one)
                scaling = xp.add(2.0, xp.divide(1.0, denominator))
            elif order == 2:
                scaling = xp.full(population.shape, 2.0, dtype=state_front.dtype)
            elif multiplicity == 3:
                denominator_one = xp.maximum(xp.subtract(population, one), one)
                denominator_two = xp.maximum(xp.subtract(population, 2.0), one)
                scaling = xp.add(
                    3.0,
                    xp.add(
                        xp.divide(1.0, denominator_one),
                        xp.divide(2.0, denominator_two),
                    ),
                )
            elif multiplicity == 2:
                denominator = xp.maximum(xp.subtract(population, one), one)
                scaling = xp.multiply(
                    1.5,
                    xp.add(2.0, xp.divide(1.0, denominator)),
                )
            else:
                scaling = xp.full(population.shape, 3.0, dtype=state_front.dtype)
            rows.append(cast("Array", scaling))
        return cast("Array", xp.stack(rows, axis=0))

    def select_tau(  # noqa: PLR0914
        self,
        state: Array,
        propensity: Array,
        critical: Array,
        network: _ReactionNetwork,
        *,
        leap_tolerance: float,
    ) -> float:
        """Estimate the largest noncritical leap satisfying the leap condition.

        Returns:
            Scalar candidate tau, possibly infinity when active propensities are
            state-independent or every active reaction is critical.
        """
        xp = _namespace_of(state)
        state_order = self._axis_order(network.reaction_axis, len(state.shape))
        reaction_order = self._axis_order(
            network.reaction_axis,
            len(propensity.shape),
        )
        state_front = cast("Array", xp.permute_dims(state, state_order))
        noncritical = xp.where(critical, xp.zeros_like(propensity), propensity)
        propensity_front = cast(
            "Array",
            xp.permute_dims(noncritical, reaction_order),
        )
        propensity_2d = cast(
            "Array",
            xp.reshape(propensity_front, (network.n_reactions, -1)),
        )
        stoichiometry = cast(
            "Array",
            xp.asarray(network.stoichiometry, dtype=state.dtype),
        )
        drift = cast("Array", xp.matmul(stoichiometry, propensity_2d))
        variance = cast(
            "Array",
            xp.matmul(xp.square(stoichiometry), propensity_2d),
        )

        active_events = xp.asarray(
            xp.greater(propensity_2d, 0),
            dtype=state.dtype,
        )
        incidence = cast(
            "Array",
            xp.asarray(self.incidence, dtype=state.dtype),
        )
        active_species = xp.greater(xp.matmul(incidence, active_events), 0)
        state_2d = cast(
            "Array",
            xp.reshape(state_front, (network.n_species, -1)),
        )
        scaling_2d = cast(
            "Array",
            xp.reshape(
                self._species_scaling(state_front),
                (network.n_species, -1),
            ),
        )
        bound = xp.maximum(
            xp.divide(xp.multiply(leap_tolerance, state_2d), scaling_2d),
            1.0,
        )
        infinity = xp.full(drift.shape, np.inf, dtype=state.dtype)
        absolute_drift = xp.abs(drift)
        nonzero_drift = xp.greater(absolute_drift, 0)
        nonzero_variance = xp.greater(variance, 0)
        safe_drift = xp.where(nonzero_drift, absolute_drift, xp.ones_like(drift))
        safe_variance = xp.where(
            nonzero_variance,
            variance,
            xp.ones_like(variance),
        )
        mean_tau = xp.where(
            xp.logical_and(active_species, nonzero_drift),
            xp.divide(bound, safe_drift),
            infinity,
        )
        variance_tau = xp.where(
            xp.logical_and(active_species, nonzero_variance),
            xp.divide(xp.square(bound), safe_variance),
            infinity,
        )
        return float(xp.min(xp.minimum(mean_tau, variance_tau)).item())


@dataclass(slots=True, frozen=True)
class _LeapResult:
    """One accepted adaptive leap and updated sampler indices."""

    state: Array
    dt: float
    poisson_draw_index: int
    ssa_draw_index: int


class AdaptiveTauLeapingSolver:
    """Adaptive non-negative Poisson tau-leaping with exact critical events.

    The implementation follows the Cao-Gillespie-Petzold species-based
    pre-leap selector. Reactions near exhaustion are bounded to at most one
    collective exact event. A negative post-leap proposal is rejected and
    retried at half the attempted tau; populations and firing counts are never
    clipped.
    """

    def __init__(
        self,
        core: ModelCore,
        stoichiometry: Array,
        reactant_stoichiometry: Array,
        *,
        reaction_axis: str | int = "state",
    ) -> None:
        """Initialize the reaction and reactant structures.

        Args:
            core: Model state and output-time container.
            stoichiometry: Net integer state changes by reaction.
            reactant_stoichiometry: Non-negative integer reactant counts by
                species and reaction. This cannot be inferred safely from net
                stoichiometry for catalytic reactions.
            reaction_axis: State axis changed by reaction firings.
        """
        self.core = core
        self._network = _ReactionNetwork(core, stoichiometry, reaction_axis)
        self._reactants = _ReactantStructure(
            reactant_stoichiometry,
            self._network,
        )

    @property
    def n_reactions(self) -> int:
        """Return the number of reaction channels."""
        return self._network.n_reactions

    def _sample_poisson(
        self,
        mean: Array,
        poisson_sampler: PoissonSampler,
        *,
        draw_index: int,
    ) -> Array:
        """Draw and validate one noncritical firing-count array.

        Returns:
            Firing counts in the active namespace and state dtype.

        Raises:
            TypeError: If the sampler changes array namespaces.
            ValueError: If the sample shape or values are invalid.
        """
        xp = _namespace_of(mean)
        firings = poisson_sampler(mean, draw_index)
        if _namespace_of(firings) is not xp:
            msg = "poisson_sampler must preserve the state array namespace"
            raise TypeError(msg)
        if firings.shape != self._network.reaction_shape:
            msg = (
                f"firing-count shape {firings.shape} does not match expected "
                f"{self._network.reaction_shape}"
            )
            raise ValueError(msg)
        firings = cast("Array", xp.asarray(firings, dtype=mean.dtype))
        self._network.validate_counts(firings)
        return firings

    def _attempt_leap(  # noqa: PLR0913
        self,
        state: Array,
        propensity: Array,
        critical: Array,
        poisson_sampler: PoissonSampler,
        ssa_sampler: SSASampler,
        *,
        tau_candidate: float,
        remaining: float,
        poisson_draw_index: int,
        ssa_draw_index: int,
        config: AdaptiveTauLeapingConfig,
    ) -> _LeapResult:
        """Sample until a non-negative adaptive leap is accepted.

        Returns:
            Accepted state, duration, and updated deterministic draw indices.

        Raises:
            RuntimeError: If the retry limit or floating-point step size fails.
        """
        xp = _namespace_of(state)
        tau_cap = tau_candidate
        noncritical_propensity = cast(
            "Array",
            xp.where(critical, xp.zeros_like(propensity), propensity),
        )
        critical_propensity = cast(
            "Array",
            xp.where(critical, propensity, xp.zeros_like(propensity)),
        )

        for retry in range(config.max_retries + 1):
            critical_event = _draw_ssa_event(
                critical_propensity,
                ssa_sampler,
                draw_index=ssa_draw_index,
            )
            if critical_event is None:
                critical_wait = np.inf
                critical_index = None
            else:
                critical_wait, critical_index = critical_event
                ssa_draw_index += 1

            dt = min(tau_cap, critical_wait, remaining)
            if dt <= 0.0:
                raise RuntimeError(_STEP_UNDERFLOW)
            mean = cast(
                "Array",
                xp.multiply(noncritical_propensity, dt),
            )
            firings = self._sample_poisson(
                mean,
                poisson_sampler,
                draw_index=poisson_draw_index,
            )
            poisson_draw_index += 1
            proposed = self._network.apply_firings(firings, state)
            if (
                critical_index is not None
                and critical_wait <= tau_cap
                and critical_wait <= remaining
            ):
                proposed = self._network.apply_event(critical_index, proposed)

            try:
                self._network.validate_finite_nonnegative(
                    proposed,
                    message=_ADAPTIVE_NEGATIVE_STATE,
                )
            except ValueError as error:
                if retry >= config.max_retries:
                    raise RuntimeError(_ADAPTIVE_NEGATIVE_STATE) from error
                tau_cap = 0.5 * dt
                continue
            return _LeapResult(
                proposed,
                dt,
                poisson_draw_index,
                ssa_draw_index,
            )

        raise RuntimeError(_ADAPTIVE_NEGATIVE_STATE)

    def _apply_exact_event(self, state: Array, event_index: int) -> Array:
        """Apply and validate one exact event.

        Returns:
            Updated state.

        Raises:
            RuntimeError: If the propensity allowed an impossible reaction.
        """
        proposed = self._network.apply_event(event_index, state)
        try:
            self._network.validate_finite_nonnegative(
                proposed,
                message=_EXACT_NEGATIVE_STATE,
            )
        except ValueError as error:
            raise RuntimeError(_EXACT_NEGATIVE_STATE) from error
        return proposed

    def run(  # noqa: C901, PLR0914, PLR0915
        self,
        propensity_func: PropensityFunction,
        poisson_sampler: PoissonSampler,
        ssa_sampler: SSASampler,
        *,
        config: AdaptiveTauLeapingConfig | None = None,
    ) -> None:
        """Advance a bounded adaptive trajectory through the output grid.

        Args:
            propensity_func: Reaction-channel propensities.
            poisson_sampler: Backend-specific noncritical Poisson sampler.
            ssa_sampler: Backend-specific exact-event sampler.
            config: Optional adaptive and retry controls.

        Raises:
            RuntimeError: If a step guard, retry guard, or non-negativity
                invariant fails.
        """
        cfg = config or AdaptiveTauLeapingConfig()
        state = self.core.get_current_state()
        self._network.validate_finite_nonnegative(
            state,
            message="Initial state is invalid",
        )
        time_grid = np.asarray(self.core.time_grid, dtype=float)
        t = float(time_grid[0])
        poisson_draw_index = 0
        ssa_draw_index = 0
        pending_exact_time: float | None = None
        pending_exact_index: int | None = None
        absorbing = False

        for output_index in range(int(self.core.n_timesteps) - 1):
            target = float(time_grid[output_index + 1])
            interval_steps = 0
            while not absorbing and t < target:
                if pending_exact_time is not None:
                    if pending_exact_time > target:
                        break
                    if interval_steps >= cfg.max_steps:
                        raise RuntimeError(_MAX_STEPS)
                    if pending_exact_index is None:
                        msg = "Adaptive tau exact-event state is inconsistent"
                        raise RuntimeError(msg)
                    state = self._apply_exact_event(state, pending_exact_index)
                    t = pending_exact_time
                    pending_exact_time = None
                    pending_exact_index = None
                    interval_steps += 1
                    continue

                if interval_steps >= cfg.max_steps:
                    raise RuntimeError(_MAX_STEPS)
                propensity = self._network.evaluate_propensity(
                    propensity_func,
                    t=t,
                    state=state,
                )
                xp = _namespace_of(state)
                total_rate = cast(
                    "Array",
                    xp.asarray(xp.sum(propensity), dtype=state.dtype),
                )
                total_rate_value = float(total_rate.item())
                if total_rate_value == 0.0:
                    absorbing = True
                    break

                critical = self._reactants.critical_mask(
                    state,
                    propensity,
                    self._network,
                    threshold=cfg.critical_threshold,
                )
                noncritical_active = _array_any(
                    cast(
                        "Array",
                        xp.logical_and(
                            xp.logical_not(critical),
                            xp.greater(propensity, 0),
                        ),
                    )
                )
                tau_candidate = self._reactants.select_tau(
                    state,
                    propensity,
                    critical,
                    self._network,
                    leap_tolerance=cfg.leap_tolerance,
                )
                fallback_threshold = cfg.exact_fallback_multiplier / total_rate_value
                use_exact = not noncritical_active or tau_candidate < fallback_threshold
                if use_exact:
                    event = _draw_ssa_event(
                        propensity,
                        ssa_sampler,
                        draw_index=ssa_draw_index,
                    )
                    if event is None:
                        absorbing = True
                        break
                    waiting_time, pending_exact_index = event
                    pending_exact_time = t + waiting_time
                    ssa_draw_index += 1
                    continue

                result = self._attempt_leap(
                    state,
                    propensity,
                    critical,
                    poisson_sampler,
                    ssa_sampler,
                    tau_candidate=tau_candidate,
                    remaining=target - t,
                    poisson_draw_index=poisson_draw_index,
                    ssa_draw_index=ssa_draw_index,
                    config=cfg,
                )
                state = result.state
                if t + result.dt == t:
                    raise RuntimeError(_STEP_UNDERFLOW)
                t += result.dt
                poisson_draw_index = result.poisson_draw_index
                ssa_draw_index = result.ssa_draw_index
                interval_steps += 1

            self.core.advance_timestep(state)


__all__ = ["AdaptiveTauLeapingConfig", "AdaptiveTauLeapingSolver"]
