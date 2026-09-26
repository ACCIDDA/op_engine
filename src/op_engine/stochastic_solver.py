"""Stochastic reaction-network integration methods.

This module keeps stochastic event semantics separate from deterministic ODE
right-hand sides. Tau-leaping and exact direct SSA consume the same validated
reaction network: reaction-channel propensities, a stoichiometric matrix, and
one configured reaction axis. Random sampling is injected so the numerical
methods remain independent of NumPy, JAX, or another array ecosystem's PRNG
API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from ._typing import Array
    from .model_core import ModelCore


_ARRAY_API_ERROR = (
    "Stochastic solver arrays must implement __array_namespace__(); got {type_name}."
)
_INVALID_PROPENSITY = "Reaction propensities must be finite and non-negative"
_INVALID_FIRINGS = "Poisson firing counts must be finite non-negative integers"
_TAU_NEGATIVE_STATE = (
    "Tau-leaping produced a negative population; reduce max_step or use a "
    "bounded/adaptive tau-leaping method"
)
_MAX_STEPS = "Exceeded max_steps while advancing to an output time"
_SSA_NEGATIVE_STATE = (
    "Direct SSA produced a negative population; consuming reactions must have "
    "zero propensity when insufficient population is available"
)
_MAX_EVENTS = "Exceeded max_events while advancing to an output time"
_INVALID_SSA_WAIT = "SSA waiting_time must be a finite positive scalar"
_INVALID_SSA_INDEX = "SSA flat_event_index must be a valid scalar integer"


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace advertised by ``value``.

    Raises:
        TypeError: If ``value`` does not advertise an array namespace.
    """
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        raise TypeError(_ARRAY_API_ERROR.format(type_name=type(value).__name__))
    return namespace()


def _array_any(value: Array) -> bool:
    """Extract an eager boolean reduction from an Array-API array.

    Returns:
        Whether any element is true.
    """
    xp = _namespace_of(value)
    return bool(xp.any(value).item())


class PoissonSampler(Protocol):
    """Backend-specific Poisson sampler injected into tau-leaping.

    ``step_index`` is a stable, zero-based accepted-step index. Functional
    PRNG ecosystems can derive a distinct key from it without mutable state.
    """

    def __call__(self, mean: Array, step_index: int, /) -> Array:
        """Draw independent Poisson counts with the shape of ``mean``."""


class SSASample(NamedTuple):
    """One direct-SSA random draw in the active array namespace.

    Attributes:
        waiting_time: Scalar exponential waiting time.
        flat_event_index: Scalar categorical index into the flattened batched
            propensity array.
    """

    waiting_time: Array
    flat_event_index: Array


class SSASampler(Protocol):
    """Backend-specific direct-SSA sampler.

    ``draw_index`` is a stable zero-based index for each random draw. Functional
    PRNG ecosystems can derive a distinct key from it without mutable state.
    Probabilities have the full batched reaction shape; the returned event index
    addresses their flattened representation.
    """

    def __call__(
        self,
        total_rate: Array,
        probabilities: Array,
        draw_index: int,
        /,
    ) -> SSASample:
        """Draw an exponential wait and one categorical event index."""


PropensityFunction: TypeAlias = Callable[[float, "Array"], "Array"]


@dataclass(slots=True, frozen=True)
class TauLeapingConfig:
    """Configuration for fixed-step explicit tau-leaping.

    Attributes:
        max_step: Maximum internal tau. ``None`` takes one leap per output
            interval.
        max_steps: Maximum number of internal leaps per output interval.
    """

    max_step: float | None = None
    max_steps: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate fixed tau controls.

        Raises:
            ValueError: If a control is outside its valid range.
        """
        if self.max_step is not None and (
            not np.isfinite(self.max_step) or self.max_step <= 0.0
        ):
            msg = "max_step must be finite and positive when provided"
            raise ValueError(msg)
        if (
            not isinstance(self.max_steps, Integral)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            msg = "max_steps must be a positive integer"
            raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class DirectSSAConfig:
    """Configuration for Gillespie's exact direct SSA.

    Attributes:
        max_events: Maximum number of events applied in one output interval.
            This guard detects explosive or otherwise pathological processes.
    """

    max_events: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate the event-count guard.

        Raises:
            ValueError: If ``max_events`` is not a positive integer.
        """
        if (
            not isinstance(self.max_events, Integral)
            or isinstance(self.max_events, bool)
            or self.max_events < 1
        ):
            msg = "max_events must be a positive integer"
            raise ValueError(msg)


class NumpyPoissonSampler:
    """Seeded stateful Poisson sampler for NumPy tau-leaping runs."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a NumPy generator.

        Args:
            seed: Optional seed passed to :func:`numpy.random.default_rng`.
        """
        self._rng = np.random.default_rng(seed)

    def __call__(self, mean: Array, step_index: int, /) -> Array:
        """Draw Poisson firing counts in the NumPy namespace.

        Args:
            mean: Non-negative Poisson means.
            step_index: Accepted-step index; unused by the stateful generator.

        Returns:
            NumPy integer firing-count array.

        Raises:
            TypeError: If ``mean`` is not a NumPy array.
        """
        del step_index
        if not isinstance(mean, np.ndarray):
            msg = "NumpyPoissonSampler requires a NumPy mean array"
            raise TypeError(msg)
        return cast("Array", self._rng.poisson(mean))


class NumpySSASampler:
    """Seeded stateful direct-SSA sampler for NumPy runs."""

    def __init__(self, seed: int | None = None) -> None:
        """Create a NumPy generator.

        Args:
            seed: Optional seed passed to :func:`numpy.random.default_rng`.
        """
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        total_rate: Array,
        probabilities: Array,
        draw_index: int,
        /,
    ) -> SSASample:
        """Draw an exponential wait and flattened categorical event.

        Args:
            total_rate: Positive scalar sum of all propensities.
            probabilities: Normalized batched reaction probabilities.
            draw_index: Draw index; unused by the stateful generator.

        Returns:
            A NumPy scalar waiting time and integer event index.

        Raises:
            TypeError: If either input is not a NumPy array.
        """
        del draw_index
        if not isinstance(total_rate, np.ndarray) or not isinstance(
            probabilities, np.ndarray
        ):
            msg = "NumpySSASampler requires NumPy rate and probability arrays"
            raise TypeError(msg)
        rate = float(total_rate.item())
        probability_vector = np.reshape(probabilities, (-1,))
        waiting_time = np.asarray(
            self._rng.exponential(scale=1.0 / rate),
            dtype=probabilities.dtype,
        )
        flat_event_index = np.asarray(
            self._rng.choice(probability_vector.size, p=probability_vector),
            dtype=np.int64,
        )
        return SSASample(
            cast("Array", waiting_time),
            cast("Array", flat_event_index),
        )


def _validate_ssa_sample(
    sample: SSASample,
    *,
    xp: Any,  # noqa: ANN401
    n_events: int,
) -> tuple[float, int]:
    """Validate and extract one eager backend-specific random draw.

    Returns:
        Python waiting time and flattened event index.

    Raises:
        TypeError: If the sampler changes array namespaces.
        ValueError: If either sampled value is invalid.
    """
    waiting_time = sample.waiting_time
    flat_event_index = sample.flat_event_index
    if (
        _namespace_of(waiting_time) is not xp
        or _namespace_of(flat_event_index) is not xp
    ):
        msg = "ssa_sampler must preserve the state array namespace"
        raise TypeError(msg)
    if waiting_time.shape != ():
        raise ValueError(_INVALID_SSA_WAIT)
    invalid_wait = xp.logical_or(
        xp.logical_not(xp.isfinite(waiting_time)),
        xp.less_equal(waiting_time, 0),
    )
    if bool(invalid_wait.item()):
        raise ValueError(_INVALID_SSA_WAIT)

    if flat_event_index.shape != ():
        raise ValueError(_INVALID_SSA_INDEX)
    invalid_index = xp.logical_or(
        xp.logical_not(xp.isfinite(flat_event_index)),
        xp.not_equal(flat_event_index, xp.round(flat_event_index)),
    )
    if bool(invalid_index.item()):
        raise ValueError(_INVALID_SSA_INDEX)
    index = int(flat_event_index.item())
    if not 0 <= index < n_events:
        raise ValueError(_INVALID_SSA_INDEX)
    return float(waiting_time.item()), index


def _draw_ssa_event(
    propensity: Array,
    ssa_sampler: SSASampler,
    *,
    draw_index: int,
) -> tuple[float, int] | None:
    """Draw an SSA event, or return no event for an absorbing state.

    Returns:
        Waiting time and event index, or no value when total rate is zero.
    """
    xp = _namespace_of(propensity)
    total_rate = cast(
        "Array",
        xp.asarray(xp.sum(propensity), dtype=propensity.dtype),
    )
    if float(total_rate.item()) == 0.0:
        return None
    probabilities = cast("Array", xp.divide(propensity, total_rate))
    sample = ssa_sampler(total_rate, probabilities, draw_index)
    return _validate_ssa_sample(
        sample,
        xp=xp,
        n_events=int(np.prod(propensity.shape)),
    )


class _ReactionNetwork:
    """Validated stoichiometry and shared batched reaction operations."""

    def __init__(
        self,
        core: ModelCore,
        stoichiometry: Array,
        reaction_axis: str | int,
    ) -> None:
        if len(stoichiometry.shape) != 2:
            msg = "stoichiometry must be a 2D (n_species, n_reactions) array"
            raise ValueError(msg)
        self.n_species = int(stoichiometry.shape[0])
        self.n_reactions = int(stoichiometry.shape[1])
        if self.n_species < 1 or self.n_reactions < 1:
            msg = "stoichiometry must contain at least one species and reaction"
            raise ValueError(msg)

        self.core = core
        self.reaction_axis = core.axis_index(reaction_axis)
        if int(core.state_shape[self.reaction_axis]) != self.n_species:
            msg = (
                "stoichiometry species dimension does not match the configured "
                "reaction axis"
            )
            raise ValueError(msg)
        xp = _namespace_of(stoichiometry)
        if _array_any(cast("Array", xp.logical_not(xp.isfinite(stoichiometry)))):
            msg = "stoichiometry values must be finite"
            raise ValueError(msg)
        if _array_any(
            cast("Array", xp.not_equal(stoichiometry, xp.round(stoichiometry)))
        ):
            msg = "stoichiometry values must be integers"
            raise ValueError(msg)
        self.stoichiometry = stoichiometry

    @property
    def reaction_shape(self) -> tuple[int, ...]:
        """Return the expected batched propensity/firing shape."""
        shape = list(self.core.state_shape)
        shape[self.reaction_axis] = self.n_reactions
        return tuple(shape)

    @staticmethod
    def validate_finite_nonnegative(value: Array, *, message: str) -> None:
        """Validate eager Array-API values.

        Raises:
            ValueError: If any value is non-finite or negative.
        """
        xp = _namespace_of(value)
        invalid = xp.logical_or(xp.logical_not(xp.isfinite(value)), xp.less(value, 0))
        if _array_any(cast("Array", invalid)):
            raise ValueError(message)

    @classmethod
    def validate_counts(cls, value: Array) -> None:
        """Validate sampled firing counts.

        Raises:
            ValueError: If any count is non-finite, negative, or non-integral.
        """
        cls.validate_finite_nonnegative(value, message=_INVALID_FIRINGS)
        xp = _namespace_of(value)
        if _array_any(cast("Array", xp.not_equal(value, xp.round(value)))):
            raise ValueError(_INVALID_FIRINGS)

    def evaluate_propensity(
        self,
        propensity_func: PropensityFunction,
        *,
        t: float,
        state: Array,
    ) -> Array:
        """Evaluate and validate one batched propensity array.

        Returns:
            Propensities in the state array's namespace and dtype.

        Raises:
            TypeError: If the callback changes array namespaces.
            ValueError: If the returned values or shape are invalid.
        """
        xp = _namespace_of(state)
        propensity = propensity_func(t, state)
        if _namespace_of(propensity) is not xp:
            msg = "propensity_func must preserve the state array namespace"
            raise TypeError(msg)
        propensity = cast("Array", xp.asarray(propensity, dtype=state.dtype))
        if propensity.shape != self.reaction_shape:
            msg = (
                f"propensity shape {propensity.shape} does not match expected "
                f"{self.reaction_shape}"
            )
            raise ValueError(msg)
        self.validate_finite_nonnegative(propensity, message=_INVALID_PROPENSITY)
        return propensity

    def apply_firings(self, firings: Array, state: Array) -> Array:
        """Map reaction firing counts to a state increment.

        Returns:
            State after applying every firing count.
        """
        xp = _namespace_of(state)
        stoichiometry = cast(
            "Array",
            xp.asarray(self.stoichiometry, dtype=state.dtype),
        )
        axis_order = (
            self.reaction_axis,
            *(axis for axis in range(len(state.shape)) if axis != self.reaction_axis),
        )
        firings_front = cast("Array", xp.permute_dims(firings, axis_order))
        firings_2d = cast(
            "Array",
            xp.reshape(firings_front, (self.n_reactions, -1)),
        )
        delta_2d = cast("Array", xp.matmul(stoichiometry, firings_2d))
        delta_front = cast(
            "Array",
            xp.reshape(delta_2d, (self.n_species, *firings_front.shape[1:])),
        )
        inverse_order = tuple(axis_order.index(axis) for axis in range(len(axis_order)))
        delta = cast("Array", xp.permute_dims(delta_front, inverse_order))
        return cast("Array", xp.add(state, delta))

    def apply_event(self, flat_event_index: int, state: Array) -> Array:
        """Apply one event selected from the flattened propensity array.

        Returns:
            State after applying the selected reaction once.
        """
        xp = _namespace_of(state)
        n_events = int(np.prod(self.reaction_shape))
        event_ids = xp.arange(n_events)
        selected = xp.equal(event_ids, flat_event_index)
        firings = cast(
            "Array",
            xp.reshape(
                xp.asarray(selected, dtype=state.dtype),
                self.reaction_shape,
            ),
        )
        return self.apply_firings(firings, state)


class TauLeapingSolver:
    """Fixed-step explicit Poisson tau-leaping on a ``ModelCore`` state.

    The stoichiometric matrix has shape ``(n_species, n_reactions)``. The
    configured reaction axis identifies the species dimension in the state;
    every other dimension is treated as an independent batch. A propensity
    function returns the same batched shape with ``n_reactions`` replacing
    ``n_species`` on that axis.
    """

    def __init__(
        self,
        core: ModelCore,
        stoichiometry: Array,
        *,
        reaction_axis: str | int = "state",
    ) -> None:
        """Initialize a reaction network.

        Args:
            core: Model state and output-time container.
            stoichiometry: Integer-valued matrix shaped
                ``(n_species, n_reactions)``.
            reaction_axis: State axis changed by reaction firings.

        """
        self.core = core
        self._network = _ReactionNetwork(core, stoichiometry, reaction_axis)

    @property
    def n_reactions(self) -> int:
        """Return the number of reaction channels."""
        return self._network.n_reactions

    def _step(  # noqa: PLR0913
        self,
        propensity_func: PropensityFunction,
        poisson_sampler: PoissonSampler,
        *,
        t: float,
        dt: float,
        state: Array,
        step_index: int,
    ) -> Array:
        """Take one explicit Poisson leap.

        Returns:
            Proposed state after one leap.

        Raises:
            TypeError: If a callback changes array namespaces.
            ValueError: If callback values or shapes are invalid.
            RuntimeError: If the leap produces a negative population.
        """
        xp = _namespace_of(state)
        propensity = self._network.evaluate_propensity(
            propensity_func,
            t=t,
            state=state,
        )
        expected_shape = self._network.reaction_shape

        mean = cast("Array", xp.multiply(propensity, dt))
        firings = poisson_sampler(mean, step_index)
        if _namespace_of(firings) is not xp:
            msg = "poisson_sampler must preserve the state array namespace"
            raise TypeError(msg)
        if firings.shape != expected_shape:
            msg = (
                f"firing-count shape {firings.shape} does not match expected "
                f"{expected_shape}"
            )
            raise ValueError(msg)
        firings = cast("Array", xp.asarray(firings, dtype=state.dtype))
        self._network.validate_counts(firings)

        proposed = self._network.apply_firings(firings, state)
        try:
            self._network.validate_finite_nonnegative(
                proposed,
                message=_TAU_NEGATIVE_STATE,
            )
        except ValueError as error:
            raise RuntimeError(_TAU_NEGATIVE_STATE) from error
        return proposed

    def run(
        self,
        propensity_func: PropensityFunction,
        poisson_sampler: PoissonSampler,
        *,
        config: TauLeapingConfig | None = None,
    ) -> None:
        """Advance the reaction network through the core output grid.

        Args:
            propensity_func: Reaction-channel propensity function.
            poisson_sampler: Backend-specific Poisson sampler.
            config: Optional fixed-tau controls.

        Raises:
            RuntimeError: If an interval exceeds its step limit or a leap
                produces a negative population.
        """
        cfg = config or TauLeapingConfig()
        state = self.core.get_current_state()
        self._network.validate_finite_nonnegative(
            state,
            message="Initial state is invalid",
        )

        time_grid = np.asarray(self.core.time_grid, dtype=float)
        step_index = 0
        for output_index in range(int(self.core.n_timesteps) - 1):
            t = float(time_grid[output_index])
            target = float(time_grid[output_index + 1])
            state = self.core.get_current_state()
            interval_steps = 0
            while t < target:
                if interval_steps >= cfg.max_steps:
                    raise RuntimeError(_MAX_STEPS)
                remaining = target - t
                dt = remaining if cfg.max_step is None else min(cfg.max_step, remaining)
                state = self._step(
                    propensity_func,
                    poisson_sampler,
                    t=t,
                    dt=dt,
                    state=state,
                    step_index=step_index,
                )
                t += dt
                step_index += 1
                interval_steps += 1
            self.core.advance_timestep(state)


class DirectSSASolver:
    """Gillespie direct SSA for an exact continuous-time reaction process.

    All reaction channels and batch cells form one flattened categorical event
    space. This is the superposition of the independent batched processes, so
    each accepted event changes exactly one batch cell. Propensities must be
    time-homogeneous between events; requested output times only observe the
    path and never cause an event to be resampled.
    """

    def __init__(
        self,
        core: ModelCore,
        stoichiometry: Array,
        *,
        reaction_axis: str | int = "state",
    ) -> None:
        """Initialize a reaction network.

        Args:
            core: Model state and output-time container.
            stoichiometry: Integer-valued matrix shaped
                (n_species, n_reactions).
            reaction_axis: State axis changed by reaction firings.

        """
        self.core = core
        self._network = _ReactionNetwork(core, stoichiometry, reaction_axis)

    @property
    def n_reactions(self) -> int:
        """Return the number of reaction channels."""
        return self._network.n_reactions

    def run(
        self,
        propensity_func: PropensityFunction,
        ssa_sampler: SSASampler,
        *,
        config: DirectSSAConfig | None = None,
    ) -> None:
        """Advance an exact reaction trajectory through the core output grid.

        An event drawn beyond an output boundary is retained for the following
        interval. A zero total propensity is absorbing, and the sampler is not
        called again.

        Args:
            propensity_func: Time-homogeneous reaction-channel propensities.
            ssa_sampler: Backend-specific exponential/categorical sampler.
            config: Optional event-count guard.

        Raises:
            RuntimeError: If an interval exceeds its event limit or a reaction
                produces a negative population.
        """
        cfg = config or DirectSSAConfig()
        state = self.core.get_current_state()
        self._network.validate_finite_nonnegative(
            state,
            message="Initial state is invalid",
        )

        time_grid = np.asarray(self.core.time_grid, dtype=float)
        t = float(time_grid[0])
        draw_index = 0
        pending_time: float | None = None
        pending_event_index: int | None = None
        absorbing = False

        for output_index in range(int(self.core.n_timesteps) - 1):
            target = float(time_grid[output_index + 1])
            interval_events = 0
            while not absorbing and t < target:
                if pending_time is None:
                    propensity = self._network.evaluate_propensity(
                        propensity_func,
                        t=t,
                        state=state,
                    )
                    event = _draw_ssa_event(
                        propensity,
                        ssa_sampler,
                        draw_index=draw_index,
                    )
                    if event is None:
                        absorbing = True
                        break
                    waiting_time, pending_event_index = event
                    pending_time = t + waiting_time
                    draw_index += 1

                if pending_time > target:
                    break
                if interval_events >= cfg.max_events:
                    raise RuntimeError(_MAX_EVENTS)
                if pending_event_index is None:
                    msg = "Direct SSA event state is inconsistent"
                    raise RuntimeError(msg)

                proposed = self._network.apply_event(pending_event_index, state)
                try:
                    self._network.validate_finite_nonnegative(
                        proposed,
                        message=_SSA_NEGATIVE_STATE,
                    )
                except ValueError as error:
                    raise RuntimeError(_SSA_NEGATIVE_STATE) from error
                state = proposed
                t = pending_time
                pending_time = None
                pending_event_index = None
                interval_events += 1

            self.core.advance_timestep(state)


__all__ = [
    "DirectSSAConfig",
    "DirectSSASolver",
    "NumpyPoissonSampler",
    "NumpySSASampler",
    "PoissonSampler",
    "PropensityFunction",
    "SSASample",
    "SSASampler",
    "TauLeapingConfig",
    "TauLeapingSolver",
]
