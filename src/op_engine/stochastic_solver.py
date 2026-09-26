"""Stochastic reaction-network integration methods.

This module keeps stochastic event semantics separate from deterministic ODE
right-hand sides. Tau-leaping consumes reaction-channel propensities and a
stoichiometric matrix, then samples one Poisson firing count per channel and
batch cell. Random sampling is injected so the numerical method remains
independent of NumPy, JAX, or another array ecosystem's PRNG API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from ._typing import Array
    from .model_core import ModelCore


_ARRAY_API_ERROR = (
    "Tau-leaping arrays must implement __array_namespace__(); got {type_name}."
)
_INVALID_PROPENSITY = "Tau-leaping propensities must be finite and non-negative"
_INVALID_FIRINGS = "Poisson firing counts must be finite non-negative integers"
_NEGATIVE_STATE = (
    "Tau-leaping produced a negative population; reduce max_step or use a "
    "bounded/adaptive tau-leaping method"
)
_MAX_STEPS = "Exceeded max_steps while advancing to an output time"


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

        Raises:
            ValueError: If the stoichiometry shape or values are invalid.
        """
        if len(stoichiometry.shape) != 2:
            msg = "stoichiometry must be a 2D (n_species, n_reactions) array"
            raise ValueError(msg)
        self._n_species = int(stoichiometry.shape[0])
        self._n_reactions = int(stoichiometry.shape[1])
        if self._n_species < 1 or self._n_reactions < 1:
            msg = "stoichiometry must contain at least one species and reaction"
            raise ValueError(msg)

        self.core = core
        self._reaction_axis = core.axis_index(reaction_axis)
        if int(core.state_shape[self._reaction_axis]) != self._n_species:
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
        self._stoichiometry = stoichiometry

    @property
    def n_reactions(self) -> int:
        """Return the number of reaction channels."""
        return self._n_reactions

    def _reaction_shape(self) -> tuple[int, ...]:
        """Return the expected batched propensity/firing shape."""
        shape = list(self.core.state_shape)
        shape[self._reaction_axis] = self._n_reactions
        return tuple(shape)

    @staticmethod
    def _validate_finite_nonnegative(value: Array, *, message: str) -> None:
        """Validate eager Array-API values.

        Raises:
            ValueError: If any value is non-finite or negative.
        """
        xp = _namespace_of(value)
        invalid = xp.logical_or(xp.logical_not(xp.isfinite(value)), xp.less(value, 0))
        if _array_any(cast("Array", invalid)):
            raise ValueError(message)

    @classmethod
    def _validate_counts(cls, value: Array) -> None:
        """Validate sampled firing counts.

        Raises:
            ValueError: If counts are non-finite, negative, or non-integral.
        """
        cls._validate_finite_nonnegative(value, message=_INVALID_FIRINGS)
        xp = _namespace_of(value)
        if _array_any(cast("Array", xp.not_equal(value, xp.round(value)))):
            raise ValueError(_INVALID_FIRINGS)

    def _apply_firings(self, firings: Array, state: Array) -> Array:
        """Map reaction firing counts to a state increment.

        Returns:
            Proposed state after applying all reaction channels.
        """
        xp = _namespace_of(state)
        stoichiometry = cast(
            "Array",
            xp.asarray(self._stoichiometry, dtype=state.dtype),
        )
        axis_order = (
            self._reaction_axis,
            *(axis for axis in range(len(state.shape)) if axis != self._reaction_axis),
        )
        firings_front = cast("Array", xp.permute_dims(firings, axis_order))
        firings_2d = cast(
            "Array",
            xp.reshape(firings_front, (self._n_reactions, -1)),
        )
        delta_2d = cast("Array", xp.matmul(stoichiometry, firings_2d))
        delta_front = cast(
            "Array",
            xp.reshape(delta_2d, (self._n_species, *firings_front.shape[1:])),
        )
        inverse_order = tuple(axis_order.index(axis) for axis in range(len(axis_order)))
        delta = cast("Array", xp.permute_dims(delta_front, inverse_order))
        return cast("Array", xp.add(state, delta))

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
        propensity = propensity_func(t, state)
        if _namespace_of(propensity) is not xp:
            msg = "propensity_func must preserve the state array namespace"
            raise TypeError(msg)
        propensity = cast("Array", xp.asarray(propensity, dtype=state.dtype))
        expected_shape = self._reaction_shape()
        if propensity.shape != expected_shape:
            msg = (
                f"propensity shape {propensity.shape} does not match expected "
                f"{expected_shape}"
            )
            raise ValueError(msg)
        self._validate_finite_nonnegative(propensity, message=_INVALID_PROPENSITY)

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
        self._validate_counts(firings)

        proposed = self._apply_firings(firings, state)
        try:
            self._validate_finite_nonnegative(proposed, message=_NEGATIVE_STATE)
        except ValueError as error:
            raise RuntimeError(_NEGATIVE_STATE) from error
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
        self._validate_finite_nonnegative(state, message="Initial state is invalid")

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


__all__ = [
    "NumpyPoissonSampler",
    "PoissonSampler",
    "PropensityFunction",
    "TauLeapingConfig",
    "TauLeapingSolver",
]
