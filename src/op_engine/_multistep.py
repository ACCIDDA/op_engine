"""Validated linear-multistep coefficients and backend-neutral history."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Generic, TypeVar, cast

from ._typing import Array

StateArray = TypeVar("StateArray", bound=Array)
WeightedSum = Callable[
    [StateArray, float, tuple[float, ...], Sequence[StateArray]], StateArray
]
LinearSolve = Callable[[float, float, StateArray], StateArray]


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace advertised by ``value``.

    Raises:
        TypeError: If ``value`` is not an Array-API array.
    """
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        msg = "Multistep history values must implement __array_namespace__()"
        raise TypeError(msg)
    return namespace()


@dataclass(slots=True, frozen=True)
class LinearMultistepTableau:
    """Coefficients and lifecycle metadata for a linear-multistep method.

    Coefficients use the convention

    ``sum(alpha[j] * y[n + 1 - j]) =``
    ``    h * sum(beta[j] * f[n + 1 - j])``.

    The currently executable family is BDF, for which only ``beta[0]`` is
    nonzero. Keeping the complete beta vector makes the representation useful
    for future Adams or predictor-corrector families without pretending that
    their derivative-history storage already exists.

    Attributes:
        name: Human-readable method name.
        alpha: State coefficients, newest unknown first.
        beta: Derivative coefficients, newest unknown first.
        order: Formal consistency order.
        startup_methods: Method names selected with zero, one, ... stored
            older states before this method has enough history.
        requires_uniform_step: Whether these constant coefficients require a
            uniform step size.
    """

    name: str
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    order: int
    startup_methods: tuple[str, ...]
    requires_uniform_step: bool

    def __post_init__(self) -> None:  # noqa: C901
        """Validate dimensions, metadata, and declared order conditions.

        Raises:
            TypeError: If typed metadata does not use the required Python type.
            ValueError: If the method declaration is inconsistent.
        """
        if not self.name.strip():
            msg = "Linear-multistep tableau name must not be empty"
            raise ValueError(msg)
        if len(self.alpha) < 2 or len(self.alpha) != len(self.beta):
            msg = "Linear-multistep alpha and beta dimensions must agree"
            raise ValueError(msg)
        if any(not math.isfinite(value) for value in (*self.alpha, *self.beta)):
            msg = "Linear-multistep coefficients must be finite"
            raise ValueError(msg)
        if self.alpha[0] == 0.0:
            msg = "Linear-multistep alpha[0] must be nonzero"
            raise ValueError(msg)
        if not isinstance(self.order, Integral) or isinstance(self.order, bool):
            msg = "Linear-multistep order must be a positive integer"
            raise TypeError(msg)
        if self.order < 1:
            msg = "Linear-multistep order must be a positive integer"
            raise ValueError(msg)
        if not isinstance(self.requires_uniform_step, bool):
            msg = "requires_uniform_step must be boolean"
            raise TypeError(msg)
        if len(self.startup_methods) != self.stored_state_history:
            msg = "Startup sequence must cover every incomplete history depth"
            raise ValueError(msg)
        if any(not method.strip() for method in self.startup_methods):
            msg = "Startup method names must not be empty"
            raise ValueError(msg)

        for degree in range(self.order + 1):
            lhs = math.fsum(
                coefficient * ((-index) ** degree)
                for index, coefficient in enumerate(self.alpha)
            )
            rhs = (
                0.0
                if degree == 0
                else degree
                * math.fsum(
                    coefficient * ((-index) ** (degree - 1))
                    for index, coefficient in enumerate(self.beta)
                )
            )
            if not math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-14):
                msg = "Linear-multistep coefficients do not satisfy declared order"
                raise ValueError(msg)

    @property
    def n_steps(self) -> int:
        """Return the number of known states consumed by one full step."""
        return len(self.alpha) - 1

    @property
    def stored_state_history(self) -> int:
        """Return older state snapshots needed in addition to the current state."""
        return self.n_steps - 1

    @property
    def is_backward_differentiation(self) -> bool:
        """Return whether only the newest derivative has a nonzero weight."""
        return self.beta[0] != 0.0 and all(value == 0.0 for value in self.beta[1:])


@dataclass(slots=True, frozen=True)
class MultistepHistory(Generic[StateArray]):
    """Immutable, backend-preserving older-state history, newest first.

    A caller pushes only after accepting a step. ``restart`` is used after a
    discontinuity or external state replacement so the target method repeats
    its startup sequence.
    """

    capacity: int
    states: tuple[StateArray, ...] = ()

    def __post_init__(self) -> None:
        """Validate capacity and any supplied snapshots.

        Raises:
            TypeError: If states do not share one Array-API namespace.
            ValueError: If capacity or state metadata is inconsistent.
        """
        if (
            not isinstance(self.capacity, Integral)
            or isinstance(self.capacity, bool)
            or self.capacity < 0
        ):
            msg = "Multistep history capacity must be a non-negative integer"
            raise ValueError(msg)
        if len(self.states) > self.capacity:
            msg = "Multistep history contains more states than its capacity"
            raise ValueError(msg)
        if not self.states:
            return

        namespace = _namespace_of(self.states[0])
        shape = self.states[0].shape
        dtype = self.states[0].dtype
        for state in self.states[1:]:
            if _namespace_of(state) is not namespace:
                msg = "Multistep history states must share one array namespace"
                raise TypeError(msg)
            if state.shape != shape or state.dtype != dtype:
                msg = "Multistep history states must share shape and dtype"
                raise ValueError(msg)

    def push(self, state: StateArray) -> MultistepHistory[StateArray]:
        """Return a new history containing an independent state snapshot.

        Raises:
            TypeError: If the state changes the history's array namespace.
            ValueError: If the state changes the history's shape or dtype.
        """
        if self.capacity == 0:
            return self
        namespace = _namespace_of(state)
        if self.states:
            reference = self.states[0]
            if namespace is not _namespace_of(reference):
                msg = "Multistep history states must share one array namespace"
                raise TypeError(msg)
            if state.shape != reference.shape or state.dtype != reference.dtype:
                msg = "Multistep history states must share shape and dtype"
                raise ValueError(msg)
        snapshot = cast(
            "StateArray",
            namespace.asarray(state, dtype=state.dtype, copy=True),
        )
        return MultistepHistory(
            capacity=self.capacity,
            states=(snapshot, *self.states)[: self.capacity],
        )

    def restart(self) -> MultistepHistory[StateArray]:
        """Return empty history with the same capacity."""
        return MultistepHistory(capacity=self.capacity)

    def ready_for(self, tableau: LinearMultistepTableau) -> bool:
        """Return whether this history can supply ``tableau``."""
        return len(self.states) >= tableau.stored_state_history


def select_multistep_tableau(
    target: LinearMultistepTableau,
    available_history: int,
) -> LinearMultistepTableau:
    """Select the target or its startup method for a history depth.

    Returns:
        The executable tableau for this step.

    Raises:
        ValueError: If ``available_history`` is negative or startup metadata
            names an unavailable method.
    """
    if available_history < 0:
        msg = "Available multistep history must not be negative"
        raise ValueError(msg)
    if available_history >= target.stored_state_history:
        return target
    method = target.startup_methods[available_history]
    try:
        return BDF_TABLEAUS[method]
    except KeyError as error:
        msg = f"Unknown multistep startup method: {method}"
        raise ValueError(msg) from error


def evaluate_linearly_implicit_bdf(  # noqa: PLR0913
    *,
    tableau: LinearMultistepTableau,
    dt: float,
    known_states: Sequence[StateArray],
    residual: StateArray,
    solve: LinearSolve[StateArray],
    weighted_sum: WeightedSum[StateArray],
) -> StateArray:
    """Evaluate one constant-step BDF formula after RHS linearization.

    ``solve`` receives ``alpha[0]``, ``dt * beta[0]``, and the assembled
    right-hand side. The backend then solves
    ``(alpha[0] I - dt * beta[0] J) y_next = rhs``.

    Returns:
        The next state in the input array namespace.

    Raises:
        ValueError: If the tableau is not BDF or history is incomplete.
    """
    if not tableau.is_backward_differentiation:
        msg = "The linearly implicit kernel currently supports only BDF tableaus"
        raise ValueError(msg)
    if len(known_states) != tableau.n_steps:
        msg = "Known-state count does not match the multistep tableau"
        raise ValueError(msg)

    history_weights = tuple(-value for value in tableau.alpha[2:])
    rhs = weighted_sum(
        known_states[0],
        -tableau.alpha[1],
        (*history_weights, dt * tableau.beta[0]),
        (*known_states[1:], residual),
    )
    return solve(tableau.alpha[0], dt * tableau.beta[0], rhs)


BDF1 = LinearMultistepTableau(
    name="BDF1",
    alpha=(1.0, -1.0),
    beta=(1.0, 0.0),
    order=1,
    startup_methods=(),
    requires_uniform_step=False,
)

BDF2 = LinearMultistepTableau(
    name="BDF2",
    alpha=(3.0 / 2.0, -2.0, 1.0 / 2.0),
    beta=(1.0, 0.0, 0.0),
    order=2,
    startup_methods=("bdf1",),
    requires_uniform_step=True,
)

# Retained for stability/workload evaluation only. It is intentionally not a
# CoreSolver method: see docs/guides/linear-multistep-methods.md.
BDF3 = LinearMultistepTableau(
    name="BDF3",
    alpha=(11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0),
    beta=(1.0, 0.0, 0.0, 0.0),
    order=3,
    startup_methods=("bdf1", "bdf2"),
    requires_uniform_step=True,
)

BDF_TABLEAUS = {"bdf1": BDF1, "bdf2": BDF2, "bdf3": BDF3}


__all__ = [
    "BDF1",
    "BDF2",
    "BDF3",
    "BDF_TABLEAUS",
    "LinearMultistepTableau",
    "MultistepHistory",
    "evaluate_linearly_implicit_bdf",
    "select_multistep_tableau",
]
