"""Validated paired tableaus and kernels for additive Runge--Kutta methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, TypeAlias, cast

from ._typing import Array

AdditiveRhs: TypeAlias = Callable[[float, Array], Array]
ImplicitStageSolve: TypeAlias = Callable[[int, float, float, Array], Array]
WeightedState: TypeAlias = Callable[
    [Array, float, tuple[float, ...], Sequence[Array]], Array
]


@dataclass(slots=True, frozen=True)
class AdditiveRungeKuttaTableau:
    """Paired explicit/DIRK coefficients for ``y' = F(t, y) + G(t, y)``.

    Rows of ``a_explicit`` contain exactly ``i`` coefficients. Rows of
    ``a_implicit`` contain exactly ``i + 1`` coefficients, including the
    diagonal. Both halves use the shared stage abscissae in ``c``.

    A zero implicit diagonal is supported only when that stage's implicit
    derivative has zero weight everywhere else. This covers ARS-type methods'
    explicit first stage without requiring a second callback for ``G``.
    """

    name: str
    a_explicit: tuple[tuple[float, ...], ...]
    a_implicit: tuple[tuple[float, ...], ...]
    b_explicit: tuple[float, ...]
    b_implicit: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    b_explicit_embedded: tuple[float, ...]
    b_implicit_embedded: tuple[float, ...]
    embedded_order: int

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate dimensions, consistency, and recoverable DIRK stages.

        Raises:
            ValueError: If the paired tableau is inconsistent.
        """
        n_stages = len(self.c)
        weights = (
            self.b_explicit,
            self.b_implicit,
            self.b_explicit_embedded,
            self.b_implicit_embedded,
        )
        if not self.name.strip():
            msg = "Additive Runge-Kutta tableau name must not be empty"
            raise ValueError(msg)
        if n_stages < 1:
            msg = "Additive Runge-Kutta tableau must contain stages"
            raise ValueError(msg)
        if len(self.a_explicit) != n_stages or len(self.a_implicit) != n_stages:
            msg = "Additive Runge-Kutta tableau dimensions must agree"
            raise ValueError(msg)
        if any(len(weight) != n_stages for weight in weights):
            msg = "Additive Runge-Kutta weights must match stages"
            raise ValueError(msg)
        if (
            not isinstance(self.order, Integral)
            or isinstance(self.order, bool)
            or self.order < 1
        ):
            msg = "Additive Runge-Kutta order must be a positive integer"
            raise ValueError(msg)
        if (
            not isinstance(self.embedded_order, Integral)
            or isinstance(self.embedded_order, bool)
            or self.embedded_order < 1
            or self.embedded_order >= self.order
        ):
            msg = "Embedded order must be positive and below solution order"
            raise ValueError(msg)

        coefficients = (
            *self.c,
            *(value for row in self.a_explicit for value in row),
            *(value for row in self.a_implicit for value in row),
            *(value for weight in weights for value in weight),
        )
        if any(not math.isfinite(value) for value in coefficients):
            msg = "Additive Runge-Kutta coefficients must be finite"
            raise ValueError(msg)

        for stage, (explicit_row, implicit_row, stage_time) in enumerate(
            zip(self.a_explicit, self.a_implicit, self.c, strict=True)
        ):
            if len(explicit_row) != stage:
                msg = "Explicit row i must contain exactly i coefficients"
                raise ValueError(msg)
            if len(implicit_row) != stage + 1:
                msg = "Implicit row i must contain exactly i + 1 coefficients"
                raise ValueError(msg)
            for row in (explicit_row, implicit_row):
                if not math.isclose(
                    math.fsum(row),
                    stage_time,
                    rel_tol=1e-12,
                    abs_tol=1e-14,
                ):
                    msg = "Explicit and implicit stage coefficients must each sum to c"
                    raise ValueError(msg)

        for weight in weights:
            if not math.isclose(math.fsum(weight), 1.0, rel_tol=1e-12, abs_tol=1e-14):
                msg = "Each additive Runge-Kutta weight must sum to one"
                raise ValueError(msg)

        for stage, row in enumerate(self.a_implicit):
            if row[-1] != 0.0:
                continue
            later_weights = (
                *(
                    self.a_implicit[index][stage]
                    for index in range(stage + 1, n_stages)
                ),
                self.b_implicit[stage],
                self.b_implicit_embedded[stage],
            )
            if any(value != 0.0 for value in later_weights):
                msg = "A zero-diagonal implicit stage must have zero downstream weights"
                raise ValueError(msg)

    @property
    def n_stages(self) -> int:
        """Return the number of shared additive stages."""
        return len(self.c)


@dataclass(slots=True, frozen=True)
class AdditiveRungeKuttaResult:
    """High- and embedded-order results from one paired-tableau evaluation."""

    state: Array
    embedded_state: Array


def evaluate_additive_runge_kutta(  # noqa: PLR0913
    *,
    tableau: AdditiveRungeKuttaTableau,
    t: float,
    dt: float,
    y: Array,
    explicit_rhs: AdditiveRhs,
    solve_implicit_stage: ImplicitStageSolve,
    weighted_state: WeightedState,
) -> AdditiveRungeKuttaResult:
    """Evaluate a paired explicit/DIRK tableau with linear implicit solves.

    Nonzero-diagonal implicit derivatives are recovered from the stage
    equation, ``G_i = (Y_i - base_i) / (dt * a_ii)``. Validated zero-diagonal
    stages need no implicit derivative because all corresponding weights vanish.

    Returns:
        High- and embedded-order states.
    """
    explicit_stages: list[Array] = []
    implicit_stages: list[Array] = []
    xp = cast("Any", y.__array_namespace__())

    for stage_index, (explicit_row, implicit_row, stage_time) in enumerate(
        zip(tableau.a_explicit, tableau.a_implicit, tableau.c, strict=True)
    ):
        base = weighted_state(y, dt, explicit_row, explicit_stages)
        base = weighted_state(base, dt, implicit_row[:-1], implicit_stages)
        diagonal = implicit_row[-1]
        if diagonal == 0.0:
            stage_state = base
            implicit_derivative = cast("Array", xp.zeros_like(y))
        else:
            stage_state = solve_implicit_stage(
                stage_index,
                t + stage_time * dt,
                diagonal,
                base,
            )
            implicit_derivative = cast(
                "Array",
                xp.divide(
                    xp.subtract(stage_state, base),
                    dt * diagonal,
                ),
            )
        explicit_stages.append(explicit_rhs(t + stage_time * dt, stage_state))
        implicit_stages.append(implicit_derivative)

    high = weighted_state(y, dt, tableau.b_explicit, explicit_stages)
    high = weighted_state(high, dt, tableau.b_implicit, implicit_stages)
    embedded = weighted_state(
        y,
        dt,
        tableau.b_explicit_embedded,
        explicit_stages,
    )
    embedded = weighted_state(
        embedded,
        dt,
        tableau.b_implicit_embedded,
        implicit_stages,
    )
    return AdditiveRungeKuttaResult(high, embedded)


# Ascher--Ruuth--Spiteri ARS(4,4,3), written with its explicit initial stage.
# The embedded formula selects the shared c=1/2 stage, giving explicit and
# implicit midpoint weights. Each color therefore satisfies the additive
# second-order conditions sum(b)=1 and b.c=1/2.
ARS443_32 = AdditiveRungeKuttaTableau(
    name="ARS(4,4,3) 3(2)",
    a_explicit=(
        (),
        (1.0 / 2.0,),
        (11.0 / 18.0, 1.0 / 18.0),
        (5.0 / 6.0, -5.0 / 6.0, 1.0 / 2.0),
        (1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, -7.0 / 4.0),
    ),
    a_implicit=(
        (0.0,),
        (0.0, 1.0 / 2.0),
        (0.0, 1.0 / 6.0, 1.0 / 2.0),
        (0.0, -1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
        (0.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
    ),
    b_explicit=(1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, -7.0 / 4.0, 0.0),
    b_implicit=(0.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
    c=(0.0, 1.0 / 2.0, 2.0 / 3.0, 1.0 / 2.0, 1.0),
    order=3,
    b_explicit_embedded=(0.0, 1.0, 0.0, 0.0, 0.0),
    b_implicit_embedded=(0.0, 1.0, 0.0, 0.0, 0.0),
    embedded_order=2,
)


ADDITIVE_RUNGE_KUTTA_TABLEAUS = {"imex-ark3": ARS443_32}


__all__ = [
    "ADDITIVE_RUNGE_KUTTA_TABLEAUS",
    "ARS443_32",
    "AdditiveRungeKuttaResult",
    "AdditiveRungeKuttaTableau",
    "evaluate_additive_runge_kutta",
]
