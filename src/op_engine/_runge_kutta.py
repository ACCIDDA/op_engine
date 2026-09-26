"""Validated coefficient tables for explicit Runge--Kutta methods."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TypeVar

    from ._typing import Scalar

    StateT = TypeVar("StateT")


@dataclass(slots=True, frozen=True)
class ExplicitRungeKuttaTableau:
    """Compact Butcher tableau for an explicit Runge--Kutta method.

    Row ``i`` of ``a`` contains exactly ``i`` coefficients, so an implicit
    diagonal or upper-triangular coefficient cannot be represented. The
    optional embedded weights estimate local error without another RHS call.

    Attributes:
        name: Human-readable method name.
        a: Strictly lower-triangular stage coefficients in compact row form.
        b: Weights for the accepted, high-order solution.
        c: Stage-time coefficients.
        order: Formal order of the accepted solution.
        b_embedded: Optional lower-order weights for an error estimate.
        embedded_order: Formal order of the embedded solution.
        fsal: Whether the final stage is the first stage of the next step.
    """

    name: str
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    b_embedded: tuple[float, ...] | None = None
    embedded_order: int | None = None
    fsal: bool = False

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate tableau dimensions and consistency.

        Raises:
            TypeError: If a boolean flag has the wrong type.
            ValueError: If the coefficients do not define a consistent
                explicit Runge--Kutta method.
        """
        n_stages = len(self.c)
        if not self.name.strip():
            msg = "Runge-Kutta tableau name must not be empty"
            raise ValueError(msg)
        if n_stages < 1:
            msg = "Runge-Kutta tableau must contain at least one stage"
            raise ValueError(msg)
        if len(self.a) != n_stages or len(self.b) != n_stages:
            msg = "Runge-Kutta tableau a, b, and c dimensions must agree"
            raise ValueError(msg)
        if (
            not isinstance(self.order, Integral)
            or isinstance(self.order, bool)
            or self.order < 1
        ):
            msg = "Runge-Kutta order must be a positive integer"
            raise ValueError(msg)
        if not isinstance(self.fsal, bool):
            msg = "Runge-Kutta fsal flag must be boolean"
            raise TypeError(msg)

        coefficients = (*self.b, *self.c, *(value for row in self.a for value in row))
        if any(not math.isfinite(value) for value in coefficients):
            msg = "Runge-Kutta coefficients must be finite"
            raise ValueError(msg)
        for stage, (row, stage_time) in enumerate(zip(self.a, self.c, strict=True)):
            if len(row) != stage:
                msg = "Explicit Runge-Kutta row i must contain exactly i coefficients"
                raise ValueError(msg)
            if not math.isclose(
                math.fsum(row),
                stage_time,
                rel_tol=1e-13,
                abs_tol=1e-15,
            ):
                msg = "Runge-Kutta stage coefficients must sum to c"
                raise ValueError(msg)
        if not math.isclose(math.fsum(self.b), 1.0, rel_tol=1e-13, abs_tol=1e-15):
            msg = "Runge-Kutta solution weights must sum to one"
            raise ValueError(msg)

        if (self.b_embedded is None) != (self.embedded_order is None):
            msg = "Embedded weights and order must be provided together"
            raise ValueError(msg)
        if self.b_embedded is not None:
            if len(self.b_embedded) != n_stages or any(
                not math.isfinite(value) for value in self.b_embedded
            ):
                msg = "Embedded Runge-Kutta weights must be finite and match stages"
                raise ValueError(msg)
            if not math.isclose(
                math.fsum(self.b_embedded),
                1.0,
                rel_tol=1e-13,
                abs_tol=1e-15,
            ):
                msg = "Embedded Runge-Kutta weights must sum to one"
                raise ValueError(msg)
            if (
                not isinstance(self.embedded_order, Integral)
                or isinstance(self.embedded_order, bool)
                or self.embedded_order < 1
                or self.embedded_order >= self.order
            ):
                msg = "Embedded order must be positive and below the solution order"
                raise ValueError(msg)

        if self.fsal and (
            not math.isclose(self.c[-1], 1.0, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(self.b[-1], 0.0, rel_tol=0.0, abs_tol=1e-15)
            or any(
                not math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-15)
                for left, right in zip(self.a[-1], self.b[:-1], strict=True)
            )
        ):
            msg = "FSAL tableau must end at c=1 with its final row equal to b"
            raise ValueError(msg)

    @property
    def n_stages(self) -> int:
        """Return the number of RHS stages per uncached step."""
        return len(self.c)


def evaluate_explicit_runge_kutta(  # noqa: PLR0913
    tableau: ExplicitRungeKuttaTableau,
    *,
    t: Scalar,
    dt: Scalar,
    y: StateT,
    rhs: Callable[[Scalar, StateT], StateT],
    weighted_sum: Callable[
        [StateT, Scalar, tuple[float, ...], Sequence[StateT]],
        StateT,
    ],
    first_stage: StateT | None = None,
) -> tuple[StateT, StateT | None, StateT, StateT | None]:
    """Evaluate one explicit tableau over an arbitrary state algebra.

    ``weighted_sum`` owns the state representation and computes
    ``y + dt * sum(weights[i] * stages[i])``. This keeps the validated method
    coefficients shared by dense arrays and structured provider states.

    Returns:
        High-order state, optional embedded state, first stage, and optional
        FSAL stage for the next accepted step.
    """
    stages: list[StateT] = []
    for stage_index, (row, stage_time) in enumerate(
        zip(tableau.a, tableau.c, strict=True)
    ):
        if stage_index == 0 and first_stage is not None:
            derivative = first_stage
        else:
            stage_state = weighted_sum(y, dt, row, stages)
            derivative = rhs(
                cast("Scalar", cast("Any", t) + stage_time * cast("Any", dt)),
                stage_state,
            )
        stages.append(derivative)

    high = weighted_sum(y, dt, tableau.b, stages)
    embedded = (
        None
        if tableau.b_embedded is None
        else weighted_sum(y, dt, tableau.b_embedded, stages)
    )
    last_stage = stages[-1] if tableau.fsal else None
    return high, embedded, stages[0], last_stage


HEUN_EULER = ExplicitRungeKuttaTableau(
    name="Heun--Euler 2(1)",
    a=((), (1.0,)),
    b=(0.5, 0.5),
    c=(0.0, 1.0),
    order=2,
    b_embedded=(1.0, 0.0),
    embedded_order=1,
)

CLASSIC_RK4 = ExplicitRungeKuttaTableau(
    name="Classic Runge--Kutta 4",
    a=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c=(0.0, 0.5, 0.5, 1.0),
    order=4,
)

DORMAND_PRINCE_54 = ExplicitRungeKuttaTableau(
    name="Dormand--Prince 5(4)",
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ),
        (
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ),
    ),
    b=(
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ),
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0),
    order=5,
    b_embedded=(
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ),
    embedded_order=4,
    fsal=True,
)


EXPLICIT_TABLEAUS = {
    "heun": HEUN_EULER,
    "rk4": CLASSIC_RK4,
    "dopri5": DORMAND_PRINCE_54,
}
