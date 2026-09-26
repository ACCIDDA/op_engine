"""Validated coefficient tables and stage orchestration for Rosenbrock-W methods."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TypeVar

StageArray = TypeVar("StageArray")
WeightedSum = Callable[
    [StageArray, float, tuple[float, ...], Sequence[StageArray]], StageArray
]


@dataclass(slots=True, frozen=True)
class RosenbrockWTableau:
    """Coefficients for a Rosenbrock-W method in direct-coupling form.

    The stage increments use the convention

    ``(I - gamma*h*J) K_i = h*f(t + c_i*h, y + sum(a_ij*K_j))``
    ``                         + sum(g_ij*K_j)``.

    The accepted and embedded states are ``y + sum(b_i*K_i)`` and
    ``y + sum(b_embedded_i*K_i)``. Rows of ``a`` and ``g`` are stored in
    compact strictly lower-triangular form.

    Attributes:
        name: Human-readable method name.
        gamma: Shared diagonal coefficient in the stage solve.
        a: Stage-state coupling coefficients.
        g: Transformed Jacobian-coupling coefficients applied directly to
            earlier stage increments.
        b: Weights for the accepted, high-order state.
        c: Stage-time coefficients.
        order: Formal order of the accepted state.
        b_embedded: Weights for the embedded state.
        embedded_order: Formal order of the embedded state.
        controller_order: Order used by the adaptive step-size controller.
    """

    name: str
    gamma: float
    a: tuple[tuple[float, ...], ...]
    g: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    order: int
    b_embedded: tuple[float, ...]
    embedded_order: int
    controller_order: int

    def __post_init__(self) -> None:  # noqa: C901
        """Validate tableau dimensions and metadata.

        Raises:
            ValueError: If coefficients or order metadata are inconsistent.
        """
        n_stages = len(self.c)
        if not self.name.strip():
            msg = "Rosenbrock-W tableau name must not be empty"
            raise ValueError(msg)
        if n_stages < 1:
            msg = "Rosenbrock-W tableau must contain at least one stage"
            raise ValueError(msg)
        if not math.isfinite(self.gamma) or self.gamma == 0.0:
            msg = "Rosenbrock-W gamma must be finite and nonzero"
            raise ValueError(msg)
        if any(
            len(values) != n_stages
            for values in (self.a, self.g, self.b, self.b_embedded)
        ):
            msg = "Rosenbrock-W tableau dimensions must agree"
            raise ValueError(msg)

        for label, value in (
            ("order", self.order),
            ("embedded order", self.embedded_order),
            ("controller order", self.controller_order),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                msg = f"Rosenbrock-W {label} must be a positive integer"
                raise ValueError(msg)
        if self.embedded_order >= self.order:
            msg = "Rosenbrock-W embedded order must be below the solution order"
            raise ValueError(msg)
        if self.controller_order > self.order:
            msg = "Rosenbrock-W controller order must not exceed the solution order"
            raise ValueError(msg)

        coefficients = (
            self.gamma,
            *self.b,
            *self.b_embedded,
            *self.c,
            *(value for row in self.a for value in row),
            *(value for row in self.g for value in row),
        )
        if any(not math.isfinite(value) for value in coefficients):
            msg = "Rosenbrock-W coefficients must be finite"
            raise ValueError(msg)

        for stage, (a_row, g_row, stage_time) in enumerate(
            zip(self.a, self.g, self.c, strict=True)
        ):
            if len(a_row) != stage or len(g_row) != stage:
                msg = "Rosenbrock-W row i must contain exactly i coefficients"
                raise ValueError(msg)
            if not math.isclose(
                math.fsum(a_row),
                stage_time,
                rel_tol=1e-13,
                abs_tol=1e-15,
            ):
                msg = "Rosenbrock-W stage coefficients must sum to c"
                raise ValueError(msg)

    @property
    def n_stages(self) -> int:
        """Return the number of linearly implicit stages."""
        return len(self.c)


def evaluate_rosenbrock_w(  # noqa: PLR0913
    *,
    tableau: RosenbrockWTableau,
    t: float,
    dt: float,
    y: StageArray,
    rhs: Callable[[float, StageArray], StageArray],
    solve: Callable[[StageArray], StageArray],
    weighted_sum: WeightedSum[StageArray],
) -> tuple[StageArray, StageArray]:
    """Evaluate a Rosenbrock-W tableau with backend-provided array operations.

    Returns:
        Accepted and embedded states.
    """
    stages: list[StageArray] = []
    for a_row, g_row, stage_time in zip(
        tableau.a,
        tableau.g,
        tableau.c,
        strict=True,
    ):
        stage_state = weighted_sum(y, 1.0, a_row, stages)
        derivative = rhs(t + stage_time * dt, stage_state)
        stage_rhs = weighted_sum(derivative, dt, g_row, stages)
        stages.append(solve(stage_rhs))

    high = weighted_sum(y, 1.0, tableau.b, stages)
    embedded = weighted_sum(y, 1.0, tableau.b_embedded, stages)
    return high, embedded


# Two-stage, L-stable ROS2 formula expressed in increment form.
ROS2 = RosenbrockWTableau(
    name="ROS2 2(1)",
    gamma=1.0 - 1.0 / math.sqrt(2.0),
    a=((), (1.0,)),
    g=((), (-2.0,)),
    b=(1.5, 0.5),
    c=(0.0, 1.0),
    order=2,
    b_embedded=(1.0, 0.0),
    embedded_order=1,
    controller_order=1,
)


ROSENBROCK_W_TABLEAUS = {"ros2": ROS2}
