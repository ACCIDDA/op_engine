"""Tests for reusable explicit Runge--Kutta tableaus."""

# ruff: noqa: PLC2701

from __future__ import annotations

import pytest

from op_engine._runge_kutta import (
    CLASSIC_RK4,
    DORMAND_PRINCE_54,
    HEUN_EULER,
    ExplicitRungeKuttaTableau,
)


def test_builtin_tableaus_expose_expected_orders_and_stages() -> None:
    """Named tableaus retain their mathematical metadata."""
    assert (HEUN_EULER.order, HEUN_EULER.embedded_order) == (2, 1)
    assert (CLASSIC_RK4.order, CLASSIC_RK4.n_stages) == (4, 4)
    assert (
        DORMAND_PRINCE_54.order,
        DORMAND_PRINCE_54.embedded_order,
        DORMAND_PRINCE_54.n_stages,
        DORMAND_PRINCE_54.fsal,
    ) == (5, 4, 7, True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"a": ((),), "b": (0.5, 0.5)}, "dimensions"),
        ({"a": ((1.0,),), "b": (1.0,), "c": (0.0,)}, "row i"),
        ({"b": (0.25, 0.25)}, "sum to one"),
        ({"a": ((), (0.25,)), "c": (0.0, 0.5)}, "sum to c"),
        (
            {"b_embedded": (1.0, 0.0), "embedded_order": None},
            "provided together",
        ),
    ],
)
def test_tableau_rejects_inconsistent_coefficients(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Malformed coefficient families fail when they are declared."""
    defaults: dict[str, object] = {
        "name": "test 2(1)",
        "a": ((), (1.0,)),
        "b": (0.5, 0.5),
        "c": (0.0, 1.0),
        "order": 2,
        "b_embedded": (1.0, 0.0),
        "embedded_order": 1,
    }
    with pytest.raises(ValueError, match=match):
        ExplicitRungeKuttaTableau(**(defaults | kwargs))  # type: ignore[arg-type]


def test_tableau_rejects_inconsistent_fsal_claim() -> None:
    """FSAL is validated from coefficients rather than trusted as metadata."""
    with pytest.raises(ValueError, match="FSAL"):
        ExplicitRungeKuttaTableau(
            name="not FSAL",
            a=((), (1.0,)),
            b=(0.5, 0.5),
            c=(0.0, 1.0),
            order=2,
            fsal=True,
        )
