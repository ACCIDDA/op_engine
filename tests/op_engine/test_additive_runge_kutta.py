"""Tests for reusable paired additive Runge--Kutta tableaus."""

# ruff: noqa: PLC2701

from __future__ import annotations

import pytest

from op_engine._additive_runge_kutta import (
    ARS443_32,
    AdditiveRungeKuttaTableau,
)


def test_ars443_exposes_validated_pair_metadata() -> None:
    """The built-in pair retains its shared-stage orders and coefficients."""
    assert ARS443_32.name == "ARS(4,4,3) 3(2)"
    assert (ARS443_32.order, ARS443_32.embedded_order) == (3, 2)
    assert ARS443_32.n_stages == 5
    assert ARS443_32.c == (0.0, 0.5, 2.0 / 3.0, 0.5, 1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"a_explicit": ((),)}, "dimensions"),
        ({"a_explicit": ((0.0,), (1.0,))}, "Explicit row"),
        ({"a_implicit": ((0.0,), (1.0,))}, "Implicit row"),
        ({"a_explicit": ((), (0.5,))}, "sum to c"),
        ({"b_explicit": (0.25, 0.25)}, "sum to one"),
        ({"embedded_order": 2}, "below solution order"),
        (
            {
                "a_implicit": ((0.0,), (1.0, 0.0)),
                "b_implicit": (0.0, 1.0),
            },
            "zero-diagonal",
        ),
    ],
)
def test_additive_tableau_rejects_inconsistent_coefficients(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Malformed paired coefficients fail when they are declared."""
    defaults: dict[str, object] = {
        "name": "test 2(1)",
        "a_explicit": ((), (1.0,)),
        "a_implicit": ((0.0,), (0.0, 1.0)),
        "b_explicit": (0.5, 0.5),
        "b_implicit": (0.0, 1.0),
        "c": (0.0, 1.0),
        "order": 2,
        "b_explicit_embedded": (1.0, 0.0),
        "b_implicit_embedded": (0.0, 1.0),
        "embedded_order": 1,
    }
    with pytest.raises(ValueError, match=match):
        AdditiveRungeKuttaTableau(**(defaults | kwargs))  # type: ignore[arg-type]
