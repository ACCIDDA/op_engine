"""Tests for reusable Rosenbrock-W coefficient tables."""

# ruff: noqa: PLC2701

from __future__ import annotations

import pytest

from op_engine._rosenbrock import ROS2, RosenbrockWTableau


def test_ros2_exposes_method_and_controller_orders() -> None:
    """ROS2 publishes the metadata used for accuracy and step control."""
    assert (ROS2.order, ROS2.embedded_order, ROS2.controller_order) == (2, 1, 1)
    assert ROS2.n_stages == 2
    assert ROS2.a == ((), (1.0,))
    assert ROS2.g == ((), (-2.0,))
    assert ROS2.b == (1.5, 0.5)
    assert ROS2.b_embedded == (1.0, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"gamma": 0.0}, "gamma"),
        ({"a": ((),), "g": ((),), "b": (1.0,)}, "dimensions"),
        (
            {
                "a": ((1.0,),),
                "g": ((),),
                "b": (1.0,),
                "b_embedded": (1.0,),
                "c": (0.0,),
            },
            "row i",
        ),
        ({"a": ((), (0.5,))}, "sum to c"),
        ({"embedded_order": 2}, "below the solution order"),
        ({"controller_order": 0}, "positive integer"),
        ({"controller_order": 3}, "must not exceed"),
    ],
)
def test_tableau_rejects_inconsistent_coefficients(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Malformed Rosenbrock-W families fail when they are declared."""
    defaults: dict[str, object] = {
        "name": "test 2(1)",
        "gamma": 0.25,
        "a": ((), (1.0,)),
        "g": ((), (-2.0,)),
        "b": (1.5, 0.5),
        "c": (0.0, 1.0),
        "order": 2,
        "b_embedded": (1.0, 0.0),
        "embedded_order": 1,
        "controller_order": 1,
    }
    with pytest.raises(ValueError, match=match):
        RosenbrockWTableau(**(defaults | kwargs))  # type: ignore[arg-type]
