"""Current compiler expression of the engine's independent hierarchical fixture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from op_engine import array_namespace
from op_engine.diffrax import compiled_diffrax_term

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarks.hierarchical_ode import HierarchicalProblem
    from op_engine import Array


def compiled_hierarchical_term(
    problem: HierarchicalProblem,
) -> tuple[Any, Callable[[Array], dict[str, Array]]]:
    """Compile the same nonlinear balances, keeping rates as native runtime args.

    Returns:
        One reusable PyTree ODETerm and the named runtime-parameter packing map.
    """
    from op_system import compile_spec  # noqa: PLC0415 - optional benchmark producer

    driver = "A[group,cell]"
    if problem.coupled:
        driver += f" + 0.15*sum_over(A[group:g,cell],group=g)/{problem.groups}"
    compiled = compile_spec({
        "kind": "expr",
        "axes": [
            {"name": "group", "coords": [f"g{i}" for i in range(problem.groups)]},
            {
                "name": "cell",
                "coords": [f"c{i}" for i in range(problem.local_dimension)],
            },
        ],
        "state": ["A[group,cell]", "B[group,cell]"],
        "aliases": {
            "driver[group,cell]": driver,
            "net[group,cell]": (
                "gain*local_rate[group,cell]*A[group,cell]*driver[group,cell]"
                "/(1+A[group,cell]) - recovery*B[group,cell]"
            ),
        },
        "equations": {
            "A[group,cell]": "-net[group,cell]",
            "B[group,cell]": "net[group,cell]",
        },
    })

    def parameters(values: Array) -> dict[str, Array]:
        xp = array_namespace(values)
        return {
            "gain": cast("Array", xp.exp(values[0])),
            "recovery": cast("Array", xp.exp(values[1])),
            "local_rate": cast(
                "Array",
                xp.exp(
                    xp.reshape(
                        values[2 : 2 + problem.cells],
                        (problem.groups, problem.local_dimension),
                    )
                ),
            ),
        }

    return compiled_diffrax_term(compiled), parameters
