# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Provider loop selection against independent discrete maps and gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from flepimop2.axis import ResolvedShape
from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
from flepimop2.typing import StateChangeEnum

from benchmarks.prepared_execution import (
    _configure_structured_options,  # noqa: PLC2701
    _RateSystem,  # noqa: PLC2701
)
from flepimop2.engine.op_engine import OpEngineEngineConfig, OpEngineFlepimop2Engine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    """Enable float64 without leaking a dtype policy to other provider tests."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    yield
    jax.config.update("jax_enable_x64", previous)


def _case(
    settings: dict[str, object],
    path: str,
    backend: str = "jax",
    dtype: str = "float64",
) -> tuple[Callable[[Any, Any], Any], Any, Any]:
    """Build a public ordinary/prepared solve with dynamic vector rates.

    Returns:
        Solve callable, rate vector and initial state vector.
    """
    xp = np if backend == "numpy" else pytest.importorskip("jax.numpy")
    if settings.get("state_layout") == "block":
        settings = {**settings, "block_axis": "batch"}
    config = OpEngineEngineConfig.model_validate({
        "method": "rk4",
        "fixed_max_step": 0.125,
        **settings,
    })
    engine = OpEngineFlepimop2Engine(state_change=StateChangeEnum.FLOW, config=config)
    system = _RateSystem()
    _configure_structured_options(system, 3)
    shape = ResolvedShape(("batch",), (3,))
    model_state = ModelStateSpecification(parameter_names=("x",))
    times = np.asarray([0.0, 0.25, 1.0])
    rate = xp.asarray([-1.3, -0.4, 0.2], dtype=dtype)
    initial = xp.asarray([2.0, 1.5, 0.7], dtype=dtype)
    if path == "prepared":
        prepared = engine.prepare(
            system,
            times,
            {"x": ParameterValue(initial, shape)},
            {"rate": ParameterValue(rate, shape)},
            model_state=model_state,
        )

        def solve(active_rate: Any, active_initial: Any) -> Any:  # noqa: ANN401
            return prepared({"x": active_initial}, {"rate": active_rate})

    else:

        def solve(active_rate: Any, active_initial: Any) -> Any:  # noqa: ANN401
            return engine.run(
                system,
                times,
                {"x": ParameterValue(active_initial, shape)},
                {"rate": ParameterValue(active_rate, shape)},
                model_state=model_state,
            )

    return solve, rate, initial


_METHOD_LAYOUTS = [
    ("euler", "flat"),
    ("heun", "flat"),
    *(
        (method, layout)
        for method in ("rk4", "dopri5")
        for layout in ("flat", "pytree", "block")
    ),
]
_COEFFICIENTS = {
    "euler": [1.0, 1.0],
    "heun": [1.0, 1.0, 0.5],
    "rk4": [1.0, 1.0, 0.5, 1 / 6, 1 / 24],
    "dopri5": [1.0, 1.0, 0.5, 1 / 6, 1 / 24, 1 / 120, 1 / 600],
}


@pytest.mark.parametrize(("method", "layout"), _METHOD_LAYOUTS)
@pytest.mark.parametrize("path", ["ordinary", "prepared"])
@pytest.mark.parametrize(
    ("loop", "checkpoint"), [("unroll", False), ("scan", False), ("scan", True)]
)
def test_fixed_loop_preserves_discrete_values_and_gradients(
    method: str, layout: str, path: str, loop: str, *, checkpoint: bool
) -> None:
    """Loop choice changes the graph, not the method or static step map."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    solve, rate, initial = _case(
        {
            "method": method,
            "state_layout": layout,
            "fixed_loop": loop,
            "fixed_scan_checkpoint": checkpoint,
        },
        path,
    )
    graph = jax.make_jaxpr(solve)(rate, initial)
    scans = [e for e in graph.jaxpr.eqns if e.primitive.name == "scan"]
    assert len(scans) == (0 if loop == "unroll" else 1)
    if checkpoint:
        assert "remat" in str(graph)

    def objective(r: Any, y: Any) -> Any:  # noqa: ANN401
        trajectory = solve(r, y)
        return jnp.sum(trajectory[-1, 1:]), trajectory

    (_value, trajectory), gradients = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    )(rate, initial)
    coefficients = _COEFFICIENTS[method]
    z = np.asarray(rate) / 8
    factor = np.polynomial.polynomial.polyval(z, coefficients)
    derivative = np.polynomial.polynomial.polyval(
        z, np.polynomial.polynomial.polyder(coefficients)
    )
    expected = np.asarray(initial) * factor ** np.asarray([0, 2, 8])[:, None]
    np.testing.assert_allclose(trajectory[:, 1:], expected, rtol=3e-13)
    np.testing.assert_allclose(
        gradients[0], initial * factor**7 * derivative, rtol=3e-12
    )
    np.testing.assert_allclose(gradients[1], factor**8, rtol=3e-13)
    np.testing.assert_array_equal(trajectory[:, 0], [0.0, 0.25, 1.0])
    assert trajectory.dtype == initial.dtype


@pytest.mark.parametrize("layout", ["flat", "pytree", "block"])
@pytest.mark.parametrize("path", ["ordinary", "prepared"])
def test_auto_preserves_legacy_jax_scan(layout: str, path: str) -> None:
    """Omitted and automatic controls retain the existing scan and arithmetic."""
    jax = pytest.importorskip("jax")
    legacy, rate, initial = _case({"state_layout": layout}, path)
    auto, _, _ = _case({"state_layout": layout, "fixed_loop": "auto"}, path)
    graph = jax.make_jaxpr(auto)(rate, initial)
    assert sum(e.primitive.name == "scan" for e in graph.jaxpr.eqns) == 1
    np.testing.assert_array_equal(
        jax.jit(legacy)(rate, initial), jax.jit(auto)(rate, initial)
    )


@pytest.mark.parametrize("layout", ["flat", "pytree"])
@pytest.mark.parametrize("path", ["ordinary", "prepared"])
@pytest.mark.parametrize("loop", ["auto", "unroll"])
def test_numpy_loop_controls_preserve_eager_bytes(
    layout: str, path: str, loop: str
) -> None:
    """NumPy keeps its eager method when no scan is forced."""
    legacy, rate, initial = _case({"state_layout": layout}, path, "numpy")
    selected, _, _ = _case({"state_layout": layout, "fixed_loop": loop}, path, "numpy")
    assert legacy(rate, initial).tobytes() == selected(rate, initial).tobytes()


@pytest.mark.parametrize("layout", ["flat", "pytree"])
@pytest.mark.parametrize("path", ["ordinary", "prepared"])
def test_forced_scan_rejects_numpy(layout: str, path: str) -> None:
    """An explicitly requested scan never silently falls back to eager NumPy."""
    solve, rate, initial = _case(
        {"state_layout": layout, "fixed_loop": "scan"}, path, "numpy"
    )
    with pytest.raises(TypeError, match=r"fixed_loop='scan'.*JAX"):
        solve(rate, initial)


@pytest.mark.parametrize("path", ["ordinary", "prepared"])
@pytest.mark.parametrize("loop", ["scan", "unroll"])
@pytest.mark.parametrize("enable_x64", [False, True])
def test_loop_choice_preserves_provider_precision_policy(
    path: str, loop: str, *, enable_x64: bool
) -> None:
    """Both drivers retain the provider's existing default-float promotion rule."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", enable_x64)
    solve, rate, initial = _case({"fixed_loop": loop}, path, dtype="float32")
    result = jax.jit(solve)(rate, initial)
    assert result.dtype == np.dtype("float64" if enable_x64 else "float32")
    factor = np.polynomial.polynomial.polyval(
        np.asarray(rate, dtype=np.float64) / 8, _COEFFICIENTS["rk4"]
    )
    np.testing.assert_allclose(result[-1, 1:], initial * factor**8, rtol=2e-6)


def test_prepared_cache_separates_controls_and_keeps_old_plan() -> None:
    """New loop settings miss the cache without mutating an existing executor."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    engine = OpEngineFlepimop2Engine(
        state_change=StateChangeEnum.FLOW,
        config=OpEngineEngineConfig(method="rk4", fixed_max_step=0.125),
    )
    system = _RateSystem()
    shape = ResolvedShape()
    initial = {"x": ParameterValue(jnp.asarray(1.0), shape)}
    params = {"rate": ParameterValue(jnp.asarray(-0.2), shape)}
    model_state = ModelStateSpecification(parameter_names=("x",))
    arguments = (system, np.asarray([0.0, 1.0]), initial, params)
    legacy = engine.prepare(*arguments, model_state=model_state)
    engine.config.fixed_loop = "unroll"
    unrolled = engine.prepare(*arguments, model_state=model_state)
    engine.config.fixed_loop = "scan"
    engine.config.fixed_scan_checkpoint = True
    checkpointed = engine.prepare(*arguments, model_state=model_state)
    assert engine.prepared_cache_size == 3
    assert legacy is not unrolled
    assert unrolled is not checkpointed
    graphs = [
        jax.make_jaxpr(plan)({"x": jnp.asarray(1.0)}, {"rate": jnp.asarray(-0.2)})
        for plan in (legacy, unrolled, checkpointed)
    ]
    assert [sum(e.primitive.name == "scan" for e in g.jaxpr.eqns) for g in graphs] == [
        1,
        0,
        1,
    ]
    assert "remat" in str(graphs[-1])
