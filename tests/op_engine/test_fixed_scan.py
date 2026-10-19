"""Fixed-mesh scan equations, transforms and bounded traced control flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore, Scalar, array_namespace, loop_ops
from op_engine.core_solver import RunConfig
from op_engine.loop_ops import LoopAdapter
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_long_fixed_scan_example() -> None:
    """The public example checks its solution and gradient independently."""
    import runpy  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    pytest.importorskip("jax")
    example = Path(__file__).resolve().parents[2] / "examples/fixed_scan.py"
    runpy.run_path(str(example))["main"]()


@pytest.fixture
def jax_x64() -> Iterator[None]:
    """Scope precision for independent trajectory and derivative checks."""
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        yield


def _solve(  # noqa: PLR0913
    parameters: Array,
    times: np.ndarray,
    *,
    method: str = "rk4",
    loop: str = "scan",
    checkpoint: bool = False,
    store_history: bool = True,
    max_step: float | None = 0.125,
) -> tuple[Array, Array | None]:
    """Solve scalar exponential dynamics with dynamic rate and initial state.

    Returns:
        Final state and optional trajectory.
    """
    xp = array_namespace(parameters)
    core = ModelCore(
        1,
        1,
        times,
        options=ModelCoreOptions(dtype=np.float64, store_history=store_history),
    )
    core.set_initial_state(xp.reshape(parameters[1], (1, 1)))

    def rhs(_time: Scalar, state: Array) -> Array:
        return cast("Array", parameters[0] * state)

    solver = CoreSolver(core)
    solver.run(
        rhs,
        config=RunConfig(
            method=method,
            fixed_loop=loop,
            fixed_checkpoint=checkpoint,
            fixed_max_step=max_step,
        ),
    )
    assert solver.last_adaptive_schedule is None
    assert solver.last_nonlinear_diagnostics is None
    assert core.current_step == core.n_timesteps - 1
    return core.get_current_state(), core.state_array


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
@pytest.mark.parametrize("store_history", [False, True])
def test_fixed_scan_preserves_method_and_output_mesh(
    method: str, *, store_history: bool
) -> None:
    """Independent stability polynomials distinguish full from half steps."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    coefficients = {
        "euler": [1, 1],
        "heun": [1, 1, 1 / 2],
        "rk4": [1, 1, 1 / 2, 1 / 6, 1 / 24],
        "dopri5": [1, 1, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 600],
    }
    parameters = jnp.asarray([-1.3, 2.0])
    times = np.asarray([0.0, 0.25, 1.0])
    final, history = jax.jit(
        lambda p: _solve(p, times, method=method, store_history=store_history)
    )(parameters)
    factor = np.polynomial.polynomial.polyval(-1.3 * 0.125, coefficients[method])
    expected = 2.0 * factor ** np.asarray([0, 2, 8])
    np.testing.assert_allclose(final, [[expected[-1]]], rtol=2e-14)
    assert array_namespace(final) is jnp
    if store_history:
        np.testing.assert_allclose(history, expected[:, None, None], rtol=2e-14)
    else:
        assert history is None


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_fixed_scan_gradients_and_vmap(*, checkpoint: bool) -> None:
    """Rates and initial states stay dynamic through jit, grad and vmap."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    times = np.asarray([0.0, 0.25, 1.0])

    def objective(parameters: Array) -> Array:
        final, _ = _solve(parameters, times, checkpoint=checkpoint, store_history=False)
        return final[0, 0]

    parameters = np.asarray([[-1.3, 2.0], [-0.4, 1.5]])
    actual = jax.jit(jax.vmap(jax.value_and_grad(objective)))(jnp.asarray(parameters))
    rate, initial = parameters.T
    z = rate / 8
    factor = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    derivative = 1 + z + z**2 / 2 + z**3 / 6
    expected = initial * factor**8
    np.testing.assert_allclose(actual[0], expected, rtol=2e-14)
    np.testing.assert_allclose(actual[1][:, 0], initial * factor**7 * derivative)
    np.testing.assert_allclose(actual[1][:, 1], factor**8)


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("times", [[0.0], [0.0, 0.125]])
def test_empty_and_one_step_mesh(times: list[float]) -> None:
    """No scan iterations are needed for an initial state or a single step."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    actual, _ = jax.jit(lambda p: _solve(p, np.asarray(times)))(jnp.asarray([0.0, 3.0]))
    np.testing.assert_array_equal(actual, [[3.0]])


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize("method", ["heun", "rk4", "dopri5"])
def test_scan_splits_declared_forcing_jumps(method: str) -> None:
    """A jump inside an output interval uses one-sided endpoint stages."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    def solve(initial: Array) -> Array:
        core = ModelCore(1, 1, np.asarray([0.0, 1.0]))
        core.set_initial_state(initial)

        def rhs(time: Scalar, state: Array) -> Array:
            return jnp.ones_like(state) * jnp.where(time < 0.375, 2.0, -1.0)

        CoreSolver(core).run(
            rhs,
            config=RunConfig(
                method=method,
                fixed_loop="scan",
                forcing_breakpoints=(0.375,),
                fixed_max_step=0.25,
            ),
        )
        return core.get_current_state()

    actual = jax.jit(solve)(jnp.asarray([[1.0]]))
    np.testing.assert_allclose(actual, [[1.0 + 2.0 * 0.375 - 0.625]], atol=2e-15)


@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
def test_numpy_auto_preserves_eager_bytes(method: str) -> None:
    """An unregistered namespace retains the existing arithmetic path."""
    parameters = np.asarray([-1.3, 2.0])
    times = np.asarray([0.0, 0.25, 1.0])
    eager = _solve(parameters, times, method=method, loop="unroll")
    automatic = _solve(parameters, times, method=method, loop="auto")
    for actual, expected in zip(automatic, eager, strict=True):
        assert actual is not None
        assert expected is not None
        assert actual.tobytes() == expected.tobytes()


def test_fixed_loop_validation() -> None:
    """Unsupported modes fail before silently choosing another algorithm."""
    assert RunConfig().fixed_loop == "unroll"
    with pytest.raises(ValueError, match="fixed_loop"):
        RunConfig(fixed_loop="unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fixed_checkpoint"):
        RunConfig(fixed_checkpoint=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fixed explicit"):
        RunConfig(fixed_loop="scan", adaptive=True)
    with pytest.raises(ValueError, match="fixed explicit"):
        RunConfig(fixed_loop="auto", method="implicit-euler")
    with pytest.raises(ValueError, match="requires a loop adapter"):
        _solve(np.asarray([-1.0, 1.0]), np.asarray([0.0, 1.0]))


def test_checkpoint_requires_adapter_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom adapter must explicitly provide rematerialization."""
    namespace = array_namespace(np.asarray([1.0]))
    monkeypatch.setattr(
        loop_ops,
        "_LOOP_ADAPTERS",
        {
            namespace.__name__: LoopAdapter(
                scan=lambda _body, initial, _xs: (initial, None)
            )
        },
    )
    with pytest.raises(ValueError, match="does not support checkpointing"):
        _solve(np.asarray([-1.0, 1.0]), np.asarray([0.0, 1.0]), checkpoint=True)


@pytest.mark.usefixtures("jax_x64")
def test_traced_graph_keeps_one_scan_as_step_count_grows() -> None:
    """Graph structure, rather than noisy wall time, guards against unrolling."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    graphs = [
        jax.make_jaxpr(
            lambda p, count=count: _solve(
                p, np.linspace(0.0, 1.0, count + 1), max_step=None
            )[0]
        )(jnp.asarray([-1.0, 2.0])).jaxpr
        for count in (50, 200)
    ]
    scans = [
        [eq for eq in graph.eqns if eq.primitive.name == "scan"] for graph in graphs
    ]
    assert [len(items) for items in scans] == [1, 1]
    assert [items[0].params["length"] for items in scans] == [49, 199]
    assert len(graphs[0].eqns) == len(graphs[1].eqns)
    assert len(scans[0][0].params["jaxpr"].jaxpr.eqns) == len(
        scans[1][0].params["jaxpr"].jaxpr.eqns
    )
