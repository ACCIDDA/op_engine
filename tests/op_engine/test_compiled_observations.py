"""Compiler-to-engine physical definitions on independently checked meshes."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from op_engine import (
    Array,
    CoreSolver,
    ModelCore,
    ObservationResult,
    array_namespace,
    observation_from_compiled,
    observe_model,
)
from op_engine.core_solver import RunConfig

TIMES = np.array([0.0, 0.15, 0.4])
MESHES = (
    ([0.25, 0.75], [0.5, 0.5]),
    ([0.25, 0.55, 0.8], [0.5, 0.1, 0.4]),
)


def _compiler(coords: list[float], widths: list[float]) -> Any:  # noqa: ANN401
    op_system = pytest.importorskip("op_system")
    return op_system.compile_spec({
        "axes": [
            {
                "name": "x",
                "type": "continuous",
                "coords": coords,
                "measure": {"kind": "cell", "weights": widths, "units": "m"},
            }
        ],
        "state": ["rho[x]"],
        "equations": {"rho[x]": "-k*rho[x]"},
        "state_metadata": {
            "rho": {
                "representation": "density",
                "measure_axes": ["x"],
                "units": "items/m",
            }
        },
        "parameters": {"gain": {"support": "positive"}},
        "observables": {
            "total": {
                "expr": "gain*rho[x]",
                "reduction": {"kind": "integral", "axes": ["x"]},
                "units": "items",
            },
            "point": {"expr": "rho[0.25]", "units": "items/m"},
            "field": {"expr": "rho[x]", "axes": ["x"]},
        },
    })


def _solve(compiled: Any, initial: Array, k: Array, gain: Array) -> ObservationResult:  # noqa: ANN401
    core = ModelCore(len(compiled.state_names), 1, TIMES)
    core.set_initial_state(initial)
    xp = array_namespace(initial)

    def rhs(t: object, y: Array) -> Array:
        flat = compiled.eval_fn(t, xp.reshape(y, (-1,)), k=k)
        return cast("Array", xp.reshape(flat, y.shape))

    CoreSolver(core).run(rhs, config=RunConfig(method="rk4", fixed_max_step=0.025))
    return observe_model(
        core,
        [observation_from_compiled(item) for item in compiled.observables.values()],
        params={"gain": gain},
    )


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_compiler_engine_two_meshes_physical_values(backend: str) -> None:
    """The cell integral is 2*gain*exp(-k*t); x=.25 samples 1.5*exp(-k*t)."""
    module = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    if backend == "jax":
        jax = pytest.importorskip("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        with enable():
            _check_mesh_values(module, backend)
    else:
        _check_mesh_values(module, backend)


def _check_mesh_values(module: Any, backend: str) -> None:  # noqa: ANN401
    results = []
    for coords, widths in MESHES:
        compiled = _compiler(coords, widths)
        initial = module.asarray(
            (1 + 2 * np.asarray(coords)).reshape(-1, 1), dtype=module.float64
        )
        result = _solve(
            compiled,
            initial,
            module.asarray(0.3, dtype=module.float64),
            module.asarray(2.0, dtype=module.float64),
        )
        values = result.series["total"].values
        host = values.detach().numpy() if backend == "torch" else values
        np.testing.assert_allclose(host, 4 * np.exp(-0.3 * TIMES), rtol=2e-10)
        point = result.series["point"].values
        np.testing.assert_allclose(
            point.detach().numpy() if backend == "torch" else point,
            1.5 * np.exp(-0.3 * TIMES),
            rtol=2e-10,
        )
        assert array_namespace(values) is array_namespace(initial)
        assert values.dtype == initial.dtype
        assert result.times == tuple(TIMES)
        assert result.series["field"].values.shape == (len(TIMES), len(coords))
        metadata = result.series["total"].spec.metadata
        assert metadata.units == "items"
        assert metadata.parameter_shapes == (("gain", ()),)
        assert metadata.measure_requirements[0][1].weights == tuple(widths)
        assert result.series["field"].spec.metadata.coordinates == (
            ("x", tuple(map(str, coords))),
        )
        results.append(np.asarray(host))
    np.testing.assert_allclose(results[0], results[1], rtol=1e-14)


def test_jax_compiler_engine_parameter_and_initial_gradients() -> None:
    """JIT/AD differentiates a fixed RK4 trajectory and observation-only gain."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        coords, widths = MESHES[1]
        compiled = _compiler(coords, widths)
        initial = jnp.asarray((1 + 2 * np.asarray(coords)).reshape(-1, 1))

        def objective(y: Array, k: Array, gain: Array) -> Array:
            return cast(
                "Array", jnp.sum(_solve(compiled, y, k, gain).series["total"].values)
            )

        actual = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(
            initial, jnp.asarray(0.3), jnp.asarray(2.0)
        )
        decay = np.exp(-0.3 * TIMES)
        np.testing.assert_allclose(
            actual[0], (2 * np.asarray(widths) * decay.sum()).reshape(-1, 1), rtol=2e-8
        )
        np.testing.assert_allclose(actual[1], -4 * np.sum(TIMES * decay), rtol=2e-8)
        np.testing.assert_allclose(actual[2], 2 * decay.sum(), rtol=2e-8)


def test_torch_compiler_engine_parameter_and_initial_gradients() -> None:
    """CPU Torch AD retains state/rate/gain sensitivities through the adapter."""
    torch = pytest.importorskip("torch")
    coords, widths = MESHES[1]
    compiled = _compiler(coords, widths)
    initial = torch.tensor(
        (1 + 2 * np.asarray(coords)).reshape(-1, 1),
        dtype=torch.float64,
        requires_grad=True,
    )
    k = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    gain = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    values = _solve(compiled, initial, k, gain).series["total"].values
    actual = torch.autograd.grad(torch.sum(values), (initial, k, gain))
    decay = np.exp(-0.3 * TIMES)
    np.testing.assert_allclose(
        actual[0].detach().numpy(),
        (2 * np.asarray(widths) * decay.sum()).reshape(-1, 1),
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        actual[1].detach().numpy(), -4 * np.sum(TIMES * decay), rtol=2e-8
    )
    np.testing.assert_allclose(actual[2].detach().numpy(), 2 * decay.sum(), rtol=2e-8)
