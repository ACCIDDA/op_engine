"""Required real Inductor execution, derivative and failure contracts."""

from __future__ import annotations

import os
import subprocess  # noqa: S404
import sys
from typing import Any

import numpy as np
import pytest

from op_engine import CoreSolver, ModelCore, array_namespace, loop_ops
from op_engine.core_solver import AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions
from op_engine.torch_loop import torch_compile_loop_adapter


def _torch() -> Any:  # noqa: ANN401
    torch = pytest.importorskip("torch")
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 14):
        pytest.skip("Separate compiled-loop contract requires Torch >=2.14")
    return torch


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("method", ["euler", "heun", "rk4", "dopri5"])
def test_torch_native_fixed_methods_and_dynamic_gradients(
    method: str, dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent stability polynomials check full-step values and derivatives."""
    torch = _torch()
    adapter = torch_compile_loop_adapter(max_steps=8)
    monkeypatch.setattr(loop_ops, "_LOOP_ADAPTERS", {})
    loop_ops.register_loop_adapter(array_namespace(torch.ones(1)).__name__, adapter)
    polynomials: dict[str, list[float]] = {
        "euler": [1, 1],
        "heun": [1, 1, 0.5],
        "rk4": [1, 1, 0.5, 1 / 6, 1 / 24],
        "dopri5": [1, 1, 0.5, 1 / 6, 1 / 24, 1 / 120, 1 / 600],
    }
    coefficients = polynomials[method]
    for values in ((-0.7, 2.0), (-0.3, 1.2)):
        parameters = torch.tensor(
            values, dtype=getattr(torch, dtype), requires_grad=True
        )
        core = ModelCore(
            1,
            1,
            np.array([0.0, 0.5, 1.0]),
            options=ModelCoreOptions(dtype=getattr(np, dtype)),
        )
        core.set_initial_state(parameters[1].reshape(1, 1))
        solver = CoreSolver(core)
        solver.run(
            lambda _t, y, current=parameters: current[0] * y,
            config=RunConfig(method=method, fixed_loop="scan", fixed_max_step=0.25),
        )
        result: Any = core.get_current_state()
        gradient = torch.autograd.grad(result.sum(), parameters)[0]
        z = values[0] * 0.25
        factor = np.polynomial.polynomial.polyval(z, coefficients)
        derivative = np.polynomial.polynomial.polyval(
            z, np.polynomial.polynomial.polyder(coefficients)
        )
        tolerance = 3e-6 if dtype == "float32" else 3e-13
        np.testing.assert_allclose(
            result.detach().numpy(), values[1] * factor**4, rtol=tolerance
        )
        np.testing.assert_allclose(
            gradient.detach().numpy(),
            [values[1] * factor**3 * derivative, factor**4],
            rtol=tolerance,
        )
        assert result.dtype == parameters.dtype
        history: Any = core.state_array
        assert history is not None
        np.testing.assert_allclose(
            history.detach().numpy().reshape(-1),
            values[1] * factor ** np.array([0, 2, 4]),
            rtol=tolerance,
        )


def test_torch_nested_none_and_observation_trees() -> None:
    """Nested sequential loops preserve optional carry and measured output trees."""
    torch = _torch()
    adapter = torch_compile_loop_adapter(max_steps=8)
    initial = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)

    def advance(carry: Any, step: Any) -> tuple[Any, None]:  # noqa: ANN401
        return {"state": carry["state"] * (1 + rate * step), "cache": None}, None

    def interval(carry: Any, steps: Any) -> tuple[Any, Any]:  # noqa: ANN401
        carry, unused = adapter.scan(advance, carry, steps)
        assert unused is None
        return carry, (carry["state"], {"square": carry["state"] ** 2, "unused": None})

    final, output = adapter.scan(
        interval,
        {"state": initial, "cache": None},
        torch.full((2, 2), 0.25, dtype=torch.float64),
    )
    objective = final["state"] + output[1]["square"].sum()
    grad_initial, grad_rate = torch.autograd.grad(objective, (initial, rate))
    factor = 1 - 0.7 * 0.25
    powers = np.array([2, 4])
    np.testing.assert_allclose(output[0].detach().numpy(), 2 * factor**powers)
    np.testing.assert_allclose(
        float(grad_initial), factor**4 + 4 * np.sum(factor ** (2 * powers))
    )
    np.testing.assert_allclose(
        float(grad_rate),
        2 * factor**3 + 2 * np.sum(powers * factor ** (2 * powers - 1)),
    )
    assert final["cache"] is None
    assert output[1]["unused"] is None


@pytest.mark.parametrize("replay", [False, True])
def test_torch_one_step_has_no_empty_adapter_call(
    *, replay: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bootstrap step requires no unknown empty-output inference."""
    torch = pytest.importorskip("torch")

    def forbidden(*_args: Any) -> Any:  # noqa: ANN401
        pytest.fail("A one-step solve called an empty scan")

    monkeypatch.setattr(
        loop_ops,
        "_LOOP_ADAPTERS",
        {array_namespace(torch.ones(1)).__name__: loop_ops.LoopAdapter(scan=forbidden)},
    )
    initial = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    core = ModelCore(1, 1, np.array([0.0, 0.125]))
    core.set_initial_state(initial)
    solver = CoreSolver(core)
    if replay:
        solver.replay_adaptive_schedule(
            lambda _t, y: -y,
            AdaptiveStepSchedule((0.0, 0.125), ((0.125,),)),
            config=RunConfig(method="rk4", adaptive=True, replay_loop="scan"),
        )
    else:
        solver.run(lambda _t, y: -y, config=RunConfig(method="rk4", fixed_loop="scan"))
    final: Any = core.get_current_state()
    final.sum().backward()
    assert initial.grad is not None
    np.testing.assert_allclose(final.detach().numpy(), 2 * initial.grad.numpy())


def test_torch_rejects_unsupported_bounds_dtypes_and_inputs() -> None:
    """Explicit compilation requests fail before silently selecting another path."""
    torch = _torch()
    for bound in (0, 129, True):
        with pytest.raises(ValueError, match="max_steps"):
            torch_compile_loop_adapter(max_steps=bound)
    adapter = torch_compile_loop_adapter(max_steps=4, max_elements=8)
    initial = torch.tensor(1.0)
    for inputs, message in (
        (torch.ones(0), "length"),
        (torch.ones(5), "length"),
        (torch.tensor(1.0), "leading"),
        (torch.ones(4, 4), "max_elements"),
        (torch.ones(2, dtype=torch.int64), "float32/float64"),
        (torch.ones(2, device="meta"), "CPU"),
        ((torch.ones(2), torch.ones(3)), "leading"),
    ):
        with pytest.raises(ValueError, match=message):
            adapter.scan(lambda carry, _step: (carry, carry), initial, inputs)
    with pytest.raises(TypeError, match="Tensor/None"):
        adapter.scan(lambda carry, _step: (carry, carry), 1.0, torch.ones(2))
    with pytest.raises(ValueError, match="one floating dtype"):
        adapter.scan(
            lambda carry, _step: (carry, carry), initial.double(), torch.ones(2)
        )
    assert adapter.checkpoint is None


def test_torch_graph_break_with_suppression_still_raises() -> None:
    """Fullgraph compilation does not adopt global eager fallback settings."""
    torch = _torch()
    adapter = torch_compile_loop_adapter()

    def broken(carry: Any, step: Any) -> tuple[Any, Any]:  # noqa: ANN401
        torch._dynamo.graph_break()  # noqa: SLF001 - exercise a real compiler break
        return carry + step, carry

    with (
        torch._dynamo.config.patch(suppress_errors=True),  # noqa: SLF001
        pytest.raises(Exception, match=r"graph_break|Graph break"),
    ):
        adapter.scan(broken, torch.tensor(1.0), torch.ones(2))


def test_torch_disabled_compilation_raises_in_fresh_process() -> None:
    """The startup environment switch cannot silently turn the adapter eager."""
    _torch()
    program = """
import torch
from op_engine.torch_loop import torch_compile_loop_adapter
adapter = torch_compile_loop_adapter()
try:
    adapter.scan(
        lambda carry, step: (carry + step, carry), torch.tensor(1.), torch.ones(2)
    )
except RuntimeError as error:
    assert 'disabled' in str(error), str(error)
else:
    raise AssertionError('Disabled compiler returned a silent eager result')
"""
    subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        env={**os.environ, "TORCH_COMPILE_DISABLE": "1"},
        timeout=60,
        check=True,
        capture_output=True,
        text=True,
    )


def test_torch_minimum_version_is_explicitly_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager minimum support does not imply a qualified compiler version."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "__version__", "2.6.0+cpu")
    with pytest.raises(ValueError, match=r"Torch >=2\.14"):
        torch_compile_loop_adapter()


def test_torch_body_contract_violations_cannot_return_a_result() -> None:
    """Changing carry or oversized output cannot escape the compiled bounds."""
    torch = _torch()
    cases = (
        lambda carry, step: ((carry, None), carry + step),
        lambda carry, step: (carry + step, torch.ones(8)),
        lambda carry, step: (carry.double() + step, carry + step),
    )
    for body in cases:
        adapter = torch_compile_loop_adapter(max_steps=4, max_elements=8)
        with pytest.raises(
            Exception, match=r"Observed exception|preserve|max_elements"
        ):
            adapter.scan(body, torch.tensor(1.0), torch.ones(2))
