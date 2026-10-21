"""State adoption retains graphs on the oldest and current supported Torch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import RunConfig
from op_engine.model_core import ModelCoreOptions


@pytest.mark.parametrize("operation", ["initial", "next", "deltas", "trajectory"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("store_history", [False, True])
def test_torch_native_state_adoption_keeps_graphs(
    operation: str, dtype: str, *, store_history: bool
) -> None:
    """Independent linear derivatives detect detachment, including dtype casts."""
    torch = pytest.importorskip("torch")
    value = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 1.0]),
        options=ModelCoreOptions(
            dtype=getattr(np, dtype),
            store_history=store_history,
        ),
    )
    core.set_initial_state(value.reshape(1, 1))
    if operation == "next":
        core.apply_next_state((3 * value).reshape(1, 1))
    elif operation == "deltas":
        core.apply_deltas((3 * value).reshape(1, 1))
    elif operation == "trajectory":
        core.apply_trajectory(torch.stack((value, 3 * value)).reshape(2, 1, 1))
    result: Any = core.get_current_state()
    coefficient = {"initial": 1.0, "next": 3.0, "deltas": 4.0, "trajectory": 3.0}[
        operation
    ]
    if store_history:
        result = result.sum() + core.state_array.sum()
        coefficient += 1.0 if operation == "initial" else 1.0 + coefficient
    else:
        result = result.sum()
    result.backward()
    assert value.grad.item() == coefficient
    assert core.get_current_state().dtype == getattr(torch, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_torch_fixed_solver_rate_and_initial_gradients(dtype: str) -> None:
    """A separate stability-polynomial derivative checks the entire state lifecycle."""
    torch = pytest.importorskip("torch")
    parameters = torch.tensor(
        [-0.7, 2.0], dtype=getattr(torch, dtype), requires_grad=True
    )
    core = ModelCore(
        1,
        1,
        np.asarray([0.0, 0.5, 1.0]),
        options=ModelCoreOptions(
            dtype=getattr(np, dtype),
            store_history=False,
        ),
    )
    core.set_initial_state(parameters[1].reshape(1, 1))

    def rhs(_time: object, y: Any) -> Any:  # noqa: ANN401
        return parameters[0] * y

    CoreSolver(core).run(rhs, config=RunConfig(method="rk4", fixed_max_step=0.25))
    core.get_current_state().sum().backward()
    factor = np.polynomial.polynomial.polyval(-0.7 * 0.25, [1, 1, 1 / 2, 1 / 6, 1 / 24])
    derivative = np.polynomial.polynomial.polyval(-0.7 * 0.25, [1, 1, 1 / 2, 1 / 6])
    np.testing.assert_allclose(
        parameters.grad.detach().numpy(),
        [2 * 4 * 0.25 * factor**3 * derivative, factor**4],
        rtol=2e-6,
        atol=1e-7,
    )
