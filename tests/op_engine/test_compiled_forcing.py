"""Pinned compiler held-table contracts in both native state precisions."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from op_engine import CoreSolver, ModelCore, array_namespace
from op_engine.core_solver import RunConfig
from op_engine.model_core import ModelCoreOptions

op_system = pytest.importorskip("op_system")


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compiled_previous_native_precision(backend: str, dtype: str) -> None:
    """An actual float64 table grid cannot pull outgoing float32 times left."""
    module = (
        np
        if backend == "numpy"
        else pytest.importorskip("jax.numpy" if backend == "jax" else "torch")
    )
    precision = nullcontext()
    if backend == "jax":
        jax = pytest.importorskip("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        precision = enable()
    with precision:
        compiled = op_system.compile_spec({
            "kind": "transitions",
            "time_interpolation": "previous",
            "axes": [{"name": "time", "type": "continuous", "coords": [0.0, 0.7, 1.0]}],
            "state": ["x", "z"],
            "transitions": [
                {"name": "transfer", "from": "x", "to": "z", "rate": "rate[time]"}
            ],
        })
        initial = module.asarray([[1.0], [0.0]], dtype=getattr(module, dtype))
        xp = array_namespace(initial)
        rates = xp.asarray([1.0, 0.0, 2.0], dtype=initial.dtype)
        core = ModelCore(
            2,
            1,
            np.asarray([0.0, 0.9, 1.5]),
            options=ModelCoreOptions(dtype=np.dtype(dtype)),
        )
        core.set_initial_state(initial)
        CoreSolver(core).run(
            lambda t, state: xp.reshape(
                compiled.eval_fn(t, xp.reshape(state, (2,)), rate=rates), state.shape
            ),
            config=RunConfig(
                method="dopri5", fixed_max_step=0.05, forcing_breakpoints=(0.7, 1.0)
            ),
        )
        result = core.get_current_state()
        host = np.asarray(
            result.detach().numpy() if hasattr(result, "detach") else result
        )
        expected_x = np.exp(-1.7)
        np.testing.assert_allclose(
            host[:, 0],
            [expected_x, 1.0 - expected_x],
            rtol=3e-6 if dtype == "float32" else 1e-8,
        )
        assert result.dtype == initial.dtype
        assert array_namespace(result) is xp
