"""Differentiate a fixed-mesh solve without unrolling its internal steps.

Requires the optional JAX extra. Run ``uv run python examples/fixed_scan.py``.
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    """Check a long fixed solve and its derivative against exponential decay."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from op_engine import CoreSolver, ModelCore  # noqa: PLC0415
    from op_engine.core_solver import RunConfig  # noqa: PLC0415
    from op_engine.model_core import ModelCoreOptions  # noqa: PLC0415

    def final_value(rate: jax.Array) -> jax.Array:
        # Requested outputs and internal step size are static under jit.
        core = ModelCore(
            1,
            1,
            np.asarray([0.0, 10.0]),
            options=ModelCoreOptions(store_history=False),
        )
        core.set_initial_state(jnp.asarray([[1.0]], dtype=rate.dtype))
        CoreSolver(core).run(
            lambda _time, state: -rate * state,
            config=RunConfig(method="rk4", fixed_max_step=0.01, fixed_loop="scan"),
        )
        return core.get_current_state()[0, 0]

    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64
    with enable_x64():
        value, derivative = jax.jit(jax.value_and_grad(final_value))(
            jnp.asarray(0.2, dtype=jnp.float64)
        )
    expected = np.exp(-2.0)
    np.testing.assert_allclose(value, expected, rtol=1e-9)
    np.testing.assert_allclose(derivative, -10.0 * expected, rtol=1e-9)
    print(  # noqa: T201 - command-line example output
        f"Final value: {float(value):.6f}; rate derivative: {float(derivative):.6f}"
    )


if __name__ == "__main__":
    main()
