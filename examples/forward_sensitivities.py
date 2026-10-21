"""Compare bounded forward equations with derivatives of the same fixed solve.

Requires the optional op_system derivative artifact; core op_engine has no
compiler dependency. Run each backend/mode in a fresh process for timing.
Both modes return the full Jacobian of two final states and one measurement
with respect to two rates, measurement gain and initial amplitude.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import resource
import statistics
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import RunConfig
from op_engine.model_core import ModelCoreOptions
from op_engine.sensitivities import (
    SensitivityInitial,
    SensitivityOptions,
    SensitivitySystem,
    solve_sensitivities,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def _analytic() -> NDArray[np.float64]:
    a, b, gain, amplitude = 0.7, 1.1, 1.5, 2.0
    left, right = np.exp(-a), np.exp(-b)
    difference, factor = left - right, a / (b - a)
    x = np.array([-amplitude * left, 0.0, 0.0, left])
    z = np.array([
        amplitude * (b / (b - a) ** 2 * difference - factor * left),
        amplitude * (-a / (b - a) ** 2 * difference + factor * right),
        0.0,
        factor * difference,
    ])
    observed = gain * (x + 2 * z)
    observed[2] = amplitude * (left + 2 * factor * difference)
    return np.stack((x, z, observed))


def _functions(
    xp: Any,  # noqa: ANN401
    steps: int,
) -> tuple[Callable[..., Any], Callable[..., Any], str]:
    from op_system import compile_spec, symbolic_derivatives  # noqa: PLC0415

    compiled = compile_spec({
        "kind": "expr",
        "state": ["x", "z"],
        "aliases": {"transfer": "a*x-b*z"},
        "equations": {"x": "-a*x", "z": "transfer"},
        "observables": {"measurement": {"expr": "gain*(x+2*z)"}},
    })
    system = SensitivitySystem(
        compiled.eval_fn, symbolic_derivatives(compiled), ("amplitude",)
    )
    config = RunConfig(
        method="rk4",
        fixed_max_step=1 / steps,
        fixed_loop="scan" if xp.__name__ == "jax.numpy" else "unroll",
    )

    def initial(vector: Any) -> SensitivityInitial:  # noqa: ANN401
        zero, one = xp.zeros_like(vector[0]), xp.ones_like(vector[0])
        return SensitivityInitial(
            xp.stack((vector[3], zero)),
            xp.stack((
                xp.stack((zero, zero, zero, one)),
                xp.stack((zero, zero, zero, zero)),
            )),
        )

    def augmented(vector: Any) -> Any:  # noqa: ANN401
        solution = solve_sensitivities(
            system,
            dict(zip(system.parameter_names, vector, strict=True)),
            initial(vector),
            np.asarray([0.0, 1.0]),
            options=SensitivityOptions(solver=config),
        )
        return xp.concat(
            (solution.parameter_derivatives[0], solution.observable_derivatives[0]),
            axis=0,
        )

    def primal(vector: Any) -> Any:  # noqa: ANN401
        bindings = dict(
            zip(system.derivatives.parameter_names, vector[:3], strict=True)
        )
        core = ModelCore(
            2,
            1,
            np.asarray([0.0, 1.0]),
            options=ModelCoreOptions(store_history=False, dtype=np.float64),
        )
        core.set_initial_state(xp.reshape(initial(vector).state, (2, 1)))

        def rhs(t: Any, y: Any) -> Any:  # noqa: ANN401
            return xp.reshape(
                compiled.eval_fn(t, xp.reshape(y, (2,)), **bindings), (2, 1)
            )

        CoreSolver(core).run(rhs, config=config)
        y = xp.reshape(core.get_current_state(), (2,))
        measured = compiled.observables["measurement"].eval_fn(1.0, y, **bindings)
        return xp.concat((y, xp.reshape(measured, (1,))))

    return augmented, primal, system.derivatives.model.digest


def _transform(
    backend: str, mode: str, augmented: Callable[..., Any], primal: Callable[..., Any]
) -> Callable[..., Any]:
    if backend == "jax":
        jax = importlib.import_module("jax")
        return jax.jit(augmented if mode == "augmented" else jax.jacfwd(primal))
    if mode == "augmented":
        return augmented
    if backend == "torch":
        torch = importlib.import_module("torch")
        return lambda value: torch.autograd.functional.jacobian(primal, value)

    def difference(value: Any) -> Any:  # noqa: ANN401
        columns = []
        for index in range(len(value)):
            offset = np.zeros_like(value)
            offset[index] = 1e-5
            columns.append((primal(value + offset) - primal(value - offset)) / 2e-5)
        return np.stack(columns, axis=1)

    return difference


def _measure(backend: str, mode: str, steps: int) -> dict[str, object]:
    xp = importlib.import_module("jax.numpy" if backend == "jax" else backend)
    augmented, primal, identity = _functions(xp, steps)
    function = _transform(backend, mode, augmented, primal)
    vector = xp.asarray([0.7, 1.1, 1.5, 2.0], dtype=xp.float64)
    elapsed = []
    for _ in range(6):
        start = time.perf_counter()
        result = function(vector)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed.append(time.perf_counter() - start)
    host = result.detach().numpy() if backend == "torch" else np.asarray(result)
    tolerance = 2e-6 * (16 / steps) ** 4 + 1e-9
    np.testing.assert_allclose(host, _analytic(), rtol=1e-8, atol=tolerance)
    return {
        "backend": backend,
        "mode": mode,
        "method": "rk4",
        "steps": steps,
        "output_mesh": [0.0, 1.0],
        "adaptive": False,
        "tolerances": None,
        "dtype": "float64",
        "device": "cpu",
        "physical_states": 2,
        "parameters": ["a", "b", "gain", "amplitude"],
        "augmented_dimension": 10,
        "returned_jacobian_shape": [3, 4],
        "model": identity,
        "first_seconds": elapsed[0],
        "warm_seconds": statistics.median(elapsed[1:]),
        "warm_repetitions": 5,
        "analytic_max_absolute_error": float(np.max(np.abs(host - _analytic()))),
        "analytic_absolute_tolerance": tolerance,
        "jacobian": host.tolist(),
        "versions": {
            name: version(name) for name in ("numpy", "op-system", "op-engine", backend)
        },
        "sensitivity_source_sha256": hashlib.sha256(
            Path(inspect.getfile(solve_sensitivities)).read_bytes()
        ).hexdigest(),
        "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "whole_worker_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "derivative_method": (
            "forward equations"
            if mode == "augmented"
            else {
                "numpy": "central finite difference, h=1e-5",
                "jax": "native forward AD",
                "torch": "native reverse AD",
            }[backend]
        ),
    }


def main() -> None:
    """Check one synthetic derivative workload and print advisory timing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "jax", "torch"), required=True)
    parser.add_argument("--mode", choices=("augmented", "ad"), required=True)
    parser.add_argument("--steps", type=int, choices=(16, 64), default=16)
    args = parser.parse_args()
    context = nullcontext()
    if args.backend == "jax":
        jax = importlib.import_module("jax")
        enable = getattr(jax, "enable_x64", None)
        if enable is None:
            enable = jax.experimental.enable_x64
        context = enable()
    with context:
        result = _measure(args.backend, args.mode, args.steps)
    print(json.dumps(result, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
