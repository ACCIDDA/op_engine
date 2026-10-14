"""Check and refresh a compatible mesh after a nonlinear parameter change.

Run ``uv run python examples/replay_mesh_refresh.py``. A shared transfer rate
and two group-local multipliers drive a generic closed two-pool model. The
analytic solution independently checks replay accuracy before any mesh reuse
is trusted. Refresh is an eager decision outside a differentiated solve.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import numpy as np

from op_engine import Array, CoreSolver, ModelCore, array_namespace
from op_engine.core_solver import AdaptiveConfig, AdaptiveStepSchedule, RunConfig
from op_engine.model_core import ModelCoreOptions

INITIAL = np.asarray([[0.8, 0.6], [0.2, 0.4]])
LOCAL = np.asarray([1.0, 1.3])
CONFIG = RunConfig(
    method="heun",
    adaptive=True,
    adaptive_cfg=AdaptiveConfig(rtol=1e-4, atol=1e-7, dt_init=0.5),
    replay_model_id="quadratic-transfer:pool,group:v1",
    replay_require_provenance=True,
)


def trajectory(
    rate: float, schedule: AdaptiveStepSchedule | None = None
) -> tuple[np.ndarray, AdaptiveStepSchedule]:
    """Discover or replay this structured model at a shared transfer rate.

    Returns:
        Final state and its accepted mesh.

    Raises:
        RuntimeError: If live discovery did not return a mesh.
    """
    core = ModelCore(
        2,
        2,
        np.asarray([0.0, 1.0]),
        options=ModelCoreOptions(
            axis_names=("pool", "group"),
        ),
    )
    core.set_initial_state(INITIAL)
    solver = CoreSolver(core)

    def rhs(_time: object, state: Array) -> Array:
        xp = array_namespace(state)
        flow = rate * xp.asarray(LOCAL, dtype=state.dtype) * state[0] ** 2
        return cast("Array", xp.stack((-flow, flow)))

    config = replace(
        CONFIG, replay_reference_regime=f"shared_rate={rate}; local=[1,1.3]; A0=[.8,.6]"
    )
    if schedule is None:
        solver.run(rhs, config=config)
    else:
        solver.replay_adaptive_schedule(rhs, schedule, config=config)
    recorded = solver.last_adaptive_schedule
    if recorded is None:
        msg = "Adaptive solve did not return an accepted mesh"
        raise RuntimeError(msg)
    return np.asarray(core.get_current_state()), recorded


def exact(rate: float) -> np.ndarray:
    """Evaluate the independent closed-form two-pool solution at time one.

    Returns:
        Final state from A'=-k*A^2 and the conserved total A+B.
    """
    first = INITIAL[0] / (1 + rate * LOCAL * INITIAL[0])
    return np.stack((first, INITIAL.sum(axis=0) - first))


def assess_and_refresh() -> dict[str, float | int | bool | str]:
    """Apply an independent state-error policy at a changed parameter regime.

    Returns:
        Accuracy and mesh-count evidence for the reusable and refreshed cases.

    Raises:
        RuntimeError: If the selected mesh fails the independent accuracy check.
    """
    _, nominal = trajectory(0.05)
    near, _ = trajectory(0.055, nominal)
    changed_rate = 3.0
    changed, _ = trajectory(changed_rate, nominal)
    reference = exact(changed_rate)
    maximum_error = float(np.max(np.abs(changed - reference)))
    accuracy_limit = 1e-4
    selected = nominal
    result = changed
    if maximum_error > accuracy_limit:
        result, selected = trajectory(changed_rate)
    refreshed_error = float(np.max(np.abs(result - reference)))
    if refreshed_error > accuracy_limit:
        msg = "The selected mesh still fails the independent accuracy check"
        raise RuntimeError(msg)
    return {
        "model_id": CONFIG.replay_model_id or "",
        "nominal_rate": 0.05,
        "changed_rate": changed_rate,
        "accuracy_limit": accuracy_limit,
        "nearby_error": float(np.max(np.abs(near - exact(0.055)))),
        "old_mesh_error": maximum_error,
        "selected_mesh_error": refreshed_error,
        "nominal_steps": sum(map(len, nominal.step_sizes)),
        "selected_steps": sum(map(len, selected.step_sizes)),
        "refreshed": selected is not nominal,
    }


if __name__ == "__main__":
    print(json.dumps(assess_and_refresh(), indent=2))  # noqa: T201
