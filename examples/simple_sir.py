# op_engine/examples/simple_sir.py
"""Single-location SIR as a canonical ODE example using CoreSolver.run().

This example demonstrates the core API:

- ModelCore.time_grid is treated as *output times* (states are stored exactly here).
- CoreSolver.run(...) advances between output times using:
    * adaptive=False: exactly one step per output interval
    * adaptive=True: internal substeps that land exactly on the next output time

We model a normalized SIR system with state y = (S, I, R) and S + I + R ≈ 1.

This script saves plots to disk (no interactive windows).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from op_engine.core_solver import CoreSolver, RunConfig
from op_engine.model_core import ModelCore, ModelCoreOptions

_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "sir"
_STATE_ARRAY_NONE_ERROR = "state_array is None despite store_history=True"


def sir_rhs(
    t: float,  # noqa: ARG001 (no explicit time dependence here)
    state: np.ndarray,
    *,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """RHS for a normalized SIR model.

    Args:
        t: Current time (unused; included for API compatibility).
        state: State tensor of shape (3, 1) with rows (S, I, R).
        beta: Transmission rate.
        gamma: Recovery rate.

    Returns:
        RHS tensor of shape (3, 1): (dS/dt, dI/dt, dR/dt).
    """
    s = float(state[0, 0])
    i = float(state[1, 0])

    new_inf = beta * s * i
    recov = gamma * i

    out = np.empty_like(state)
    out[0, 0] = -new_inf
    out[1, 0] = new_inf - recov
    out[2, 0] = recov
    return out


def compute_conservation_drift(states: np.ndarray) -> float:
    """Compute max |S+I+R-1| over stored times.

    Args:
        states: State history, shape (n_steps, 3, 1).

    Returns:
        Maximum absolute conservation drift.
    """
    total = states[:, 0, 0] + states[:, 1, 0] + states[:, 2, 0]
    return float(np.max(np.abs(total - 1.0)))


def save_sir_plot(
    time: np.ndarray,
    states: np.ndarray,
    *,
    title: str,
    out_path: Path,
    drift: float | None = None,
) -> None:
    """Save S, I, R trajectories to an image file.

    Args:
        time: 1D array of times, shape (n_steps,).
        states: State history, shape (n_steps, 3, 1).
        title: Plot title.
        out_path: Output path for the saved figure.
        drift: Optional conservation drift to annotate.
    """
    s = states[:, 0, 0]
    i = states[:, 1, 0]
    r = states[:, 2, 0]

    plt.figure(figsize=(8, 5))
    plt.plot(time, s, label="S")
    plt.plot(time, i, label="I")
    plt.plot(time, r, label="R")
    plt.grid(visible=True)
    plt.legend()

    if drift is not None and np.isfinite(drift):
        title = f"{title}\nmax |S+I+R-1| = {drift:.3e}"

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _run_sir(
    *,
    time_grid: np.ndarray,
    beta: float,
    gamma: float,
    initial_infected: float,
    config: RunConfig,
) -> np.ndarray:
    """Run a single SIR simulation and return stored state history.

    Args:
        time_grid: 1D array of output times.
        beta: Transmission rate.
        gamma: Recovery rate.
        initial_infected: Initial infected fraction.
        config: Run configuration for CoreSolver.

    Raises:
        RuntimeError: If the state history is not available after the run.

    Returns:
        State history array of shape (n_steps, 3, 1).
    """
    opts = ModelCoreOptions(store_history=True, dtype=np.float64)
    core = ModelCore(n_states=3, n_subgroups=1, time_grid=time_grid, options=opts)

    s0 = 1.0 - float(initial_infected)
    i0 = float(initial_infected)
    r0 = 0.0
    y0 = np.array([[s0], [i0], [r0]], dtype=np.float64)
    core.set_initial_state(y0)

    solver = CoreSolver(core, operators=None)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return sir_rhs(t, y, beta=beta, gamma=gamma)

    solver.run(
        rhs,  # type: ignore[arg-type]  # numpy typing can be strict; runtime shape is validated
        config=config,
    )

    if core.state_array is None:
        raise RuntimeError(_STATE_ARRAY_NONE_ERROR)

    return core.state_array


def main() -> None:
    """Run and save canonical SIR simulations demonstrating output-time semantics.

    Produces two saved plots:
      1) Non-adaptive: exactly one solver step per output interval.
      2) Adaptive: internal substeps per output interval, landing exactly on each
         output time, enabling a coarser output grid without losing accuracy.

    Files are written to: examples/output/
    """
    # ---------------------------------------------------------------------
    # Model parameters
    # ---------------------------------------------------------------------
    beta = 0.30
    gamma = 1.0 / 7.0
    initial_infected = 0.01

    # ---------------------------------------------------------------------
    # Simulation horizon
    # ---------------------------------------------------------------------
    total_time = 160.0

    # ---------------------------------------------------------------------
    # (1) Non-adaptive run on a fine output grid
    # ---------------------------------------------------------------------
    time_fine = np.linspace(0.0, total_time, 801, dtype=float)
    cfg_fine = RunConfig(method="heun", adaptive=False, strict=True)

    states_fine = _run_sir(
        time_grid=time_fine,
        beta=beta,
        gamma=gamma,
        initial_infected=initial_infected,
        config=cfg_fine,
    )
    drift_fine = compute_conservation_drift(states_fine)

    save_sir_plot(
        time_fine,
        states_fine,
        title="SIR via CoreSolver.run (Heun, non-adaptive; 1 step per output interval)",
        out_path=_OUTPUT_DIR / "simple_sir_heun_nonadaptive.png",
        drift=drift_fine,
    )

    # ---------------------------------------------------------------------
    # (2) Adaptive run on a coarser output grid (Point J: demonstrate semantics)
    #     Here, states are still stored exactly at output times, but the solver
    #     may take multiple internal substeps between them.
    # ---------------------------------------------------------------------
    time_coarse = np.linspace(0.0, total_time, 201, dtype=float)
    cfg_adapt = RunConfig(method="heun", adaptive=True, strict=True)

    states_adapt = _run_sir(
        time_grid=time_coarse,
        beta=beta,
        gamma=gamma,
        initial_infected=initial_infected,
        config=cfg_adapt,
    )
    drift_adapt = compute_conservation_drift(states_adapt)

    save_sir_plot(
        time_coarse,
        states_adapt,
        title="SIR via CoreSolver.run (Heun, adaptive; substeps between outputs)",
        out_path=_OUTPUT_DIR / "simple_sir_heun_adaptive_coarse_output.png",
        drift=drift_adapt,
    )


if __name__ == "__main__":
    main()
