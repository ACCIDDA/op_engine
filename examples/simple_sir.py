# op_engine/examples/simple_sir.py
"""Single-location SIR using IMEX (explicit Heun) with A = 0.

This example uses the CoreSolver IMEX API but with no linear operators (A = 0).
With operators=None, CoreSolver.run_imex reduces to a pure explicit second-order
Heun (trapezoidal) predictor-corrector method for the reaction terms in a
normalized (total population = 1.0) SIR model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from op_engine.core_solver import CoreSolver
from op_engine.model_core import ModelCore, ModelCoreOptions

STATE_ARRAY_NONE_ERROR = "state_array is None despite store_history=True"


def sir_reaction(
    t: float,  # noqa: ARG001 (no explicit time dependence)
    state: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Pure reaction RHS F(t, y) for a normalized SIR model.

    The state is assumed to be normalized so that S + I + R ≈ 1.

    Args:
        t: Current time (unused; included for API compatibility).
        state: Current state array of shape (3, 1) with rows (S, I, R).
        beta: Transmission rate.
        gamma: Recovery rate.

    Returns:
        Reaction RHS array F(t, state) with the same shape as ``state``,
        representing (dS/dt, dI/dt, dR/dt).
    """
    susceptibles = float(state[0, 0])
    infecteds = float(state[1, 0])

    new_inf = beta * susceptibles * infecteds
    recov = gamma * infecteds

    out = np.empty_like(state)
    out[0, 0] = -new_inf
    out[1, 0] = new_inf - recov
    out[2, 0] = recov
    return out


def run_sir_imex(
    beta: float = 0.3,
    gamma: float = 1.0 / 7.0,
    initial_infected: float = 0.01,
    total_time: float = 160.0,
    n_steps: int = 801,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a normalized SIR model using IMEX with A = 0 (explicit Heun).

    The total population is normalized to 1.0, so the initial conditions are:
        S(0) = 1 - initial_infected, I(0) = initial_infected, R(0) = 0.

    Args:
        beta: Transmission rate.
        gamma: Recovery rate.
        initial_infected: Initial fraction infected (0 < I0 < 1).
        total_time: Final simulation time.
        n_steps: Number of time points in the uniform time grid.

    Returns:
        - time_grid: 1D array of shape (n_steps,) with simulation times.
        - state_array: 3D array of shape (n_steps, 3, 1) containing (S, I, R)
          over time.

    Raises:
        RuntimeError: If the core's state_array is None despite store_history=True.
    """
    time_grid = np.linspace(0.0, total_time, n_steps, dtype=float)

    opts = ModelCoreOptions(store_history=True, dtype=np.float64)
    core = ModelCore(
        n_states=3,
        n_subgroups=1,
        time_grid=time_grid,
        options=opts,
    )

    initial_susceptible = 1.0 - float(initial_infected)
    init = np.array(
        [[initial_susceptible], [float(initial_infected)], [0.0]],
        dtype=float,
    )
    core.set_initial_state(init)

    # A = 0 => operators=None => run_imex is explicit Heun on reaction term.
    solver = CoreSolver(core, operators=None)

    def reaction_f(t: float, state: np.ndarray) -> np.ndarray:
        return sir_reaction(t, state, beta=beta, gamma=gamma)

    solver.run_imex(reaction_f)

    if core.state_array is None:
        raise RuntimeError(STATE_ARRAY_NONE_ERROR)

    return time_grid, core.state_array


def plot_sir(time: np.ndarray, states: np.ndarray) -> None:
    """Plot S, I, R trajectories from a SIR simulation.

    Args:
        time: 1D array of simulation times of shape (n_steps,).
        states: 3D array of shape (n_steps, 3, 1) containing the (S, I, R)
            trajectories returned by :func:`run_sir_imex`.
    """
    susceptibles = states[:, 0, 0]
    infecteds = states[:, 1, 0]
    recovered = states[:, 2, 0]

    plt.figure(figsize=(8, 5))
    plt.plot(time, susceptibles, label="S")
    plt.plot(time, infecteds, label="I")
    plt.plot(time, recovered, label="R")
    plt.grid(visible=True)
    plt.legend()
    plt.title("SIR via IMEX (explicit Heun), A = 0")
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run and plot a simple SIR model example."""
    time, states = run_sir_imex()
    plot_sir(time, states)


if __name__ == "__main__":
    main()
