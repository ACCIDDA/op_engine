"""SIR model plugin for integration testing."""

import numpy as np


def stepper(
    time: np.float64,  # noqa: ARG001
    state: np.ndarray,
    *,
    beta: float = 0.3,
    gamma: float = 0.1,
    **kwargs: object,  # noqa: ARG001
) -> np.ndarray:
    """Return dstate/dt for a simple SIR model."""
    y_s, y_i, _ = np.asarray(state, dtype=float)
    infection = beta * y_s * y_i / np.sum(state)
    recovery = gamma * y_i
    return np.array([-infection, infection - recovery, recovery], dtype=float)
