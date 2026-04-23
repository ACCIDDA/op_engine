# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
