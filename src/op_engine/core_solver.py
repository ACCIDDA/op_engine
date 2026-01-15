# op_engine/src/op_engine/core_solver.py
"""Core semi-implicit solver for time-evolving models.

This solver advances a :class:`gempyor2.model_core.ModelCore` instance over its
time grid using either:

- Direct assignment ("euler" mode): the provided RHS is interpreted as the next
  state tensor.
- Semi-implicit linear stepping ("crank-nicolson" / "predictor-corrector"):
  the provided RHS is transformed by a linear solve

      L @ y_{n+1} = R @ x

  optionally preceded by a linear preprocessing matrix ("predictor"):

      x_tilde = predictor @ x

The linear operators are treated as acting along a single named axis of the
state tensor (by default the leading "state" axis). All remaining axes are
batched, allowing a single operator to be applied to many independent columns.

IMEX support:
    For systems of the form y' = A y + F(t, y), the :meth:`run_imex` method
    provides an explicit Heun (trapezoidal) predictor-corrector for F and uses
    the configured operators to treat A implicitly.

    If A = 0 (no configured operators), :meth:`run_imex` reduces to a pure
    explicit second-order Heun method. This makes the API work uniformly for
    ODE-only models while preserving a path to operator splitting and PDEs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from .matrix_ops import implicit_solve

if TYPE_CHECKING:
    from .model_core import ModelCore


_OPERATORS_ERROR_MSG = "operators must be a 2-tuple (CN) or 3-tuple (PC)"
_RHS_SHAPE_ERROR_MSG = "rhs shape {actual} does not match expected {expected}"
_OPERATOR_AXIS_LEN_ERROR_MSG = (
    "Operator axis length {axis_len} does not match operator size {op_size}"
)


# Type aliases --------------------------------------------------------------


RHSFunction = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]
"""RHS function used by CoreSolver.run.

Signature:
    rhs(t, state) -> rhs_array

Args:
    t: Current simulation time.
    state: Current state tensor of shape core.state_shape.

Returns:
    RHS tensor of shape core.state_shape to be fed into the solver step.
"""

ReactionRHSFunction = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]
"""Reaction-only RHS function used by CoreSolver.run_imex.

This represents the nonlinear / stochastic "reaction" part F(t, y) of a system:

    y' = A y + F(t, y),

where the linear part A is handled implicitly via configured operators and F is
treated explicitly.

Signature:
    F(t, state) -> reaction_array

Args:
    t: Current simulation time.
    state: Current state tensor of shape core.state_shape.

Returns:
    Reaction RHS tensor of shape core.state_shape. This must not include the
    linear term A y.
"""


CoreOperators2 = tuple[Any, Any]
CoreOperators3 = tuple[Any, Any, Any]
CoreOperators = CoreOperators2 | CoreOperators3


class CoreSolver:
    """Semi-implicit solver operating on a ModelCore time/state grid.

    This solver supports three modes:

    1) euler (operators=None):
        next_state = rhs  (direct assignment)

    2) crank-nicolson (operators=(L, R)):
        next_state = solve(L @ y = R @ rhs)

    3) predictor-corrector (operators=(P, L, R)):
        rhs_tilde  = P @ rhs
        next_state = solve(L @ y = R @ rhs_tilde)

    Operators are interpreted as acting along a single axis of the state tensor
    (default: "state"). All other axes are batched for efficient application.
    """

    def __init__(
        self,
        core: ModelCore,
        operators: CoreOperators | None = None,
        *,
        operator_axis: str | int = "state",
    ) -> None:
        """Initialize the CoreSolver.

        Args:
            core: ModelCore managing time and state.
            operators: Operator tuple controlling the integration method.

                - Crank-Nicolson:
                    (L_op, R_op)
                - Predictor-corrector style:
                    (predictor, L_op, R_op)
                - If None:
                    direct assignment (Euler-like semantics).

            operator_axis: Axis name or index along which operators act. By
                default, operators act along the leading "state" axis. This
                enables (for example) diffusion along "space" or couplings
                along a trait axis without changing the core tensor layout.

        Raises:
            ValueError: If operators is not None and does not have length 2 or 3.
            ValueError: If operator sizes do not match the operator_axis length.
        """
        self.core = core
        self.dtype = core.dtype
        self.state_shape = core.state_shape
        self.state_ndim = len(self.state_shape)

        self._op_axis = operator_axis
        self._op_axis_idx = core.axis_index(operator_axis)
        self._op_axis_len = int(self.state_shape[self._op_axis_idx])

        # Operator configuration
        if operators is None:
            self.predictor: Any | None = None
            self.L_op: Any | None = None
            self.R_op: Any | None = None
            self.method = "euler"
        elif len(operators) == 2:
            self.predictor = None
            self.L_op, self.R_op = operators
            self.method = "crank-nicolson"
        elif len(operators) == 3:
            self.predictor, self.L_op, self.R_op = operators
            self.method = "predictor-corrector"
        else:
            msg = _OPERATORS_ERROR_MSG
            raise ValueError(msg)

        # Validate operator sizes if provided.
        if self.L_op is not None and self.R_op is not None:
            op_n = int(np.asarray(self.L_op).shape[0])
            if op_n != self._op_axis_len:
                msg = _OPERATOR_AXIS_LEN_ERROR_MSG.format(
                    axis_len=self._op_axis_len,
                    op_size=op_n,
                )
                raise ValueError(msg)

        # Preallocate buffers (full tensor shape)
        self._rhs_buffer: NDArray[np.floating] = np.zeros(
            self.state_shape,
            dtype=self.dtype,
        )
        self._next_state_buffer: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # IMEX buffers
        self._f_n: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_pred: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._state_pred: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_operator_solve(self, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply configured linear operators to rhs and return next-state tensor.

        Args:
            rhs: RHS tensor of shape core.state_shape.

        Returns:
            Next state tensor of shape core.state_shape.
        """
        # Reshape into (axis_len, batch) for operator application.
        rhs2d, original_shape, _ = self.core.reshape_for_axis_solve(rhs, self._op_axis)

        # Optional linear preprocessing.
        if self.predictor is not None:
            rhs2d = np.asarray(self.predictor @ rhs2d, dtype=self.dtype)

        # Apply implicit solve. Prefer vectorized solve when safe.
        if self.L_op is None or self.R_op is None:
            out2d = rhs2d
        elif issparse(self.L_op) or issparse(self.R_op):
            # Conservative path: solve each RHS column to avoid backend quirks.
            out2d = np.empty_like(rhs2d)
            for j in range(rhs2d.shape[1]):
                out2d[:, j] = implicit_solve(self.L_op, self.R_op, rhs2d[:, j])
        else:
            # Dense path: allow multiple RHS columns to use BLAS/LAPACK.
            out2d = np.asarray(
                implicit_solve(self.L_op, self.R_op, rhs2d),
                dtype=self.dtype,
            )

        return self.core.unreshape_from_axis_solve(out2d, original_shape, self._op_axis)

    # ------------------------------------------------------------------
    # Single-step advance
    # ------------------------------------------------------------------

    def step(self, rhs: NDArray[np.floating]) -> None:
        """Advance one timestep using the configured stepping rule.

        Args:
            rhs: Tensor of shape core.state_shape.

                Semantics:
                    - With operators: rhs is the quantity to which the linear
                      operators are applied.
                    - Without operators: rhs is interpreted as next_state
                      (direct assignment).

        Raises:
            ValueError: If rhs does not have shape core.state_shape.
        """
        rhs_arr = np.asarray(rhs, dtype=self.dtype)
        if rhs_arr.shape != self.state_shape:
            msg = _RHS_SHAPE_ERROR_MSG.format(
                actual=rhs_arr.shape,
                expected=self.state_shape)
            raise ValueError(msg)

        np.copyto(self._rhs_buffer, rhs_arr)

        if self.L_op is None or self.R_op is None:
            # Direct assignment mode.
            np.copyto(self._next_state_buffer, self._rhs_buffer)
        else:
            next_state = self._apply_operator_solve(self._rhs_buffer)
            np.copyto(self._next_state_buffer, next_state)

        self.core.advance_timestep(self._next_state_buffer)

    # ------------------------------------------------------------------
    # Full run: legacy / direct semantics
    # ------------------------------------------------------------------

    def run(self, rhs_func: RHSFunction) -> None:
        """Run the simulation over the entire time grid using rhs_func.

        This preserves the legacy semantics: rhs_func supplies the tensor
        passed into :meth:`step`, which then applies any configured linear
        operators.

        Args:
            rhs_func: Callable rhs_func(t, state) -> rhs, where state and rhs
                both have shape core.state_shape.
        """
        time_grid = self.core.time_grid
        for t in time_grid[:-1]:
            state = self.core.get_current_state()
            rhs = rhs_func(float(t), state)
            self.step(rhs)

    # ------------------------------------------------------------------
    # Full run: IMEX Heun (explicit) + optional implicit linear solve
    # ------------------------------------------------------------------

    def run_imex(self, rhs_func: ReactionRHSFunction) -> None:
        """Run an IMEX predictor-corrector scheme over the time grid.

        Intended for systems of the form:

            y' = A y + F(t, y),

        where:
            - A is linear and (optionally) treated implicitly through operators.
            - F is nonlinear/stochastic reaction and treated explicitly.

        Per step [t_n, t_{n+1}] with dt_n:

            f_n    = F(t_n, y_n)
            y_pred = y_n + dt_n * f_n
            f_pred = F(t_{n+1}, y_pred)
            x      = y_n + 0.5 * dt_n * (f_n + f_pred)

            If operators are configured (A != 0):
                y_{n+1} = solve(L @ y = R @ x)  (plus optional predictor)
            Else (A = 0):
                y_{n+1} = x   (pure explicit Heun)

        Args:
            rhs_func: Reaction-only RHS function F(t, state) -> reaction tensor
                of shape core.state_shape.
        """
        time_grid = self.core.time_grid

        for idx in range(self.core.n_timesteps - 1):
            t_n = float(time_grid[idx])
            t_np1 = float(time_grid[idx + 1])
            dt_n = float(self.core.get_dt(idx))

            state_n = self.core.get_current_state()

            # f_n
            f_n = np.asarray(
                rhs_func(t_n, state_n),
                dtype=self.dtype)
            if f_n.shape != self.state_shape:
                msg = _RHS_SHAPE_ERROR_MSG.format(
                    actual=f_n.shape,
                    expected=self.state_shape)
                raise ValueError(msg)
            np.copyto(self._f_n, f_n)

            # y_pred
            np.multiply(self._f_n, dt_n, out=self._state_pred)
            self._state_pred += state_n

            # f_pred
            f_pred = np.asarray(
                rhs_func(t_np1, self._state_pred),
                dtype=self.dtype)
            if f_pred.shape != self.state_shape:
                msg = _RHS_SHAPE_ERROR_MSG.format(
                    actual=f_pred.shape,
                    expected=self.state_shape)
                raise ValueError(msg)
            np.copyto(self._f_pred, f_pred)

            np.add(self._f_n, self._f_pred, out=self._rhs_buffer)
            self._rhs_buffer *= 0.5 * dt_n
            self._rhs_buffer += state_n

            # If A != 0, apply operators via step(); else, commit x directly.
            if self.L_op is None or self.R_op is None:
                self.core.advance_timestep(self._rhs_buffer)
            else:
                self.step(self._rhs_buffer)


