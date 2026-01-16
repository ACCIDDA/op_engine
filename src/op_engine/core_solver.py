# op_engine/src/op_engine/core_solver.py
"""Core semi-implicit solver for time-evolving models.

This solver advances a :class:`op_engine.model_core.ModelCore` instance over its
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

TR-BDF2 support:
    For stiff split systems y' = A y + F(t, y), :meth:`run_imex_trbdf2` provides
    a robust second-order IMEX scheme with strong damping.

    We use the standard TR-BDF2 two-stage structure with parameter gamma:

        Stage 1 (TR over dt1 = gamma*dt):
            Treat A implicitly with trapezoidal operators; treat F explicitly
            with a trapezoidal predictor-corrector (Heun over dt1).

        Stage 2 (BDF2-like completion to t_{n+1}):
            Treat A implicitly with a backward-Euler-style operator pair
            (I - d*dt*A, I), and treat F explicitly via linear extrapolation.

    IMPORTANT: The completion stage uses the correct TR-BDF2 linear combination
    of y_n and y1. A common bug is missing the 1/gamma factor in the y1/y_n
    coefficients, which collapses the method to ~first order for the linear test
    equation.

    With gamma in (0, 1), define:
        denom = 2 - gamma
        d     = (1 - gamma) / denom

        a_y1  = 1 / (gamma * denom)
        b_yn  = -(1 - gamma)^2 / (gamma * denom)

    Then for the linear test equation (F=0), stage 2 is:
        (I - d*dt*A) y_{n+1} = a_y1 * y1 + b_yn * y_n

Time-/state-varying operators:
    Operator factories may depend on:
        - the step size dt
        - a stage scalar (scale)
        - the current stage time t and stage state y

    via a StageOperatorContext passed through to the factory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from .matrix_ops import StageOperatorContext, implicit_solve

if TYPE_CHECKING:
    from .model_core import ModelCore


_OPERATORS_ERROR_MSG = "operators must be a 2-tuple (CN) or 3-tuple (PC)"
_RHS_SHAPE_ERROR_MSG = "rhs shape {actual} does not match expected {expected}"
_OPERATOR_AXIS_LEN_ERROR_MSG = (
    "Operator axis length {axis_len} does not match operator size {op_size}"
)

# Type aliases --------------------------------------------------------------

RHSFunction = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]
"""RHS function used by CoreSolver.run."""

ReactionRHSFunction = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]
"""Reaction-only RHS function used by CoreSolver.run_imex and CoreSolver.run_imex_trbdf2."""

CoreOperators2 = tuple[Any, Any]
CoreOperators3 = tuple[Any, Any, Any]
CoreOperators = CoreOperators2 | CoreOperators3

StageOperatorFactory = Callable[[float, float, StageOperatorContext], CoreOperators2 | CoreOperators3]
"""Stage operator factory for TR-BDF2 and other multistage IMEX methods.

Signature:
    factory(dt, scale, ctx) -> operators

Interpretation:
    - dt is the step size for the current [t_n, t_{n+1}]
    - scale is a dimensionless multiplier; factories typically build operators
      for dt_scale = dt * scale.
    - ctx carries the stage time and the best available stage state proxy.
"""


class CoreSolver:
    """Semi-implicit solver operating on a ModelCore time/state grid."""

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

            operator_axis: Axis name or index along which operators act.

        Raises:
            ValueError: If operators is not None and does not have length 2 or 3.
            ValueError: If operator sizes do not match the operator_axis length.
        """
        self.core = core
        self.dtype = core.dtype
        self.state_shape = core.state_shape
        self.state_ndim = len(self.state_shape)

        # Store requested axis, but resolve/validate only if operators are configured.
        self._op_axis = operator_axis
        self._op_axis_idx: int | None = None
        self._op_axis_len: int | None = None

        # Operator configuration (default per-step operators; used by step() and run_imex()).
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
            raise ValueError(_OPERATORS_ERROR_MSG)

        # Validate operator sizes only when operators exist.
        if self.L_op is not None and self.R_op is not None:
            self._resolve_operator_axis()
            self._validate_operator_sizes(self.L_op, self.R_op)

        # Preallocate buffers (full tensor shape)
        self._rhs_buffer: NDArray[np.floating] = np.zeros(self.state_shape, dtype=self.dtype)
        self._next_state_buffer: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # IMEX buffers (Heun + TR-BDF2)
        self._f_n: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_pred: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._state_pred: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # TR-BDF2 additional buffers
        self._y_stage1: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_stage1: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_extrap: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_operator_axis(self) -> None:
        """Resolve operator axis index/length if not already resolved."""
        if self._op_axis_idx is None or self._op_axis_len is None:
            self._op_axis_idx = self.core.axis_index(self._op_axis)
            self._op_axis_len = int(self.state_shape[self._op_axis_idx])

    def _validate_operator_sizes(self, left_op: Any, right_op: Any) -> None:
        """Validate operator axis length matches operator matrix size."""
        self._resolve_operator_axis()
        assert self._op_axis_len is not None

        # For sparse, np.asarray(csr).shape works but materializes dtype object;
        # shape access is safe and cheap either way.
        op_n = int(left_op.shape[0])
        if op_n != self._op_axis_len:
            raise ValueError(
                _OPERATOR_AXIS_LEN_ERROR_MSG.format(
                    axis_len=self._op_axis_len,
                    op_size=op_n,
                )
            )

    def _reshape_for_solve(
        self, rhs: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], tuple[int, ...]]:
        """Reshape rhs into (axis_len, batch) for operator solve."""
        self._resolve_operator_axis()
        rhs2d, original_shape, _ = self.core.reshape_for_axis_solve(rhs, self._op_axis)
        return rhs2d, original_shape

    def _unreshape_after_solve(
        self, out2d: NDArray[np.floating], original_shape: tuple[int, ...]
    ) -> NDArray[np.floating]:
        """Unreshape solved (axis_len, batch) back to state tensor shape."""
        return self.core.unreshape_from_axis_solve(out2d, original_shape, self._op_axis)

    def _apply_operator_solve_with_ops(
        self,
        rhs: NDArray[np.floating],
        *,
        predictor: Any | None,
        left_op: Any | None,
        right_op: Any | None,
    ) -> NDArray[np.floating]:
        """Apply a provided (predictor, L, R) operator set to rhs."""
        if left_op is None or right_op is None:
            return rhs

        self._validate_operator_sizes(left_op, right_op)
        rhs2d, original_shape = self._reshape_for_solve(rhs)

        if predictor is not None:
            rhs2d = np.asarray(predictor @ rhs2d, dtype=self.dtype)

        # matrix_ops.implicit_solve supports both 1D and 2D RHS for dense and sparse.
        out2d = np.asarray(implicit_solve(left_op, right_op, rhs2d), dtype=self.dtype)

        return self._unreshape_after_solve(out2d, original_shape)

    def _apply_operator_solve(self, rhs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply configured default linear operators to rhs and return next-state tensor."""
        return self._apply_operator_solve_with_ops(
            rhs,
            predictor=self.predictor,
            left_op=self.L_op,
            right_op=self.R_op,
        )

    def _resolve_stage_operators(
        self,
        spec: CoreOperators | StageOperatorFactory | None,
        *,
        dt: float,
        scale: float,
        ctx: StageOperatorContext,
    ) -> tuple[Any | None, Any | None, Any | None]:
        """Resolve a stage operator specification into (predictor, L, R)."""
        if spec is None:
            return None, None, None

        ops: CoreOperators2 | CoreOperators3
        ops = spec(dt, scale, ctx) if callable(spec) else spec

        if len(ops) == 2:
            left_op, right_op = ops
            return None, left_op, right_op
        if len(ops) == 3:
            predictor, left_op, right_op = ops
            return predictor, left_op, right_op

        raise ValueError(_OPERATORS_ERROR_MSG)

    # ------------------------------------------------------------------
    # Single-step advance
    # ------------------------------------------------------------------

    def step(self, rhs: NDArray[np.floating]) -> None:
        """Advance one timestep using the configured stepping rule."""
        rhs_arr = np.asarray(rhs, dtype=self.dtype)
        if rhs_arr.shape != self.state_shape:
            raise ValueError(_RHS_SHAPE_ERROR_MSG.format(actual=rhs_arr.shape, expected=self.state_shape))

        np.copyto(self._rhs_buffer, rhs_arr)

        if self.L_op is None or self.R_op is None:
            np.copyto(self._next_state_buffer, self._rhs_buffer)
        else:
            next_state = self._apply_operator_solve(self._rhs_buffer)
            np.copyto(self._next_state_buffer, next_state)

        self.core.advance_timestep(self._next_state_buffer)

    # ------------------------------------------------------------------
    # Full run: legacy / direct semantics
    # ------------------------------------------------------------------

    def run(self, rhs_func: RHSFunction) -> None:
        """Run the simulation over the entire time grid using rhs_func."""
        time_grid = self.core.time_grid
        for t in time_grid[:-1]:
            state = self.core.get_current_state()
            rhs = rhs_func(float(t), state)
            self.step(rhs)

    # ------------------------------------------------------------------
    # Full run: IMEX Heun (explicit) + optional implicit linear solve
    # ------------------------------------------------------------------

    def run_imex(self, rhs_func: ReactionRHSFunction) -> None:
        """Run an IMEX Heun predictor-corrector scheme over the time grid."""
        time_grid = self.core.time_grid

        for idx in range(self.core.n_timesteps - 1):
            t_n = float(time_grid[idx])
            t_np1 = float(time_grid[idx + 1])
            dt_n = float(self.core.get_dt(idx))

            state_n = self.core.get_current_state()

            # f_n
            f_n = np.asarray(rhs_func(t_n, state_n), dtype=self.dtype)
            if f_n.shape != self.state_shape:
                raise ValueError(_RHS_SHAPE_ERROR_MSG.format(actual=f_n.shape, expected=self.state_shape))
            np.copyto(self._f_n, f_n)

            # y_pred = y_n + dt * f_n
            np.multiply(self._f_n, dt_n, out=self._state_pred)
            self._state_pred += state_n

            # f_pred = f(t_{n+1}, y_pred)
            f_pred = np.asarray(rhs_func(t_np1, self._state_pred), dtype=self.dtype)
            if f_pred.shape != self.state_shape:
                raise ValueError(_RHS_SHAPE_ERROR_MSG.format(actual=f_pred.shape, expected=self.state_shape))
            np.copyto(self._f_pred, f_pred)

            # x = y_n + 0.5*dt*(f_n + f_pred)
            np.add(self._f_n, self._f_pred, out=self._rhs_buffer)
            self._rhs_buffer *= 0.5 * dt_n
            self._rhs_buffer += state_n

            if self.L_op is None or self.R_op is None:
                self.core.advance_timestep(self._rhs_buffer)
            else:
                self.step(self._rhs_buffer)

    # ------------------------------------------------------------------
    # Full run: IMEX TR-BDF2 (second order, stiff-friendly)
    # ------------------------------------------------------------------

    def run_imex_trbdf2(
        self,
        rhs_func: ReactionRHSFunction,
        *,
        operators_tr: CoreOperators | StageOperatorFactory | None = None,
        operators_bdf2: CoreOperators | StageOperatorFactory | None = None,
        gamma: float | None = None,
    ) -> None:
        """Run an IMEX TR-BDF2 scheme over the time grid.

        Intended for stiff split systems:
            y' = A y + F(t, y)

        Args:
            rhs_func: Reaction-only RHS function F(t, state) -> reaction tensor.
            operators_tr: Operator spec/factory for the TR stage.
            operators_bdf2: Operator spec/factory for the BDF2 completion stage.
            gamma: TR-BDF2 parameter. If None, uses gamma = 2 - sqrt(2).
        """
        time_grid = self.core.time_grid

        if gamma is None:
            gamma = float(2.0 - np.sqrt(2.0))

        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1); got {gamma}")

        # --- Correct TR-BDF2 completion coefficients (critical for 2nd order) ---
        denom = 2.0 - gamma
        d = (1.0 - gamma) / denom  # dt multiplier for implicit A part and explicit F part

        a_y1 = 1.0 / (gamma * denom)
        b_yn = -((1.0 - gamma) ** 2) / (gamma * denom)

        for idx in range(self.core.n_timesteps - 1):
            t_n = float(time_grid[idx])
            dt_n = float(self.core.get_dt(idx))

            y_n = self.core.get_current_state()

            # ------------------------------------------------------------------
            # Stage 0: evaluate F_n
            # ------------------------------------------------------------------
            f_n = np.asarray(rhs_func(t_n, y_n), dtype=self.dtype)
            if f_n.shape != self.state_shape:
                raise ValueError(_RHS_SHAPE_ERROR_MSG.format(actual=f_n.shape, expected=self.state_shape))
            np.copyto(self._f_n, f_n)

            # ------------------------------------------------------------------
            # Stage 1 (TR): advance to t1 = t_n + gamma*dt
            # ------------------------------------------------------------------
            dt1 = gamma * dt_n
            t1 = t_n + dt1

            # y_pred = y_n + dt1 * F_n  (explicit predictor for stage state)
            np.multiply(self._f_n, dt1, out=self._state_pred)
            self._state_pred += y_n

            # F at predicted stage state (explicit)
            f1_pred = np.asarray(rhs_func(t1, self._state_pred), dtype=self.dtype)
            if f1_pred.shape != self.state_shape:
                raise ValueError(_RHS_SHAPE_ERROR_MSG.format(actual=f1_pred.shape, expected=self.state_shape))

            # x1 = y_n + 0.5*dt1*(F_n + F1_pred)
            np.add(self._f_n, f1_pred, out=self._rhs_buffer)
            self._rhs_buffer *= 0.5 * dt1
            self._rhs_buffer += y_n

            # Resolve TR-stage operators.
            # We pass ctx=(t1, y_pred) so factories can be time- and stage-state dependent.
            if operators_tr is None:
                predictor_tr = self.predictor
                left_tr = self.L_op
                right_tr = self.R_op
            else:
                ctx_tr = StageOperatorContext(t=t1, y=self._state_pred, stage="tr")
                predictor_tr, left_tr, right_tr = self._resolve_stage_operators(
                    operators_tr, dt=dt_n, scale=gamma, ctx=ctx_tr
                )

            if left_tr is None or right_tr is None:
                np.copyto(self._y_stage1, self._rhs_buffer)
            else:
                y1 = self._apply_operator_solve_with_ops(
                    self._rhs_buffer,
                    predictor=predictor_tr,
                    left_op=left_tr,
                    right_op=right_tr,
                )
                np.copyto(self._y_stage1, y1)

            # Re-evaluate F at (t1, y1) for extrapolation into stage 2.
            f1 = np.asarray(rhs_func(t1, self._y_stage1), dtype=self.dtype)
            if f1.shape != self.state_shape:
                raise ValueError(_RHS_SHAPE_ERROR_MSG.format(actual=f1.shape, expected=self.state_shape))
            np.copyto(self._f_stage1, f1)

            # ------------------------------------------------------------------
            # Stage 2 (BDF2-like completion): to t_{n+1}
            # ------------------------------------------------------------------
            # F_{n+1}^* = (1/gamma) * F1 - ((1-gamma)/gamma) * F_n
            c1 = 1.0 / gamma
            c0 = -((1.0 - gamma) / gamma)
            np.multiply(self._f_stage1, c1, out=self._f_extrap)
            self._f_extrap += c0 * self._f_n

            # (I - d*dt*A) y_{n+1} = a_y1*y1 + b_yn*y_n + d*dt*F_{n+1}^*
            np.multiply(self._y_stage1, a_y1, out=self._rhs_buffer)
            self._rhs_buffer += b_yn * y_n
            self._rhs_buffer += (d * dt_n) * self._f_extrap

            t_np1 = t_n + dt_n

            # Resolve BDF2-stage operators.
            # We pass ctx=(t_{n+1}, y_stage1) as the best available stage-state proxy.
            if operators_bdf2 is None:
                predictor_bdf = self.predictor
                left_bdf = self.L_op
                right_bdf = self.R_op
            else:
                ctx_bdf2 = StageOperatorContext(t=t_np1, y=self._y_stage1, stage="bdf2")
                predictor_bdf, left_bdf, right_bdf = self._resolve_stage_operators(
                    operators_bdf2, dt=dt_n, scale=d, ctx=ctx_bdf2
                )

            if left_bdf is None or right_bdf is None:
                self.core.advance_timestep(self._rhs_buffer)
            else:
                y_np1 = self._apply_operator_solve_with_ops(
                    self._rhs_buffer,
                    predictor=predictor_bdf,
                    left_op=left_bdf,
                    right_op=right_bdf,
                )
                self.core.advance_timestep(y_np1)
