# op_engine/src/op_engine/core_solver.py
"""Core semi-implicit solver for time-evolving models (ODE + IMEX multiphysics).

This solver advances a :class:`op_engine.model_core.ModelCore` instance over its
configured time grid. In the updated semantics, ModelCore.time_grid is treated
as *output times*: the times at which the user wants a stored solution state.

Between consecutive output times, the solver may take either:
- exactly one step of size dt = t_{i+1} - t_i (adaptive=False), or
- multiple internal adaptive substeps that land exactly on t_{i+1} (adaptive=True).

Supported methods (keyword `method=`):
    - "euler":        Explicit Euler (order 1), adaptive via step-doubling.
    - "heun":         Explicit Heun / RK2 (order 2), embedded Euler estimator.
    - "imex-euler":   IMEX Euler: explicit Euler on F(t,y), implicit Euler on A.
                      Adaptive via step-doubling (IMEX step-doubling).
    - "imex-heun-tr": IMEX Heun-Trapezoidal: Heun on F, trapezoidal/CN on A.
                      Adaptive via embedded low/high (Euler vs Heun) mapped by the
                      same implicit operator solve.
    - "imex-trbdf2":  IMEX TR-BDF2 (order 2), adaptive via step-doubling.

IMEX structure:
    We assume a split system:
        y' = A(t,y) y + F(t,y)
    where F is provided by rhs_func(t, y), and A is represented by linear operators
    applied along a single tensor axis. Operators may be:
        - None (ODE-only / explicit-only behavior), or
        - provided as tuples (predictor?, L, R), or
        - provided as factories depending on dt, stage-scale, and context.

Operator application:
    Operators act along a configured axis (default "state"). All other axes are
    batched. The solve form is:
        L @ y_next = R @ x
    optionally with a preprocessing predictor:
        x_tilde = predictor @ x

Non-uniform dt:
    - Explicit methods naturally support non-uniform dt.
    - Implicit/IMEX methods require operator factories whenever dt varies across
      steps (non-uniform output grid or adaptive stepping), because L/R depend on dt.

Performance hygiene:
    - All major scratch arrays are preallocated.
    - Inner loops use in-place NumPy ops and np.copyto.
    - Implicit solves are delegated to matrix_ops.implicit_solve (cached factorization).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .matrix_ops import StageOperatorContext, implicit_solve

if TYPE_CHECKING:
    from .model_core import ModelCore


# =============================================================================
# Errors / messages
# =============================================================================

_OPERATORS_ERROR_MSG = "operators must be a 2-tuple (L,R) or 3-tuple (predictor,L,R)"
_RHS_SHAPE_ERROR_MSG = "rhs shape {actual} does not match expected {expected}"
_OPERATOR_AXIS_LEN_ERROR_MSG = (
    "Operator axis length {axis_len} does not match operator size {op_size}"
)
_UNKNOWN_METHOD_ERROR_MSG = "Unknown method: {method}"
_MISSING_OPERATORS_ERROR_MSG = "Method '{method}' requires operators (or factories)"
_FACTORY_REQUIRED_ERROR_MSG = (
    "Method '{method}' with variable dt requires a StageOperatorFactory (callable) "
    "for operators; static tuples are invalid because L/R depend on dt."
)
_TOO_MANY_REJECTS_ERROR_MSG = "Too many rejected steps while advancing to output time"
_DT_UNDERFLOW_ERROR_MSG = "dt fell below dt_min while advancing to output time"
_MAX_STEPS_ERROR_MSG = "Exceeded max_steps while advancing to output time"
_TIME_GRID_INCREASING_ERROR_MSG = "time_grid must be strictly increasing"
_GAMMA_RANGE_ERROR_MSG = "For imex-trbdf2, gamma must be in (0, 1)"
_INTERNAL_ERROR_OP_AXIS_MSG = "Internal error: operator axis not resolved"
_INTERNAL_ERROR_GAMMA_MSG = "Internal error: gamma was not resolved for imex-trbdf2"
_INTERNAL_ERROR_ERR_OUT_MSG = "Internal error: err_out is required for this step"
_UNSUPPORTED_OPERATOR_TYPE_MSG = (
    "Unsupported operator type for current backend. "
    "Expected numpy.ndarray or scipy.sparse.csr_matrix."
)


# =============================================================================
# Type aliases / protocols
# =============================================================================

RHSFunction = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]
MethodName = Literal["euler", "heun", "imex-euler", "imex-heun-tr", "imex-trbdf2"]


class OperatorLike(Protocol):
    """Minimal operator interface required by CoreSolver.

    Implementations are expected to behave like 2D linear operators suitable for
    implicit_solve(L, R, rhs2d). Only shape is required for validation.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the operator shape."""
        ...


class PredictorLike(Protocol):
    """Minimal predictor interface required by CoreSolver.

    The predictor is an optional preprocessing operator applied as:
        rhs2d = predictor @ rhs2d
    """

    def __matmul__(self, other: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply the predictor to a 2D array."""
        ...


CoreOperators2 = tuple[OperatorLike, OperatorLike]
CoreOperators3 = tuple[PredictorLike, OperatorLike, OperatorLike]
CoreOperators = CoreOperators2 | CoreOperators3
StageOperatorFactory = Callable[[float, float, StageOperatorContext], CoreOperators]

# What the current (NumPy/SciPy) implicit_solve backend actually accepts.
ScipyOperator: TypeAlias = NDArray[np.floating] | csr_matrix


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass(slots=True, frozen=True)
class DtControllerConfig:
    """Configuration for adaptive timestep control.

    Attributes:
        dt_min: Minimum allowed dt.
        dt_max: Maximum allowed dt.
        safety: Safety factor applied to dt updates.
        fac_min: Minimum multiplicative change factor.
        fac_max: Maximum multiplicative change factor.
    """

    dt_min: float = 0.0
    dt_max: float = float("inf")
    safety: float = 0.9
    fac_min: float = 0.2
    fac_max: float = 5.0


@dataclass(slots=True, frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive stepping.

    Attributes:
        rtol: Relative tolerance.
        atol: Absolute tolerance (scalar or array-like).
        dt_init: Optional initial dt guess; if None, use output dt.
        max_reject: Maximum number of rejected attempts per accepted step.
        max_steps: Maximum number of internal substeps per output interval.
    """

    rtol: float = 1e-6
    atol: float | NDArray[np.floating] = 1e-9
    dt_init: float | None = None
    max_reject: int = 25
    max_steps: int = 1_000_000


@dataclass(slots=True, frozen=True)
class OperatorSpecs:
    """Operator specifications for implicit/IMEX methods.

    Attributes:
        default: Default operator spec (tuple or factory) used by IMEX Euler/Heun-TR
            and as a fallback for TR/BDF2 stages.
        tr: Operator spec for trapezoidal stage of TR-BDF2 (optional).
        bdf2: Operator spec for BDF2 stage of TR-BDF2 (optional).
    """

    default: CoreOperators | StageOperatorFactory | None = None
    tr: CoreOperators | StageOperatorFactory | None = None
    bdf2: CoreOperators | StageOperatorFactory | None = None


@dataclass(slots=True, frozen=True)
class RunConfig:
    """Configuration for CoreSolver.run.

    Attributes:
        method: Method name.
        adaptive: Whether to use adaptive substepping between output times.
        strict: If True, invalid configurations raise; otherwise warnings and
            method downshifts may occur.
        dt_controller: Parameters for dt controller when adaptive=True.
        adaptive_cfg: Parameters controlling error tolerances and limits.
        operators: Operator specifications for implicit/IMEX methods.
        gamma: Optional TR-BDF2 gamma (if None, uses default).
    """

    method: str = "heun"
    adaptive: bool = False
    strict: bool = True
    dt_controller: DtControllerConfig = DtControllerConfig()
    adaptive_cfg: AdaptiveConfig = AdaptiveConfig()
    operators: OperatorSpecs = OperatorSpecs()
    gamma: float | None = None


@dataclass(slots=True, frozen=True)
class RunPlan:
    """Resolved execution plan derived from RunConfig.

    This is the internal, validated form used by the stepping loops.

    Attributes:
        method: Final method after any strict=False downshifts.
        gamma: TR-BDF2 gamma, or None for non-TR-BDF2 methods.
        op_default: Operator spec for IMEX Euler/Heun-TR.
        op_tr: TR-stage operator spec for TR-BDF2.
        op_bdf2: BDF2-stage operator spec for TR-BDF2.
    """

    method: MethodName
    gamma: float | None
    op_default: CoreOperators | StageOperatorFactory | None
    op_tr: CoreOperators | StageOperatorFactory | None
    op_bdf2: CoreOperators | StageOperatorFactory | None


@dataclass(slots=True)
class StepIO:
    """Bundle of per-step state for stepping kernels.

    Attributes:
        t: Current time.
        dt: Step size.
        y: Current state array (input).
        out: Output state array (written in-place).
        err_out: Error estimate array (written in-place) for adaptive methods.
    """

    t: float
    dt: float
    y: NDArray[np.floating]
    out: NDArray[np.floating]
    err_out: NDArray[np.floating] | None = None


@dataclass(slots=True)
class ImplicitStageParams:
    """Bundle of parameters for one implicit operator application.

    Attributes:
        spec: Operator spec (tuple or factory) or None for identity.
        dt: Full-step dt for operator factory context.
        scale: Stage scaling factor for dt-dependent operators.
        t_stage: Stage time.
        y_stage: Stage state proxy for operator factories.
        stage: Stage label (e.g., "be", "tr", "bdf2").
        x: Input array to map.
        out: Output array (written in-place).
    """

    spec: CoreOperators | StageOperatorFactory | None
    dt: float
    scale: float
    t_stage: float
    y_stage: NDArray[np.floating]
    stage: str
    x: NDArray[np.floating]
    out: NDArray[np.floating]


@dataclass(slots=True)
class ImexEulerOnceParams:
    """Bundle of parameters for one IMEX Euler step (non-doubling).

    Attributes:
        t: Current time.
        y: Current state.
        dt: Step size.
        op_spec: Operator spec for implicit stage.
        out: Output state array (written in-place).
    """

    t: float
    y: NDArray[np.floating]
    dt: float
    op_spec: CoreOperators | StageOperatorFactory | None
    out: NDArray[np.floating]


@dataclass(slots=True)
class Trbdf2OnceParams:
    """Bundle of parameters for one TR-BDF2 step (non-doubling).

    Attributes:
        t: Current time.
        y: Current state.
        dt: Step size.
        operators_tr: TR stage operator spec.
        operators_bdf2: BDF2 stage operator spec.
        gamma: TR-BDF2 gamma.
        out: Output state array (written in-place).
    """

    t: float
    y: NDArray[np.floating]
    dt: float
    operators_tr: CoreOperators | StageOperatorFactory | None
    operators_bdf2: CoreOperators | StageOperatorFactory | None
    gamma: float
    out: NDArray[np.floating]


@dataclass(slots=True)
class AdaptiveAdvanceParams:
    """Bundle of parameters for adaptive advancement to an output time.

    Attributes:
        plan: Resolved run plan.
        t0: Start time.
        t1: End/output time.
        y0: Initial state at t0.
        adaptive_cfg: Adaptive stepping configuration.
        dt_ctrl: dt controller configuration.
    """

    plan: RunPlan
    t0: float
    t1: float
    y0: NDArray[np.floating]
    adaptive_cfg: AdaptiveConfig
    dt_ctrl: DtControllerConfig


# =============================================================================
# CoreSolver
# =============================================================================


class CoreSolver:
    """Semi-implicit solver operating on a ModelCore time/state grid."""

    def __init__(
        self,
        core: ModelCore,
        operators: CoreOperators | StageOperatorFactory | None = None,
        *,
        operator_axis: str | int = "state",
    ) -> None:
        """Initialize CoreSolver.

        Args:
            core: ModelCore instance to solve.
            operators: Default operator spec (tuple or factory) for implicit stages.
            operator_axis: Axis along which operators act (name or index).
        """
        self.core = core
        self.dtype = core.dtype
        self.state_shape = core.state_shape
        self.state_ndim = len(self.state_shape)

        # Operator axis resolution
        self._op_axis = operator_axis
        self._op_axis_idx: int | None = None
        self._op_axis_len: int | None = None

        # Default operator spec (tuple or factory or None)
        self._default_operator_spec: CoreOperators | StageOperatorFactory | None = (
            operators
        )

        # Preallocate buffers (full tensor shape)
        self._rhs_buffer: NDArray[np.floating] = np.zeros(
            self.state_shape,
            dtype=self.dtype,
        )
        self._next_state_buffer: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # Shared stepping buffers
        self._f_n: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_pred: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._state_pred: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # Adaptive buffers
        self._y_full: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._y_half: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._y_two_half: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._y_low: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._err: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._scale: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._ratio: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # Working state buffers (avoid allocating per substep)
        self._y_curr: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._y_try: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # TR-BDF2 additional buffers
        self._y_stage1: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_stage1: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_extrap: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # Validate operator sizes if default spec is a static tuple
        if operators is not None and not callable(operators):
            _predictor, left_op, right_op = self._normalize_ops_tuple(operators)
            if left_op is not None and right_op is not None:
                self._resolve_operator_axis()
                self._validate_operator_sizes(left_op, right_op)

    # ------------------------------------------------------------------
    # Axis / operator helpers
    # ------------------------------------------------------------------

    def _resolve_operator_axis(self) -> None:
        """Resolve operator axis index/length lazily."""
        if self._op_axis_idx is None or self._op_axis_len is None:
            self._op_axis_idx = self.core.axis_index(self._op_axis)
            self._op_axis_len = int(self.state_shape[self._op_axis_idx])

    def _require_op_axis_len(self) -> int:
        """Return resolved operator axis length.

        Raises:
            RuntimeError: If axis length is not available after resolution.
        """
        self._resolve_operator_axis()
        if self._op_axis_len is None:
            raise RuntimeError(_INTERNAL_ERROR_OP_AXIS_MSG)
        return int(self._op_axis_len)

    def _validate_operator_sizes(
        self,
        left_op: OperatorLike,
        right_op: OperatorLike,
    ) -> None:
        """Validate operator sizes against configured operator axis.

        Args:
            left_op: Left operator.
            right_op: Right operator.

        Raises:
            ValueError: If operator sizes do not match the configured axis length.
        """
        axis_len = self._require_op_axis_len()

        left_n = int(left_op.shape[0])
        right_n = int(right_op.shape[0])

        if left_n != axis_len:
            raise ValueError(
                _OPERATOR_AXIS_LEN_ERROR_MSG.format(
                    axis_len=axis_len,
                    op_size=left_n,
                )
            )
        if right_n != axis_len:
            raise ValueError(
                _OPERATOR_AXIS_LEN_ERROR_MSG.format(
                    axis_len=axis_len,
                    op_size=right_n,
                )
            )

    @staticmethod
    def _as_scipy_operator(op: OperatorLike) -> ScipyOperator:
        """Convert an OperatorLike to a backend-supported operator type.

        The current backend uses NumPy/SciPy, so implicit_solve requires either a
        NumPy ndarray or a SciPy CSR matrix. This method is the single boundary
        where backend-specific operator requirements are enforced.

        Args:
            op: Operator-like object.

        Raises:
            TypeError: If op is not a supported operator type for this backend.

        Returns:
            Operator as a supported SciPy/NumPy operator type.
        """
        if isinstance(op, np.ndarray):
            return cast("NDArray[np.floating]", op)

        if csr_matrix is not None and isinstance(op, csr_matrix):
            return op

        raise TypeError(_UNSUPPORTED_OPERATOR_TYPE_MSG)

    def _reshape_for_solve(
        self,
        rhs: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], tuple[int, ...]]:
        """
        Reshape a full state tensor into 2D (axis_len, batch).

        Args:
            rhs: Full tensor input array.

        Returns:
            rhs2d: 2D reshaped array for implicit_solve.
            original_shape: Original full tensor shape.
        """
        self._resolve_operator_axis()
        rhs2d, original_shape, _ = self.core.reshape_for_axis_solve(rhs, self._op_axis)
        return rhs2d, original_shape

    def _unreshape_after_solve(
        self,
        out2d: NDArray[np.floating],
        original_shape: tuple[int, ...],
    ) -> NDArray[np.floating]:
        """
        Inverse of _reshape_for_solve.

        Args:
            out2d: 2D output array from implicit_solve.
            original_shape: Original full tensor shape.

        Returns:
            Full tensor output array.
        """
        return self.core.unreshape_from_axis_solve(out2d, original_shape, self._op_axis)

    def _apply_operator_solve_with_ops(
        self,
        x: NDArray[np.floating],
        *,
        predictor: PredictorLike | None,
        left_op: OperatorLike | None,
        right_op: OperatorLike | None,
        out: NDArray[np.floating],
    ) -> None:
        """Compute out = S(x) where S is defined by (predictor, left_op, right_op).

        If left_op/right_op are None, this degenerates to out = x (copy).

        Args:
            x: Input state-like tensor.
            predictor: Optional predictor operator.
            left_op: Left implicit operator.
            right_op: Right implicit operator.
            out: Output array to write.
        """
        if left_op is None or right_op is None:
            np.copyto(out, x)
            return

        self._validate_operator_sizes(left_op, right_op)
        x2d, original_shape = self._reshape_for_solve(x)

        if predictor is not None:
            x2d = np.asarray(predictor @ x2d, dtype=self.dtype)

        left_s = self._as_scipy_operator(left_op)
        right_s = self._as_scipy_operator(right_op)

        out2d = np.asarray(implicit_solve(left_s, right_s, x2d), dtype=self.dtype)
        out_arr = self._unreshape_after_solve(out2d, original_shape)
        np.copyto(out, out_arr)

    @staticmethod
    def _normalize_ops_tuple(
        ops: CoreOperators,
    ) -> tuple[PredictorLike | None, OperatorLike | None, OperatorLike | None]:
        """
        Normalize a 2- or 3-tuple operator spec to (predictor, L, R).

        Args:
            ops: Operator spec tuple.

        Raises:
            ValueError: If ops is not a 2- or 3-tuple.

        Returns:
            Tuple of (predictor, L, R) operators.
        """
        if len(ops) == 2:
            left_op, right_op = ops
            return None, left_op, right_op
        if len(ops) == 3:
            predictor, left_op, right_op = ops
            return predictor, left_op, right_op
        raise ValueError(_OPERATORS_ERROR_MSG)

    def _resolve_stage_operators(
        self,
        spec: CoreOperators | StageOperatorFactory | None,
        *,
        dt: float,
        scale: float,
        ctx: StageOperatorContext,
    ) -> tuple[PredictorLike | None, OperatorLike | None, OperatorLike | None]:
        """
        Resolve a stage operator spec (tuple or factory) to (predictor, L, R).

        Args:
            spec: Operator spec (tuple or factory) or None for identity.
            dt: Full-step dt for operator factory context.
            scale: Stage scaling factor for dt-dependent operators.
            ctx: Stage operator context.

        Returns:
            Tuple of (predictor, L, R) operators, or (None, None, None) if spec is None.
        """
        if spec is None:
            return None, None, None

        resolved: CoreOperators = spec(dt, scale, ctx) if callable(spec) else spec
        return self._normalize_ops_tuple(resolved)

    def _apply_implicit(self, params: ImplicitStageParams) -> None:
        """Resolve operators from params.spec and apply implicit mapping.

        Args:
            params: Implicit stage bundle.
        """
        if params.spec is None:
            np.copyto(params.out, params.x)
            return

        ctx = StageOperatorContext(
            t=float(params.t_stage),
            y=params.y_stage,
            stage=params.stage,
        )
        predictor, left_op, right_op = self._resolve_stage_operators(
            params.spec,
            dt=params.dt,
            scale=params.scale,
            ctx=ctx,
        )
        self._apply_operator_solve_with_ops(
            params.x,
            predictor=predictor,
            left_op=left_op,
            right_op=right_op,
            out=params.out,
        )

    # ------------------------------------------------------------------
    # RHS evaluation helper (shape + dtype enforcement)
    # ------------------------------------------------------------------

    def _rhs_into(
        self,
        out: NDArray[np.floating],
        rhs_func: RHSFunction,
        t: float,
        y: NDArray[np.floating],
    ) -> None:
        """Evaluate RHS into out with shape enforcement.

        Args:
            out: Output buffer to write into.
            rhs_func: RHS function F(t, y).
            t: Time.
            y: State.

        Raises:
            ValueError: If RHS returns an array with an unexpected shape.
        """
        f = np.asarray(rhs_func(float(t), y), dtype=self.dtype)
        if f.shape != self.state_shape:
            raise ValueError(
                _RHS_SHAPE_ERROR_MSG.format(
                    actual=f.shape,
                    expected=self.state_shape,
                )
            )
        np.copyto(out, f)

    # ------------------------------------------------------------------
    # dt variability checks / operator spec validation
    # ------------------------------------------------------------------

    def _output_dt_is_uniform(self) -> bool:
        """Return True if the output time grid has uniform dt."""
        if self.core.n_timesteps <= 2:
            return True
        dtg = np.asarray(self.core.dt_grid, dtype=float)
        if dtg.size == 0:
            return True
        return bool(np.allclose(dtg, dtg[0]))

    def _require_factory_if_variable_dt(
        self,
        method: str,
        *,
        adaptive: bool,
        spec: CoreOperators | StageOperatorFactory | None,
        strict: bool,
    ) -> None:
        """Enforce operator factory requirement under variable dt.

        Args:
            method: Method name.
            adaptive: Whether adaptive stepping is enabled.
            spec: Operator spec to validate.
            strict: If True, raise; else warn.

        Raises:
            ValueError: If variable dt requires a factory but a static tuple is given.
        """
        variable_dt = adaptive or (not self._output_dt_is_uniform())
        if not variable_dt or spec is None or callable(spec):
            return

        if strict:
            raise ValueError(_FACTORY_REQUIRED_ERROR_MSG.format(method=method))
        warnings.warn(
            _FACTORY_REQUIRED_ERROR_MSG.format(method=method),
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _normalize_method(method: str) -> MethodName:
        """Normalize and validate method string.

        Args:
            method: User-provided method string.

        Raises:
            ValueError: If method is unknown.

        Returns:
            Normalized method literal.
        """
        method_norm = str(method).strip().lower()
        allowed: tuple[str, ...] = (
            "euler",
            "heun",
            "imex-euler",
            "imex-heun-tr",
            "imex-trbdf2",
        )
        if method_norm not in allowed:
            raise ValueError(_UNKNOWN_METHOD_ERROR_MSG.format(method=method))
        return method_norm  # type: ignore[return-value]

    @staticmethod
    def _resolve_gamma(method: MethodName, gamma: float | None) -> float | None:
        """
        Resolve TR-BDF2 gamma parameter.

        Args:
            method: Method name.
            gamma: User-provided gamma (or None for default).

        Raises:
            ValueError: If gamma is out of range for TR-BDF2.

        Returns:
            Resolved gamma for TR-BDF2, or None for non-TR-BDF2 methods.
        """
        if method != "imex-trbdf2":
            return None
        if gamma is None:
            gamma = float(2.0 - np.sqrt(2.0))
        gamma_f = float(gamma)
        if not (0.0 < gamma_f < 1.0):
            raise ValueError(_GAMMA_RANGE_ERROR_MSG)
        return gamma_f

    # ------------------------------------------------------------------
    # Minimal helper split to reduce _resolve_run_plan complexity (C901)
    # ------------------------------------------------------------------
    @staticmethod
    def _plan_for_explicit(
        method_in: MethodName,
        op_default: CoreOperators | StageOperatorFactory | None,
        *,
        strict: bool,
    ) -> RunPlan:
        """Build a RunPlan for explicit methods.

        Args:
            method_in: Explicit method ("euler" or "heun").
            op_default: Default operator spec (ignored for explicit methods).
            strict: If True, warn when operators are provided.

        Returns:
            RunPlan for the explicit method.
        """
        if op_default is not None:
            msg = (
                f"Method '{method_in}' is explicit; provided operators ignored. "
                "Use 'imex-euler', 'imex-heun-tr', or 'imex-trbdf2' for implicit A."
            )
            if strict:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return RunPlan(
            method=method_in,
            gamma=None,
            op_default=None,
            op_tr=None,
            op_bdf2=None,
        )

    def _plan_for_imex_single(
        self,
        method_in: MethodName,
        op_default: CoreOperators | StageOperatorFactory | None,
        *,
        strict: bool,
        adaptive: bool,
    ) -> RunPlan:
        """Build a RunPlan for IMEX Euler / IMEX Heun-TR.

        Args:
            method_in: Method ("imex-euler" or "imex-heun-tr").
            op_default: Default operator spec.
            strict: If True, invalid configuration raises.
            adaptive: Whether adaptive stepping is enabled.

        Raises:
            ValueError: If required operators are missing.

        Returns:
            RunPlan for IMEX method, or an explicit fallback if strict=False and
            operators are missing.
        """
        if op_default is None:
            if strict:
                raise ValueError(_MISSING_OPERATORS_ERROR_MSG.format(method=method_in))
            warnings.warn(
                (
                    f"{_MISSING_OPERATORS_ERROR_MSG.format(method=method_in)}; "
                    "falling back to explicit method."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            fallback: MethodName = "euler" if method_in == "imex-euler" else "heun"
            return RunPlan(
                method=fallback,
                gamma=None,
                op_default=None,
                op_tr=None,
                op_bdf2=None,
            )

        self._require_factory_if_variable_dt(
            method_in,
            adaptive=adaptive,
            spec=op_default,
            strict=strict,
        )
        return RunPlan(
            method=method_in,
            gamma=None,
            op_default=op_default,
            op_tr=None,
            op_bdf2=None,
        )

    def _plan_for_trbdf2(
        self,
        method_in: MethodName,
        *,
        gamma: float | None,
        operators: OperatorSpecs,
        strict: bool,
        adaptive: bool,
    ) -> RunPlan:
        """Build a RunPlan for IMEX TR-BDF2.

        Args:
            method_in: Method (must be "imex-trbdf2").
            gamma: TR-BDF2 gamma (resolved earlier).
            operators: Operator specifications for TR-BDF2. If stage-specific specs are
                not provided, they fall back to operators.default.
            strict: If True, invalid configuration raises.
            adaptive: Whether adaptive stepping is enabled.

        Raises:
            RuntimeError: If method_in is not "imex-trbdf2".
            ValueError: If required operators are missing.

        Returns:
            RunPlan for TR-BDF2, or a Heun fallback if strict=False and operators are
            missing.
        """
        if method_in != "imex-trbdf2":
            raise RuntimeError(_UNKNOWN_METHOD_ERROR_MSG.format(method=method_in))

        op_default = operators.default
        op_tr = operators.tr
        op_bdf2 = operators.bdf2

        op_tr_eff = op_tr if op_tr is not None else op_default
        op_bdf2_eff = op_bdf2 if op_bdf2 is not None else op_default

        if op_tr_eff is None and op_bdf2_eff is None:
            if strict:
                raise ValueError(_MISSING_OPERATORS_ERROR_MSG.format(method=method_in))
            warnings.warn(
                (
                    f"{_MISSING_OPERATORS_ERROR_MSG.format(method=method_in)}; "
                    "falling back to explicit Heun."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return RunPlan(
                method="heun",
                gamma=None,
                op_default=None,
                op_tr=None,
                op_bdf2=None,
            )

        self._require_factory_if_variable_dt(
            method_in,
            adaptive=adaptive,
            spec=op_tr_eff,
            strict=strict,
        )
        self._require_factory_if_variable_dt(
            method_in,
            adaptive=adaptive,
            spec=op_bdf2_eff,
            strict=strict,
        )
        if gamma is None:
            raise RuntimeError(_INTERNAL_ERROR_GAMMA_MSG)

        return RunPlan(
            method="imex-trbdf2",
            gamma=gamma,
            op_default=None,
            op_tr=op_tr_eff,
            op_bdf2=op_bdf2_eff,
        )

    def _resolve_run_plan(self, cfg: RunConfig) -> RunPlan:
        """Resolve method/operators into an executable plan.

        Args:
            cfg: Run configuration.

        Returns:
            Resolved plan.
        """
        method_in = self._normalize_method(cfg.method)
        gamma = self._resolve_gamma(method_in, cfg.gamma)

        op_default = (
            cfg.operators.default
            if cfg.operators.default is not None
            else self._default_operator_spec
        )
        op_tr = cfg.operators.tr
        op_bdf2 = cfg.operators.bdf2

        strict = bool(cfg.strict)
        adaptive = bool(cfg.adaptive)

        if method_in in {"euler", "heun"}:
            return self._plan_for_explicit(method_in, op_default, strict=strict)

        if method_in in {"imex-euler", "imex-heun-tr"}:
            return self._plan_for_imex_single(
                method_in,
                op_default,
                strict=strict,
                adaptive=adaptive,
            )

        operators_trbdf2 = OperatorSpecs(
            default=op_default,
            tr=op_tr,
            bdf2=op_bdf2,
        )
        return self._plan_for_trbdf2(
            method_in,
            gamma=gamma,
            operators=operators_trbdf2,
            strict=strict,
            adaptive=adaptive,
        )

    # ------------------------------------------------------------------
    # Error norm + dt controller
    # ------------------------------------------------------------------

    def _error_norm(
        self,
        err: NDArray[np.floating],
        y_ref: NDArray[np.floating],
        y_prev: NDArray[np.floating],
        *,
        rtol: float,
        atol: float | NDArray[np.floating],
    ) -> float:
        """
        Compute RMS scaled error norm.

        Args:
            err: Error array.
            y_ref: Reference solution array.
            y_prev: Previous solution array.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            RMS scaled error norm.
        """
        np.abs(y_ref, out=self._scale)
        np.abs(y_prev, out=self._ratio)
        np.maximum(self._scale, self._ratio, out=self._scale)

        if isinstance(atol, (float, int, np.floating)):
            self._scale *= float(rtol)
            self._scale += float(atol)
        else:
            atol_arr = np.asarray(atol, dtype=self.dtype)
            self._scale *= float(rtol)
            self._scale += atol_arr

        np.divide(err, self._scale, out=self._ratio)
        v = float(np.sqrt(np.mean(self._ratio * self._ratio)))
        if not np.isfinite(v):
            return float("inf")
        return v

    @staticmethod
    def _propose_dt(
        dt: float,
        err_norm: float,
        order: int,
        *,
        cfg: DtControllerConfig,
    ) -> float:
        """
        Propose a new dt based on error norm and method order.

        Args:
            dt: Current dt.
            err_norm: Current error norm.
            order: Method order.
            cfg: Dt controller configuration.

        Returns:
            Proposed new dt.
        """
        if err_norm <= 0.0:
            fac = cfg.fac_max
        else:
            exp = 1.0 / float(order + 1)
            fac = cfg.safety * (err_norm ** (-exp))
            fac = min(cfg.fac_max, max(cfg.fac_min, fac))

        dt_new = dt * fac
        if dt_new < cfg.dt_min:
            return cfg.dt_min
        if dt_new > cfg.dt_max:
            return cfg.dt_max
        return dt_new

    # ------------------------------------------------------------------
    # One-step kernels (write into provided out arrays)
    # ------------------------------------------------------------------
    @staticmethod
    def _require_err_out(step: StepIO) -> NDArray[np.floating]:
        """
        Return err_out for a step, raising if missing.

        Args:
            step: Step bundle.

        Raises:
            RuntimeError: If err_out is None.

        Returns:
            err_out array.
        """
        if step.err_out is None:
            raise RuntimeError(_INTERNAL_ERROR_ERR_OUT_MSG)
        return step.err_out

    def _step_euler_doubling(self, rhs_func: RHSFunction, step: StepIO) -> int:
        """Explicit Euler step with step-doubling error estimate.

        Args:
            rhs_func: RHS function.
            step: Step bundle.

        Returns:
            Method order (1).
        """
        err_out = self._require_err_out(step)

        self._rhs_into(self._f_n, rhs_func, step.t, step.y)
        np.multiply(self._f_n, step.dt, out=self._y_full)
        self._y_full += step.y

        np.multiply(self._f_n, 0.5 * step.dt, out=self._y_half)
        self._y_half += step.y

        self._rhs_into(self._f_pred, rhs_func, step.t + 0.5 * step.dt, self._y_half)
        np.multiply(self._f_pred, 0.5 * step.dt, out=self._y_two_half)
        self._y_two_half += self._y_half

        np.subtract(self._y_two_half, self._y_full, out=err_out)
        np.copyto(step.out, self._y_two_half)
        return 1

    def _step_heun(self, rhs_func: RHSFunction, step: StepIO) -> int:
        """Explicit Heun (RK2) step with embedded Euler estimator.

        Args:
            rhs_func: RHS function.
            step: Step bundle.

        Returns:
            Method order (2).
        """
        err_out = self._require_err_out(step)

        self._rhs_into(self._f_n, rhs_func, step.t, step.y)

        np.multiply(self._f_n, step.dt, out=self._state_pred)
        self._state_pred += step.y

        self._rhs_into(self._f_pred, rhs_func, step.t + step.dt, self._state_pred)

        np.add(self._f_n, self._f_pred, out=self._rhs_buffer)
        self._rhs_buffer *= 0.5 * step.dt
        self._rhs_buffer += step.y

        np.subtract(self._rhs_buffer, self._state_pred, out=err_out)
        np.copyto(step.out, self._rhs_buffer)
        return 2

    def _imex_euler_step_once(
        self,
        rhs_func: RHSFunction,
        params: ImexEulerOnceParams,
    ) -> None:
        """One IMEX Euler step (no step-doubling).

        Args:
            rhs_func: RHS function F(t, y).
            params: IMEX Euler parameters.
        """
        self._rhs_into(self._f_n, rhs_func, params.t, params.y)
        np.multiply(self._f_n, params.dt, out=self._rhs_buffer)
        self._rhs_buffer += params.y

        self._apply_implicit(
            ImplicitStageParams(
                spec=params.op_spec,
                dt=params.dt,
                scale=1.0,
                t_stage=params.t + params.dt,
                y_stage=params.y,
                stage="be",
                x=self._rhs_buffer,
                out=params.out,
            )
        )

    def _step_imex_euler_doubling(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        op_spec: CoreOperators | StageOperatorFactory | None,
    ) -> int:
        """
        IMEX Euler step with step-doubling error estimate.

        Args:
            rhs_func: RHS function.
            step: Step bundle.
            op_spec: Implicit stage operator spec.

        Returns:
            Method order (1).
        """
        err_out = self._require_err_out(step)

        self._imex_euler_step_once(
            rhs_func,
            ImexEulerOnceParams(
                t=step.t,
                y=step.y,
                dt=step.dt,
                op_spec=op_spec,
                out=self._y_full,
            ),
        )

        dt2 = 0.5 * step.dt
        self._imex_euler_step_once(
            rhs_func,
            ImexEulerOnceParams(
                t=step.t,
                y=step.y,
                dt=dt2,
                op_spec=op_spec,
                out=self._y_half,
            ),
        )
        self._imex_euler_step_once(
            rhs_func,
            ImexEulerOnceParams(
                t=step.t + dt2,
                y=self._y_half,
                dt=dt2,
                op_spec=op_spec,
                out=self._y_two_half,
            ),
        )

        np.subtract(self._y_two_half, self._y_full, out=err_out)
        np.copyto(step.out, self._y_two_half)
        return 1

    def _step_imex_heun_tr(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        op_spec: CoreOperators | StageOperatorFactory | None,
    ) -> int:
        """
        IMEX Heun-Trapezoidal step with embedded low/high estimate.

        Args:
            rhs_func: RHS function.
            step: Step bundle.
            op_spec: Implicit stage operator spec.

        Returns:
            Method order (2).
        """
        err_out = self._require_err_out(step)

        self._rhs_into(self._f_n, rhs_func, step.t, step.y)

        np.multiply(self._f_n, step.dt, out=self._state_pred)
        self._state_pred += step.y  # x_low

        self._rhs_into(self._f_pred, rhs_func, step.t + step.dt, self._state_pred)

        np.add(self._f_n, self._f_pred, out=self._rhs_buffer)
        self._rhs_buffer *= 0.5 * step.dt
        self._rhs_buffer += step.y  # x_high

        ctx_y = self._state_pred
        self._apply_implicit(
            ImplicitStageParams(
                spec=op_spec,
                dt=step.dt,
                scale=1.0,
                t_stage=step.t + step.dt,
                y_stage=ctx_y,
                stage="tr",
                x=self._state_pred,
                out=self._y_low,
            )
        )
        self._apply_implicit(
            ImplicitStageParams(
                spec=op_spec,
                dt=step.dt,
                scale=1.0,
                t_stage=step.t + step.dt,
                y_stage=ctx_y,
                stage="tr",
                x=self._rhs_buffer,
                out=step.out,
            )
        )

        np.subtract(step.out, self._y_low, out=err_out)
        return 2

    def _trbdf2_step_once(
        self,
        rhs_func: RHSFunction,
        params: Trbdf2OnceParams,
    ) -> None:
        """One TR-BDF2 step (no step-doubling).

        Args:
            rhs_func: RHS function.
            params: TR-BDF2 parameters.
        """
        gamma = float(params.gamma)
        denom = 2.0 - gamma
        d = (1.0 - gamma) / denom
        a_y1 = 1.0 / (gamma * denom)
        b_yn = -((1.0 - gamma) ** 2) / (gamma * denom)

        self._rhs_into(self._f_n, rhs_func, params.t, params.y)

        dt1 = gamma * params.dt
        t1 = params.t + dt1

        np.multiply(self._f_n, dt1, out=self._state_pred)
        self._state_pred += params.y

        self._rhs_into(self._f_pred, rhs_func, t1, self._state_pred)

        np.add(self._f_n, self._f_pred, out=self._rhs_buffer)
        self._rhs_buffer *= 0.5 * dt1
        self._rhs_buffer += params.y

        self._apply_implicit(
            ImplicitStageParams(
                spec=params.operators_tr,
                dt=params.dt,
                scale=gamma,
                t_stage=t1,
                y_stage=self._state_pred,
                stage="tr",
                x=self._rhs_buffer,
                out=self._y_stage1,
            )
        )

        self._rhs_into(self._f_stage1, rhs_func, t1, self._y_stage1)

        c1 = 1.0 / gamma
        c0 = -((1.0 - gamma) / gamma)
        np.multiply(self._f_stage1, c1, out=self._f_extrap)
        self._f_extrap += c0 * self._f_n

        np.multiply(self._y_stage1, a_y1, out=self._rhs_buffer)
        self._rhs_buffer += b_yn * params.y
        self._rhs_buffer += (d * params.dt) * self._f_extrap

        t_np1 = params.t + params.dt
        self._apply_implicit(
            ImplicitStageParams(
                spec=params.operators_bdf2,
                dt=params.dt,
                scale=d,
                t_stage=t_np1,
                y_stage=self._y_stage1,
                stage="bdf2",
                x=self._rhs_buffer,
                out=params.out,
            )
        )

    def _step_imex_trbdf2_doubling(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        operators_tr: CoreOperators | StageOperatorFactory | None,
        operators_bdf2: CoreOperators | StageOperatorFactory | None,
        gamma: float,
    ) -> int:
        """
        TR-BDF2 step with step-doubling error estimate.

        Args:
            rhs_func: RHS function.
            step: Step bundle.
            operators_tr: TR stage operator spec.
            operators_bdf2: BDF2 stage operator spec.
            gamma: TR-BDF2 gamma parameter.

        Returns:
            Method order (2).
        """
        err_out = self._require_err_out(step)

        self._trbdf2_step_once(
            rhs_func,
            Trbdf2OnceParams(
                t=step.t,
                y=step.y,
                dt=step.dt,
                operators_tr=operators_tr,
                operators_bdf2=operators_bdf2,
                gamma=float(gamma),
                out=self._y_full,
            ),
        )

        dt2 = 0.5 * step.dt
        self._trbdf2_step_once(
            rhs_func,
            Trbdf2OnceParams(
                t=step.t,
                y=step.y,
                dt=dt2,
                operators_tr=operators_tr,
                operators_bdf2=operators_bdf2,
                gamma=float(gamma),
                out=self._y_half,
            ),
        )
        self._trbdf2_step_once(
            rhs_func,
            Trbdf2OnceParams(
                t=step.t + dt2,
                y=self._y_half,
                dt=dt2,
                operators_tr=operators_tr,
                operators_bdf2=operators_bdf2,
                gamma=float(gamma),
                out=self._y_two_half,
            ),
        )

        np.subtract(self._y_two_half, self._y_full, out=err_out)
        np.copyto(step.out, self._y_two_half)
        return 2

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _attempt_step(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        step: StepIO,
    ) -> int:
        """
        Dispatch a single attempted step and return the method order.

        Args:
            rhs_func: RHS function.
            plan: Run plan.
            step: Step bundle.

        Raises:
            RuntimeError: If an unknown method is encountered or internal errors occur.

        Returns:
            Method order.
        """
        if plan.method == "euler":
            return self._step_euler_doubling(rhs_func, step)
        if plan.method == "heun":
            return self._step_heun(rhs_func, step)
        if plan.method == "imex-euler":
            return self._step_imex_euler_doubling(
                rhs_func,
                step=step,
                op_spec=plan.op_default,
            )
        if plan.method == "imex-heun-tr":
            return self._step_imex_heun_tr(
                rhs_func,
                step=step,
                op_spec=plan.op_default,
            )
        if plan.method != "imex-trbdf2":
            raise RuntimeError(_UNKNOWN_METHOD_ERROR_MSG.format(method=plan.method))
        if plan.gamma is None:
            raise RuntimeError(_INTERNAL_ERROR_GAMMA_MSG)

        return self._step_imex_trbdf2_doubling(
            rhs_func,
            step=step,
            operators_tr=plan.op_tr,
            operators_bdf2=plan.op_bdf2,
            gamma=plan.gamma,
        )

    def _advance_nonadaptive_to_time(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        t0: float,
        dt_out: float,
        y0: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Advance exactly one step to the next output time.

        Args:
            rhs_func: RHS function.
            plan: Run plan.
            t0: Initial time.
            dt_out: Output time step.
            y0: Initial state.

        Returns:
            State at t0 + dt_out.
        """
        step = StepIO(
            t=t0,
            dt=dt_out,
            y=y0,
            out=self._y_try,
            err_out=self._err,
        )
        _ = self._attempt_step(rhs_func, plan=plan, step=step)
        return self._y_try

    def _advance_adaptive_to_time(
        self,
        rhs_func: RHSFunction,
        params: AdaptiveAdvanceParams,
    ) -> NDArray[np.floating]:
        """Advance with adaptive substepping to land exactly on params.t1.

        Args:
            rhs_func: RHS function.
            params: Adaptive advance parameters.

        Raises:
            RuntimeError: If step rejection limits or dt bounds are violated.

        Returns:
            State at params.t1.
        """
        t0 = float(params.t0)
        t1 = float(params.t1)
        t = t0

        dt_out = float(t1 - t0)
        dt = (
            float(params.adaptive_cfg.dt_init)
            if (
                params.adaptive_cfg.dt_init is not None
                and np.isfinite(params.adaptive_cfg.dt_init)
                and params.adaptive_cfg.dt_init > 0.0
            )
            else dt_out
        )

        if dt > params.dt_ctrl.dt_max:
            dt = float(params.dt_ctrl.dt_max)
        if dt <= 0.0:
            dt = dt_out

        np.copyto(self._y_curr, params.y0)

        n_internal = 0
        while t < t1:
            if n_internal >= params.adaptive_cfg.max_steps:
                raise RuntimeError(_MAX_STEPS_ERROR_MSG)

            remaining = t1 - t
            if remaining <= 0.0:
                break
            dt = min(dt, remaining)

            rejects = 0
            while True:
                if rejects >= params.adaptive_cfg.max_reject:
                    raise RuntimeError(_TOO_MANY_REJECTS_ERROR_MSG)

                step = StepIO(
                    t=t,
                    dt=dt,
                    y=self._y_curr,
                    out=self._y_try,
                    err_out=self._err,
                )
                order = self._attempt_step(rhs_func, plan=params.plan, step=step)

                err_norm = self._error_norm(
                    self._err,
                    self._y_try,
                    self._y_curr,
                    rtol=params.adaptive_cfg.rtol,
                    atol=params.adaptive_cfg.atol,
                )

                if err_norm <= 1.0:
                    t += dt
                    self._y_curr, self._y_try = self._y_try, self._y_curr
                    dt = self._propose_dt(dt, err_norm, order, cfg=params.dt_ctrl)
                    break

                dt_new = self._propose_dt(dt, err_norm, order, cfg=params.dt_ctrl)
                if dt_new <= params.dt_ctrl.dt_min and params.dt_ctrl.dt_min > 0.0:
                    raise RuntimeError(_DT_UNDERFLOW_ERROR_MSG)
                dt = dt_new
                rejects += 1

            n_internal += 1

        return self._y_curr

    # ------------------------------------------------------------------
    # Public run loop
    # ------------------------------------------------------------------

    def run(self, rhs_func: RHSFunction, *, config: RunConfig | None = None) -> None:
        """Advance the ModelCore state through its time grid.

        Args:
            rhs_func: Function computing the explicit RHS F(t, y).
            config: Optional run configuration. If None, defaults are used.

        Raises:
            ValueError: If invalid parameters are provided.
        """
        cfg = config or RunConfig()
        plan = self._resolve_run_plan(cfg)

        time_grid = np.asarray(self.core.time_grid, dtype=float)
        n_steps = int(self.core.n_timesteps)

        for idx in range(n_steps - 1):
            t0 = float(time_grid[idx])
            t1 = float(time_grid[idx + 1])
            if t1 <= t0:
                raise ValueError(_TIME_GRID_INCREASING_ERROR_MSG)
            dt_out = t1 - t0

            np.copyto(self._y_curr, self.core.get_current_state())

            if not cfg.adaptive:
                y_next = self._advance_nonadaptive_to_time(
                    rhs_func,
                    plan=plan,
                    t0=t0,
                    dt_out=dt_out,
                    y0=self._y_curr,
                )
                self.core.advance_timestep(y_next)
                continue

            y_end = self._advance_adaptive_to_time(
                rhs_func,
                AdaptiveAdvanceParams(
                    plan=plan,
                    t0=t0,
                    t1=t1,
                    y0=self._y_curr,
                    adaptive_cfg=cfg.adaptive_cfg,
                    dt_ctrl=cfg.dt_controller,
                ),
            )
            self.core.advance_timestep(y_end)
