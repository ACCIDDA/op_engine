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
    - "rk4":          Classic explicit Runge--Kutta (order 4).
    - "dopri5":       Dormand--Prince 5(4), embedded adaptive estimator.
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
    - NumPy paths retain preallocated scratch arrays and in-place operations.
    - Immutable namespaces use functional stepping operations.
    - Dense implicit solves use Array-API linalg; sparse adapters cache factors.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, identity

from ._rosenbrock import ROSENBROCK_W_TABLEAUS, RosenbrockWTableau
from ._rosenbrock import evaluate_rosenbrock_w as evaluate_rosenbrock_w_tableau
from ._runge_kutta import EXPLICIT_TABLEAUS, ExplicitRungeKuttaTableau
from ._typing import Array
from .matrix_ops import (
    StageOperatorContext,
    build_implicit_euler_operators,
    build_trapezoidal_operators,
    implicit_solve,
)

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
_ARRAY_API_ERROR_MSG = (
    "Explicit solver arrays must implement __array_namespace__(); got {type_name}."
)
_SCHEDULE_TIME_GRID_ERROR_MSG = (
    "Adaptive schedule output_times must match the ModelCore time_grid"
)


# =============================================================================
# Type aliases / protocols
# =============================================================================

RHSFunction = Callable[..., Array]
JacobianFunction = Callable[..., "OperatorLike"]
MethodName = Literal[
    "euler",
    "heun",
    "rk4",
    "dopri5",
    "imex-euler",
    "imex-heun-tr",
    "imex-trbdf2",
    "implicit-euler",
    "trapezoidal",
    "bdf2",
    "ros2",
]

_METHOD_ALIASES: dict[str, MethodName] = {
    "cn": "trapezoidal",
    "crank-nicolson": "trapezoidal",
    "trap": "trapezoidal",
    "rosenbrock": "ros2",
    "rosenbrock-w": "ros2",
    "dormand-prince": "dopri5",
    "dormand-prince-5(4)": "dopri5",
    "rk45": "dopri5",
    "runge-kutta-4": "rk4",
}
_ALLOWED_METHODS: tuple[MethodName, ...] = (
    "euler",
    "heun",
    "rk4",
    "dopri5",
    "imex-euler",
    "imex-heun-tr",
    "imex-trbdf2",
    "implicit-euler",
    "trapezoidal",
    "bdf2",
    "ros2",
)
_EXPLICIT_METHODS = frozenset({"euler", "heun", "rk4", "dopri5"})


def _normalize_method(method: str) -> MethodName:
    """Normalize and validate a user-provided method string.

    Returns:
        Canonical method name.

    Raises:
        ValueError: If the method is unknown.
    """
    method_norm = str(method).strip().lower()
    method_norm = _METHOD_ALIASES.get(method_norm, cast("MethodName", method_norm))
    if method_norm not in _ALLOWED_METHODS:
        raise ValueError(_UNKNOWN_METHOD_ERROR_MSG.format(method=method))
    return method_norm


class OperatorLike(Protocol):
    """Minimal operator interface required by CoreSolver.

    Implementations are expected to behave like 2D linear operators suitable for
    implicit_solve(L, R, rhs2d). Only shape is required for validation.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Operator shape."""
        ...


class PredictorLike(Protocol):
    """Minimal predictor interface required by CoreSolver.

    The predictor is an optional preprocessing operator applied as:
        rhs2d = predictor @ rhs2d
    """

    def __matmul__(self, other: Array) -> Array:
        """Apply the predictor to a 2D array."""
        ...


CoreOperators2 = tuple[OperatorLike, OperatorLike]
CoreOperators3 = tuple[PredictorLike, OperatorLike, OperatorLike]
CoreOperators = CoreOperators2 | CoreOperators3
StageOperatorFactory = Callable[[float, float, StageOperatorContext], CoreOperators]

# What the current (NumPy/SciPy) implicit_solve backend actually accepts.
ScipyOperator: TypeAlias = NDArray[np.floating] | csr_matrix


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace advertised by ``value``.

    Raises:
        TypeError: If ``value`` does not advertise an array namespace.
    """
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        raise TypeError(
            _ARRAY_API_ERROR_MSG.format(type_name=type(value).__name__),
        )
    return namespace()


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

    def __post_init__(self) -> None:
        """Validate static timestep-controller parameters.

        Raises:
            ValueError: If any controller parameter is outside its valid range.
        """
        if not np.isfinite(self.dt_min) or self.dt_min < 0.0:
            msg = "dt_min must be finite and non-negative"
            raise ValueError(msg)
        if np.isnan(self.dt_max) or self.dt_max <= 0.0:
            msg = "dt_max must be positive and not NaN"
            raise ValueError(msg)
        if self.dt_max < self.dt_min:
            msg = "dt_max must be greater than or equal to dt_min"
            raise ValueError(msg)
        if not np.isfinite(self.safety) or self.safety <= 0.0:
            msg = "safety must be finite and positive"
            raise ValueError(msg)
        if not np.isfinite(self.fac_min) or self.fac_min <= 0.0:
            msg = "fac_min must be finite and positive"
            raise ValueError(msg)
        if not np.isfinite(self.fac_max) or self.fac_max <= 0.0:
            msg = "fac_max must be finite and positive"
            raise ValueError(msg)
        if self.fac_max < self.fac_min:
            msg = "fac_max must be greater than or equal to fac_min"
            raise ValueError(msg)


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
    atol: float | Array = 1e-9
    dt_init: float | None = None
    max_reject: int = 25
    max_steps: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate static adaptive-step parameters without coercing arrays.

        Raises:
            ValueError: If any static adaptive parameter is invalid.
        """
        if not np.isfinite(self.rtol) or self.rtol < 0.0:
            msg = "rtol must be finite and non-negative"
            raise ValueError(msg)

        if isinstance(self.atol, (float, int, np.floating, np.integer)):
            if not np.isfinite(self.atol) or self.atol < 0.0:
                msg = "scalar atol must be finite and non-negative"
                raise ValueError(msg)
        elif isinstance(self.atol, np.ndarray) and (
            not np.all(np.isfinite(self.atol)) or np.any(self.atol < 0.0)
        ):
            msg = "NumPy atol values must be finite and non-negative"
            raise ValueError(msg)

        if self.dt_init is not None and (
            not np.isfinite(self.dt_init) or self.dt_init <= 0.0
        ):
            msg = "dt_init must be finite and positive when provided"
            raise ValueError(msg)
        if (
            not isinstance(self.max_reject, Integral)
            or isinstance(self.max_reject, bool)
            or self.max_reject < 1
        ):
            msg = "max_reject must be a positive integer"
            raise ValueError(msg)
        if (
            not isinstance(self.max_steps, Integral)
            or isinstance(self.max_steps, bool)
            or self.max_steps < 1
        ):
            msg = "max_steps must be a positive integer"
            raise ValueError(msg)


@dataclass(slots=True, frozen=True)
class AdaptiveStepSchedule:
    """Accepted step sizes for replaying one adaptive solve.

    The schedule records controller decisions, not array values. Replaying it
    therefore uses the same numerical kernels and active Array-API namespace as
    a live solve while keeping loop lengths and step sizes static.

    Attributes:
        output_times: Output grid used to create the schedule.
        step_sizes: Accepted internal step sizes for each output interval.
    """

    output_times: tuple[float, ...]
    step_sizes: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Normalize and validate the recorded mesh.

        Raises:
            ValueError: If times or step sizes do not define a valid mesh.
        """
        output_times = tuple(float(value) for value in self.output_times)
        step_sizes = tuple(
            tuple(float(step_size) for step_size in interval)
            for interval in self.step_sizes
        )
        object.__setattr__(self, "output_times", output_times)
        object.__setattr__(self, "step_sizes", step_sizes)

        if not output_times:
            msg = "Adaptive schedule must contain at least one output time"
            raise ValueError(msg)
        if any(not math.isfinite(value) for value in output_times):
            msg = "Adaptive schedule output times must be finite"
            raise ValueError(msg)
        if any(end <= start for start, end in pairwise(output_times)):
            msg = "Adaptive schedule output times must be strictly increasing"
            raise ValueError(msg)
        if len(step_sizes) != len(output_times) - 1:
            msg = "Adaptive schedule must contain one step group per output interval"
            raise ValueError(msg)

        for interval_index, interval_steps in enumerate(step_sizes):
            if not interval_steps:
                msg = "Adaptive schedule step groups must not be empty"
                raise ValueError(msg)
            if any(
                not math.isfinite(step_size) or step_size <= 0.0
                for step_size in interval_steps
            ):
                msg = "Adaptive schedule step sizes must be finite and positive"
                raise ValueError(msg)

            interval = output_times[interval_index + 1] - output_times[interval_index]
            if not math.isclose(
                math.fsum(interval_steps),
                interval,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                msg = "Adaptive schedule steps must sum to each output interval"
                raise ValueError(msg)


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
        jacobian: Optional Jacobian function for fully implicit / Rosenbrock methods.
        gamma: Optional TR-BDF2 gamma (if None, uses default).
    """

    method: str = "heun"
    adaptive: bool = False
    strict: bool = True
    dt_controller: DtControllerConfig = DtControllerConfig()
    adaptive_cfg: AdaptiveConfig = AdaptiveConfig()
    operators: OperatorSpecs = OperatorSpecs()
    jacobian: JacobianFunction | None = None
    gamma: float | None = None

    def __post_init__(self) -> None:
        """Normalize the method and validate context-free configuration.

        Raises:
            TypeError: If a nested configuration has the wrong type.
            ValueError: If the method or TR-BDF2 gamma is invalid.
        """
        method = _normalize_method(self.method)
        object.__setattr__(self, "method", method)

        if not isinstance(self.dt_controller, DtControllerConfig):
            msg = "dt_controller must be a DtControllerConfig"
            raise TypeError(msg)
        if not isinstance(self.adaptive_cfg, AdaptiveConfig):
            msg = "adaptive_cfg must be an AdaptiveConfig"
            raise TypeError(msg)
        if not isinstance(self.operators, OperatorSpecs):
            msg = "operators must be an OperatorSpecs"
            raise TypeError(msg)

        if method == "imex-trbdf2" and self.gamma is not None:
            gamma = float(self.gamma)
            if not np.isfinite(gamma) or not (0.0 < gamma < 1.0):
                raise ValueError(_GAMMA_RANGE_ERROR_MSG)


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
        jacobian: Optional Jacobian function for fully implicit / Rosenbrock methods.
    """

    method: MethodName
    gamma: float | None
    op_default: CoreOperators | StageOperatorFactory | None
    op_tr: CoreOperators | StageOperatorFactory | None
    op_bdf2: CoreOperators | StageOperatorFactory | None
    jacobian: JacobianFunction | None


@dataclass(slots=True)
class StepIO:
    """Bundle of per-step state for stepping kernels.

    Attributes:
        t: Current time.
        dt: Step size.
        y: Current state array (input).
        out: Output state array (written in-place).
        err_out: Error estimate array (written in-place) for adaptive methods.
        y_prev: Optional previous state (for multistep methods).
    """

    t: float
    dt: float
    y: NDArray[np.floating]
    out: NDArray[np.floating]
    err_out: NDArray[np.floating] | None = None
    y_prev: NDArray[np.floating] | None = None


@dataclass(slots=True, frozen=True)
class ExplicitStepResult:
    """Result and reusable stages from one explicit adaptive attempt.

    Attributes:
        state: Accepted-order candidate state.
        error: Local error estimate.
        controller_order: Order supplied to the existing step-size controller.
        first_stage: Derivative at the attempted step's initial state.
        last_stage: Derivative reusable by an FSAL method after acceptance.
    """

    state: Array
    error: Array
    controller_order: int
    first_stage: Array
    last_stage: Array | None


class _LinearizedStepFunction(Protocol):
    """Shared signature for NumPy/SciPy linearized step implementations."""

    def __call__(
        self,
        solver: CoreSolver,
        rhs_func: RHSFunction,
        /,
        *,
        step: StepIO,
        jacobian: JacobianFunction,
    ) -> int:
        """Attempt one step and return its method order."""


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

        # Working state buffers (avoid allocating per substep)
        self._y_curr: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._y_try: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # TR-BDF2 additional buffers
        self._y_stage1: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_stage1: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._f_extrap: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)

        # Previous-step cache (for multistep methods like BDF2)
        self._prev_state_cache: NDArray[np.floating] = np.zeros_like(self._rhs_buffer)
        self._has_prev_state = False
        self._last_adaptive_schedule: AdaptiveStepSchedule | None = None

        # Validate operator sizes if default spec is a static tuple
        if operators is not None and not callable(operators):
            _predictor, left_op, right_op = self._normalize_ops_tuple(operators)
            if left_op is not None and right_op is not None:
                self._resolve_operator_axis()
                self._validate_operator_sizes(left_op, right_op)

    @property
    def last_adaptive_schedule(self) -> AdaptiveStepSchedule | None:
        """Return the schedule recorded or replayed by the latest adaptive run."""
        return self._last_adaptive_schedule

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

        Returns:
            Operator as a supported SciPy/NumPy operator type.

        Raises:
            TypeError: If op is not a supported operator type for this backend.
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

    def _apply_operator_matmul(
        self,
        op: OperatorLike,
        x: NDArray[np.floating],
        out: NDArray[np.floating],
    ) -> None:
        """Compute out = op @ x along the operator axis.

        Args:
            op: Operator acting along the configured axis.
            x: Input state-like tensor.
            out: Output array to write.
        """
        self._validate_operator_sizes(op, op)  # single-op: left=right
        x2d, original_shape = self._reshape_for_solve(x)
        op_s = self._as_scipy_operator(op)
        y2d = op_s @ x2d
        y_arr = self._unreshape_after_solve(
            np.asarray(y2d, dtype=self.dtype), original_shape
        )
        np.copyto(out, y_arr)

    @staticmethod
    def _normalize_ops_tuple(
        ops: CoreOperators,
    ) -> tuple[PredictorLike | None, OperatorLike | None, OperatorLike | None]:
        """
        Normalize a 2- or 3-tuple operator spec to (predictor, L, R).

        Args:
            ops: Operator spec tuple.

        Returns:
            Tuple of (predictor, L, R) operators.

        Raises:
            ValueError: If ops is not a 2- or 3-tuple.
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

    def _identity_operator(self) -> NDArray[np.floating]:
        """Return a dense identity operator sized to the operator axis."""
        n = self._require_op_axis_len()
        return np.eye(n, dtype=self.dtype)

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

    def _reshape_array_for_solve(
        self,
        value: Array,
    ) -> tuple[Array, tuple[int, ...], tuple[int, ...]]:
        """Move the operator axis first and flatten the remaining axes.

        Returns:
            Two-dimensional value, original shape, and forward permutation.

        Raises:
            RuntimeError: If the configured operator axis cannot be resolved.
        """
        self._resolve_operator_axis()
        if self._op_axis_idx is None:
            raise RuntimeError(_INTERNAL_ERROR_OP_AXIS_MSG)

        xp = _namespace_of(value)
        original_shape = value.shape
        axes = (
            self._op_axis_idx,
            *(
                index
                for index in range(len(original_shape))
                if index != self._op_axis_idx
            ),
        )
        moved = xp.permute_dims(value, axes)
        reshaped = xp.reshape(moved, (original_shape[self._op_axis_idx], -1))
        return cast("Array", reshaped), original_shape, axes

    @staticmethod
    def _unreshape_array_after_solve(
        value: Array,
        original_shape: tuple[int, ...],
        axes: tuple[int, ...],
    ) -> Array:
        """Undo :meth:`_reshape_array_for_solve` in the active namespace.

        Returns:
            State tensor with ``original_shape``.
        """
        xp = _namespace_of(value)
        moved_shape = (
            original_shape[axes[0]],
            *(original_shape[index] for index in axes[1:]),
        )
        moved = xp.reshape(value, moved_shape)
        inverse = tuple(axes.index(index) for index in range(len(axes)))
        return cast("Array", xp.permute_dims(moved, inverse))

    @staticmethod
    def _operator_array(op: object, reference: Array) -> Array:
        """Convert a dense-capable operator to ``reference``'s namespace.

        Returns:
            Dense operator array in the reference namespace.
        """
        xp = _namespace_of(reference)
        toarray = getattr(op, "toarray", None)
        value = toarray() if callable(toarray) else op
        return cast("Array", xp.asarray(value, dtype=reference.dtype))

    def _apply_operator_matmul_array(self, op: OperatorLike, x: Array) -> Array:
        """Apply an operator along the configured axis without host coercion.

        Returns:
            Operator product in ``x``'s namespace.
        """
        self._validate_operator_sizes(op, op)
        xp = _namespace_of(x)
        x2d, original_shape, axes = self._reshape_array_for_solve(x)
        op_array = self._operator_array(op, x)
        y2d = cast("Array", xp.matmul(op_array, x2d))
        return self._unreshape_array_after_solve(y2d, original_shape, axes)

    def _apply_operator_solve_array(
        self,
        x: Array,
        *,
        predictor: PredictorLike | None,
        left_op: OperatorLike | None,
        right_op: OperatorLike | None,
    ) -> Array:
        """Apply an implicit mapping in ``x``'s namespace.

        Returns:
            Mapped state tensor.
        """
        if left_op is None or right_op is None:
            return x

        self._validate_operator_sizes(left_op, right_op)
        xp = _namespace_of(x)
        x2d, original_shape, axes = self._reshape_array_for_solve(x)
        if predictor is not None:
            predictor_array = self._operator_array(predictor, x)
            x2d = cast("Array", xp.matmul(predictor_array, x2d))

        out2d = implicit_solve(left_op, right_op, x2d)
        return self._unreshape_array_after_solve(out2d, original_shape, axes)

    def _apply_implicit_array(  # noqa: PLR0913
        self,
        spec: CoreOperators | StageOperatorFactory | None,
        *,
        dt: float,
        scale: float,
        t_stage: float,
        y_stage: Array,
        stage: str,
        x: Array,
    ) -> Array:
        """Resolve and apply an implicit stage without mutable scratch arrays.

        Returns:
            Stage result in ``x``'s namespace.
        """
        if spec is None:
            return x

        ctx = StageOperatorContext(t=float(t_stage), y=y_stage, stage=stage)
        predictor, left_op, right_op = self._resolve_stage_operators(
            spec,
            dt=dt,
            scale=scale,
            ctx=ctx,
        )
        return self._apply_operator_solve_array(
            x,
            predictor=predictor,
            left_op=left_op,
            right_op=right_op,
        )

    def _identity_array(self, reference: Array) -> Array:
        """Return an operator-axis identity in ``reference``'s namespace."""
        xp = _namespace_of(reference)
        return cast(
            "Array",
            xp.eye(self._require_op_axis_len(), dtype=reference.dtype),
        )

    def _build_dense_stage_operators(
        self,
        base_op: OperatorLike,
        reference: Array,
        *,
        dt_scale: float,
        trapezoidal: bool,
    ) -> tuple[Array, Array]:
        """Build implicit Euler or trapezoidal dense operators natively.

        Returns:
            Left and right stage operators.
        """
        xp = _namespace_of(reference)
        base = self._operator_array(base_op, reference)
        identity_op = self._identity_array(reference)
        scale = 0.5 * dt_scale if trapezoidal else dt_scale
        scaled = xp.multiply(base, scale)
        left_op = cast("Array", xp.subtract(identity_op, scaled))
        right_op = (
            cast("Array", xp.add(identity_op, scaled)) if trapezoidal else identity_op
        )
        return left_op, right_op

    def _linearized_residual_array(
        self,
        rhs_func: RHSFunction,
        jacobian: JacobianFunction,
        *,
        t: float,
        y: Array,
    ) -> tuple[OperatorLike, Array]:
        """Return a Jacobian and ``f(t, y) - J @ y`` natively.

        Returns:
            Jacobian operator and linearization residual.
        """
        xp = _namespace_of(y)
        rhs = self._rhs_array(rhs_func, t, y)
        jac = jacobian(float(t), y)
        jac_y = self._apply_operator_matmul_array(jac, y)
        residual = cast("Array", xp.subtract(rhs, jac_y))
        return jac, residual

    def _imex_euler_array_once(
        self,
        rhs_func: RHSFunction,
        *,
        t: float,
        dt: float,
        y: Array,
        op_spec: CoreOperators | StageOperatorFactory | None,
    ) -> Array:
        """Take one functional IMEX Euler step.

        Returns:
            Next state in ``y``'s namespace.
        """
        xp = _namespace_of(y)
        rhs = self._rhs_array(rhs_func, t, y)
        explicit = cast("Array", xp.add(y, xp.multiply(rhs, dt)))
        return self._apply_implicit_array(
            op_spec,
            dt=dt,
            scale=1.0,
            t_stage=t + dt,
            y_stage=y,
            stage="be",
            x=explicit,
        )

    def _imex_trbdf2_array_once(  # noqa: PLR0913, PLR0914
        self,
        rhs_func: RHSFunction,
        *,
        t: float,
        dt: float,
        y: Array,
        operators_tr: CoreOperators | StageOperatorFactory | None,
        operators_bdf2: CoreOperators | StageOperatorFactory | None,
        gamma: float,
    ) -> Array:
        """Take one functional IMEX TR-BDF2 step.

        Returns:
            Next state in ``y``'s namespace.
        """
        xp = _namespace_of(y)
        denom = 2.0 - gamma
        d = (1.0 - gamma) / denom
        a_y1 = 1.0 / (gamma * denom)
        b_yn = -((1.0 - gamma) ** 2) / (gamma * denom)

        f_n = self._rhs_array(rhs_func, t, y)
        dt1 = gamma * dt
        t1 = t + dt1
        state_pred = cast("Array", xp.add(y, xp.multiply(f_n, dt1)))
        f_pred = self._rhs_array(rhs_func, t1, state_pred)
        average_rhs = xp.multiply(xp.add(f_n, f_pred), 0.5 * dt1)
        stage_rhs = cast("Array", xp.add(y, average_rhs))
        y_stage1 = self._apply_implicit_array(
            operators_tr,
            dt=dt,
            scale=gamma,
            t_stage=t1,
            y_stage=state_pred,
            stage="tr",
            x=stage_rhs,
        )

        f_stage1 = self._rhs_array(rhs_func, t1, y_stage1)
        extrapolated = xp.add(
            xp.multiply(f_stage1, 1.0 / gamma),
            xp.multiply(f_n, -((1.0 - gamma) / gamma)),
        )
        final_rhs = xp.add(
            xp.add(
                xp.multiply(y_stage1, a_y1),
                xp.multiply(y, b_yn),
            ),
            xp.multiply(extrapolated, d * dt),
        )
        return self._apply_implicit_array(
            operators_bdf2,
            dt=dt,
            scale=d,
            t_stage=t + dt,
            y_stage=y_stage1,
            stage="bdf2",
            x=cast("Array", final_rhs),
        )

    def _implicit_euler_array_once(
        self,
        rhs_func: RHSFunction,
        jacobian: JacobianFunction,
        *,
        t: float,
        dt: float,
        y: Array,
    ) -> Array:
        """Take one linearly implicit Euler step natively.

        Returns:
            Next state in ``y``'s namespace.
        """
        xp = _namespace_of(y)
        jac, residual = self._linearized_residual_array(
            rhs_func,
            jacobian,
            t=t,
            y=y,
        )
        left_op, right_op = self._build_dense_stage_operators(
            jac,
            y,
            dt_scale=dt,
            trapezoidal=False,
        )
        solve_rhs = cast("Array", xp.add(y, xp.multiply(residual, dt)))
        return self._apply_operator_solve_array(
            solve_rhs,
            predictor=None,
            left_op=left_op,
            right_op=right_op,
        )

    @staticmethod
    def _weighted_array_state(
        base: Array,
        base_weight: float,
        weights: tuple[float, ...],
        stages: Sequence[Array],
    ) -> Array:
        """Return a weighted state/stage sum in ``base``'s namespace."""
        xp = _namespace_of(base)
        result = cast("Array", xp.multiply(base, base_weight))
        for weight, stage in zip(weights, stages, strict=True):
            if weight != 0.0:
                result = cast(
                    "Array",
                    xp.add(result, xp.multiply(stage, weight)),
                )
        return result

    def _evaluate_rosenbrock_w_array(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        jacobian: JacobianFunction,
        *,
        tableau: RosenbrockWTableau,
        t: float,
        dt: float,
        y: Array,
    ) -> tuple[Array, Array]:
        """Evaluate one dense Rosenbrock-W tableau in ``y``'s namespace.

        Returns:
            Accepted and embedded states.
        """
        jac = jacobian(float(t), y)
        left_op, right_op = self._build_dense_stage_operators(
            jac,
            y,
            dt_scale=tableau.gamma * dt,
            trapezoidal=False,
        )

        def solve(stage_rhs: Array) -> Array:
            return self._apply_operator_solve_array(
                stage_rhs,
                predictor=None,
                left_op=left_op,
                right_op=right_op,
            )

        return evaluate_rosenbrock_w_tableau(
            tableau=tableau,
            t=t,
            dt=dt,
            y=y,
            rhs=lambda stage_time, stage_state: self._rhs_array(
                rhs_func,
                stage_time,
                stage_state,
            ),
            solve=solve,
            weighted_sum=self._weighted_array_state,
        )

    def _attempt_array_implicit_step(  # noqa: C901, PLR0911, PLR0913, PLR0914
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        t: float,
        dt: float,
        y: Array,
        y_prev: Array | None = None,
    ) -> tuple[Array, Array, int]:
        """Attempt any implicit/IMEX method with functional array operations.

        Returns:
            Candidate state, error estimate, and method order.

        Raises:
            RuntimeError: If the resolved plan is internally inconsistent.
        """
        xp = _namespace_of(y)

        if plan.method == "imex-euler":
            y_full = self._imex_euler_array_once(
                rhs_func,
                t=t,
                dt=dt,
                y=y,
                op_spec=plan.op_default,
            )
            y_half = self._imex_euler_array_once(
                rhs_func,
                t=t,
                dt=0.5 * dt,
                y=y,
                op_spec=plan.op_default,
            )
            y_two_half = self._imex_euler_array_once(
                rhs_func,
                t=t + 0.5 * dt,
                dt=0.5 * dt,
                y=y_half,
                op_spec=plan.op_default,
            )
            error = cast("Array", xp.subtract(y_two_half, y_full))
            return y_two_half, error, 1

        if plan.method == "imex-heun-tr":
            f_n = self._rhs_array(rhs_func, t, y)
            state_pred = cast("Array", xp.add(y, xp.multiply(f_n, dt)))
            f_pred = self._rhs_array(rhs_func, t + dt, state_pred)
            high_rhs = cast(
                "Array",
                xp.add(y, xp.multiply(xp.add(f_n, f_pred), 0.5 * dt)),
            )
            y_low = self._apply_implicit_array(
                plan.op_default,
                dt=dt,
                scale=1.0,
                t_stage=t + dt,
                y_stage=state_pred,
                stage="tr",
                x=state_pred,
            )
            y_high = self._apply_implicit_array(
                plan.op_default,
                dt=dt,
                scale=1.0,
                t_stage=t + dt,
                y_stage=state_pred,
                stage="tr",
                x=high_rhs,
            )
            return y_high, cast("Array", xp.subtract(y_high, y_low)), 2

        if plan.method == "imex-trbdf2":
            if plan.gamma is None:
                raise RuntimeError(_INTERNAL_ERROR_GAMMA_MSG)
            y_full = self._imex_trbdf2_array_once(
                rhs_func,
                t=t,
                dt=dt,
                y=y,
                operators_tr=plan.op_tr,
                operators_bdf2=plan.op_bdf2,
                gamma=plan.gamma,
            )
            y_half = self._imex_trbdf2_array_once(
                rhs_func,
                t=t,
                dt=0.5 * dt,
                y=y,
                operators_tr=plan.op_tr,
                operators_bdf2=plan.op_bdf2,
                gamma=plan.gamma,
            )
            y_two_half = self._imex_trbdf2_array_once(
                rhs_func,
                t=t + 0.5 * dt,
                dt=0.5 * dt,
                y=y_half,
                operators_tr=plan.op_tr,
                operators_bdf2=plan.op_bdf2,
                gamma=plan.gamma,
            )
            return y_two_half, cast("Array", xp.subtract(y_two_half, y_full)), 2

        if plan.jacobian is None:
            raise RuntimeError(_INTERNAL_ERROR_OP_AXIS_MSG)

        if plan.method == "implicit-euler":
            y_full = self._implicit_euler_array_once(
                rhs_func,
                plan.jacobian,
                t=t,
                dt=dt,
                y=y,
            )
            y_half = self._implicit_euler_array_once(
                rhs_func,
                plan.jacobian,
                t=t,
                dt=0.5 * dt,
                y=y,
            )
            y_two_half = self._implicit_euler_array_once(
                rhs_func,
                plan.jacobian,
                t=t + 0.5 * dt,
                dt=0.5 * dt,
                y=y_half,
            )
            return y_two_half, cast("Array", xp.subtract(y_two_half, y_full)), 1

        if plan.method == "trapezoidal":
            jac, residual = self._linearized_residual_array(
                rhs_func,
                plan.jacobian,
                t=t,
                y=y,
            )
            left_op, right_op = self._build_dense_stage_operators(
                jac,
                y,
                dt_scale=dt,
                trapezoidal=True,
            )
            mapped = self._apply_operator_matmul_array(right_op, y)
            solve_rhs = cast("Array", xp.add(mapped, xp.multiply(residual, dt)))
            y_high = self._apply_operator_solve_array(
                solve_rhs,
                predictor=None,
                left_op=left_op,
                right_op=self._identity_array(y),
            )
            y_low = self._implicit_euler_array_once(
                rhs_func,
                plan.jacobian,
                t=t,
                dt=dt,
                y=y,
            )
            return y_high, cast("Array", xp.subtract(y_high, y_low)), 2

        if plan.method == "bdf2":
            if y_prev is None:
                y_next = self._implicit_euler_array_once(
                    rhs_func,
                    plan.jacobian,
                    t=t,
                    dt=dt,
                    y=y,
                )
                return y_next, cast("Array", xp.zeros_like(y_next)), 1

            jac, residual = self._linearized_residual_array(
                rhs_func,
                plan.jacobian,
                t=t,
                y=y,
            )
            jac_array = self._operator_array(jac, y)
            identity_op = self._identity_array(y)
            left_op = cast(
                "Array",
                xp.subtract(
                    xp.multiply(identity_op, 1.5),
                    xp.multiply(jac_array, dt),
                ),
            )
            solve_rhs = xp.add(
                xp.subtract(xp.multiply(y, 2.0), xp.multiply(y_prev, 0.5)),
                xp.multiply(residual, dt),
            )
            y_high = self._apply_operator_solve_array(
                cast("Array", solve_rhs),
                predictor=None,
                left_op=left_op,
                right_op=identity_op,
            )
            y_low = self._implicit_euler_array_once(
                rhs_func,
                plan.jacobian,
                t=t,
                dt=dt,
                y=y,
            )
            return y_high, cast("Array", xp.subtract(y_high, y_low)), 2

        if plan.method == "ros2":
            tableau = ROSENBROCK_W_TABLEAUS[plan.method]
            y_high, y_low = self._evaluate_rosenbrock_w_array(
                rhs_func,
                plan.jacobian,
                tableau=tableau,
                t=t,
                dt=dt,
                y=y,
            )
            error = cast("Array", xp.subtract(y_high, y_low))
            return y_high, error, tableau.controller_order

        raise RuntimeError(_UNKNOWN_METHOD_ERROR_MSG.format(method=plan.method))

    # ------------------------------------------------------------------
    # RHS evaluation helper (shape + dtype enforcement)
    # ------------------------------------------------------------------

    def _rhs_array(self, rhs_func: RHSFunction, t: float, y: Array) -> Array:
        """Evaluate an RHS without converting it out of ``y``'s namespace.

        Args:
            rhs_func: RHS function F(t, y).
            t: Time.
            y: State array whose namespace owns this evaluation.

        Returns:
            RHS array in the same numerical ecosystem as ``y``.

        Raises:
            TypeError: If the RHS result is not an Array-API array.
            ValueError: If the RHS result has an unexpected shape.
        """
        xp = _namespace_of(y)
        result = rhs_func(float(t), y)
        result_xp = _namespace_of(result)
        if result_xp is not xp:
            msg = (
                "rhs_func must preserve the input array namespace; "
                f"got {result_xp!r}, expected {xp!r}."
            )
            raise TypeError(msg)
        if result.shape != self.state_shape:
            raise ValueError(
                _RHS_SHAPE_ERROR_MSG.format(
                    actual=result.shape,
                    expected=self.state_shape,
                )
            )
        return result

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

    def _compute_linearized_residual(
        self,
        rhs_func: RHSFunction,
        jacobian: JacobianFunction,
        t: float,
        y: NDArray[np.floating],
    ) -> OperatorLike:
        """Compute J(t, y) and residual r = f - J y for linearized solves.

        Writes residual into ``self._f_extrap``.

        Returns:
            Jacobian operator J(t, y).
        """
        self._rhs_into(self._f_n, rhs_func, t, y)
        jac = jacobian(float(t), y)
        self._apply_operator_matmul(jac, y, self._f_pred)
        np.subtract(self._f_n, self._f_pred, out=self._f_extrap)
        return jac

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
    def _require_jacobian(
        method: MethodName,
        jacobian: JacobianFunction | None,
    ) -> JacobianFunction:
        """Ensure a Jacobian is provided for fully implicit methods.

        Args:
            method: Method name.
            jacobian: Optional Jacobian function.

        Returns:
            Jacobian function.

        Raises:
            ValueError: If Jacobian is missing.
        """
        if jacobian is None:
            msg = (
                f"Method '{method}' requires a jacobian(t, y) callable; none provided."
            )
            raise ValueError(msg)
        return jacobian

    @staticmethod
    def _resolve_gamma(method: MethodName, gamma: float | None) -> float | None:
        """
        Resolve TR-BDF2 gamma parameter.

        Args:
            method: Method name.
            gamma: User-provided gamma (or None for default).

        Returns:
            Resolved gamma for TR-BDF2, or None for non-TR-BDF2 methods.

        Raises:
            ValueError: If gamma is out of range for TR-BDF2.
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
            method_in: Explicit method name.
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
            jacobian=None,
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

        Returns:
            RunPlan for IMEX method, or an explicit fallback if strict=False and
            operators are missing.

        Raises:
            ValueError: If required operators are missing.
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
                jacobian=None,
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
            jacobian=None,
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

        Returns:
            RunPlan for TR-BDF2, or a Heun fallback if strict=False and operators are
            missing.

        Raises:
            RuntimeError: If method_in is not "imex-trbdf2".
            ValueError: If required operators are missing.
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
                jacobian=None,
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
            jacobian=None,
        )

    def _resolve_run_plan(self, cfg: RunConfig) -> RunPlan:
        """Resolve method/operators into an executable plan.

        Args:
            cfg: Run configuration.

        Returns:
            Resolved plan.

        Raises:
            ValueError: If invalid parameters are provided.
        """
        method_in = _normalize_method(cfg.method)
        gamma = self._resolve_gamma(method_in, cfg.gamma)

        op_default = (
            cfg.operators.default
            if cfg.operators.default is not None
            else self._default_operator_spec
        )
        op_tr = cfg.operators.tr
        op_bdf2 = cfg.operators.bdf2
        jacobian = cfg.jacobian

        strict = bool(cfg.strict)
        adaptive = bool(cfg.adaptive)

        if method_in in _EXPLICIT_METHODS:
            return self._plan_for_explicit(method_in, op_default, strict=strict)

        if method_in in {"imex-euler", "imex-heun-tr"}:
            return self._plan_for_imex_single(
                method_in,
                op_default,
                strict=strict,
                adaptive=adaptive,
            )

        if method_in in {"implicit-euler", "trapezoidal", "bdf2", "ros2"}:
            jac_required = self._require_jacobian(method_in, jacobian)

            if method_in == "bdf2":
                if adaptive:
                    msg = "Method 'bdf2' currently supports adaptive=False"
                    raise ValueError(msg)
                if not self._output_dt_is_uniform():
                    msg = "Method 'bdf2' currently requires a uniform output time grid"
                    raise ValueError(msg)

            return RunPlan(
                method=method_in,
                gamma=None,
                op_default=None,
                op_tr=None,
                op_bdf2=None,
                jacobian=jac_required,
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

    @staticmethod
    def _error_norm(
        err: Array,
        y_ref: Array,
        y_prev: Array,
        *,
        rtol: float,
        atol: float | Array,
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
        xp = _namespace_of(err)
        scale = xp.maximum(xp.abs(y_ref), xp.abs(y_prev))
        scale = xp.multiply(scale, rtol)
        if isinstance(atol, (float, int, np.floating)):
            scale = xp.add(scale, float(atol))
        else:
            atol_arr = xp.asarray(atol, dtype=err.dtype)
            scale = xp.add(scale, atol_arr)

        ratio = xp.divide(err, scale)
        squared = xp.multiply(ratio, ratio)
        norm = cast("Array", xp.sqrt(xp.mean(squared)))
        value = float(norm.item())
        if not np.isfinite(value):
            return float("inf")
        return value

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

    def _step_explicit_euler_once(
        self,
        rhs_func: RHSFunction,
        *,
        t: float,
        dt: float,
        y: Array,
    ) -> Array:
        """Return one explicit Euler step in ``y``'s namespace."""
        xp = _namespace_of(y)
        f_n = self._rhs_array(rhs_func, t, y)
        return cast("Array", xp.add(y, xp.multiply(f_n, dt)))

    def _step_explicit_euler_doubling(
        self,
        rhs_func: RHSFunction,
        *,
        t: float,
        dt: float,
        y: Array,
        first_stage: Array | None = None,
    ) -> ExplicitStepResult:
        """Return an Euler step and doubling error in ``y``'s namespace."""
        xp = _namespace_of(y)
        f_n = (
            first_stage if first_stage is not None else self._rhs_array(rhs_func, t, y)
        )
        y_full = cast("Array", xp.add(y, xp.multiply(f_n, dt)))
        y_half = cast("Array", xp.add(y, xp.multiply(f_n, 0.5 * dt)))
        f_half = self._rhs_array(rhs_func, t + 0.5 * dt, y_half)
        y_two_half = cast(
            "Array",
            xp.add(y_half, xp.multiply(f_half, 0.5 * dt)),
        )
        error = cast("Array", xp.subtract(y_two_half, y_full))
        return ExplicitStepResult(y_two_half, error, 1, f_n, None)

    @staticmethod
    def _weighted_rk_state(
        y: Array,
        dt: float,
        weights: tuple[float, ...],
        stages: list[Array],
    ) -> Array:
        """Return ``y + dt * sum(weights[i] * stages[i])`` natively."""
        xp = _namespace_of(y)
        result = y
        for weight, stage in zip(weights, stages, strict=True):
            if weight != 0.0:
                result = cast(
                    "Array",
                    xp.add(result, xp.multiply(stage, dt * weight)),
                )
        return result

    def _evaluate_explicit_rk(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        tableau: ExplicitRungeKuttaTableau,
        t: float,
        dt: float,
        y: Array,
        first_stage: Array | None = None,
    ) -> tuple[Array, Array | None, Array, Array | None]:
        """Evaluate one explicit tableau with optional first-stage reuse.

        Returns:
            High-order state, optional embedded state, first stage, and an
            optional FSAL stage for the next accepted step.
        """
        stages: list[Array] = []
        for stage_index, (row, stage_time) in enumerate(
            zip(tableau.a, tableau.c, strict=True)
        ):
            if stage_index == 0 and first_stage is not None:
                derivative = first_stage
            else:
                stage_state = self._weighted_rk_state(y, dt, row, stages)
                derivative = self._rhs_array(
                    rhs_func,
                    t + stage_time * dt,
                    stage_state,
                )
            stages.append(derivative)

        high = self._weighted_rk_state(y, dt, tableau.b, stages)
        embedded = (
            None
            if tableau.b_embedded is None
            else self._weighted_rk_state(y, dt, tableau.b_embedded, stages)
        )
        last_stage = stages[-1] if tableau.fsal else None
        return high, embedded, stages[0], last_stage

    def _attempt_explicit_step(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        method: MethodName,
        t: float,
        dt: float,
        y: Array,
        first_stage: Array | None = None,
    ) -> ExplicitStepResult:
        """Dispatch one functional explicit step.

        Returns:
            Candidate state, error estimate, controller order, and reusable
            derivative stages.

        Raises:
            RuntimeError: If called with a non-explicit method.
        """
        if method == "euler":
            return self._step_explicit_euler_doubling(
                rhs_func,
                t=t,
                dt=dt,
                y=y,
                first_stage=first_stage,
            )
        tableau = EXPLICIT_TABLEAUS.get(method)
        if tableau is None:
            raise RuntimeError(_UNKNOWN_METHOD_ERROR_MSG.format(method=method))

        high, embedded, stage_zero, last_stage = self._evaluate_explicit_rk(
            rhs_func,
            tableau=tableau,
            t=t,
            dt=dt,
            y=y,
            first_stage=first_stage,
        )
        xp = _namespace_of(y)
        if embedded is not None:
            if tableau.embedded_order is None:
                raise RuntimeError(_INTERNAL_ERROR_ERR_OUT_MSG)
            error = cast("Array", xp.subtract(high, embedded))
            return ExplicitStepResult(
                high,
                error,
                tableau.embedded_order,
                stage_zero,
                last_stage,
            )

        half, _embedded, _half_first, _half_last = self._evaluate_explicit_rk(
            rhs_func,
            tableau=tableau,
            t=t,
            dt=0.5 * dt,
            y=y,
            first_stage=stage_zero,
        )
        two_half, _embedded, _half_first, _half_last = self._evaluate_explicit_rk(
            rhs_func,
            tableau=tableau,
            t=t + 0.5 * dt,
            dt=0.5 * dt,
            y=half,
        )
        error_scale = 1.0 / (2.0**tableau.order - 1.0)
        error = cast(
            "Array",
            xp.multiply(xp.subtract(two_half, high), error_scale),
        )
        return ExplicitStepResult(
            two_half,
            error,
            tableau.order,
            stage_zero,
            None,
        )

    def _step_explicit_fixed(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        method: MethodName,
        t: float,
        dt: float,
        y: Array,
        first_stage: Array | None = None,
    ) -> tuple[Array, Array | None]:
        """Take one fixed explicit step without adaptive-only error work.

        Returns:
            Next state and an optional FSAL stage for the following step.

        Raises:
            RuntimeError: If called with a non-explicit method.
        """
        if method == "euler":
            return self._step_explicit_euler_once(
                rhs_func,
                t=t,
                dt=dt,
                y=y,
            ), None
        tableau = EXPLICIT_TABLEAUS.get(method)
        if tableau is None:
            raise RuntimeError(_UNKNOWN_METHOD_ERROR_MSG.format(method=method))
        high, _embedded, _stage_zero, last_stage = self._evaluate_explicit_rk(
            rhs_func,
            tableau=tableau,
            t=t,
            dt=dt,
            y=y,
            first_stage=first_stage,
        )
        return high, last_stage

    @staticmethod
    def _require_err_out(step: StepIO) -> NDArray[np.floating]:
        """
        Return err_out for a step, raising if missing.

        Args:
            step: Step bundle.

        Returns:
            err_out array.

        Raises:
            RuntimeError: If err_out is None.
        """
        if step.err_out is None:
            raise RuntimeError(_INTERNAL_ERROR_ERR_OUT_MSG)
        return step.err_out

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
    # Fully implicit + Rosenbrock (linearly implicit) methods
    # ------------------------------------------------------------------

    def _implicit_euler_linearized_once(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        t: float,
        y: NDArray[np.floating],
        dt: float,
        jacobian: JacobianFunction,
        out: NDArray[np.floating],
    ) -> None:
        jac = self._compute_linearized_residual(rhs_func, jacobian, t, y)
        base_op = self._as_scipy_operator(jac)
        left_op, right_op = build_implicit_euler_operators(base_op, dt_scale=dt)

        np.copyto(self._rhs_buffer, y)
        self._rhs_buffer += dt * self._f_extrap

        self._apply_operator_solve_with_ops(
            self._rhs_buffer,
            predictor=None,
            left_op=left_op,
            right_op=right_op,
            out=out,
        )

    def _step_implicit_euler_linearized(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        jacobian: JacobianFunction,
    ) -> int:
        """Implicit Euler with linearization + step-doubling estimator.

        Returns:
            Method order (1).
        """
        err_out = self._require_err_out(step)

        self._implicit_euler_linearized_once(
            rhs_func,
            t=step.t,
            y=step.y,
            dt=step.dt,
            jacobian=jacobian,
            out=self._y_full,
        )

        dt2 = 0.5 * step.dt
        self._implicit_euler_linearized_once(
            rhs_func,
            t=step.t,
            y=step.y,
            dt=dt2,
            jacobian=jacobian,
            out=self._y_half,
        )
        self._implicit_euler_linearized_once(
            rhs_func,
            t=step.t + dt2,
            y=self._y_half,
            dt=dt2,
            jacobian=jacobian,
            out=self._y_two_half,
        )

        np.subtract(self._y_two_half, self._y_full, out=err_out)
        np.copyto(step.out, self._y_two_half)
        return 1

    def _trapezoidal_linearized_step(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        jacobian: JacobianFunction,
    ) -> int:
        """Trapezoidal (Crank-Nicolson-style) with linearized residual.

        Uses implicit Euler (linearized) as the embedded low-order estimator.

        Returns:
            Method order (2).
        """
        err_out = self._require_err_out(step)

        jac = self._compute_linearized_residual(rhs_func, jacobian, step.t, step.y)
        base_op = self._as_scipy_operator(jac)
        left_op, right_op = build_trapezoidal_operators(base_op, dt_scale=step.dt)

        self._apply_operator_matmul(right_op, step.y, self._rhs_buffer)
        self._rhs_buffer += step.dt * self._f_extrap

        self._apply_operator_solve_with_ops(
            self._rhs_buffer,
            predictor=None,
            left_op=left_op,
            right_op=self._identity_operator(),
            out=step.out,
        )

        self._implicit_euler_linearized_once(
            rhs_func,
            t=step.t,
            y=step.y,
            dt=step.dt,
            jacobian=jacobian,
            out=self._y_full,
        )
        np.subtract(step.out, self._y_full, out=err_out)
        return 2

    def _step_bdf2_linearized(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        jacobian: JacobianFunction,
    ) -> int:
        """BDF2 with linearized residual (uniform dt, non-adaptive).

        Returns:
            Method order (2).
        """
        if step.y_prev is None:
            err_out = self._require_err_out(step)
            self._implicit_euler_linearized_once(
                rhs_func,
                t=step.t,
                y=step.y,
                dt=step.dt,
                jacobian=jacobian,
                out=step.out,
            )
            err_out.fill(0.0)
            return 1

        err_out = self._require_err_out(step)

        jac = self._compute_linearized_residual(rhs_func, jacobian, step.t, step.y)
        jac_op = self._as_scipy_operator(jac)
        n = self._require_op_axis_len()

        left_op: OperatorLike
        right_op: OperatorLike

        if isinstance(jac_op, csr_matrix):
            identity_op = identity(n, format="csr", dtype=jac_op.dtype)
            left_op = (1.5 * identity_op - step.dt * jac_op).tocsr()
            right_op = identity_op
        else:
            jac_arr = np.asarray(jac_op, dtype=self.dtype)
            left_op = 1.5 * np.eye(n, dtype=self.dtype) - step.dt * jac_arr
            right_op = np.eye(n, dtype=self.dtype)

        np.copyto(self._rhs_buffer, step.y)
        self._rhs_buffer *= 2.0
        self._rhs_buffer -= 0.5 * step.y_prev
        self._rhs_buffer += step.dt * self._f_extrap

        self._apply_operator_solve_with_ops(
            self._rhs_buffer,
            predictor=None,
            left_op=left_op,
            right_op=right_op,
            out=step.out,
        )

        # Embedded implicit Euler (order 1) for error estimate
        self._implicit_euler_linearized_once(
            rhs_func,
            t=step.t,
            y=step.y,
            dt=step.dt,
            jacobian=jacobian,
            out=self._y_full,
        )
        np.subtract(step.out, self._y_full, out=err_out)
        return 2

    def _step_rosenbrock_w2(
        self,
        rhs_func: RHSFunction,
        *,
        step: StepIO,
        jacobian: JacobianFunction,
    ) -> int:
        """Rosenbrock-W 2(1) pair (linearly implicit, stiff ODEs).

        Returns:
            Controller order for the embedded error estimate.
        """
        err_out = self._require_err_out(step)
        tableau = ROSENBROCK_W_TABLEAUS["ros2"]
        jac_op = self._as_scipy_operator(jacobian(float(step.t), step.y))
        left_op, right_op = build_implicit_euler_operators(
            jac_op,
            dt_scale=tableau.gamma * step.dt,
        )

        def solve(stage_rhs: Array) -> Array:
            stage = np.empty_like(step.y)
            self._apply_operator_solve_with_ops(
                cast("NDArray[np.floating]", stage_rhs),
                predictor=None,
                left_op=left_op,
                right_op=right_op,
                out=stage,
            )
            return stage

        high, embedded = evaluate_rosenbrock_w_tableau(
            tableau=tableau,
            t=step.t,
            dt=step.dt,
            y=cast("Array", step.y),
            rhs=lambda stage_time, stage_state: self._rhs_array(
                rhs_func,
                stage_time,
                stage_state,
            ),
            solve=solve,
            weighted_sum=self._weighted_array_state,
        )
        high_numpy = cast("NDArray[np.floating]", high)
        embedded_numpy = cast("NDArray[np.floating]", embedded)
        np.copyto(step.out, high_numpy)
        np.subtract(high_numpy, embedded_numpy, out=err_out)
        return tableau.controller_order

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    _LINEARIZED_STEP_FUNCTIONS: ClassVar[dict[MethodName, _LinearizedStepFunction]] = {
        "implicit-euler": _step_implicit_euler_linearized,
        "trapezoidal": _trapezoidal_linearized_step,
        "bdf2": _step_bdf2_linearized,
        "ros2": _step_rosenbrock_w2,
    }

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

        Returns:
            Method order.

        Raises:
            RuntimeError: If an unknown method is encountered or internal errors occur.
        """
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
        linearized_step = self._LINEARIZED_STEP_FUNCTIONS.get(plan.method)
        if linearized_step is not None:
            if plan.jacobian is None:
                raise RuntimeError(_INTERNAL_ERROR_OP_AXIS_MSG)
            return linearized_step(
                self,
                rhs_func,
                step=step,
                jacobian=plan.jacobian,
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

    def _advance_explicit_adaptive_to_time(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        method: MethodName,
        t0: float,
        t1: float,
        y0: Array,
        adaptive_cfg: AdaptiveConfig,
        dt_ctrl: DtControllerConfig,
        accepted_steps: list[float],
        first_stage: Array | None = None,
    ) -> tuple[Array, Array | None]:
        """Advance an explicit method with functional adaptive substeps.

        Returns:
            State at ``t1`` and an optional FSAL stage for the next interval.

        Raises:
            RuntimeError: If step rejection, minimum-dt, or step-count limits
                are exceeded.
        """
        t = float(t0)
        target = float(t1)
        dt_out = target - t
        dt = (
            float(adaptive_cfg.dt_init)
            if (
                adaptive_cfg.dt_init is not None
                and np.isfinite(adaptive_cfg.dt_init)
                and adaptive_cfg.dt_init > 0.0
            )
            else dt_out
        )
        dt = min(dt, dt_ctrl.dt_max)
        if dt <= 0.0:
            dt = dt_out

        y_current = y0
        cached_first_stage = first_stage
        n_internal = 0
        while t < target:
            if n_internal >= adaptive_cfg.max_steps:
                raise RuntimeError(_MAX_STEPS_ERROR_MSG)

            remaining = target - t
            if remaining <= 0.0:
                break
            dt = min(dt, remaining)

            rejects = 0
            while True:
                if rejects >= adaptive_cfg.max_reject:
                    raise RuntimeError(_TOO_MANY_REJECTS_ERROR_MSG)

                result = self._attempt_explicit_step(
                    rhs_func,
                    method=method,
                    t=t,
                    dt=dt,
                    y=y_current,
                    first_stage=cached_first_stage,
                )
                error_norm = self._error_norm(
                    result.error,
                    result.state,
                    y_current,
                    rtol=adaptive_cfg.rtol,
                    atol=adaptive_cfg.atol,
                )

                if error_norm <= 1.0:
                    accepted_steps.append(float(dt))
                    t += dt
                    y_current = result.state
                    cached_first_stage = result.last_stage
                    dt = self._propose_dt(
                        dt,
                        error_norm,
                        result.controller_order,
                        cfg=dt_ctrl,
                    )
                    break

                cached_first_stage = result.first_stage
                dt_new = self._propose_dt(
                    dt,
                    error_norm,
                    result.controller_order,
                    cfg=dt_ctrl,
                )
                if dt_new <= dt_ctrl.dt_min and dt_ctrl.dt_min > 0.0:
                    raise RuntimeError(_DT_UNDERFLOW_ERROR_MSG)
                dt = dt_new
                rejects += 1

            n_internal += 1

        return y_current, cached_first_stage

    def _run_explicit(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        config: RunConfig,
    ) -> None:
        """Run an explicit method without mutable NumPy scratch buffers.

        Raises:
            ValueError: If an output interval is not strictly increasing.
        """
        time_grid = np.asarray(self.core.time_grid, dtype=float)
        n_steps = int(self.core.n_timesteps)
        schedule_steps: list[tuple[float, ...]] = []
        first_stage: Array | None = None

        for idx in range(n_steps - 1):
            t0 = float(time_grid[idx])
            t1 = float(time_grid[idx + 1])
            if t1 <= t0:
                raise ValueError(_TIME_GRID_INCREASING_ERROR_MSG)

            y_current = self.core.get_current_state()
            if config.adaptive:
                accepted_steps: list[float] = []
                y_next, first_stage = self._advance_explicit_adaptive_to_time(
                    rhs_func,
                    method=plan.method,
                    t0=t0,
                    t1=t1,
                    y0=y_current,
                    adaptive_cfg=config.adaptive_cfg,
                    dt_ctrl=config.dt_controller,
                    accepted_steps=accepted_steps,
                    first_stage=first_stage,
                )
                schedule_steps.append(tuple(accepted_steps))
            else:
                y_next, first_stage = self._step_explicit_fixed(
                    rhs_func,
                    method=plan.method,
                    t=t0,
                    dt=t1 - t0,
                    y=y_current,
                    first_stage=first_stage,
                )
            self.core.advance_timestep(y_next)

        if config.adaptive:
            self._last_adaptive_schedule = AdaptiveStepSchedule(
                output_times=tuple(float(value) for value in time_grid),
                step_sizes=tuple(schedule_steps),
            )

    def _advance_array_implicit_adaptive_to_time(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        t0: float,
        t1: float,
        y0: Array,
        adaptive_cfg: AdaptiveConfig,
        dt_ctrl: DtControllerConfig,
        accepted_steps: list[float],
    ) -> Array:
        """Advance a non-NumPy implicit method with adaptive substeps.

        Returns:
            State at ``t1`` in the input namespace.

        Raises:
            RuntimeError: If rejection, minimum-dt, or step limits are exceeded.
        """
        t = float(t0)
        target = float(t1)
        dt_out = target - t
        dt = (
            float(adaptive_cfg.dt_init)
            if (
                adaptive_cfg.dt_init is not None
                and np.isfinite(adaptive_cfg.dt_init)
                and adaptive_cfg.dt_init > 0.0
            )
            else dt_out
        )
        dt = min(dt, dt_ctrl.dt_max)
        if dt <= 0.0:
            dt = dt_out

        y_current = y0
        n_internal = 0
        while t < target:
            if n_internal >= adaptive_cfg.max_steps:
                raise RuntimeError(_MAX_STEPS_ERROR_MSG)

            remaining = target - t
            if remaining <= 0.0:
                break
            dt = min(dt, remaining)

            rejects = 0
            while True:
                if rejects >= adaptive_cfg.max_reject:
                    raise RuntimeError(_TOO_MANY_REJECTS_ERROR_MSG)

                y_try, error, order = self._attempt_array_implicit_step(
                    rhs_func,
                    plan=plan,
                    t=t,
                    dt=dt,
                    y=y_current,
                )
                error_norm = self._error_norm(
                    error,
                    y_try,
                    y_current,
                    rtol=adaptive_cfg.rtol,
                    atol=adaptive_cfg.atol,
                )
                if error_norm <= 1.0:
                    accepted_steps.append(float(dt))
                    t += dt
                    y_current = y_try
                    dt = self._propose_dt(dt, error_norm, order, cfg=dt_ctrl)
                    break

                dt_new = self._propose_dt(dt, error_norm, order, cfg=dt_ctrl)
                if dt_new <= dt_ctrl.dt_min and dt_ctrl.dt_min > 0.0:
                    raise RuntimeError(_DT_UNDERFLOW_ERROR_MSG)
                dt = dt_new
                rejects += 1

            n_internal += 1

        return y_current

    def _run_array_implicit(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        config: RunConfig,
    ) -> None:
        """Run an implicit/IMEX plan in a non-NumPy Array-API namespace.

        Raises:
            ValueError: If an output interval is not strictly increasing.
        """
        time_grid = np.asarray(self.core.time_grid, dtype=float)
        previous_state: Array | None = None
        schedule_steps: list[tuple[float, ...]] = []

        for index in range(int(self.core.n_timesteps) - 1):
            t0 = float(time_grid[index])
            t1 = float(time_grid[index + 1])
            if t1 <= t0:
                raise ValueError(_TIME_GRID_INCREASING_ERROR_MSG)

            current_state = self.core.get_current_state()
            if config.adaptive:
                accepted_steps: list[float] = []
                next_state = self._advance_array_implicit_adaptive_to_time(
                    rhs_func,
                    plan=plan,
                    t0=t0,
                    t1=t1,
                    y0=current_state,
                    adaptive_cfg=config.adaptive_cfg,
                    dt_ctrl=config.dt_controller,
                    accepted_steps=accepted_steps,
                )
                schedule_steps.append(tuple(accepted_steps))
            else:
                next_state, _error, _order = self._attempt_array_implicit_step(
                    rhs_func,
                    plan=plan,
                    t=t0,
                    dt=t1 - t0,
                    y=current_state,
                    y_prev=previous_state,
                )

            self.core.advance_timestep(next_state)
            previous_state = current_state

        if config.adaptive:
            self._last_adaptive_schedule = AdaptiveStepSchedule(
                output_times=tuple(float(value) for value in time_grid),
                step_sizes=tuple(schedule_steps),
            )

    def _advance_nonadaptive_to_time(  # noqa: PLR0913
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        t0: float,
        dt_out: float,
        y0: NDArray[np.floating],
        y_prev: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """
        Advance exactly one step to the next output time.

        Args:
            rhs_func: RHS function.
            plan: Run plan.
            t0: Initial time.
            dt_out: Output time step.
            y0: Initial state.
            y_prev: Previous state (required for multistep methods).

        Returns:
            State at t0 + dt_out.
        """
        step = StepIO(
            t=t0,
            dt=dt_out,
            y=y0,
            out=self._y_try,
            err_out=self._err,
            y_prev=y_prev,
        )
        _ = self._attempt_step(rhs_func, plan=plan, step=step)
        return self._y_try

    def _advance_adaptive_to_time(
        self,
        rhs_func: RHSFunction,
        params: AdaptiveAdvanceParams,
        accepted_steps: list[float],
    ) -> NDArray[np.floating]:
        """Advance with adaptive substepping to land exactly on params.t1.

        Args:
            rhs_func: RHS function.
            params: Adaptive advance parameters.
            accepted_steps: Destination for accepted step sizes.

        Returns:
            State at params.t1.

        Raises:
            RuntimeError: If step rejection limits or dt bounds are violated.
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
                    accepted_steps.append(float(dt))
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

    def _validate_schedule_time_grid(self, schedule: AdaptiveStepSchedule) -> None:
        """Require a replay schedule to describe this core's output grid.

        Raises:
            ValueError: If the schedule and core output grids differ.
        """
        time_grid = np.asarray(self.core.time_grid, dtype=float)
        if len(schedule.output_times) != len(time_grid) or not np.allclose(
            schedule.output_times,
            time_grid,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(_SCHEDULE_TIME_GRID_ERROR_MSG)

    def _replay_explicit_schedule(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        schedule: AdaptiveStepSchedule,
    ) -> None:
        """Replay accepted explicit steps with functional Array-API kernels."""
        first_stage: Array | None = None
        for t0, interval_steps in zip(
            schedule.output_times[:-1],
            schedule.step_sizes,
            strict=True,
        ):
            t = t0
            state = self.core.get_current_state()
            for dt in interval_steps:
                result = self._attempt_explicit_step(
                    rhs_func,
                    method=plan.method,
                    t=t,
                    dt=dt,
                    y=state,
                    first_stage=first_stage,
                )
                state = result.state
                first_stage = result.last_stage
                t += dt
            self.core.advance_timestep(state)

    def _replay_array_implicit_schedule(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        schedule: AdaptiveStepSchedule,
    ) -> None:
        """Replay accepted implicit steps with functional Array-API kernels."""
        for t0, interval_steps in zip(
            schedule.output_times[:-1],
            schedule.step_sizes,
            strict=True,
        ):
            t = t0
            state = self.core.get_current_state()
            for dt in interval_steps:
                state, _error, _order = self._attempt_array_implicit_step(
                    rhs_func,
                    plan=plan,
                    t=t,
                    dt=dt,
                    y=state,
                )
                t += dt
            self.core.advance_timestep(state)

    def _replay_numpy_implicit_schedule(
        self,
        rhs_func: RHSFunction,
        *,
        plan: RunPlan,
        schedule: AdaptiveStepSchedule,
    ) -> None:
        """Replay accepted implicit steps with NumPy/SciPy scratch buffers.

        Raises:
            TypeError: If the state changes array ecosystems during replay.
        """
        for t0, interval_steps in zip(
            schedule.output_times[:-1],
            schedule.step_sizes,
            strict=True,
        ):
            state = self.core.get_current_state()
            if not isinstance(state, np.ndarray):
                raise TypeError(
                    _ARRAY_API_ERROR_MSG.format(type_name=type(state).__name__)
                )
            np.copyto(self._y_curr, state)
            t = t0
            for dt in interval_steps:
                step = StepIO(
                    t=t,
                    dt=dt,
                    y=self._y_curr,
                    out=self._y_try,
                    err_out=self._err,
                )
                self._attempt_step(rhs_func, plan=plan, step=step)
                self._y_curr, self._y_try = self._y_try, self._y_curr
                t += dt
            self.core.advance_timestep(self._y_curr)

    def replay_adaptive_schedule(
        self,
        rhs_func: RHSFunction,
        schedule: AdaptiveStepSchedule,
        *,
        config: RunConfig,
    ) -> None:
        """Replay a recorded adaptive mesh through the configured method.

        Replay bypasses error norms and accept/reject decisions while invoking
        the same high-order step kernels used by a live adaptive solve. With a
        JAX state, the static Python schedule can therefore be traced by
        ``jax.jit`` and differentiated with respect to array-valued model inputs.

        Args:
            rhs_func: Function computing the explicit RHS F(t, y).
            schedule: Accepted step mesh recorded by an adaptive run.
            config: Matching adaptive run configuration.

        Raises:
            TypeError: If schedule has the wrong type or the state changes array
                ecosystems during replay.
            ValueError: If config is not adaptive or the output grid differs.
        """
        if not isinstance(schedule, AdaptiveStepSchedule):
            msg = "schedule must be an AdaptiveStepSchedule"
            raise TypeError(msg)
        if not config.adaptive:
            msg = "Schedule replay requires config.adaptive=True"
            raise ValueError(msg)

        self._last_adaptive_schedule = None
        self._validate_schedule_time_grid(schedule)
        plan = self._resolve_run_plan(config)

        if plan.method in _EXPLICIT_METHODS:
            self._replay_explicit_schedule(rhs_func, plan=plan, schedule=schedule)
        else:
            current_state = self.core.get_current_state()
            if isinstance(current_state, np.ndarray):
                self._replay_numpy_implicit_schedule(
                    rhs_func,
                    plan=plan,
                    schedule=schedule,
                )
            else:
                self._replay_array_implicit_schedule(
                    rhs_func,
                    plan=plan,
                    schedule=schedule,
                )

        self._last_adaptive_schedule = schedule

    def run(self, rhs_func: RHSFunction, *, config: RunConfig | None = None) -> None:
        """Advance the ModelCore state through its time grid.

        Args:
            rhs_func: Function computing the explicit RHS F(t, y).
            config: Optional run configuration. If None, defaults are used.

        Raises:
            TypeError: If the state changes array ecosystems during a run.
            ValueError: If invalid parameters are provided.
        """
        self._last_adaptive_schedule = None
        cfg = config or RunConfig()
        plan = self._resolve_run_plan(cfg)

        if plan.method in _EXPLICIT_METHODS:
            self._run_explicit(rhs_func, plan=plan, config=cfg)
            return

        current_state = self.core.get_current_state()
        if not isinstance(current_state, np.ndarray):
            self._run_array_implicit(rhs_func, plan=plan, config=cfg)
            return

        time_grid = np.asarray(self.core.time_grid, dtype=float)
        n_steps = int(self.core.n_timesteps)
        schedule_steps: list[tuple[float, ...]] = []

        for idx in range(n_steps - 1):
            t0 = float(time_grid[idx])
            t1 = float(time_grid[idx + 1])
            if t1 <= t0:
                raise ValueError(_TIME_GRID_INCREASING_ERROR_MSG)
            dt_out = t1 - t0

            state = self.core.get_current_state()
            if not isinstance(state, np.ndarray):
                raise TypeError(
                    _ARRAY_API_ERROR_MSG.format(type_name=type(state).__name__)
                )
            np.copyto(self._y_curr, state)

            y_prev: NDArray[np.floating] | None = (
                self._prev_state_cache if self._has_prev_state else None
            )

            if not cfg.adaptive:
                y_next = self._advance_nonadaptive_to_time(
                    rhs_func,
                    plan=plan,
                    t0=t0,
                    dt_out=dt_out,
                    y0=self._y_curr,
                    y_prev=y_prev,
                )
                self.core.advance_timestep(y_next)
                self._has_prev_state = True
                np.copyto(self._prev_state_cache, self._y_curr)
                continue

            accepted_steps: list[float] = []
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
                accepted_steps,
            )
            schedule_steps.append(tuple(accepted_steps))
            self.core.advance_timestep(y_end)
            self._has_prev_state = True
            np.copyto(self._prev_state_cache, self._y_curr)

        if cfg.adaptive:
            self._last_adaptive_schedule = AdaptiveStepSchedule(
                output_times=tuple(float(value) for value in time_grid),
                step_sizes=tuple(schedule_steps),
            )
