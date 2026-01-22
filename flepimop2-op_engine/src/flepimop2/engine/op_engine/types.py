"""Type definitions for the op_engine.flepimop2 integration layer.

This module intentionally avoids runtime imports from flepimop2 so op_engine
can be installed without the optional extra.

It mirrors flepimop2's public interfaces in a dependency-free way and defines
adapter-specific configuration types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypedDict, runtime_checkable

import numpy as np
from flepimop2.configuration import IdentifierString  # noqa: TC002
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Core numeric aliases
# -----------------------------------------------------------------------------

Float64Array: TypeAlias = NDArray[np.float64]

# Shape-intent aliases (NumPy typing does not encode shapes; these are semantic).
Float64Array2D: TypeAlias = NDArray[np.float64]

# Generic floating tensor used by internal solver interfaces.
FloatArray: TypeAlias = NDArray[np.floating]

_TIME_NOT_1D_MSG: Final[str] = "{name} must be a 1D array"
_TIME_NOT_INCREASING_MSG: Final[str] = "{name} must be strictly increasing"

# -----------------------------------------------------------------------------
# flepimop2-compatible protocol mirrors
# -----------------------------------------------------------------------------


@runtime_checkable
class SystemStepper(Protocol):
    """Protocol for flepimop2-compatible system stepper functions."""

    def __call__(
        self,
        time: np.float64,
        state: Float64Array,
        **params: object,
    ) -> Float64Array:
        """Compute dstate/dt at a given time/state."""
        ...


@runtime_checkable
class EngineRunner(Protocol):
    """Protocol for flepimop2-compatible engine runner functions."""

    def __call__(
        self,
        stepper: SystemStepper,
        times: Float64Array,
        state: Float64Array,
        params: Mapping[IdentifierString, object],
        **engine_kwargs: object,
    ) -> Float64Array2D:
        """Run the stepper over times and return (T, n_state) output."""
        ...


# -----------------------------------------------------------------------------
# Adapter-level configuration typing
# -----------------------------------------------------------------------------

MethodName: TypeAlias = Literal[
    "euler",
    "heun",
    "imex-euler",
    "imex-heun-tr",
    "imex-trbdf2",
]

OperatorMode: TypeAlias = Literal[
    "none",
    "static",
    "time",
    "stage_state",
]


class OperatorSpecDict(TypedDict, total=False):
    """Dictionary form of operator specifications for IMEX methods."""

    default: object
    tr: object
    bdf2: object


class AdapterKwargs(TypedDict, total=False):
    """Keyword arguments accepted by the flepimop2 adapter."""

    method: MethodName
    adaptive: bool
    strict: bool

    # tolerances
    rtol: float
    atol: float | Float64Array
    dt_init: float | None

    # controller
    dt_min: float
    dt_max: float
    safety: float
    fac_min: float
    fac_max: float

    # limits
    max_reject: int
    max_steps: int

    # IMEX
    operators: OperatorSpecDict
    gamma: float | None

    # axis routing
    operator_axis: str | int


@dataclass(frozen=True, slots=True)
class EngineAdapterConfig:
    """Normalized internal adapter configuration."""

    method: MethodName = "heun"
    adaptive: bool = True
    strict: bool = True

    rtol: float = 1e-6
    atol: float | Float64Array = 1e-9
    dt_init: float | None = None

    dt_min: float = 0.0
    dt_max: float = float("inf")
    safety: float = 0.9
    fac_min: float = 0.2
    fac_max: float = 5.0

    max_reject: int = 25
    max_steps: int = 1_000_000

    operators: OperatorSpecDict | None = None
    gamma: float | None = None

    operator_axis: str | int = "state"


# -----------------------------------------------------------------------------
# Adapter helpers
# -----------------------------------------------------------------------------


def as_float64_1d(x: object, *, name: str = "array") -> Float64Array:
    """Convert input to contiguous float64 1D array.

    Args:
        x: Input array-like.
        name: Name used in error messages.

    Returns:
        Contiguous float64 1D array.

    Raises:
        ValueError: If input cannot be represented as a 1D array.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(_TIME_NOT_1D_MSG.format(name=name))
    return np.ascontiguousarray(arr)


def as_float64_1d_times(x: object, *, name: str = "times") -> Float64Array:
    """Convert input to a contiguous float64 1D time array.

    Args:
        x: Input time array-like.
        name: Name used in error messages.

    Returns:
        Contiguous float64 1D time array.
    """
    return as_float64_1d(x, name=name)


def as_float64_state(x: object, *, name: str = "state") -> Float64Array:
    """Convert input to a contiguous float64 1D state array.

    Args:
        x: Input state array-like.
        name: Name used in error messages.

    Returns:
        Contiguous float64 1D state array.
    """
    return as_float64_1d(x, name=name)


def ensure_strictly_increasing_times(
    times: Float64Array, *, name: str = "times"
) -> None:
    """Validate that a time vector is strictly increasing.

    Args:
        times: 1D float64 time array.
        name: Name used in error messages.

    Raises:
        ValueError: If times is not strictly increasing.
    """
    if times.size <= 1:
        return
    dt = np.diff(np.asarray(times, dtype=np.float64))
    if np.any(dt <= 0.0):
        raise ValueError(_TIME_NOT_INCREASING_MSG.format(name=name))


def normalize_params(
    params: Mapping[IdentifierString, object] | None,
) -> dict[IdentifierString, object]:
    """Normalize parameter mapping to a plain dictionary.

    Args:
        params: Input parameter mapping or None.

    Returns:
        Plain dictionary of parameters.
    """
    if params is None:
        return {}
    return dict(params)


# -----------------------------------------------------------------------------
# RHS tensor function alias (op_engine internal)
# -----------------------------------------------------------------------------

RhsTensorFunc: TypeAlias = Callable[[float, FloatArray], FloatArray]
