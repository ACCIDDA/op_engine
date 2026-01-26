"""Error types and dependency-guard utilities for op_engine.flepimop2.

Design intent:
- op_engine can be installed without flepimop2
- op_engine.flepimop2 fails fast with clear, actionable messages if used without
  the optional extra dependencies.
"""

from __future__ import annotations

from enum import StrEnum
from importlib.util import find_spec
from typing import Final

_FLEPIMOP2_EXTRA_INSTALL_MSG: Final[str] = (
    "Install the optional dependency group with:\n"
    "  pip install 'op_engine[flepimop2]'\n"
    "or, if you are using uv:\n"
    "  uv pip install '.[flepimop2]'"
)


class ErrorCode(StrEnum):
    """Machine-readable classification for op_engine.flepimop2 failures.

    Use these codes to support consistent logging and (optional) programmatic
    recovery without requiring many custom exception subclasses.
    """

    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    INVALID_ENGINE_CONFIG = "invalid_engine_config"
    UNSUPPORTED_METHOD = "unsupported_method"
    INVALID_STATE_SHAPE = "invalid_state_shape"
    INVALID_PARAMETERS = "invalid_parameters"


class OpEngineFlepimop2Error(Exception):
    """Base exception for op_engine.flepimop2 integration errors.

    This exists so callers can catch integration-layer failures explicitly
    without depending on a wide taxonomy of custom subclasses.
    """

    def __init__(self, message: str, *, code: ErrorCode | None = None) -> None:
        """
        Initialize an OpEngineFlepimop2Error.

        Args:
            message: Human-readable error message.
            code: Optional machine-readable error code classifying the error.
        """
        super().__init__(message)
        self.code: ErrorCode | None = code


class OptionalDependencyMissingError(OpEngineFlepimop2Error, ImportError):
    """Raised when an optional dependency is required but missing."""


class EngineConfigError(OpEngineFlepimop2Error, ValueError):
    """Raised when a flepimop2 engine config is invalid or incomplete."""


def require_flepimop2() -> None:
    """Require that flepimop2 is importable.

    Raises:
        OptionalDependencyMissingError: If flepimop2 cannot be imported.
    """
    if find_spec("flepimop2") is not None:
        return

    msg = (
        "The op_engine.flepimop2 integration requires flepimop2, but it is not "
        "available in this environment.\n\n"
        "Import detail: Module spec not found\n\n"
        f"{_FLEPIMOP2_EXTRA_INSTALL_MSG}"
    )

    raise OptionalDependencyMissingError(
        msg, code=ErrorCode.OPTIONAL_DEPENDENCY_MISSING
    )


def raise_unsupported_imex(method: str, *, reason: str) -> None:
    """
    Raise a standardized error for IMEX configuration issues.

    Args:
        method: Name of the IMEX method.
        reason: Explanation of why the method is unsupported.

    Raises:
        ValueError: Always.
    """
    msg = (
        f"Method '{method}' is not supported under the current flepimop2 engine "
        "configuration.\n"
        f"Reason: {reason}\n\n"
        "IMEX methods require operator specifications that provide an implicit "
        "linear operator A (or factories for dt-dependent operators)."
    )
    raise ValueError(msg) from OpEngineFlepimop2Error(
        msg, code=ErrorCode.UNSUPPORTED_METHOD
    )


def raise_invalid_engine_config(
    *,
    missing: list[str] | None = None,
    detail: str | None = None,
) -> None:
    """
    Raise a standardized engine configuration error.

    Args:
        missing: List of missing required fields, if any.
        detail: Additional detail about the configuration issue.

    Raises:
        EngineConfigError: Always.
    """
    parts: list[str] = ["Invalid op_engine.flepimop2 engine configuration."]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    msg = " ".join(parts)

    raise EngineConfigError(msg, code=ErrorCode.INVALID_ENGINE_CONFIG)


def raise_state_shape_error(*, name: str, expected: str, got: object) -> None:
    """
    Raise a standardized state/time array shape error.

    Args:
        name: Name of the array (for error messages).
        expected: Description of the expected shape/value.
        got: Actual value received.

    Raises:
        ValueError: Always.
    """
    msg = f"{name} has an invalid shape/value. Expected {expected}. Got: {got!r}."
    raise ValueError(msg) from OpEngineFlepimop2Error(
        msg, code=ErrorCode.INVALID_STATE_SHAPE
    )


def raise_parameter_error(*, detail: str) -> None:
    """
    Raise a standardized parameter/type error.

    Args:
        detail: Detail text describing the parameter issue.

    Raises:
        TypeError: Always.
    """
    msg = f"Invalid parameters for op_engine.flepimop2 adapter: {detail}"
    raise TypeError(msg) from OpEngineFlepimop2Error(
        msg, code=ErrorCode.INVALID_PARAMETERS
    )
