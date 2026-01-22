# src/op_engine/flepimop2/errors.py
"""Error types and dependency-guard utilities for op_engine.flepimop2.

This module centralizes:
- explicit error classes with actionable messages, and
- small helpers to guard optional flepimop2 imports.

Design intent:
- op_engine can be installed without flepimop2
- op_engine.flepimop2 fails fast with a clear message if used without the extra
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Final

_FLEPIMOP2_EXTRA_INSTALL_MSG: Final[str] = (
    "Install the optional dependency group with:\n"
    "  pip install 'op_engine[flepimop2]'\n"
    "or, if you are using uv:\n"
    "  uv pip install '.[flepimop2]'"
)


class OpEngineFlepimop2Error(Exception):
    """Base exception for op_engine.flepimop2 integration errors."""


class OptionalDependencyMissingError(OpEngineFlepimop2Error, ImportError):
    """Raised when an optional dependency is required but missing."""


class EngineConfigError(OpEngineFlepimop2Error, ValueError):
    """Raised when a flepimop2 engine config is invalid or incomplete."""


class UnsupportedMethodError(OpEngineFlepimop2Error, ValueError):
    """Raised when an op_engine method cannot run under current engine config."""


class StateShapeError(OpEngineFlepimop2Error, ValueError):
    """Raised when state/time array shapes are incompatible with the engine adapter."""


class ParameterError(OpEngineFlepimop2Error, TypeError):
    """Raised when parameters passed from flepimop2 are invalid for op_engine."""


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Structured description of optional dependency availability."""

    package: str
    is_available: bool
    detail: str | None = None


def check_flepimop2_available() -> DependencyStatus:
    """Check whether flepimop2 is importable.

    Returns:
        DependencyStatus describing flepimop2 availability.
    """
    spec = find_spec("flepimop2")
    if spec is None:
        return DependencyStatus(
            package="flepimop2",
            is_available=False,
            detail="Module spec not found",
        )
    return DependencyStatus(package="flepimop2", is_available=True, detail=None)


def require_flepimop2() -> None:
    """Require that flepimop2 is importable.

    Raises:
        OptionalDependencyMissingError: If flepimop2 cannot be imported.
    """
    status = check_flepimop2_available()
    if status.is_available:
        return

    msg = (
        "The op_engine.flepimop2 integration requires flepimop2, but it is not "
        "available in this environment.\n\n"
        f"Import detail: {status.detail}\n\n"
        f"{_FLEPIMOP2_EXTRA_INSTALL_MSG}"
    )
    raise OptionalDependencyMissingError(msg)


def raise_unsupported_imex(method: str, *, reason: str) -> None:
    """Raise a standardized UnsupportedMethodError for IMEX configuration issues.

    Args:
        method: Requested method name (for example, "imex-euler").
        reason: Human-readable reason the method cannot run.

    Raises:
        UnsupportedMethodError: Always.
    """
    msg = (
        f"Method '{method}' is not supported under the current flepimop2 engine "
        "configuration.\n"
        f"Reason: {reason}\n\n"
        "IMEX methods require operator specifications that provide an implicit "
        "linear operator A (or factories for dt-dependent operators)."
    )
    raise UnsupportedMethodError(msg)


def raise_invalid_engine_config(
    *,
    missing: list[str] | None = None,
    detail: str | None = None,
) -> None:
    """Raise a standardized EngineConfigError.

    Args:
        missing: Required config keys that are missing.
        detail: Optional additional context.

    Raises:
        EngineConfigError: Always.
    """
    parts: list[str] = ["Invalid op_engine.flepimop2 engine configuration."]
    if missing:
        parts.append(f"Missing required field(s): {sorted(set(missing))}.")
    if detail:
        parts.append(f"Detail: {detail}")
    raise EngineConfigError(" ".join(parts))


def raise_state_shape_error(*, name: str, expected: str, got: object) -> None:
    """Raise a standardized StateShapeError.

    Args:
        name: Name of the object with the shape issue.
        expected: Human-readable expected shape description.
        got: Actual observed shape/value.

    Raises:
        StateShapeError: Always.
    """
    msg = f"{name} has an invalid shape/value. Expected {expected}. Got: {got!r}."
    raise StateShapeError(msg)
