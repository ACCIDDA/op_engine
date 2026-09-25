"""Structural typing shared by op_engine's array-polymorphic paths."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Array(Protocol):
    """Minimal Array-API duck type used at public numerical boundaries.

    The contract intentionally matches :class:`op_system.Array` and
    :class:`flepimop2.typing.Array`.  NumPy 2 arrays, JAX arrays, and other
    Array-API implementations provide these members.  Numerical operations
    are selected from the value's ``__array_namespace__`` at call time; no
    backend module is stored in solver configuration.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""

    @property
    def dtype(self) -> object:
        """Backend-defined array dtype."""

    def __array_namespace__(  # noqa: PLW3201
        self,
        *,
        api_version: Any = None,  # noqa: ANN401
    ) -> object:
        """Return the Array-API namespace for this value."""

    def item(self) -> Any:  # noqa: ANN401
        """Return a scalar value from a zero-dimensional array."""


__all__ = ["Array"]
