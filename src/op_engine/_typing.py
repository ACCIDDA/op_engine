"""Structural typing shared by op_engine's array-polymorphic paths."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class Array(Protocol):
    """Minimal Array-API duck type used at public numerical boundaries.

    The contract intentionally matches :class:`op_system.Array` and
    :class:`flepimop2.typing.Array`. It describes arrays with the standard
    ``__array_namespace__`` hook so those package boundaries remain statically
    compatible. Runtime namespace discovery additionally accepts native arrays
    such as ``torch.Tensor`` through compatibility namespaces. No backend
    module is stored in solver configuration.
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


Scalar: TypeAlias = float | Array
"""A Python float or a backend-native rank-zero numerical array."""


__all__ = ["Array", "Scalar"]
