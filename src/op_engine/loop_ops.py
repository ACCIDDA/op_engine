"""Optional namespace adapters for iteration and rematerialization."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from collections.abc import Callable

LoopBody: TypeAlias = "Callable[[Any, Any], tuple[Any, Any]]"
ScanFunction: TypeAlias = "Callable[[LoopBody, Any, Any], tuple[Any, Any]]"


@dataclass(frozen=True, slots=True)
class LoopAdapter:
    """Control-flow operations for one array namespace.

    Numerical stages remain Array-API operations. This bundle dispatches only
    their iteration strategy; namespaces without an adapter keep eager loops.
    The scan must preserve carry structure and stack the body outputs.

    Attributes:
        scan: Function with the ``scan(body, initial, inputs)`` contract.
        checkpoint: Optional transformation that recomputes body intermediates
            during reverse-mode differentiation.
    """

    scan: ScanFunction
    checkpoint: Callable[[LoopBody], LoopBody] | None = None


_LOOP_ADAPTERS: dict[str, LoopAdapter] = {}


def register_loop_adapter(namespace: str, adapter: LoopAdapter) -> None:
    """Register or replace a driver for a namespace's fully qualified name.

    Args:
        namespace: Namespace module name, such as ``"jax.numpy"``.
        adapter: Backend's scan and optional checkpoint operations.

    Raises:
        TypeError: If the adapter has the wrong type.
        ValueError: If the namespace name is empty.
    """
    if not namespace:
        msg = "Loop adapter namespace must not be empty"
        raise ValueError(msg)
    if not isinstance(adapter, LoopAdapter):
        msg = "adapter must be a LoopAdapter"
        raise TypeError(msg)
    _LOOP_ADAPTERS[namespace] = adapter


def get_loop_adapter(namespace: object) -> LoopAdapter | None:
    """Find a driver, registering JAX lazily when its namespace is supplied.

    Ordinary NumPy, Torch, and CuPy calls do not import JAX. Optional backend
    imports stay here rather than inside numerical step kernels.

    Returns:
        Registered driver, or ``None`` to retain the existing eager loop.
    """
    name = getattr(namespace, "__name__", "")
    adapter = _LOOP_ADAPTERS.get(name)
    if adapter is None and name == "jax.numpy":
        try:
            jax = import_module("jax")
        except ImportError:
            return None
        adapter = LoopAdapter(
            scan=cast("ScanFunction", jax.lax.scan),
            checkpoint=cast("Callable[[LoopBody], LoopBody]", jax.checkpoint),
        )
        register_loop_adapter(name, adapter)
    return adapter


__all__ = ["LoopAdapter", "get_loop_adapter", "register_loop_adapter"]
