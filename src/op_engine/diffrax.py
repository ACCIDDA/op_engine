"""Optional native CompiledRhs-to-Diffrax terms for smooth deterministic ODEs.

Term ownership belongs to the caller. This module creates no global solve
cache, changes no JAX settings and imports neither optional package until the
factory is called. Solver, save policy and adjoint selection belong to Diffrax.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from ._typing import Array


class DiffraxCompiledRhs(Protocol):
    """Current compiler surface, without a compiler runtime dependency."""

    @property
    def state_names(self) -> tuple[str, ...]:
        """Give the compiler's expanded flat state ordering."""
        ...

    @property
    def param_names(self) -> tuple[str, ...]:
        """Give scalar parameters; shaped parameters are declared in metadata."""
        ...

    @property
    def eval_fn(self) -> Callable[..., object]:
        """Evaluate the flat RHS from native state and named parameters."""
        ...

    @property
    def pytree_eval_fn(self) -> Callable[..., object] | None:
        """Evaluate native StateDict leaves without flattening, when available."""
        ...

    @property
    def template_shapes(self) -> Mapping[str, tuple[int, ...]] | None:
        """Give complete shapes for the native StateDict templates."""
        ...

    @property
    def meta(self) -> Mapping[str, Any]:
        """Expose parameter axes, time tables and other numerical requirements."""
        ...

    @property
    def operators(self) -> Sequence[object]:
        """Declare operator contributions that this scoped adapter rejects."""
        ...

    @property
    def history_requirements(self) -> Sequence[object]:
        """Declare nonlocal-in-time contributions that need another driver."""
        ...


@dataclass(frozen=True, slots=True)
class _Contract:
    layout: Literal["flat", "pytree"]
    state_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    evaluate: Callable[..., object]
    n_states: int


def _contract(compiled: DiffraxCompiledRhs, layout: str) -> _Contract:
    """Snapshot supported public metadata before building a numerical term.

    Returns:
        An immutable layout and evaluator, with no captured numerical inputs.

    Raises:
        ValueError: The requested layout or compiler shape metadata is invalid.
        NotImplementedError: Contributions need an unsupported numerical path.
    """
    if layout not in {"flat", "pytree"}:
        msg = "Diffrax state_layout must be 'flat' or 'pytree'"
        raise ValueError(msg)
    meta = compiled.meta
    if any((
        compiled.operators,
        meta.get("operators"),
        compiled.history_requirements,
        meta.get("op_system_may_have_history"),
        meta.get("time_varying_params"),
        meta.get("forcing_breakpoints"),
    )):
        msg = (
            "Diffrax terms require a smooth ODE without operators/history/table forcing"
        )
        raise NotImplementedError(msg)
    n_states = len(compiled.state_names)
    shapes = tuple(
        (name, tuple(shape)) for name, shape in (compiled.template_shapes or {}).items()
    )
    if not n_states or (
        shapes and sum(math.prod(shape) for _, shape in shapes) != n_states
    ):
        msg = "Diffrax compiler state layout is empty or inconsistent"
        raise ValueError(msg)
    evaluate = compiled.eval_fn
    if layout == "pytree":
        if not shapes or compiled.pytree_eval_fn is None:
            msg = (
                "Diffrax pytree layout requires the compiler's native PyTree evaluator"
            )
            raise NotImplementedError(msg)
        evaluate = compiled.pytree_eval_fn
    axis_sizes = {axis["name"]: axis["size"] for axis in meta.get("axes", ())}
    parameters: dict[str, tuple[int, ...]] = dict.fromkeys(compiled.param_names, ())
    parameters.update({
        name: tuple(axis_sizes[axis] for axis in axes)
        for name, axes in meta.get("shaped_params", ())
    })
    return _Contract(
        cast("Literal['flat', 'pytree']", layout),
        shapes,
        tuple(sorted(parameters.items())),
        evaluate,
        n_states,
    )


def _array(value: object, shape: tuple[int, ...], name: str, jax: ModuleType) -> Array:
    """Check static native-array metadata, including inside a JAX trace.

    Returns:
        The original native array or tracer without a transfer or cast.

    Raises:
        TypeError: A leaf is not a native real floating JAX value.
        ValueError: Its static shape does not match the compiler contract.
    """
    if not isinstance(value, (jax.Array, jax.core.Tracer)):
        msg = f"Diffrax {name} must be a native JAX array"
        raise TypeError(msg)
    array = cast("Array", value)
    if str(array.dtype) not in {"float32", "float64"}:
        msg = f"Diffrax {name} requires float32 or float64"
        raise TypeError(msg)
    if array.shape != shape:
        msg = f"Diffrax {name} shape must be {shape}, got {array.shape}"
        raise ValueError(msg)
    return array


def _state(value: object, contract: _Contract, name: str, jax: ModuleType) -> object:
    """Check a flat array or the compiler's StateDict without materializing it.

    Returns:
        The common native leaf dtype.

    Raises:
        TypeError: State structure or leaf dtypes are incompatible.
        ValueError: Keys or static shapes differ from the compiler layout.
    """
    if contract.layout == "flat":
        return _array(value, (contract.n_states,), name, jax).dtype
    if not isinstance(value, dict):
        msg = f"Diffrax {name} requires a dict of compiler state templates"
        raise TypeError(msg)
    if value.keys() != dict(contract.state_shapes).keys():
        msg = f"Diffrax {name} keys must match compiler state templates"
        raise ValueError(msg)
    dtypes = {
        _array(value[key], shape, f"{name}[{key}]", jax).dtype
        for key, shape in contract.state_shapes
    }
    if len(dtypes) != 1:
        msg = f"Diffrax {name} leaves require a common dtype"
        raise TypeError(msg)
    return next(iter(dtypes))


def compiled_diffrax_term(
    compiled: DiffraxCompiledRhs,
    *,
    state_layout: Literal["flat", "pytree"] = "pytree",
) -> Any:  # noqa: ANN401 - optional third-party ODETerm is the adapter boundary
    """Build one caller-owned Diffrax ODETerm from a current compiled RHS.

    Pass a plain dict of named native JAX parameter arrays through Diffrax
    ``args``. State and parameters have a common float32/float64 dtype and the
    compiler's exact shapes. PyTree state is a dict of base-template arrays;
    flat state follows ``compiled.state_names``. No values are captured as
    static parameters, coerced through NumPy or placed in a global cache.

    The returned object is a third-party ``diffrax.ODETerm``. Its type remains
    dynamic here so importing/checking the numerical core needs no optional
    Diffrax installation. The caller owns the solve, output policy, adjoint,
    result/failure handling and smoothness assumptions. CPU operation and
    individual derivative transforms are qualified separately.

    Returns:
        A native Diffrax ODETerm; construct it once for repeated parameter calls.

    Raises:
        ImportError: The optional Diffrax/JAX installation is absent.
        ValueError: Layout metadata is inconsistent.
        NotImplementedError: The RHS needs unsupported operators/history/forcing.
    """  # noqa: DOC502 - layout errors originate in the shared metadata validator
    contract = _contract(compiled, state_layout)
    try:
        diffrax = importlib.import_module("diffrax")
        jax = importlib.import_module("jax")
    except ImportError as error:
        msg = "Install op_engine[diffrax] to build a compiled Diffrax term"
        raise ImportError(msg) from error

    def evaluate(t: object, y: object, args: object) -> object:
        dtype = _state(y, contract, "state", jax)
        if not isinstance(args, dict):
            msg = "Diffrax args must be a dict of named native parameter arrays"
            raise TypeError(msg)
        if args.keys() != dict(contract.parameter_shapes).keys():
            msg = "Diffrax args keys must match compiler parameter names"
            raise ValueError(msg)
        for name, shape in contract.parameter_shapes:
            if _array(args[name], shape, f"args[{name}]", jax).dtype != dtype:
                msg = "Diffrax parameter and state dtypes must match"
                raise TypeError(msg)
        result = contract.evaluate(t, y, **args)
        if _state(result, contract, "RHS output", jax) != dtype:
            msg = "Diffrax RHS output dtype must match the state"
            raise TypeError(msg)
        return result

    return diffrax.ODETerm(evaluate)


__all__ = ["DiffraxCompiledRhs", "compiled_diffrax_term"]
