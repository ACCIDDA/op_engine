# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""flepimop2 Engine integration for op_engine (thin, single-file)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from flepimop2.engine.abc import EngineABC
from flepimop2.exceptions import ValidationIssue
from flepimop2.typing import IdentifierString, StateChangeEnum  # noqa: TC002
from pydantic import Field

from op_engine.core_solver import (
    CoreSolver,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

from .config import (
    OpEngineEngineConfig,
    SolverMethod,
    _coerce_operator_specs,
    _has_operator_specs,
)
from .operators import (
    compile_operator_descriptors,
    typed_operator_descriptors,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from flepimop2.parameter.abc import ModelStateSpecification, ParameterValue
    from flepimop2.system.abc import SystemABC, SystemProtocol
    from flepimop2.typing import Array, Float64NDArray
    from numpy.typing import DTypeLike


class _IndexableArray(Protocol):
    """Internal indexing surface kept out of the public Array protocol."""

    def __getitem__(self, key: object) -> Array:
        """Return one array item or slice."""
        ...


_ARRAY_API_ERROR = (
    "op_engine provider inputs must implement __array_namespace__(); got {type_name}."
)


def _namespace_of(value: object) -> Any:  # noqa: ANN401
    """Return the Array-API namespace advertised by ``value``."""
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        raise TypeError(_ARRAY_API_ERROR.format(type_name=type(value).__name__))
    return namespace()


def _array_item(value: Array, key: object) -> Array:
    """Index an Array while keeping the shared public protocol minimal."""
    return cast("_IndexableArray", value)[key]


def _state_reference(  # noqa: PLR0911
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
) -> object | None:
    """Return the first payload that contributes to the initial state."""
    state_names = system.option("state_names", None)
    init_map = system.option("initial_state", None)
    if isinstance(state_names, tuple | list) and isinstance(init_map, Mapping):
        for state_name_obj in state_names:
            state_name = str(state_name_obj)
            if state_name in initial_state:
                return initial_state[state_name].value

            entry = init_map.get(state_name)
            if isinstance(entry, str) and entry in params:
                return params[entry].value
            if isinstance(entry, Mapping):
                shaped_name = entry.get("shaped")
                if isinstance(shaped_name, str) and shaped_name in params:
                    return params[shaped_name].value
            elif getattr(entry, "__array_namespace__", None) is not None:
                return entry

    if initial_state:
        return next(iter(initial_state.values())).value
    if params:
        return next(iter(params.values())).value
    return None


def _numerical_context(
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
) -> tuple[Any, object]:
    """Select the state namespace and floating dtype for a run.

    Returns:
        Array namespace and result dtype selected from incoming payloads.
    """
    reference = _state_reference(system, initial_state, params)
    if reference is None:
        xp = np
        return xp, xp.asarray(0.0).dtype

    reference_array = cast("Array", reference)
    xp = _namespace_of(reference_array)
    default_float_dtype = xp.asarray(0.0).dtype
    dtype = xp.result_type(
        default_float_dtype, cast("DTypeLike", reference_array.dtype)
    )
    return xp, dtype


def _as_float64_1d(x: object, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"{name} must be a 1D array"
        raise ValueError(msg)
    return np.ascontiguousarray(arr)


def _ensure_strictly_increasing(times: np.ndarray, *, name: str) -> None:
    if times.size <= 1:
        return
    if np.any(np.diff(times) <= 0.0):
        msg = f"{name} must be strictly increasing"
        raise ValueError(msg)


def _rhs_from_stepper(
    stepper: SystemProtocol,
    *,
    n_state: int,
) -> Callable[[float, Array], Array]:
    def rhs(time: float, state: Array) -> Array:
        xp = _namespace_of(state)
        state_arr = state
        if state_arr.shape == (n_state,):
            state_arr = cast("Array", xp.reshape(state_arr, (n_state, 1)))
        expected_shape = (n_state, 1)
        if state_arr.shape != expected_shape:
            msg = (
                f"RHS received unexpected state shape {state_arr.shape}; "
                f"expected {expected_shape}."
            )
            raise ValueError(msg)

        flat_state = _array_item(state_arr, (slice(None), 0))
        out = stepper(np.float64(time), cast("Any", flat_state))
        out_xp = _namespace_of(out)
        if out_xp is not xp:
            msg = "System stepper must preserve the state array namespace."
            raise TypeError(msg)
        out_arr = cast("Array", xp.asarray(out, dtype=state_arr.dtype))
        if out_arr.shape != (n_state,):
            msg = f"Stepper returned shape {out_arr.shape}; expected {(n_state,)}."
            raise ValueError(msg)
        return cast("Array", xp.reshape(out_arr, expected_shape))

    return rhs


def _extract_states_2d(core: ModelCore, *, n_state: int) -> Array:
    state_array = getattr(core, "state_array", None)
    if state_array is None:
        msg = "ModelCore does not expose state_array; store_history must be enabled."
        raise RuntimeError(msg)
    arr = cast("Array", state_array)
    if len(arr.shape) == 3 and arr.shape[1] == n_state and arr.shape[2] == 1:
        return _array_item(arr, (slice(None), slice(None), 0))
    if len(arr.shape) == 2 and arr.shape[1] == n_state:
        return arr
    msg = (
        f"Unexpected state shape {arr.shape}; "
        f"expected (T, {n_state}, 1) or (T, {n_state})."
    )
    raise RuntimeError(msg)


def _make_core(times: np.ndarray, y0: Array) -> ModelCore:
    if len(y0.shape) != 1:
        msg = f"Initial state must be one-dimensional; got {y0.shape}."
        raise ValueError(msg)
    n_states = int(y0.shape[0])
    core = ModelCore(
        n_states,
        1,
        np.asarray(times, dtype=np.float64),
        options=ModelCoreOptions(
            other_axes=(),
            store_history=True,
            dtype=cast("DTypeLike", y0.dtype),
        ),
    )
    xp = _namespace_of(y0)
    core.set_initial_state(cast("Array", xp.reshape(y0, (n_states, 1))))
    return core


def _unwrap_parameter_values(
    values: Mapping[IdentifierString, ParameterValue],
) -> dict[IdentifierString, object]:
    """Return raw parameter payloads for binding to a system stepper."""
    return {name: value.value for name, value in values.items()}


def _as_initial_scalar(
    value: object,
    *,
    name: str,
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array:
    """Coerce one state-cell value without leaving the run namespace."""
    arr = cast("Array", xp.asarray(value, dtype=dtype))
    if arr.shape != ():
        msg = f"Initial state value for {name!r} must be scalar; got {arr.shape}."
        raise ValueError(msg)
    return arr


def _shaped_initial_scalar(
    *,
    state_name: str,
    entry: Mapping[str, object],
    params: Mapping[IdentifierString, ParameterValue],
    raw_params: Mapping[IdentifierString, object],
    axis_labels: Mapping[str, object],
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array:
    """Resolve one expanded state cell from a shaped parameter value."""
    shaped_name = entry.get("shaped")
    if not isinstance(shaped_name, str):
        msg = f"Initial state entry for {state_name!r} has no shaped parameter name."
        raise TypeError(msg)
    if shaped_name not in params:
        msg = (
            f"Initial state for {state_name!r} references shaped parameter "
            f"{shaped_name!r}, which is not present in params."
        )
        raise KeyError(msg)
    coords = entry.get("coords")
    if not isinstance(coords, Mapping):
        msg = f"Initial state entry for {state_name!r} has no coordinate mapping."
        raise TypeError(msg)

    pv = params[shaped_name]
    indices: list[int] = []
    for axis_name in pv.shape.axis_names:
        coord = coords.get(axis_name)
        labels = axis_labels.get(axis_name)
        if not isinstance(coord, str) or not isinstance(labels, tuple | list):
            msg = (
                f"Initial state for {state_name!r} cannot resolve coordinate "
                f"{coord!r} on axis {axis_name!r}."
            )
            raise KeyError(msg)
        string_labels = tuple(str(label) for label in labels)
        if coord not in string_labels:
            msg = (
                f"Initial state for {state_name!r} references unknown coordinate "
                f"{coord!r} on axis {axis_name!r}."
            )
            raise KeyError(msg)
        indices.append(string_labels.index(coord))

    shaped_value = cast("Array", raw_params[shaped_name])
    value = _array_item(shaped_value, tuple(indices))
    return _as_initial_scalar(value, name=state_name, xp=xp, dtype=dtype)


def _assemble_option_initial_state(
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array | None:
    """Assemble state from the metadata contract published by op_system."""
    state_names = system.option("state_names", None)
    if not isinstance(state_names, tuple | list):
        return None

    init_map = system.option("initial_state", None) or {}
    if not isinstance(init_map, Mapping):
        msg = "system option 'initial_state' must be a mapping when provided."
        raise TypeError(msg)
    raw_initial_state = _unwrap_parameter_values(initial_state)
    raw_params = _unwrap_parameter_values(params)
    axis_labels = system.option("axis_labels", None) or {}
    if not isinstance(axis_labels, Mapping):
        msg = "system option 'axis_labels' must be a mapping when provided."
        raise TypeError(msg)

    values: list[Array] = []
    for state_name_obj in state_names:
        state_name = str(state_name_obj)
        if state_name in raw_initial_state:
            values.append(
                _as_initial_scalar(
                    raw_initial_state[state_name],
                    name=state_name,
                    xp=xp,
                    dtype=dtype,
                )
            )
            continue

        entry = init_map.get(state_name)
        if entry is None:
            values.append(_as_initial_scalar(0.0, name=state_name, xp=xp, dtype=dtype))
        elif isinstance(entry, str):
            if entry not in raw_params:
                msg = (
                    f"Initial state for {state_name!r} references parameter "
                    f"{entry!r}, which is not present in params."
                )
                raise KeyError(msg)
            values.append(
                _as_initial_scalar(
                    raw_params[entry],
                    name=state_name,
                    xp=xp,
                    dtype=dtype,
                )
            )
        elif isinstance(entry, Mapping):
            values.append(
                _shaped_initial_scalar(
                    state_name=state_name,
                    entry=entry,
                    params=params,
                    raw_params=raw_params,
                    axis_labels=axis_labels,
                    xp=xp,
                    dtype=dtype,
                )
            )
        else:
            values.append(
                _as_initial_scalar(entry, name=state_name, xp=xp, dtype=dtype)
            )

    return cast("Array", xp.stack(tuple(values), axis=0))


def _assemble_initial_state(
    system: SystemABC,
    initial_state: Mapping[IdentifierString, ParameterValue],
    params: Mapping[IdentifierString, ParameterValue],
    model_state: ModelStateSpecification | None,
) -> Array:
    """Assemble flepimop2 state entries in their declared semantic order."""
    xp, dtype = _numerical_context(system, initial_state, params)
    option_state = _assemble_option_initial_state(
        system,
        initial_state,
        params,
        xp=xp,
        dtype=dtype,
    )
    if option_state is not None:
        return option_state
    if model_state is None:
        msg = "model_state must be provided to assemble the initial state."
        raise ValueError(msg)
    values = [
        cast(
            "Array",
            xp.reshape(xp.asarray(initial_state[name].value, dtype=dtype), (-1,)),
        )
        for name in model_state.parameter_names
    ]
    return cast("Array", xp.concat(tuple(values), axis=0))


class OpEngineFlepimop2Engine(EngineABC):
    """flepimop2 engine adapter backed by op_engine.CoreSolver."""

    module: Literal["flepimop2.engine.op_engine"] = "flepimop2.engine.op_engine"
    state_change: StateChangeEnum
    config: OpEngineEngineConfig = Field(default_factory=OpEngineEngineConfig)

    def validate_system(self, system: SystemABC) -> list[ValidationIssue] | None:
        """Validate system compatibility with engine config."""
        issues: list[ValidationIssue] = []

        if system.state_change != self.state_change:
            issues.append(
                ValidationIssue(
                    msg=(
                        f"Engine state change type, '{self.state_change}', is not "
                        "compatible with system state change type "
                        f"'{system.state_change}'."
                    ),
                    kind="incompatible_system",
                ),
            )

        method = self.config.method
        is_imex = method.is_imex

        if is_imex and not _has_operator_specs(
            _coerce_operator_specs(self.config.operators),
        ):
            sys_ops = system.option("operators", None)
            descriptors = typed_operator_descriptors(sys_ops)
            has_system_operators = bool(descriptors) or _has_operator_specs(
                _coerce_operator_specs(sys_ops)
            )
            if not has_system_operators:
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"IMEX method '{method}' requires operator matrices, "
                            "but neither the engine config nor "
                            "system.option('operators') provides them."
                        ),
                        kind="missing_operators",
                    ),
                )

        if method.is_implicit:
            jac = system.option("jacobian", None)
            if jac is None:
                issues.append(
                    ValidationIssue(
                        msg=(
                            f"Implicit/Rosenbrock method '{method}' requires a "
                            "Jacobian callable, but system.option('jacobian') "
                            "is not provided."
                        ),
                        kind="missing_jacobian",
                    ),
                )

        return issues or None

    def run(
        self,
        system: SystemABC,
        eval_times: Float64NDArray,
        initial_state: dict[IdentifierString, ParameterValue],
        params: Mapping[IdentifierString, ParameterValue],
        model_state: ModelStateSpecification | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Float64NDArray:
        """Execute simulation using op_engine and return `(time, state...)` output."""
        del kwargs

        times = _as_float64_1d(eval_times, name="eval_times")
        _ensure_strictly_increasing(times, name="eval_times")
        raw_params = _unwrap_parameter_values(params)
        y0 = _assemble_initial_state(system, initial_state, params, model_state)
        n_state = int(y0.shape[0])

        run_cfg = self.config.to_run_config()
        method = self.config.method
        is_imex = method.is_imex
        operators = run_cfg.operators
        compiled_system_operators = False

        if is_imex and not _has_operator_specs(operators):
            system_operators = system.option("operators", None)
            descriptors = typed_operator_descriptors(system_operators)
            if descriptors:
                operators = compile_operator_descriptors(
                    descriptors,
                    method=method.value,
                    state_names=system.option("state_names", None),
                    axis_order=system.option("axis_order", None),
                    axis_labels=system.option("axis_labels", None),
                    params=raw_params,
                    reference=y0,
                )
                compiled_system_operators = True
            else:
                operators = _coerce_operator_specs(system_operators) or operators
        run_cfg = replace(run_cfg, operators=operators)

        if is_imex and not _has_operator_specs(operators):
            msg = (
                f"IMEX method '{run_cfg.method}' requires operators from engine config "
                "or system option 'operators'."
            )
            raise ValueError(msg)

        if method.is_implicit:
            jacobian = system.option("jacobian", None)
            if callable(jacobian):
                run_cfg = replace(run_cfg, jacobian=jacobian)

        operator_axis: str | int = (
            "state" if compiled_system_operators else self.config.operator_axis
        )
        if operator_axis == "state":
            system_axis = system.option("operator_axis", None)
            if not compiled_system_operators and isinstance(system_axis, str | int):
                operator_axis = system_axis

        # Bind raw payloads once. Systems such as op_system merge their own
        # static mixing kernels and should never receive ParameterValue wrappers.
        stepper: SystemProtocol = system.bind(params=raw_params)
        rhs = _rhs_from_stepper(stepper, n_state=n_state)
        core = _make_core(times, y0)

        solver = CoreSolver(
            core,
            operators=operators.default if is_imex else None,
            operator_axis=operator_axis,
        )
        solver.run(rhs, config=run_cfg)

        states = _extract_states_2d(core, n_state=n_state)
        xp = _namespace_of(states)
        time_values = cast("Array", xp.asarray(times, dtype=states.dtype))
        time_column = cast("Array", xp.expand_dims(time_values, axis=1))
        result = cast("Array", xp.concat((time_column, states), axis=1))
        # flepimop2#343 will generalize this inherited annotation to Array.
        return cast("Float64NDArray", result)


__all__ = ["OpEngineEngineConfig", "OpEngineFlepimop2Engine", "SolverMethod"]
