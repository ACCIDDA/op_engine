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

"""Compile typed op_system operator metadata for op_engine."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from op_engine.core_solver import OperatorSpecs
from op_engine.matrix_ops import (
    build_advection_matrix,
    build_diffusion_matrix,
    make_constant_base_builder,
    make_stage_operator_factory,
)

if TYPE_CHECKING:
    from types import ModuleType

    from flepimop2.typing import Array
    from numpy.typing import NDArray
    from op_system import OperatorDescriptor

    from op_engine.core_solver import CoreOperators, StageOperatorFactory
    from op_engine.matrix_ops import StageOperatorContext

_STATE_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def _optional_op_system() -> ModuleType | None:
    """Import op_system only when typed descriptors need it."""
    try:
        return import_module("op_system")
    except ModuleNotFoundError:
        return None


def typed_operator_descriptors(
    value: object,
) -> tuple[OperatorDescriptor, ...] | None:
    """Return a typed descriptor tuple, or ``None`` for another option shape."""
    if not isinstance(value, tuple):
        return None
    module = _optional_op_system()
    if module is None:
        return None
    descriptor_type = getattr(module, "OperatorDescriptor", None)
    if not isinstance(descriptor_type, type) or not all(
        isinstance(item, descriptor_type) for item in value
    ):
        return None
    return cast("tuple[OperatorDescriptor, ...]", value)


def _require_string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    """Validate one system-option sequence of strings."""
    if not isinstance(value, tuple | list) or not all(
        isinstance(item, str) for item in value
    ):
        msg = f"system option {name!r} must be a sequence of strings."
        raise TypeError(msg)
    return tuple(value)


def _axis_label_map(value: object) -> dict[str, tuple[str, ...]]:
    """Validate and normalize the system axis-label option."""
    if not isinstance(value, Mapping):
        msg = "system option 'axis_labels' must be a mapping."
        raise TypeError(msg)
    labels: dict[str, tuple[str, ...]] = {}
    for axis, raw_labels in value.items():
        if not isinstance(axis, str) or not isinstance(raw_labels, tuple | list):
            msg = "system option 'axis_labels' must map strings to sequences."
            raise TypeError(msg)
        axis_values = tuple(
            _STATE_LABEL_SAFE_RE.sub("_", str(label)) for label in raw_labels
        )
        if len(axis_values) != len(set(axis_values)):
            msg = f"Axis {axis!r} contains duplicate coordinate labels."
            raise ValueError(msg)
        labels[axis] = axis_values
    return labels


def _parse_expanded_state_name(
    state_name: str,
    *,
    axes: tuple[str, ...],
) -> tuple[str, dict[str, str]]:
    """Split one op_system expanded state name into base and coordinates."""
    parts = state_name.split("__")
    coords: dict[str, str] = {}
    suffix_start = len(parts)
    candidate_axes = tuple(
        sorted(
            (axis for axis in axes if axis not in {"state", "subgroup"}),
            key=len,
            reverse=True,
        )
    )
    for index in range(len(parts) - 1, 0, -1):
        part = parts[index]
        matches = [axis for axis in candidate_axes if part.startswith(f"{axis}_")]
        if not matches:
            break
        axis = matches[0]
        if axis in coords:
            msg = f"Expanded state name {state_name!r} repeats axis {axis!r}."
            raise ValueError(msg)
        coords[axis] = part[len(axis) + 1 :]
        suffix_start = index
    base = "__".join(parts[:suffix_start])
    return base, coords


def _apply_to_bases(descriptor: OperatorDescriptor) -> frozenset[str]:
    """Normalize an optional operator state selector to base state names."""
    if descriptor.apply_to is None:
        return frozenset()
    return frozenset(
        name.split("[", 1)[0].split("__", 1)[0] for name in descriptor.apply_to
    )


def _resolve_scalar(
    value: str | float | None,
    *,
    params: Mapping[str, object],
    field: str,
) -> float:
    """Resolve a finite scalar literal or parameter reference."""
    resolved: object = 1.0 if value is None else value
    if isinstance(resolved, str):
        if resolved not in params:
            msg = f"{field} references missing parameter {resolved!r}."
            raise KeyError(msg)
        resolved = params[resolved]
    arr = np.asarray(resolved, dtype=np.float64)
    if arr.shape != ():
        msg = f"{field} must resolve to a scalar; got shape {arr.shape}."
        raise ValueError(msg)
    scalar = float(arr)
    if not np.isfinite(scalar):
        msg = f"{field} must resolve to a finite scalar."
        raise ValueError(msg)
    return scalar


def _uniform_axis_spacing(
    axis: str,
    *,
    axis_coords: Mapping[str, object],
    size: int,
) -> float:
    """Return the positive spacing of one static uniform coordinate axis."""
    if axis not in axis_coords:
        msg = f"Operator axis {axis!r} has no axis_coords metadata."
        raise KeyError(msg)
    coordinates = np.asarray(axis_coords[axis], dtype=np.float64)
    if coordinates.shape != (size,):
        msg = (
            f"Coordinates for operator axis {axis!r} must have shape {(size,)}; "
            f"got {coordinates.shape}."
        )
        raise ValueError(msg)
    if size < 2:
        msg = f"Operator axis {axis!r} must contain at least two cells."
        raise ValueError(msg)
    if not np.isfinite(coordinates).all():
        msg = f"Coordinates for operator axis {axis!r} must be finite."
        raise ValueError(msg)
    spacings = np.diff(coordinates)
    if not np.all(spacings > 0.0):
        msg = f"Coordinates for operator axis {axis!r} must be strictly increasing."
        raise ValueError(msg)
    if not np.allclose(spacings, spacings[0], rtol=1e-10, atol=1e-12):
        msg = (
            f"Operator axis {axis!r} must be uniformly spaced; "
            "non-uniform grids are not yet supported."
        )
        raise ValueError(msg)
    return float(spacings[0])


def _resolve_generator(
    descriptor: OperatorDescriptor,
    *,
    params: Mapping[str, object],
    size: int,
) -> NDArray[np.float64]:
    """Resolve and validate an axis-kernel generator matrix."""
    kernel = descriptor.kernel
    if not isinstance(kernel, Mapping) or kernel.get("form") != "generator":
        msg = (
            "Only op_system axis_kernel operators with "
            "kernel.form='generator' are supported by the IMEX provider."
        )
        raise ValueError(msg)
    kernel_params = kernel.get("params")
    if not isinstance(kernel_params, Mapping) or "matrix" not in kernel_params:
        msg = "axis_kernel generator requires kernel.params.matrix."
        raise ValueError(msg)
    matrix_ref = kernel_params["matrix"]
    matrix_value: object = matrix_ref
    if isinstance(matrix_ref, str):
        if matrix_ref not in params:
            msg = f"axis_kernel matrix references missing parameter {matrix_ref!r}."
            raise KeyError(msg)
        matrix_value = params[matrix_ref]
    generator = np.asarray(matrix_value, dtype=np.float64)
    if generator.shape != (size, size):
        msg = (
            f"axis_kernel generator for axis {descriptor.axis!r} must have "
            f"shape {(size, size)}; got {generator.shape}."
        )
        raise ValueError(msg)
    module = _optional_op_system()
    shared_validator = (
        None if module is None else getattr(module, "validate_axis_kernel_matrix", None)
    )
    if not callable(shared_validator):
        msg = (
            "axis_kernel compilation requires an op_system version that "
            "exports validate_axis_kernel_matrix."
        )
        raise TypeError(msg)
    problems = shared_validator(generator, form="generator")
    if problems:
        msg = "Invalid axis_kernel generator: " + "; ".join(problems)
        raise ValueError(msg)
    return generator


def _lift_axis_operator(  # noqa: PLR0912, PLR0913, PLR0914
    descriptor: OperatorDescriptor,
    *,
    state_names: tuple[str, ...],
    axis_order: tuple[str, ...],
    axis_labels: Mapping[str, tuple[str, ...]],
    axis_coords: Mapping[str, object],
    params: Mapping[str, object],
) -> NDArray[np.float64]:
    """Lift one axis operator into the expanded flat state layout."""
    if descriptor.axis not in axis_labels:
        msg = f"Operator references unknown axis {descriptor.axis!r}."
        raise KeyError(msg)
    labels = axis_labels[descriptor.axis]
    if descriptor.kind == "axis_kernel":
        generator = _resolve_generator(descriptor, params=params, size=len(labels))
        velocity = _resolve_scalar(
            descriptor.velocity,
            params=params,
            field="axis_kernel velocity",
        )
        row_source_operator = velocity * generator
    elif descriptor.kind in {"advection", "transport"}:
        velocity = _resolve_scalar(
            descriptor.velocity,
            params=params,
            field="advection velocity",
        )
        dx = _uniform_axis_spacing(
            descriptor.axis,
            axis_coords=axis_coords,
            size=len(labels),
        )
        column_operator = build_advection_matrix(
            len(labels),
            dx,
            velocity,
            bc=descriptor.bc or "absorbing",
        )
        row_source_operator = np.asarray(column_operator).T
    elif descriptor.kind == "diffusion":
        coefficient = _resolve_scalar(
            descriptor.rate,
            params=params,
            field="diffusion rate",
        )
        dx = _uniform_axis_spacing(
            descriptor.axis,
            axis_coords=axis_coords,
            size=len(labels),
        )
        column_operator = build_diffusion_matrix(
            len(labels),
            dx,
            coefficient,
            bc=descriptor.bc or "neumann",
        )
        row_source_operator = np.asarray(column_operator).T
    else:  # pragma: no cover - guarded by the public compiler
        msg = f"Unsupported op_system operator kind {descriptor.kind!r}."
        raise ValueError(msg)
    selected_bases = _apply_to_bases(descriptor)
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, int]] = {}
    for index, state_name in enumerate(state_names):
        base, coords = _parse_expanded_state_name(state_name, axes=axis_order)
        if selected_bases and base not in selected_bases:
            continue
        if descriptor.axis not in coords:
            continue
        other_coords = tuple(
            (axis, coords[axis])
            for axis in axis_order
            if axis != descriptor.axis and axis in coords
        )
        group = groups.setdefault((base, other_coords), {})
        coordinate = coords[descriptor.axis]
        if coordinate in group:
            msg = (
                f"State layout repeats coordinate {coordinate!r} on axis "
                f"{descriptor.axis!r} for base {base!r}."
            )
            raise ValueError(msg)
        group[coordinate] = index

    if not groups:
        msg = (
            f"Operator axis {descriptor.axis!r} does not occur in any selected "
            "expanded state."
        )
        raise ValueError(msg)

    flat = np.zeros((len(state_names), len(state_names)), dtype=np.float64)
    expected_coordinates = set(labels)
    for (base, _other_coords), coordinate_indices in groups.items():
        if set(coordinate_indices) != expected_coordinates:
            msg = (
                f"Expanded state group for {base!r} does not cover every "
                f"coordinate of axis {descriptor.axis!r}."
            )
            raise ValueError(msg)
        for source_position, source_label in enumerate(labels):
            source_index = coordinate_indices[source_label]
            for target_position, target_label in enumerate(labels):
                target_index = coordinate_indices[target_label]
                flat[target_index, source_index] += row_source_operator[
                    source_position, target_position
                ]
    return flat


def _array_namespace(value: object) -> Any:  # noqa: ANN401
    """Return the namespace advertised by a dense operator reference."""
    namespace = getattr(value, "__array_namespace__", None)
    if namespace is None:
        msg = "Dense portable operator compilation requires an Array-API reference."
        raise TypeError(msg)
    return namespace()


def _resolve_scalar_array(
    value: str | float | None,
    *,
    params: Mapping[str, object],
    field: str,
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array:
    """Resolve a scalar without materializing dynamic values on the host."""
    resolved: object = 1.0 if value is None else value
    if isinstance(resolved, str):
        if resolved not in params:
            msg = f"{field} references missing parameter {resolved!r}."
            raise KeyError(msg)
        resolved = params[resolved]
    arr = cast("Array", xp.asarray(resolved, dtype=dtype))
    if arr.shape != ():
        msg = f"{field} must resolve to a scalar; got shape {arr.shape}."
        raise ValueError(msg)
    return arr


def _resolve_generator_array(
    descriptor: OperatorDescriptor,
    *,
    params: Mapping[str, object],
    size: int,
    xp: Any,  # noqa: ANN401
    dtype: object,
) -> Array:
    """Resolve an axis-kernel generator in the state array namespace."""
    kernel = descriptor.kernel
    if not isinstance(kernel, Mapping) or kernel.get("form") != "generator":
        msg = (
            "Only op_system axis_kernel operators with "
            "kernel.form='generator' are supported by the IMEX provider."
        )
        raise ValueError(msg)
    kernel_params = kernel.get("params")
    if not isinstance(kernel_params, Mapping) or "matrix" not in kernel_params:
        msg = "axis_kernel generator requires kernel.params.matrix."
        raise ValueError(msg)
    matrix_ref = kernel_params["matrix"]
    matrix_value: object = matrix_ref
    if isinstance(matrix_ref, str):
        if matrix_ref not in params:
            msg = f"axis_kernel matrix references missing parameter {matrix_ref!r}."
            raise KeyError(msg)
        matrix_value = params[matrix_ref]
    generator = cast("Array", xp.asarray(matrix_value, dtype=dtype))
    if generator.shape != (size, size):
        msg = (
            f"axis_kernel generator for axis {descriptor.axis!r} must have "
            f"shape {(size, size)}; got {generator.shape}."
        )
        raise ValueError(msg)
    return generator


def _lift_axis_operator_array(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    descriptor: OperatorDescriptor,
    *,
    state_names: tuple[str, ...],
    axis_order: tuple[str, ...],
    axis_labels: Mapping[str, tuple[str, ...]],
    axis_coords: Mapping[str, object],
    params: Mapping[str, object],
    reference: Array,
) -> Array:
    """Lift an axis operator with a static map and namespace-native matmul."""
    if descriptor.axis not in axis_labels:
        msg = f"Operator references unknown axis {descriptor.axis!r}."
        raise KeyError(msg)
    labels = axis_labels[descriptor.axis]
    xp = _array_namespace(reference)
    if descriptor.kind == "axis_kernel":
        generator = _resolve_generator_array(
            descriptor,
            params=params,
            size=len(labels),
            xp=xp,
            dtype=reference.dtype,
        )
        velocity = _resolve_scalar_array(
            descriptor.velocity,
            params=params,
            field="axis_kernel velocity",
            xp=xp,
            dtype=reference.dtype,
        )
        row_source_operator = xp.multiply(generator, velocity)
    elif descriptor.kind in {"advection", "transport"}:
        velocity = _resolve_scalar_array(
            descriptor.velocity,
            params=params,
            field="advection velocity",
            xp=xp,
            dtype=reference.dtype,
        )
        dx = _uniform_axis_spacing(
            descriptor.axis,
            axis_coords=axis_coords,
            size=len(labels),
        )
        column_operator = build_advection_matrix(
            len(labels),
            dx,
            velocity,
            bc=descriptor.bc or "absorbing",
            reference=reference,
        )
        row_source_operator = xp.permute_dims(column_operator, (1, 0))
    elif descriptor.kind == "diffusion":
        coefficient = _resolve_scalar_array(
            descriptor.rate,
            params=params,
            field="diffusion rate",
            xp=xp,
            dtype=reference.dtype,
        )
        dx = _uniform_axis_spacing(
            descriptor.axis,
            axis_coords=axis_coords,
            size=len(labels),
        )
        column_operator = build_diffusion_matrix(
            len(labels),
            dx,
            coefficient,
            bc=descriptor.bc or "neumann",
            reference=reference,
        )
        row_source_operator = xp.permute_dims(column_operator, (1, 0))
    else:  # pragma: no cover - guarded by the public compiler
        msg = f"Unsupported op_system operator kind {descriptor.kind!r}."
        raise ValueError(msg)

    selected_bases = _apply_to_bases(descriptor)
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, int]] = {}
    for index, state_name in enumerate(state_names):
        base, coords = _parse_expanded_state_name(state_name, axes=axis_order)
        if selected_bases and base not in selected_bases:
            continue
        if descriptor.axis not in coords:
            continue
        other_coords = tuple(
            (axis, coords[axis])
            for axis in axis_order
            if axis != descriptor.axis and axis in coords
        )
        group = groups.setdefault((base, other_coords), {})
        coordinate = coords[descriptor.axis]
        if coordinate in group:
            msg = (
                f"State layout repeats coordinate {coordinate!r} on axis "
                f"{descriptor.axis!r} for base {base!r}."
            )
            raise ValueError(msg)
        group[coordinate] = index

    if not groups:
        msg = (
            f"Operator axis {descriptor.axis!r} does not occur in any selected "
            "expanded state."
        )
        raise ValueError(msg)

    flat_size = len(state_names)
    kernel_size = len(labels)
    lift_map = np.zeros(
        (flat_size * flat_size, kernel_size * kernel_size),
        dtype=np.float64,
    )
    expected_coordinates = set(labels)
    for (base, _other_coords), coordinate_indices in groups.items():
        if set(coordinate_indices) != expected_coordinates:
            msg = (
                f"Expanded state group for {base!r} does not cover every "
                f"coordinate of axis {descriptor.axis!r}."
            )
            raise ValueError(msg)
        for source_position, source_label in enumerate(labels):
            source_index = coordinate_indices[source_label]
            for target_position, target_label in enumerate(labels):
                target_index = coordinate_indices[target_label]
                flat_position = target_index * flat_size + source_index
                kernel_position = source_position * kernel_size + target_position
                lift_map[flat_position, kernel_position] += 1.0

    lift_array = xp.asarray(lift_map, dtype=reference.dtype)
    kernel_values = xp.reshape(row_source_operator, (kernel_size * kernel_size,))
    flat_values = xp.matmul(lift_array, kernel_values)
    return cast("Array", xp.reshape(flat_values, (flat_size, flat_size)))


def _make_array_stage_factory(
    base_operator: Array,
    *,
    scheme: str,
) -> StageOperatorFactory:
    """Build dense stage matrices in the stage state's namespace."""

    def factory(
        dt: float,
        scale: float,
        ctx: StageOperatorContext,
    ) -> CoreOperators:
        xp = _array_namespace(ctx.y)
        operator = xp.asarray(base_operator, dtype=ctx.y.dtype)
        identity = xp.eye(operator.shape[0], dtype=ctx.y.dtype)
        dt_scale = dt * scale
        if scheme == "implicit-euler":
            left = xp.subtract(identity, xp.multiply(operator, dt_scale))
            return cast("CoreOperators", (left, identity))
        if scheme == "trapezoidal":
            half_scale = 0.5 * dt_scale
            scaled = xp.multiply(operator, half_scale)
            left = xp.subtract(identity, scaled)
            right = xp.add(identity, scaled)
            return cast("CoreOperators", (left, right))
        msg = f"Unknown dense Array-API operator scheme {scheme!r}."
        raise ValueError(msg)

    return factory


def _compile_array_operator_descriptors(  # noqa: PLR0913
    descriptors: tuple[OperatorDescriptor, ...],
    *,
    method: str,
    state_names: tuple[str, ...],
    axis_order: tuple[str, ...],
    axis_labels: Mapping[str, tuple[str, ...]],
    axis_coords: Mapping[str, object],
    params: Mapping[str, object],
    reference: Array,
) -> OperatorSpecs:
    """Compile typed descriptors without coercing dynamic values through NumPy."""
    xp = _array_namespace(reference)
    base_operator = xp.zeros(
        (len(state_names), len(state_names)),
        dtype=reference.dtype,
    )
    for descriptor in descriptors:
        if descriptor.kind not in {
            "axis_kernel",
            "advection",
            "diffusion",
            "transport",
        }:
            msg = (
                f"Unsupported op_system operator kind {descriptor.kind!r}; "
                "axis_kernel, advection, and diffusion operators are supported."
            )
            raise ValueError(msg)
        contribution = _lift_axis_operator_array(
            descriptor,
            state_names=state_names,
            axis_order=axis_order,
            axis_labels=axis_labels,
            axis_coords=axis_coords,
            params=params,
            reference=reference,
        )
        base_operator = xp.add(base_operator, contribution)

    implicit_euler = _make_array_stage_factory(
        cast("Array", base_operator),
        scheme="implicit-euler",
    )
    trapezoidal = _make_array_stage_factory(
        cast("Array", base_operator),
        scheme="trapezoidal",
    )
    if method == "imex-euler":
        return OperatorSpecs(default=implicit_euler)
    if method == "imex-heun-tr":
        return OperatorSpecs(default=trapezoidal)
    if method == "imex-trbdf2":
        return OperatorSpecs(tr=trapezoidal, bdf2=implicit_euler)
    if method == "imex-ark3":
        return OperatorSpecs(default=implicit_euler)
    msg = f"Typed system operators require an IMEX method; got {method!r}."
    raise ValueError(msg)


def compile_operator_descriptors(  # noqa: PLR0913
    descriptors: tuple[OperatorDescriptor, ...],
    *,
    method: str,
    state_names: object,
    axis_order: object,
    axis_labels: object,
    axis_coords: object | None = None,
    params: Mapping[str, object],
    reference: Array | None = None,
) -> OperatorSpecs:
    """Compile supported typed descriptors into method-specific stage factories."""
    names = _require_string_sequence(state_names, name="state_names")
    axes = _require_string_sequence(axis_order, name="axis_order")
    labels = _axis_label_map(axis_labels)
    if axis_coords is None:
        coordinates: Mapping[str, object] = {}
    elif isinstance(axis_coords, Mapping) and all(
        isinstance(axis, str) for axis in axis_coords
    ):
        coordinates = cast("Mapping[str, object]", axis_coords)
    else:
        msg = "system option 'axis_coords' must be a mapping with string keys."
        raise TypeError(msg)
    if reference is not None and not isinstance(reference, np.ndarray):
        return _compile_array_operator_descriptors(
            descriptors,
            method=method,
            state_names=names,
            axis_order=axes,
            axis_labels=labels,
            axis_coords=coordinates,
            params=params,
            reference=reference,
        )

    base_operator = np.zeros((len(names), len(names)), dtype=np.float64)
    for descriptor in descriptors:
        if descriptor.kind not in {
            "axis_kernel",
            "advection",
            "diffusion",
            "transport",
        }:
            msg = (
                f"Unsupported op_system operator kind {descriptor.kind!r}; "
                "axis_kernel, advection, and diffusion operators are supported."
            )
            raise ValueError(msg)
        base_operator += _lift_axis_operator(
            descriptor,
            state_names=names,
            axis_order=axes,
            axis_labels=labels,
            axis_coords=coordinates,
            params=params,
        )

    builder = make_constant_base_builder(base_operator)
    implicit_euler = make_stage_operator_factory(builder, scheme="implicit-euler")
    trapezoidal = make_stage_operator_factory(builder, scheme="trapezoidal")
    if method == "imex-euler":
        return OperatorSpecs(default=implicit_euler)
    if method == "imex-heun-tr":
        return OperatorSpecs(default=trapezoidal)
    if method == "imex-trbdf2":
        return OperatorSpecs(tr=trapezoidal, bdf2=implicit_euler)
    if method == "imex-ark3":
        return OperatorSpecs(default=implicit_euler)
    msg = f"Typed system operators require an IMEX method; got {method!r}."
    raise ValueError(msg)


__all__ = ["compile_operator_descriptors", "typed_operator_descriptors"]
