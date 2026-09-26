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
from typing import TYPE_CHECKING, cast

import numpy as np

from op_engine.core_solver import OperatorSpecs
from op_engine.matrix_ops import (
    make_constant_base_builder,
    make_stage_operator_factory,
)

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray
    from op_system import OperatorDescriptor


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


def _lift_generator(
    descriptor: OperatorDescriptor,
    *,
    state_names: tuple[str, ...],
    axis_order: tuple[str, ...],
    axis_labels: Mapping[str, tuple[str, ...]],
    params: Mapping[str, object],
) -> NDArray[np.float64]:
    """Lift one row-source generator into the expanded flat state layout."""
    if descriptor.axis not in axis_labels:
        msg = f"Operator references unknown axis {descriptor.axis!r}."
        raise KeyError(msg)
    labels = axis_labels[descriptor.axis]
    generator = _resolve_generator(descriptor, params=params, size=len(labels))
    velocity = _resolve_scalar(
        descriptor.velocity,
        params=params,
        field="axis_kernel velocity",
    )
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
                flat[target_index, source_index] += (
                    velocity * generator[source_position, target_position]
                )
    return flat


def compile_operator_descriptors(  # noqa: PLR0913
    descriptors: tuple[OperatorDescriptor, ...],
    *,
    method: str,
    state_names: object,
    axis_order: object,
    axis_labels: object,
    params: Mapping[str, object],
) -> OperatorSpecs:
    """Compile supported typed descriptors into method-specific stage factories."""
    names = _require_string_sequence(state_names, name="state_names")
    axes = _require_string_sequence(axis_order, name="axis_order")
    labels = _axis_label_map(axis_labels)
    base_operator = np.zeros((len(names), len(names)), dtype=np.float64)
    for descriptor in descriptors:
        if descriptor.kind != "axis_kernel":
            msg = (
                f"Unsupported op_system operator kind {descriptor.kind!r}; only "
                "axis_kernel generators can currently be compiled."
            )
            raise ValueError(msg)
        base_operator += _lift_generator(
            descriptor,
            state_names=names,
            axis_order=axes,
            axis_labels=labels,
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
    msg = f"Typed system operators require an IMEX method; got {method!r}."
    raise ValueError(msg)


__all__ = ["compile_operator_descriptors", "typed_operator_descriptors"]
