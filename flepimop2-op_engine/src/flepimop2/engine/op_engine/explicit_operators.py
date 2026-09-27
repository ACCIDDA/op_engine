# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Structured, namespace-preserving explicit typed-operator drift."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from op_engine.matrix_ops import build_advection_matrix, build_diffusion_matrix

from .operators import (
    _advection_direction_sign,
    _apply_to_bases,
    _array_namespace,
    _axis_label_map,
    _diffusion_axis_geometry,
    _parse_expanded_state_name,
    _require_string_sequence,
    _resolve_generator_array,
    _resolve_scalar_array,
    _uniform_axis_spacing,
)

if TYPE_CHECKING:
    from flepimop2.typing import Array
    from op_system import OperatorDescriptor


def _row_source_operator(
    descriptor: OperatorDescriptor,
    *,
    axis_labels: Mapping[str, tuple[str, ...]],
    axis_coords: Mapping[str, object],
    params: Mapping[str, object],
    reference: Array,
) -> Array:
    """Build one small row-source operator in a structured leaf namespace."""
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
        return cast("Array", xp.multiply(generator, velocity))
    if descriptor.kind in {"advection", "transport"}:
        velocity = _resolve_scalar_array(
            descriptor.velocity,
            params=params,
            field="advection velocity",
            xp=xp,
            dtype=reference.dtype,
        )
        velocity = cast(
            "Array",
            xp.multiply(velocity, _advection_direction_sign(descriptor)),
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
        return cast("Array", xp.permute_dims(column_operator, (1, 0)))
    if descriptor.kind == "diffusion":
        coefficient = _resolve_scalar_array(
            descriptor.rate,
            params=params,
            field="diffusion rate",
            xp=xp,
            dtype=reference.dtype,
        )
        diffusion_dx, grid = _diffusion_axis_geometry(
            descriptor.axis,
            axis_coords=axis_coords,
            size=len(labels),
        )
        column_operator = build_diffusion_matrix(
            len(labels),
            diffusion_dx,
            coefficient,
            grid=grid,
            bc=descriptor.bc or "neumann",
            reference=reference,
        )
        return cast("Array", xp.permute_dims(column_operator, (1, 0)))
    msg = f"Unsupported op_system operator kind {descriptor.kind!r}."
    raise ValueError(msg)


def _template_axes(
    *,
    state_names: tuple[str, ...],
    axis_order: tuple[str, ...],
    axis_labels: Mapping[str, tuple[str, ...]],
    reference: Mapping[str, Array],
    excluded_axes: frozenset[str],
) -> dict[str, tuple[str, ...]]:
    """Recover structured leaf axes from expanded op_system state names."""
    result: dict[str, tuple[str, ...]] = {}
    for state_name in state_names:
        base, coords = _parse_expanded_state_name(state_name, axes=axis_order)
        if base not in reference or base in result:
            continue
        axes = tuple(
            axis
            for axis in axis_order
            if axis not in {"state", "subgroup"}
            and axis not in excluded_axes
            and axis in coords
        )
        value = reference[base]
        if len(axes) != len(value.shape):
            continue
        if any(
            axis not in axis_labels
            or len(axis_labels[axis]) != int(value.shape[position])
            for position, axis in enumerate(axes)
        ):
            continue
        result[base] = axes
    missing = sorted(set(reference) - set(result))
    if missing:
        msg = (
            "Expanded state metadata cannot recover structured axes for "
            f"templates {missing!r}."
        )
        raise ValueError(msg)
    return result


def _apply_axis_operator(
    value: Array,
    row_source_operator: Array,
    *,
    axis: int,
) -> Array:
    """Apply a row-source matrix along one leaf axis without flattening state."""
    xp = _array_namespace(value)
    ndim = len(value.shape)
    permutation = (*tuple(index for index in range(ndim) if index != axis), axis)
    inverse = tuple(permutation.index(index) for index in range(ndim))
    transposed = cast("Array", xp.permute_dims(value, permutation))
    width = int(value.shape[axis])
    rows = cast("Array", xp.reshape(transposed, (-1, width)))
    operator = cast("Array", xp.asarray(row_source_operator, dtype=value.dtype))
    updated = cast("Array", xp.matmul(rows, operator))
    restored = cast("Array", xp.reshape(updated, transposed.shape))
    return cast("Array", xp.permute_dims(restored, inverse))


def compile_structured_operator_drift(
    descriptors: tuple[OperatorDescriptor, ...],
    *,
    state_names: object,
    axis_order: object,
    axis_labels: object,
    axis_coords: object | None = None,
    params: Mapping[str, object],
    reference: Mapping[str, Array],
    excluded_axes: tuple[str, ...] = (),
) -> Callable[[Mapping[str, Array]], dict[str, Array]]:
    """Compile typed descriptors into an additive structured explicit drift.

    The callable applies small axis-local matrices directly to selected state
    leaves. It preserves the array namespace and avoids a dense full-state
    operator, including for block-factorized execution.
    """
    if not descriptors:
        msg = "At least one typed operator descriptor is required."
        raise ValueError(msg)
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
    template_axes = _template_axes(
        state_names=names,
        axis_order=axes,
        axis_labels=labels,
        reference=reference,
        excluded_axes=frozenset(excluded_axes),
    )

    operations: list[tuple[str, int, Array]] = []
    for descriptor in descriptors:
        if descriptor.kind not in {
            "axis_kernel",
            "advection",
            "diffusion",
            "transport",
        }:
            msg = f"Unsupported op_system operator kind {descriptor.kind!r}."
            raise ValueError(msg)
        selected_bases = _apply_to_bases(descriptor)
        matched = False
        for base, value in reference.items():
            if selected_bases and base not in selected_bases:
                continue
            base_axes = template_axes[base]
            if descriptor.axis not in base_axes:
                continue
            axis_position = base_axes.index(descriptor.axis)
            operator = _row_source_operator(
                descriptor,
                axis_labels=labels,
                axis_coords=coordinates,
                params=params,
                reference=value,
            )
            operations.append((base, axis_position, operator))
            matched = True
        if not matched:
            msg = (
                f"Operator axis {descriptor.axis!r} does not occur in any "
                "selected structured state template."
            )
            raise ValueError(msg)

    def drift(state: Mapping[str, Array]) -> dict[str, Array]:
        if set(state) != set(reference):
            msg = "Structured operator state templates differ from the compiled layout."
            raise ValueError(msg)
        result = {
            name: cast("Array", _array_namespace(value).zeros_like(value))
            for name, value in state.items()
        }
        for base, axis_position, operator in operations:
            value = state[base]
            contribution = _apply_axis_operator(
                value,
                operator,
                axis=axis_position,
            )
            xp = _array_namespace(value)
            result[base] = cast("Array", xp.add(result[base], contribution))
        return result

    return drift


__all__ = ["compile_structured_operator_drift"]
