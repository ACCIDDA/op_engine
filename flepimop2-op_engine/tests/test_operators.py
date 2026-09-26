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

"""Tests for typed op_system operator compilation."""

from __future__ import annotations

import numpy as np
import pytest
from op_engine.matrix_ops import (
    StageOperatorContext,
    build_advection_matrix,
    build_diffusion_matrix,
)
from op_system import OperatorDescriptor

from flepimop2.engine.op_engine.operators import (
    compile_operator_descriptors,
    typed_operator_descriptors,
)


def _generator_descriptor(
    *,
    apply_to: tuple[str, ...] | None = None,
) -> OperatorDescriptor:
    """Build a typed generator descriptor for the test immune axis.

    Returns:
        An axis-kernel generator descriptor.
    """
    return OperatorDescriptor(
        axis="imm",
        kind="axis_kernel",
        velocity=2.0,
        kernel={"form": "generator", "params": {"matrix": "G"}},
        apply_to=apply_to,
    )


def _advection_descriptor() -> OperatorDescriptor:
    """Build a typed advection descriptor for the test immune axis.

    Returns:
        An advection descriptor for a periodic uniform grid.
    """
    return OperatorDescriptor(
        axis="imm",
        kind="advection",
        velocity=2.0,
        bc="periodic",
        apply_to=("X[imm]",),
    )


def _diffusion_descriptor() -> OperatorDescriptor:
    """Build a typed diffusion descriptor for the test immune axis.

    Returns:
        A diffusion descriptor for a no-flux uniform grid.
    """
    return OperatorDescriptor(
        axis="imm",
        kind="diffusion",
        rate=0.3,
        bc="neumann",
        apply_to=("X[imm]",),
    )


def test_typed_operator_descriptors_rejects_untyped_options() -> None:
    """Only op_system's immutable typed tuple is accepted as system metadata."""
    descriptor = _generator_descriptor()

    assert typed_operator_descriptors((descriptor,)) == (descriptor,)
    assert typed_operator_descriptors([descriptor]) is None
    assert typed_operator_descriptors({"default": "legacy"}) is None


def test_generator_lifts_over_other_axes_and_apply_to() -> None:
    """Row-source generators become flat column-vector operators per group."""
    descriptor = _generator_descriptor(apply_to=("X[age, imm]",))
    generator = np.asarray([[-1.0, 1.0], [0.25, -0.25]])
    state_names = (
        "X__age_young__imm_x_0",
        "X__age_young__imm_x_1",
        "X__age_old__imm_x_0",
        "X__age_old__imm_x_1",
        "Y__age_young__imm_x_0",
        "Y__age_young__imm_x_1",
    )

    specs = compile_operator_descriptors(
        (descriptor,),
        method="imex-euler",
        state_names=state_names,
        axis_order=("state", "subgroup", "age", "imm"),
        axis_labels={"age": ("young", "old"), "imm": ("x-0", "x 1")},
        params={"G": generator},
    )

    assert callable(specs.default)
    dt = 0.2
    left, right = specs.default(
        dt,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((len(state_names), 1))),
    )
    observed = (np.eye(len(state_names)) - np.asarray(left)) / dt
    block = 2.0 * generator.T
    expected = np.zeros_like(observed)
    expected[0:2, 0:2] = block
    expected[2:4, 2:4] = block

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(right), np.eye(len(state_names)))


def test_ark3_compiles_typed_descriptors_to_implicit_euler_factory() -> None:
    """The provider gives each ARK DIRK stage the required left operator."""
    specs = compile_operator_descriptors(
        (_generator_descriptor(apply_to=("X[imm]",)),),
        method="imex-ark3",
        state_names=("X__imm_x_0", "X__imm_x_1"),
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x-0", "x 1")},
        params={"G": np.asarray([[-1.0, 1.0], [0.25, -0.25]])},
    )

    assert callable(specs.default)
    left, right = specs.default(
        0.2,
        0.5,
        StageOperatorContext(t=0.1, y=np.zeros((2, 1)), stage="ark3-1"),
    )
    np.testing.assert_array_equal(np.asarray(right), np.eye(2))
    assert not np.array_equal(np.asarray(left), np.eye(2))


def test_advection_lifts_portable_operator_over_selected_states() -> None:
    """Typed advection uses axis coordinates and the portable upwind stencil."""
    state_names = (
        "X__imm_x0",
        "X__imm_x1",
        "X__imm_x2",
        "Y__imm_x0",
        "Y__imm_x1",
        "Y__imm_x2",
    )
    specs = compile_operator_descriptors(
        (_advection_descriptor(),),
        method="imex-euler",
        state_names=state_names,
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_coords={"imm": np.asarray([0.0, 0.5, 1.0])},
        params={},
    )

    assert callable(specs.default)
    dt = 0.2
    left, right = specs.default(
        dt,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((len(state_names), 1))),
    )
    observed = (np.eye(len(state_names)) - np.asarray(left)) / dt
    expected = np.zeros_like(observed)
    expected[:3, :3] = build_advection_matrix(
        3,
        0.5,
        2.0,
        bc="periodic",
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(right), np.eye(len(state_names)))


def test_diffusion_lifts_portable_operator_over_selected_states() -> None:
    """Typed diffusion uses axis coordinates and the portable Laplacian."""
    state_names = (
        "X__imm_x0",
        "X__imm_x1",
        "X__imm_x2",
        "Y__imm_x0",
        "Y__imm_x1",
        "Y__imm_x2",
    )
    specs = compile_operator_descriptors(
        (_diffusion_descriptor(),),
        method="imex-euler",
        state_names=state_names,
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_coords={"imm": np.asarray([0.0, 0.5, 1.0])},
        params={},
    )

    assert callable(specs.default)
    dt = 0.2
    left, right = specs.default(
        dt,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((len(state_names), 1))),
    )
    observed = (np.eye(len(state_names)) - np.asarray(left)) / dt
    expected = np.zeros_like(observed)
    expected[:3, :3] = build_diffusion_matrix(
        3,
        0.5,
        0.3,
        bc="neumann",
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(right), np.eye(len(state_names)))


def test_advection_rejects_nonuniform_axis_coordinates() -> None:
    """The current finite-volume compiler reports its uniform-grid boundary."""
    with pytest.raises(ValueError, match="must be uniformly spaced"):
        compile_operator_descriptors(
            (_advection_descriptor(),),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1", "x2")},
            axis_coords={"imm": np.asarray([0.0, 0.5, 1.5])},
            params={},
        )


def test_generator_uses_shared_op_system_validation() -> None:
    """Invalid row sums are rejected before an IMEX factory is constructed."""
    descriptor = _generator_descriptor()

    with pytest.raises(ValueError, match="generator rows must sum to zero"):
        compile_operator_descriptors(
            (descriptor,),
            method="imex-heun-tr",
            state_names=("X__imm_x0", "X__imm_x1"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1")},
            params={"G": np.ones((2, 2))},
        )
