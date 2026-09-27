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

from flepimop2.engine.op_engine.explicit_operators import (
    compile_structured_operator_drift,
)
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


def _advection_descriptor(
    *,
    direction: str | None = None,
    bc: str = "periodic",
) -> OperatorDescriptor:
    """Build a typed advection descriptor for the test immune axis.

    Returns:
        An advection descriptor for a periodic uniform grid.
    """
    return OperatorDescriptor(
        axis="imm",
        kind="advection",
        velocity=2.0,
        bc=bc,
        direction=direction,
        apply_to=("X[imm]",),
    )


def _diffusion_descriptor(
    rate: str | float = 0.3,
) -> OperatorDescriptor:
    """Build a typed diffusion descriptor for the test immune axis.

    Args:
        rate: Literal diffusivity or parameter name.

    Returns:
        A diffusion descriptor with no-flux boundaries.
    """
    return OperatorDescriptor(
        axis="imm",
        kind="diffusion",
        rate=rate,
        bc="neumann",
        apply_to=("X[imm]",),
    )


def _jump_descriptor(
    *,
    rate: str | float = "nu",
    direction: str = "both",
    apply_to: tuple[str, ...] | None = ("X[imm]",),
    kernel: dict[str, object] | None = None,
) -> OperatorDescriptor:
    """Build a typed conservative jump-integral descriptor.

    Args:
        rate: Literal jump-rate multiplier or parameter name.
        direction: Coordinate-order direction mask.
        apply_to: Optional state-template selector.
        kernel: Optional forged kernel metadata for rejection tests.

    Returns:
        A jump-integral descriptor for the test immune axis.
    """
    return OperatorDescriptor(
        axis="imm",
        kind="jump_integral",
        rate=rate,
        bc="reflecting",
        direction=direction,
        kernel=kernel
        or {
            "form": "matrix",
            "params": {"matrix": "J"},
            "param_axes": {"J": ["imm", "imm"]},
        },
        apply_to=apply_to,
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


@pytest.mark.parametrize(
    ("direction", "generator"),
    [
        (
            "up",
            np.asarray([[-5.0, 2.0, 3.0], [0.0, -7.0, 7.0], [0.0, 0.0, 0.0]]),
        ),
        (
            "down",
            np.asarray([[0.0, 0.0, 0.0], [5.0, -5.0, 0.0], [11.0, 13.0, -24.0]]),
        ),
        (
            "both",
            np.asarray([[-5.0, 2.0, 3.0], [5.0, -12.0, 7.0], [11.0, 13.0, -24.0]]),
        ),
    ],
)
def test_jump_integral_orientation_direction_and_conservation(
    direction: str,
    generator: np.ndarray,
) -> None:
    """Provider matrices match row-source semantics after column-state lifting."""
    kernel = np.asarray([[0.0, 2.0, 3.0], [5.0, 0.0, 7.0], [11.0, 13.0, 0.0]])
    specs = compile_operator_descriptors(
        (_jump_descriptor(direction=direction),),
        method="imex-euler",
        state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_types={"imm": "ordinal"},
        params={"J": kernel, "nu": 0.4},
    )

    assert callable(specs.default)
    dt = 0.2
    left, right = specs.default(
        dt,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((3, 1))),
    )
    observed = (np.eye(3) - np.asarray(left)) / dt

    np.testing.assert_allclose(observed, 0.4 * generator.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(observed.sum(axis=0), 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(right), np.eye(3))


def test_jump_integral_reuses_multi_axis_lifting_and_apply_to() -> None:
    """A selected jump template gets one independent block per other-axis slice."""
    state_names = (
        "X__age_young__imm_x0",
        "X__age_young__imm_x1",
        "X__age_old__imm_x0",
        "X__age_old__imm_x1",
        "Y__age_young__imm_x0",
        "Y__age_young__imm_x1",
    )
    kernel = np.asarray([[0.0, 1.0], [3.0, 0.0]])
    specs = compile_operator_descriptors(
        (_jump_descriptor(rate=0.5, apply_to=("X[age, imm]",)),),
        method="imex-euler",
        state_names=state_names,
        axis_order=("state", "subgroup", "age", "imm"),
        axis_labels={"age": ("young", "old"), "imm": ("x0", "x1")},
        axis_types={"age": "categorical", "imm": "ordinal"},
        params={"J": kernel},
    )

    assert callable(specs.default)
    left, _right = specs.default(
        0.2,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((len(state_names), 1))),
    )
    observed = (np.eye(len(state_names)) - np.asarray(left)) / 0.2
    block = 0.5 * np.asarray([[-1.0, 1.0], [3.0, -3.0]]).T
    expected = np.zeros_like(observed)
    expected[0:2, 0:2] = block
    expected[2:4, 2:4] = block

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)


def test_continuous_jump_integral_applies_target_trapezoidal_weights() -> None:
    """Non-uniform continuous coordinates weight destination columns exactly."""
    coordinates = np.asarray([0.0, 0.4, 1.0])
    weights = np.asarray([0.2, 0.5, 0.3])
    kernel = np.asarray([[0.0, 2.0, 3.0], [5.0, 0.0, 7.0], [11.0, 13.0, 0.0]])
    specs = compile_operator_descriptors(
        (_jump_descriptor(rate=1.0),),
        method="imex-euler",
        state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_coords={"imm": coordinates},
        axis_types={"imm": "continuous"},
        params={"J": kernel},
    )

    assert callable(specs.default)
    left, _right = specs.default(
        0.2,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((3, 1))),
    )
    observed = (np.eye(3) - np.asarray(left)) / 0.2
    rates = kernel * weights[np.newaxis, :]
    expected_generator = rates - np.diag(rates.sum(axis=1))

    np.testing.assert_allclose(
        observed,
        expected_generator.T,
        rtol=0.0,
        atol=1e-14,
    )


def test_structured_jump_integral_uses_the_same_axis_local_operator() -> None:
    """Explicit PyTree execution applies the shared row-source jump matrix."""
    state = {"X": np.asarray([1.0, 2.0, 4.0])}
    kernel = np.asarray([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    drift = compile_structured_operator_drift(
        (_jump_descriptor(rate=0.5, direction="up"),),
        state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_types={"imm": "ordinal"},
        params={"J": kernel},
        reference=state,
    )
    generator = np.asarray([[-2.0, 2.0, 0.0], [0.0, -3.0, 3.0], [0.0, 0.0, 0.0]])

    np.testing.assert_allclose(
        drift(state)["X"],
        0.5 * state["X"] @ generator,
        rtol=0.0,
        atol=1e-14,
    )


def test_jump_integral_rejects_unsupported_form() -> None:
    """A manually forged non-matrix descriptor fails closed in the provider."""
    descriptor = _jump_descriptor(
        kernel={"form": "gaussian", "params": {"matrix": "J"}}
    )
    with pytest.raises(ValueError, match=r"kernel\.form='matrix'"):
        compile_operator_descriptors(
            (descriptor,),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1")},
            axis_types={"imm": "ordinal"},
            params={"J": np.zeros((2, 2)), "nu": 1.0},
        )


def test_jump_integral_requires_axis_type_metadata() -> None:
    """Numeric-looking coordinates never substitute for a declared axis type."""
    with pytest.raises(KeyError, match="no axis_types metadata"):
        compile_operator_descriptors(
            (_jump_descriptor(),),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1")},
            axis_coords={"imm": np.asarray([0.0, 1.0])},
            params={"J": np.zeros((2, 2)), "nu": 1.0},
        )


def test_continuous_jump_integral_requires_coordinate_mapping() -> None:
    """Continuous quadrature refuses incomplete provider geometry."""
    with pytest.raises(KeyError, match="no axis_coords metadata"):
        compile_operator_descriptors(
            (_jump_descriptor(),),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1")},
            axis_types={"imm": "continuous"},
            params={"J": np.zeros((2, 2)), "nu": 1.0},
        )


def test_jump_integral_uses_shared_eager_value_validation() -> None:
    """Invalid concrete kernel signs are rejected before factory construction."""
    with pytest.raises(ValueError, match="nonnegative"):
        compile_operator_descriptors(
            (_jump_descriptor(),),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1")},
            axis_types={"imm": "ordinal"},
            params={"J": np.asarray([[0.0, -1.0], [0.0, 0.0]]), "nu": 1.0},
        )


def test_jump_integral_rate_and_kernel_are_jittable_and_differentiable() -> None:
    """Array-API jump assembly keeps both numerical parameters dynamic."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    descriptor = _jump_descriptor(direction="up")
    kernel_template = jnp.asarray(
        [[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    state = jnp.zeros((3, 1), dtype=jnp.float32)

    def objective(rate: object, kernel_scale: object) -> object:
        specs = compile_operator_descriptors(
            (descriptor,),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1", "x2")},
            axis_types={"imm": "ordinal"},
            params={"J": kernel_template * kernel_scale, "nu": rate},
            reference=state,
        )
        assert callable(specs.default)
        left, _right = specs.default(
            0.2,
            1.0,
            StageOperatorContext(t=0.0, y=state),
        )
        change = jnp.eye(3, dtype=state.dtype) - left
        return jnp.sum(change * change)

    rate = jnp.asarray(0.4, dtype=jnp.float32)
    kernel_scale = jnp.asarray(1.3, dtype=jnp.float32)
    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        rate,
        kernel_scale,
    )

    assert gradients[0] == pytest.approx(2.0 * float(value) / float(rate), rel=1e-6)
    assert gradients[1] == pytest.approx(
        2.0 * float(value) / float(kernel_scale),
        rel=1e-6,
    )


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


@pytest.mark.parametrize(
    ("direction", "resolved_velocity"),
    [(None, 2.0), ("increasing", 2.0), ("decreasing", -2.0)],
)
def test_advection_resolves_explicit_direction(
    direction: str | None,
    resolved_velocity: float,
) -> None:
    """Direction orients a non-negative coefficient before stencil assembly."""
    state_names = ("X__imm_x0", "X__imm_x1", "X__imm_x2")
    specs = compile_operator_descriptors(
        (_advection_descriptor(direction=direction, bc="reflecting"),),
        method="imex-euler",
        state_names=state_names,
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_coords={"imm": np.asarray([0.0, 0.5, 1.0])},
        params={},
    )

    assert callable(specs.default)
    dt = 0.2
    left, _right = specs.default(
        dt,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((len(state_names), 1))),
    )
    observed = (np.eye(len(state_names)) - np.asarray(left)) / dt
    expected = np.asarray(
        build_advection_matrix(
            3,
            0.5,
            resolved_velocity,
            bc="reflecting",
        )
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)


def test_advection_rejects_unknown_direction() -> None:
    """Provider compilation fails closed for a manually forged descriptor."""
    with pytest.raises(ValueError, match="advection direction must be"):
        compile_operator_descriptors(
            (_advection_descriptor(direction="sideways"),),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1", "x2")},
            axis_coords={"imm": np.asarray([0.0, 0.5, 1.0])},
            params={},
        )


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


def test_diffusion_lifts_nonuniform_axis_coordinates() -> None:
    """Typed IMEX diffusion accepts monotone non-uniform cell centers."""
    coordinates = np.asarray([0.0, 0.25, 1.0])
    state_names = ("X__imm_x0", "X__imm_x1", "X__imm_x2")
    specs = compile_operator_descriptors(
        (_diffusion_descriptor(),),
        method="imex-euler",
        state_names=state_names,
        axis_order=("state", "subgroup", "imm"),
        axis_labels={"imm": ("x0", "x1", "x2")},
        axis_coords={"imm": coordinates},
        params={},
    )

    assert callable(specs.default)
    dt = 0.2
    left, _right = specs.default(
        dt,
        1.0,
        StageOperatorContext(t=0.0, y=np.zeros((len(state_names), 1))),
    )
    observed = (np.eye(len(state_names)) - np.asarray(left)) / dt
    expected = build_diffusion_matrix(
        3,
        None,
        0.3,
        grid=coordinates,
        bc="neumann",
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)


def test_nonuniform_diffusion_imex_preserves_jax_gradient() -> None:
    """Flat Array-API operator assembly keeps diffusivity traceable."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    coordinates = np.asarray([0.0, 0.25, 1.0])
    state = jnp.asarray([[1.0], [2.0], [4.0]], dtype=jnp.float32)

    def objective(rate: object) -> object:
        specs = compile_operator_descriptors(
            (_diffusion_descriptor("D"),),
            method="imex-euler",
            state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1", "x2")},
            axis_coords={"imm": coordinates},
            params={"D": rate},
            reference=state,
        )
        assert callable(specs.default)
        left, _right = specs.default(
            0.2,
            1.0,
            StageOperatorContext(t=0.0, y=state),
        )
        change = jnp.eye(3, dtype=state.dtype) - left
        return jnp.sum(change * change)

    rate = jnp.asarray(0.3, dtype=jnp.float32)
    value, gradient = jax.jit(jax.value_and_grad(objective))(rate)

    assert gradient == pytest.approx(2.0 * float(value) / float(rate), rel=1e-6)


def test_nonuniform_structured_diffusion_preserves_jax_gradient() -> None:
    """Structured explicit drift shares the non-uniform JAX operator path."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    coordinates = np.asarray([0.0, 0.25, 1.0])
    state = {"X": jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float32)}

    def objective(rate: object) -> object:
        drift = compile_structured_operator_drift(
            (_diffusion_descriptor("D"),),
            state_names=("X__imm_x0", "X__imm_x1", "X__imm_x2"),
            axis_order=("state", "subgroup", "imm"),
            axis_labels={"imm": ("x0", "x1", "x2")},
            axis_coords={"imm": coordinates},
            params={"D": rate},
            reference=state,
        )
        tendency = drift(state)["X"]
        return jnp.sum(tendency * tendency)

    rate = jnp.asarray(0.3, dtype=jnp.float32)
    value, gradient = jax.jit(jax.value_and_grad(objective))(rate)

    assert gradient == pytest.approx(2.0 * float(value) / float(rate), rel=1e-6)


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
