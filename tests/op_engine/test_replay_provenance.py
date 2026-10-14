"""Separate mesh validity, structural compatibility, and accuracy-driven refresh."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from runpy import run_path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from op_engine import (
    AdaptiveScheduleProvenance,
    Array,
    CoreSolver,
    DenseNewtonSolver,
    ModelCore,
    NewtonConfig,
    NonlinearMethodConfig,
    array_namespace,
)
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    OperatorSpecs,
    RunConfig,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from op_engine.core_solver import RHSFunction

INITIAL = np.asarray([[0.8, 0.6], [0.2, 0.4]])
LOCAL = np.asarray([1.0, 1.3])


def _transfer(rate: Any) -> RHSFunction:  # noqa: ANN401
    def rhs(_time: object, state: Array) -> Array:
        xp = array_namespace(state)
        flow = rate * xp.asarray(LOCAL, dtype=state.dtype) * state[0] ** 2
        return cast("Array", xp.stack((-flow, flow)))

    return rhs


def _core(
    initial: Array | None = None, *, axes: tuple[str, str] = ("pool", "group")
) -> ModelCore:
    core = ModelCore(
        2, 2, np.asarray([0.0, 1.0]), options=ModelCoreOptions(axis_names=axes)
    )
    core.set_initial_state(INITIAL if initial is None else initial)
    return core


def _config() -> RunConfig:
    return RunConfig(
        method="heun",
        adaptive=True,
        adaptive_cfg=AdaptiveConfig(rtol=1e-4, atol=1e-7, dt_init=0.5),
        replay_model_id="quadratic-transfer:pool,group:v1",
        replay_reference_regime="rate=.05; local=[1,1.3]; A0=[.8,.6]",
    )


def _record() -> tuple[AdaptiveStepSchedule, RunConfig]:
    config = _config()
    solver = CoreSolver(_core())
    solver.run(_transfer(0.05), config=config)
    assert solver.last_adaptive_schedule is not None
    return solver.last_adaptive_schedule, config


def _reference(rate: float, initial: np.ndarray = INITIAL) -> np.ndarray:
    # A'=-k*A^2, B'=+k*A^2 gives A(t)=A0/(1+k*A0*t), A+B constant.
    a = initial[0] / (1 + rate * LOCAL * initial[0])
    return np.stack((a, initial.sum(axis=0) - a))


def test_live_mesh_captures_descriptive_and_structural_metadata() -> None:
    """A live mesh identifies its method, layout, and nominal regime."""
    schedule, config = _record()
    provenance = schedule.provenance
    assert isinstance(provenance, AdaptiveScheduleProvenance)
    assert provenance.method == "heun"
    assert provenance.state_shape == INITIAL.shape
    assert provenance.axis_names == ("pool", "group")
    assert provenance.operator_axis is None
    assert provenance.model_id == config.replay_model_id
    assert provenance.reference_regime == config.replay_reference_regime
    assert dict(provenance.discovery_settings)["rtol"] == "0.0001"
    assert "numpy" in provenance.array_namespace
    assert provenance.dtype == "float64"


@pytest.mark.parametrize(
    ("field", "change"),
    [
        ("method", {"method": "rk4"}),
        ("model_id", {"replay_model_id": "different-equations:v2"}),
    ],
)
def test_known_method_and_model_mismatch_reject_before_advancing(
    field: str, change: dict[str, Any]
) -> None:
    """Known structural changes fail while retaining the untouched initial state."""
    schedule, config = _record()
    solver = CoreSolver(_core())
    with pytest.raises(ValueError, match=f"provenance {field} mismatch"):
        solver.replay_adaptive_schedule(
            _transfer(0.05), schedule, config=replace(config, **change)
        )
    assert solver.core.current_step == 0
    np.testing.assert_array_equal(solver.core.get_current_state(), INITIAL)


@pytest.mark.parametrize("layout", ["shape", "axis_names"])
def test_state_layout_mismatch_rejects(layout: str) -> None:
    """Cell shape and named axis order are structural compatibility constraints."""
    schedule, config = _record()
    core = (
        _core(axes=("state", "group"))
        if layout == "axis_names"
        else ModelCore(2, 3, np.asarray([0.0, 1.0]))
    )
    field = "axis_names" if layout == "axis_names" else "state_shape"
    with pytest.raises(
        ValueError,
        match=f"provenance {field} mismatch",
    ):
        CoreSolver(core).replay_adaptive_schedule(
            _transfer(0.05), schedule, config=config
        )
    assert core.current_step == 0


def test_legacy_mesh_is_explicitly_unverified_and_optionally_rejected() -> None:
    """Legacy constructors retain mesh-only behavior unless provenance is required."""
    manual = AdaptiveStepSchedule((0.0, 1.0), ((0.5, 0.5),))
    assert manual.provenance is None
    solver = CoreSolver(_core())
    solver.replay_adaptive_schedule(_transfer(0.05), manual, config=_config())
    assert solver.core.current_step == 1
    strict = CoreSolver(_core())
    with pytest.raises(ValueError, match="has no provenance"):
        strict.replay_adaptive_schedule(
            _transfer(0.05),
            manual,
            config=replace(_config(), replay_require_provenance=True),
        )
    assert strict.core.current_step == 0
    with pytest.raises(ValueError, match="sum to each output interval"):
        AdaptiveStepSchedule((0.0, 1.0), ((0.4, 0.4),))


def test_compatible_parameter_reuse_can_need_accuracy_refresh() -> None:
    """Independent analytic error checks distinguish reuse from an accuracy refresh."""
    schedule, config = _record()
    near = CoreSolver(_core())
    near.replay_adaptive_schedule(
        _transfer(0.055),
        schedule,
        config=replace(config, replay_reference_regime="nearby=.055"),
    )
    np.testing.assert_allclose(
        near.core.get_current_state(), _reference(0.055), rtol=1e-5, atol=1e-6
    )
    far = CoreSolver(_core())
    far.replay_adaptive_schedule(_transfer(3.0), schedule, config=config)
    discrepancy = np.max(np.abs(far.core.get_current_state() - _reference(3.0)))
    assert discrepancy > 1e-3
    fresh = CoreSolver(_core())
    refreshed_config = replace(config, replay_reference_regime="rate=3.0")
    fresh.run(_transfer(3.0), config=refreshed_config)
    refreshed = fresh.last_adaptive_schedule
    assert refreshed is not None
    assert len(refreshed.step_sizes[0]) > len(schedule.step_sizes[0])
    assert refreshed.provenance is not None
    assert refreshed.provenance.reference_regime == "rate=3.0"
    np.testing.assert_allclose(
        fresh.core.get_current_state(), _reference(3.0), rtol=1e-4, atol=1e-5
    )


def test_portable_jax_reuse_preserves_dynamic_state_and_tolerance_arrays() -> None:
    """Static compatibility checks preserve traced arrays across a backend change."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    schedule, config = _record()
    with enable():

        def objective(gain: Any, atol: Any) -> Any:  # noqa: ANN401
            solver = CoreSolver(_core(jnp.asarray(INITIAL) * gain))
            replay_config = replace(
                config,
                replay_loop="scan",
                adaptive_cfg=AdaptiveConfig(rtol=1e-5, atol=atol),
                replay_reference_regime="dynamic initial scale",
            )
            solver.replay_adaptive_schedule(
                _transfer(0.05), schedule, config=replay_config
            )
            result = solver.core.get_current_state()
            return jnp.sum(result**2)

        gain = jnp.asarray(1.0)
        value, derivative = jax.jit(jax.value_and_grad(objective))(
            gain, jnp.full((2, 2), 1e-8)
        )
        step = 1e-4
        expected = (
            np.sum(_reference(0.05, INITIAL * (1 + step)) ** 2)
            - np.sum(_reference(0.05, INITIAL * (1 - step)) ** 2)
        ) / (2 * step)
        assert np.isfinite(float(value))
        np.testing.assert_allclose(derivative, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("method", ["ros2", "imex-trbdf2", "sdirk2"])
def test_implicit_discovery_records_relevant_kernel_identity(method: str) -> None:
    """Resolved implicit kernels reject axis, gamma, or nonlinear iteration changes."""
    core = _core()
    xp = array_namespace(core.get_current_state())
    identity = xp.eye(2)

    def implicit(dt: float, scale: float, _context: object) -> tuple[Array, Array]:
        return cast("Array", identity * (1 + 0.2 * dt * scale)), cast("Array", identity)

    def trapezoidal(dt: float, scale: float, _context: object) -> tuple[Array, Array]:
        return cast("Array", identity * (1 + 0.1 * dt * scale)), cast(
            "Array", identity * (1 - 0.1 * dt * scale)
        )

    config = replace(
        _config(),
        method=method,
        jacobian=lambda _t, _y: cast("Array", -0.2 * identity),
        nonlinear=NonlinearMethodConfig(
            rhs_jacobian=lambda _t, _y: cast("Array", -0.2 * xp.eye(4))
        )
        if method == "sdirk2"
        else None,
        operators=OperatorSpecs(tr=trapezoidal, bdf2=implicit)
        if method == "imex-trbdf2"
        else OperatorSpecs(),
    )
    rhs = (
        (lambda _t, y: cast("Array", y - y))
        if method == "imex-trbdf2"
        else (lambda _t, y: cast("Array", -0.2 * y))
    )
    solver = CoreSolver(core, operator_axis="pool")
    solver.run(rhs, config=config)
    mesh = solver.last_adaptive_schedule
    assert mesh is not None
    assert mesh.provenance is not None
    assert mesh.provenance.method == method
    if method == "imex-trbdf2":
        assert mesh.provenance.gamma is not None
        changed = replace(config, gamma=0.7)
        field = "gamma"
    elif method == "sdirk2":
        assert config.nonlinear is not None
        changed = replace(
            config,
            nonlinear=replace(
                config.nonlinear,
                solver=DenseNewtonSolver(NewtonConfig(max_iterations=4)),
            ),
        )
        field = "nonlinear_signature"
    else:
        changed = config
        field = "operator_axis"
    target = CoreSolver(_core(), operator_axis="group" if method == "ros2" else "pool")
    with pytest.raises(ValueError, match=f"provenance {field} mismatch"):
        target.replay_adaptive_schedule(rhs, mesh, config=changed)


def test_array_implicit_discovery_records_jax_metadata() -> None:
    """The native implicit recording path attaches the same compatibility contract."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        solver = CoreSolver(_core(jnp.asarray(INITIAL)), operator_axis="pool")
        config = replace(
            _config(), method="ros2", jacobian=lambda _t, _y: -0.2 * jnp.eye(2)
        )
        solver.run(lambda _t, y: -0.2 * y, config=config)
        mesh = solver.last_adaptive_schedule
        assert mesh is not None
        assert mesh.provenance is not None
        assert mesh.provenance.array_namespace == "jax.numpy"
        assert mesh.provenance.operator_axis == 0
        np.testing.assert_allclose(
            solver.core.get_current_state(), INITIAL * np.exp(-0.2), rtol=1e-4
        )


def test_published_structured_refresh_example() -> None:
    """The runnable example makes and verifies an independent accuracy decision."""
    script = Path(__file__).resolve().parents[2] / "examples/replay_mesh_refresh.py"
    namespace = run_path(str(script))
    report = namespace["assess_and_refresh"]()
    assert report["refreshed"] is True
    assert float(report["nearby_error"]) < float(report["accuracy_limit"])
    assert float(report["old_mesh_error"]) > float(report["accuracy_limit"])
    assert float(report["selected_mesh_error"]) < float(report["accuracy_limit"])
    assert int(report["selected_steps"]) > int(report["nominal_steps"])
