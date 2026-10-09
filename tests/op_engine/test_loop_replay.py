"""Namespace-loop dispatch and numerical parity for frozen-mesh replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from op_engine import Array, CoreSolver, ModelCore, Scalar, array_namespace, loop_ops
from op_engine.core_solver import AdaptiveStepSchedule, OperatorSpecs, RunConfig
from op_engine.loop_ops import LoopAdapter, get_loop_adapter, register_loop_adapter
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


@pytest.fixture
def jax_x64() -> Iterator[None]:
    """Use scoped double precision for replay and derivative comparisons."""
    jax = pytest.importorskip("jax")
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        enable = jax.experimental.enable_x64
    with enable():
        yield


def _solve(  # noqa: PLR0913
    parameters: Array,
    schedule: AdaptiveStepSchedule,
    *,
    method: str = "dopri5",
    replay_loop: str = "scan",
    checkpoint: bool = False,
    store_history: bool = True,
) -> tuple[Array, Array | None]:
    """Replay a time-dependent linear problem with dynamic dense operators.

    Returns:
        Final state and optional output history.
    """
    xp = array_namespace(parameters)
    core = ModelCore(
        2,
        1,
        np.asarray(schedule.output_times),
        options=ModelCoreOptions(dtype=np.float64, store_history=store_history),
    )
    core.set_initial_state(xp.asarray([[1.2], [0.7]], dtype=parameters.dtype))
    rate, forcing = parameters

    def rhs(time: Scalar, state: Array) -> Array:
        return cast("Array", rate * (1.0 + time) * state + forcing * time)

    def jacobian(time: Scalar, _state: Array) -> Array:
        return cast("Array", xp.eye(2, dtype=parameters.dtype) * rate * (1.0 + time))

    def operators(dt: Scalar, scale: float, context: Any) -> tuple[Array, Array]:  # noqa: ANN401
        identity = xp.eye(2, dtype=parameters.dtype)
        return (
            cast("Array", identity * (1.0 - dt * scale * forcing * (1.0 + context.t))),
            cast("Array", identity),
        )

    config = RunConfig(
        method=method,
        adaptive=True,
        replay_loop=replay_loop,
        replay_checkpoint=checkpoint,
        jacobian=jacobian,
        operators=OperatorSpecs(default=operators)
        if method.startswith("imex-")
        else OperatorSpecs(),
    )
    solver = CoreSolver(core)
    solver.replay_adaptive_schedule(rhs, schedule, config=config)
    assert solver.last_adaptive_schedule == schedule
    assert solver.last_nonlinear_diagnostics is None
    assert core.current_step == core.n_timesteps - 1
    return core.get_current_state(), core.state_array


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize(
    "method",
    [
        "euler",
        "heun",
        "rk4",
        "dopri5",
        "implicit-euler",
        "trapezoidal",
        "ros2",
        "imex-euler",
        "imex-heun-tr",
        "imex-trbdf2",
        "imex-ark3",
    ],
)
@pytest.mark.parametrize("store_history", [False, True])
def test_scan_matches_unrolled_nonuniform_replay(
    method: str, *, store_history: bool
) -> None:
    """Every functional replay kernel agrees across drivers at output boundaries."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    schedule = AdaptiveStepSchedule(
        output_times=(0.0, 0.2, 0.6),
        step_sizes=((0.03, 0.07, 0.1), (0.1, 0.12, 0.18)),
    )
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float64)
    expected, expected_history = _solve(
        parameters,
        schedule,
        method=method,
        replay_loop="unroll",
        store_history=store_history,
    )
    actual, history = jax.jit(
        lambda values: _solve(
            values, schedule, method=method, store_history=store_history
        ),
    )(parameters)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert array_namespace(actual) is jnp
    if store_history:
        np.testing.assert_allclose(history, expected_history, rtol=1e-12, atol=1e-12)
    else:
        assert history is None


@pytest.mark.parametrize("method", ["euler", "dopri5", "ros2", "imex-heun-tr"])
def test_numpy_auto_keeps_byte_identical_eager_replay(method: str) -> None:
    """Non-JAX auto selection retains the original numerical path."""
    schedule = AdaptiveStepSchedule((0.0, 0.2), ((0.05, 0.15),))
    parameters = np.asarray([-0.3, 0.2])
    eager = _solve(parameters, schedule, method=method, replay_loop="unroll")
    automatic = _solve(parameters, schedule, method=method, replay_loop="auto")
    for actual, expected in zip(automatic, eager, strict=True):
        assert actual is not None
        assert expected is not None
        assert actual.tobytes() == expected.tobytes()


def test_torch_auto_keeps_eager_replay_when_available() -> None:
    """Torch tensors retain their values and autograd in auto mode."""
    torch = pytest.importorskip("torch")
    schedule = AdaptiveStepSchedule((0.0, 0.2), ((0.05, 0.15),))
    parameters = torch.tensor([-0.3, 0.2], dtype=torch.float64, requires_grad=True)
    eager, _ = _solve(parameters, schedule, replay_loop="unroll")
    automatic, _ = _solve(parameters, schedule, replay_loop="auto")
    assert torch.equal(automatic, eager)
    automatic.sum().backward()
    assert parameters.grad is not None
    assert torch.all(torch.isfinite(parameters.grad))


def test_forced_scan_requires_an_available_adapter() -> None:
    """A requested compiled strategy cannot silently become eager."""
    schedule = AdaptiveStepSchedule((0.0, 0.2), ((0.2,),))
    with pytest.raises(ValueError, match="requires a loop adapter"):
        _solve(np.asarray([-0.3, 0.2]), schedule)


def test_run_config_validates_loop_options() -> None:
    """Invalid iteration options fail at construction."""
    assert RunConfig().replay_loop == "unroll"
    with pytest.raises(ValueError, match="replay_loop"):
        RunConfig(replay_loop="unknown")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="replay_checkpoint"):
        RunConfig(replay_checkpoint=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("replay_loop", ["unroll", "auto", "scan"])
def test_bdf2_retains_fixed_step_only_contract(replay_loop: str) -> None:
    """A new driver does not enable unsupported adaptive BDF2 semantics."""
    schedule = AdaptiveStepSchedule((0.0, 0.2), ((0.2,),))
    with pytest.raises(ValueError, match=r"bdf2.*supports adaptive=False"):
        _solve(
            np.asarray([-0.3, 0.2]), schedule, method="bdf2", replay_loop=replay_loop
        )


def test_custom_namespace_adapter_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional backends can supply a scan without changing numerical kernels."""
    from types import ModuleType  # noqa: PLC0415

    namespace = ModuleType("op_engine_test_namespace")
    adapter = LoopAdapter(scan=lambda _body, initial, _inputs: (initial, None))
    monkeypatch.setattr(loop_ops, "_LOOP_ADAPTERS", {})
    assert get_loop_adapter(namespace) is None
    register_loop_adapter(namespace.__name__, adapter)
    assert get_loop_adapter(namespace) is adapter


def test_adapter_registration_validates_inputs() -> None:
    """Malformed adapters fail where they are registered."""
    adapter = LoopAdapter(scan=lambda _body, initial, _inputs: (initial, None))
    with pytest.raises(ValueError, match="namespace"):
        register_loop_adapter("", adapter)
    with pytest.raises(TypeError, match="LoopAdapter"):
        register_loop_adapter("numpy", object())  # type: ignore[arg-type]


def test_eager_namespace_does_not_import_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NumPy lookup never loads JAX or changes its dependency requirements."""

    def unexpected_import(_name: str) -> None:
        pytest.fail("An eager namespace attempted to import an optional backend")

    monkeypatch.setattr(loop_ops, "_LOOP_ADAPTERS", {})
    monkeypatch.setattr(loop_ops, "import_module", unexpected_import)
    assert get_loop_adapter(array_namespace(np.asarray([1.0]))) is None


def test_checkpoint_requires_adapter_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rematerialization request cannot silently lose its memory contract."""
    parameters = np.asarray([-0.3, 0.2])
    monkeypatch.setattr(loop_ops, "_LOOP_ADAPTERS", {})
    register_loop_adapter(
        array_namespace(parameters).__name__,
        LoopAdapter(scan=lambda _body, initial, _inputs: (initial, None)),
    )
    schedule = AdaptiveStepSchedule((0.0, 0.2), ((0.2,),))
    with pytest.raises(ValueError, match="does not support checkpointing"):
        _solve(parameters, schedule, checkpoint=True)


@pytest.mark.usefixtures("jax_x64")
@pytest.mark.parametrize(
    "schedule",
    [
        AdaptiveStepSchedule((0.0,), ()),
        AdaptiveStepSchedule((0.0, 0.2), ((0.2,),)),
    ],
)
def test_scan_handles_zero_or_one_step(schedule: AdaptiveStepSchedule) -> None:
    """Seeding an FSAL carry preserves empty and single-step mesh semantics."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    parameters = jnp.asarray([-0.3, 0.2], dtype=jnp.float64)
    actual, history = jax.jit(lambda values: _solve(values, schedule))(parameters)
    expected, expected_history = _solve(parameters, schedule, replay_loop="unroll")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(history, expected_history, rtol=1e-12, atol=1e-12)
