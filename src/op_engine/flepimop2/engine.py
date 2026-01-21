# src/op_engine/flepimop2/engine.py
"""flepimop2 Engine adapter backed by op_engine.CoreSolver.

This module provides a flepimop2-compatible Engine implementation that runs
op_engine explicit methods ("euler", "heun") and guards IMEX methods until
operator specifications are supported by the adapter configuration.

Contract:
- Accepts a flepimop2 System stepper: stepper(t, state_1d, **params) -> dstate/dt (1D)
- Accepts 1D eval-times and 1D initial-state arrays from flepimop2
- Internally uses ModelCore with state_shape (n_states, 1)
- Returns a (T, 1 + n_states) float64 array with time in the first column
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

import numpy as np
from pydantic import Field

from op_engine.core_solver import CoreSolver
from op_engine.flepimop2.config import OpEngineEngineConfig
from op_engine.flepimop2.errors import (
    OptionalDependencyMissingError,
    raise_unsupported_imex,
    require_flepimop2,
)
from op_engine.flepimop2.types import (
    Float64Array,
    Float64Array2D,
    IdentifierString,
    as_float64_1d_times,
    as_float64_state,
    ensure_strictly_increasing_times,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from flepimop2.configuration import ModuleModel
    from flepimop2.engine.abc import EngineABC
    from flepimop2.system.abc import SystemABC, SystemProtocol
    from numpy.typing import NDArray

    EngineModelType: TypeAlias = type["OpEngineFlepimop2Engine"]
else:
    EngineModelType: TypeAlias = type[object]


_FLEPIMOP2_IMPORT_MSG: Final[str] = (
    "op_engine.flepimop2.engine requires the 'flepimop2' extra.\n\n"
    "Install with:\n"
    "  pip install 'op_engine[flepimop2]'\n"
    "or, if you are using uv:\n"
    "  uv pip install '.[flepimop2]'"
)

_RHS_BAD_SHAPE_MSG: Final[str] = (
    "RHS received unexpected state shape {actual}; expected {expected}."
)

_STEPPER_BAD_SHAPE_MSG: Final[str] = (
    "Stepper returned shape {actual}; expected {expected}."
)

_STATE_ARRAY_MISSING_MSG: Final[str] = (
    "ModelCore does not expose state_array; store_history must be enabled."
)

_STATE_ARRAY_BAD_SHAPE_MSG: Final[str] = (
    "Unexpected state shape {actual}; expected (T, {n_state}, 1) or (T, {n_state})."
)

_SYSTEM_TYPE_MSG: Final[str] = "system must be a flepimop2 SystemABC instance."
_STEPPER_TYPE_MSG: Final[str] = "system stepper must satisfy flepimop2 SystemProtocol."

_MODEL_DUMP_MSG: Final[str] = "config.model_dump() must return a dict[str, object]."
_MODEL_VALIDATE_MSG: Final[str] = "engine class does not provide model_validate()."

_ENGINE_CLASS_CACHE: EngineModelType | None = None


def _require_flepimop2_runtime() -> tuple[
    type[object], type[object], type[object], type[object]
]:
    """Import flepimop2 runtime classes.

    Returns:
        Tuple of (ModuleModel, EngineABC, SystemABC, SystemProtocol) as runtime types.
    """
    require_flepimop2()

    from flepimop2.configuration import ModuleModel as _ModuleModel  # noqa: PLC0415
    from flepimop2.engine.abc import EngineABC as _EngineABC  # noqa: PLC0415
    from flepimop2.system.abc import SystemABC as _SystemABC  # noqa: PLC0415
    from flepimop2.system.abc import (  # noqa: PLC0415
        SystemProtocol as _SystemProtocol,
    )

    return (
        cast("type[object]", _ModuleModel),
        cast("type[object]", _EngineABC),
        cast("type[object]", _SystemABC),
        cast("type[object]", _SystemProtocol),
    )


def _rhs_from_stepper(
    stepper: SystemProtocol,
    *,
    params: Mapping[IdentifierString, object],
    n_state: int,
) -> Callable[[float, NDArray[np.floating]], NDArray[np.floating]]:
    """Wrap a flepimop2 stepper into an op_engine RHS function.

    The op_engine CoreSolver operates on ModelCore state tensors. For this adapter,
    ModelCore uses state_shape == (n_state, 1), while flepimop2 steppers are
    expected to consume and produce 1D arrays of length n_state.

    Args:
        stepper: flepimop2 system stepper (t, state_1d, **params) -> dstate/dt (1D).
        params: Parameter mapping passed through to the stepper.
        n_state: Flattened state length.

    Returns:
        Callable rhs(t, y) -> dy with dy having shape (n_state, 1).
    """

    def rhs(t: float, y: NDArray[np.floating]) -> NDArray[np.floating]:
        y_arr = np.asarray(y, dtype=np.float64)

        expected_2d = (n_state, 1)
        if y_arr.shape != expected_2d:
            if y_arr.shape == (n_state,):
                y_arr = y_arr.reshape(expected_2d)
            else:
                msg = _RHS_BAD_SHAPE_MSG.format(
                    actual=y_arr.shape, expected=expected_2d
                )
                raise ValueError(msg)

        y1d = y_arr[:, 0]
        out1d = stepper(np.float64(t), y1d, **dict(params))
        out1d_arr = np.asarray(out1d, dtype=np.float64)

        expected_1d = (n_state,)
        if out1d_arr.shape != expected_1d:
            msg = _STEPPER_BAD_SHAPE_MSG.format(
                actual=out1d_arr.shape, expected=expected_1d
            )
            raise ValueError(msg)

        return out1d_arr.reshape(expected_2d)

    return rhs


def _extract_states_2d(core: ModelCore, *, n_state: int) -> Float64Array2D:
    """Extract stored trajectory from ModelCore as (T, n_state) float64 array.

    Args:
        core: ModelCore instance after solver.run.
        n_state: Expected flattened state length.

    Returns:
        Array of shape (T, n_state) containing the stored state history.

    Raises:
        RuntimeError: If history is unavailable or has an unexpected shape.
    """
    state_array = getattr(core, "state_array", None)
    if state_array is None:
        raise RuntimeError(_STATE_ARRAY_MISSING_MSG)

    arr = np.asarray(state_array, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[1] == n_state and arr.shape[2] == 1:
        return arr[:, :, 0]

    if arr.ndim == 2 and arr.shape[1] == n_state:
        return arr

    msg = _STATE_ARRAY_BAD_SHAPE_MSG.format(actual=arr.shape, n_state=n_state)
    raise RuntimeError(msg)


def _make_core(times: Float64Array, y0: Float64Array) -> ModelCore:
    """Construct a ModelCore with history enabled and a singleton subgroup axis.

    Args:
        times: 1D strictly increasing time grid.
        y0: 1D initial state vector.

    Returns:
        Constructed ModelCore with history enabled and initial state set.
    """
    n_states = int(y0.size)
    n_subgroups = 1

    opts = ModelCoreOptions(
        other_axes=(),
        store_history=True,
        dtype=np.float64,
    )

    core = ModelCore(
        n_states,
        n_subgroups,
        np.asarray(times, dtype=np.float64),
        options=opts,
    )

    core.set_initial_state(np.asarray(y0, dtype=np.float64).reshape(n_states, 1))
    return core


if TYPE_CHECKING:

    class OpEngineFlepimop2Engine(ModuleModel, EngineABC):
        """Static typing view for mypy (runtime class is built lazily)."""

        module: Literal["op_engine.flepimop2.engine"]
        config: OpEngineEngineConfig

        def run(
            self,
            system: SystemABC,
            eval_times: NDArray[np.float64],
            initial_state: NDArray[np.float64],
            params: dict[IdentifierString, object],
            **kwargs: object,
        ) -> NDArray[np.float64]:
            """Run the system over eval_times and return the trajectory output."""
            ...

else:
    OpEngineFlepimop2Engine = object  # sentinel; replaced by runtime factory


def _engine_class() -> EngineModelType:
    """Return the cached Engine class, defining it lazily when dependencies exist.

    Raises:
        OptionalDependencyMissingError: If flepimop2 is not available.
        AssertionError: If invoked under TYPE_CHECKING (should be unreachable).

    Returns:
        Engine class object that implements flepimop2's EngineABC protocol.
    """
    global _ENGINE_CLASS_CACHE  # noqa: PLW0603

    if _ENGINE_CLASS_CACHE is not None:
        return _ENGINE_CLASS_CACHE

    if TYPE_CHECKING:
        raise AssertionError

    try:
        module_model_t, engine_abc_t, system_abc_t, system_protocol_t = (
            _require_flepimop2_runtime()
        )
    except OptionalDependencyMissingError as exc:  # pragma: no cover
        raise OptionalDependencyMissingError(_FLEPIMOP2_IMPORT_MSG) from exc

    module_model_base = module_model_t
    engine_abc_base = engine_abc_t
    system_abc_base = system_abc_t
    system_protocol_base = system_protocol_t

    class _RuntimeOpEngineFlepimop2Engine(module_model_base, engine_abc_base):
        """flepimop2 engine that runs op_engine.CoreSolver."""

        module: Literal["op_engine.flepimop2.engine"] = "op_engine.flepimop2.engine"
        config: OpEngineEngineConfig = Field(default_factory=OpEngineEngineConfig)

        def run(
            self,
            system: object,
            eval_times: object,
            initial_state: object,
            params: dict[IdentifierString, object],
            **kwargs: object,
        ) -> np.ndarray:
            """Run the flepimop2 system via op_engine's CoreSolver.

            Args:
                system: flepimop2 SystemABC instance.
                eval_times: 1D time grid.
                initial_state: 1D initial state vector.
                params: Parameter mapping forwarded to the system stepper.
                **kwargs: Reserved for future flepimop2 compatibility.

            Returns:
                (T, 1 + n_state) float64 array: first column is time, remaining columns
                are the state trajectory.

            Raises:
                TypeError: If system/stepper do not satisfy the expected protocols.
            """
            del kwargs  # reserved for future flepimop2 compatibility

            require_flepimop2()

            times = as_float64_1d_times(eval_times, name="eval_times")
            ensure_strictly_increasing_times(times, name="eval_times")

            y0 = as_float64_state(initial_state, name="initial_state")
            n_state = int(y0.size)

            method = str(self.config.method)
            if method.startswith("imex-"):
                raise_unsupported_imex(
                    method,
                    reason=(
                        "Operator specifications are not yet supported in the "
                        "op_engine.flepimop2 engine config. Provide an explicit method "
                        "('euler' or 'heun') for now."
                    ),
                )

            if not isinstance(system, system_abc_base):
                raise TypeError(_SYSTEM_TYPE_MSG)

            stepper_obj = system._stepper  # noqa: SLF001
            if not isinstance(stepper_obj, system_protocol_base):
                raise TypeError(_STEPPER_TYPE_MSG)
            stepper = stepper_obj

            rhs = _rhs_from_stepper(stepper, params=params, n_state=n_state)

            core = _make_core(times, y0)
            solver = CoreSolver(core, operators=None, operator_axis="state")

            run_cfg = self.config.to_run_config()
            solver.run(rhs, config=run_cfg)

            states = _extract_states_2d(core, n_state=n_state)
            out = np.column_stack((np.asarray(times, dtype=np.float64), states))
            return np.asarray(out, dtype=np.float64)

    _ENGINE_CLASS_CACHE = cast("EngineModelType", _RuntimeOpEngineFlepimop2Engine)
    return _ENGINE_CLASS_CACHE


def build(config: dict[str, object] | object) -> object:
    """Build an OpEngineFlepimop2Engine from a config dict (flepimop2 convention).

    Args:
        config: Configuration mapping (or flepimop2 ModuleModel) for the engine.

    Returns:
        Constructed OpEngineFlepimop2Engine instance.

    Raises:
        TypeError: If config cannot be converted to a dict or engine lacks validation.
    """
    require_flepimop2()

    engine_cls = _engine_class()

    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            cfg = cast("dict[str, object]", dumped)
        else:
            raise TypeError(_MODEL_DUMP_MSG)
    else:
        cfg = cast("dict[str, object]", config)

    model_validate = getattr(engine_cls, "model_validate", None)
    if not callable(model_validate):
        raise TypeError(_MODEL_VALIDATE_MSG)

    return model_validate(cfg)
