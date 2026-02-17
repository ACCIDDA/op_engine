"""flepimop2 Engine integration for op_engine (thin, single-file)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from flepimop2.configuration import IdentifierString, ModuleModel
from flepimop2.engine.abc import EngineABC
from flepimop2.exceptions import ValidationIssue
from flepimop2.system.abc import SystemABC, SystemProtocol
from pydantic import BaseModel, ConfigDict, Field, model_validator

from op_engine.core_solver import (
    AdaptiveConfig,
    CoreSolver,
    DtControllerConfig,
    OperatorSpecs,
    RunConfig,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable

StateChange = Literal["delta", "flow", "state"]


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


def _has_operator_specs(specs: OperatorSpecs | None) -> bool:
    if specs is None:
        return False
    return any(getattr(specs, key) is not None for key in ("default", "tr", "bdf2"))


def _coerce_operator_specs(specs: object) -> OperatorSpecs | None:
    if isinstance(specs, OperatorSpecs):
        return specs
    if isinstance(specs, dict):
        return OperatorSpecs(
            default=specs.get("default"),
            tr=specs.get("tr"),
            bdf2=specs.get("bdf2"),
        )
    return None


def _option(module: object, name: str) -> object | None:
    getter = getattr(module, "option", None)
    if callable(getter):
        return cast("object | None", getter(name, None))
    return None


def _rhs_from_stepper(
    stepper: SystemProtocol,
    *,
    params: dict[IdentifierString, object],
    n_state: int,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float64)
        if state_arr.shape == (n_state,):
            state_arr = state_arr.reshape((n_state, 1))
        expected_shape = (n_state, 1)
        if state_arr.shape != expected_shape:
            msg = (
                f"RHS received unexpected state shape {state_arr.shape}; "
                f"expected {expected_shape}."
            )
            raise ValueError(msg)
        out = np.asarray(
            stepper(np.float64(time), state_arr[:, 0], **params), dtype=np.float64
        )
        if out.shape != (n_state,):
            msg = f"Stepper returned shape {out.shape}; expected {(n_state,)}."
            raise ValueError(msg)
        return out.reshape(expected_shape)

    return rhs


def _extract_states_2d(core: ModelCore, *, n_state: int) -> np.ndarray:
    state_array = getattr(core, "state_array", None)
    if state_array is None:
        msg = "ModelCore does not expose state_array; store_history must be enabled."
        raise RuntimeError(msg)
    arr = np.asarray(state_array, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1] == n_state and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 2 and arr.shape[1] == n_state:
        return arr
    msg = (
        f"Unexpected state shape {arr.shape}; "
        f"expected (T, {n_state}, 1) or (T, {n_state})."
    )
    raise RuntimeError(msg)


def _make_core(times: np.ndarray, y0: np.ndarray) -> ModelCore:
    n_states = int(y0.size)
    core = ModelCore(
        n_states,
        1,
        np.asarray(times, dtype=np.float64),
        options=ModelCoreOptions(other_axes=(), store_history=True, dtype=np.float64),
    )
    core.set_initial_state(y0.reshape(n_states, 1))
    return core


class OpEngineEngineConfig(BaseModel):
    """Configuration schema for op_engine when used as a flepimop2 engine."""

    model_config = ConfigDict(extra="allow")

    method: Literal["euler", "heun", "imex-euler", "imex-heun-tr", "imex-trbdf2"] = (
        "heun"
    )
    adaptive: bool = False
    strict: bool = True
    rtol: float = Field(default=1e-6, ge=0.0)
    atol: float = Field(default=1e-9, ge=0.0)
    dt_min: float = Field(default=0.0, ge=0.0)
    dt_max: float = Field(default=float("inf"), gt=0.0)
    safety: float = Field(default=0.9, gt=0.0)
    fac_min: float = Field(default=0.2, gt=0.0)
    fac_max: float = Field(default=5.0, gt=0.0)
    gamma: float | None = Field(default=None, gt=0.0, lt=1.0)
    operator_axis: str | int = "state"
    operators: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_explicit_empty_operators(self) -> OpEngineEngineConfig:
        if (
            self.method.startswith("imex-")
            and self.operators is not None
            and not _has_operator_specs(_coerce_operator_specs(self.operators))
        ):
            msg = (
                f"IMEX method '{self.method}' received operators, "
                "but none were populated. Provide at least one stage "
                "or omit operators to use system options."
            )
            raise ValueError(msg)
        return self

    def to_run_config(self) -> RunConfig:
        """Convert this provider config to an op_engine `RunConfig`."""
        return RunConfig(
            method=self.method,
            adaptive=self.adaptive,
            strict=self.strict,
            adaptive_cfg=AdaptiveConfig(rtol=self.rtol, atol=self.atol),
            dt_controller=DtControllerConfig(
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                safety=self.safety,
                fac_min=self.fac_min,
                fac_max=self.fac_max,
            ),
            operators=_coerce_operator_specs(self.operators) or OperatorSpecs(),
            gamma=self.gamma,
        )


class OpEngineFlepimop2Engine(ModuleModel, EngineABC):
    """flepimop2 engine adapter backed by op_engine.CoreSolver."""

    module: Literal["flepimop2.engine.op_engine"] = "flepimop2.engine.op_engine"
    state_change: StateChange
    config: OpEngineEngineConfig = Field(default_factory=OpEngineEngineConfig)

    def validate_system(self, system: SystemABC) -> list[ValidationIssue] | None:
        """Validate system compatibility against the engine state-change mode."""
        system_state_change = getattr(system, "state_change", None)
        if str(system_state_change) != str(self.state_change):
            return [
                ValidationIssue(
                    msg=(
                        f"Engine state change type, '{self.state_change}', is not "
                        "compatible with system state change type "
                        f"'{system_state_change}'."
                    ),
                    kind="incompatible_system",
                )
            ]
        return None

    def run(
        self,
        system: SystemABC,
        eval_times: np.ndarray,
        initial_state: np.ndarray,
        params: dict[IdentifierString, object],
        **kwargs: object,
    ) -> np.ndarray:
        """Execute simulation using op_engine and return `(time, state...)` output."""
        del kwargs

        times = _as_float64_1d(eval_times, name="eval_times")
        _ensure_strictly_increasing(times, name="eval_times")
        y0 = _as_float64_1d(initial_state, name="initial_state")
        n_state = int(y0.size)

        run_cfg = self.config.to_run_config()
        is_imex = run_cfg.method.startswith("imex-")
        operators = run_cfg.operators

        if is_imex and not _has_operator_specs(operators):
            operators = (
                _coerce_operator_specs(_option(system, "operators")) or operators
            )
        run_cfg = replace(run_cfg, operators=operators)

        if is_imex and not _has_operator_specs(operators):
            msg = (
                f"IMEX method '{run_cfg.method}' requires operators from engine config "
                "or system option 'operators'."
            )
            raise ValueError(msg)

        operator_axis = self.config.operator_axis
        if operator_axis == "state":
            system_axis = _option(system, "operator_axis")
            if isinstance(system_axis, str | int):
                operator_axis = system_axis

        stepper = getattr(system, "_stepper", None)
        if not isinstance(stepper, SystemProtocol):
            msg = "system does not expose a valid flepimop2 SystemProtocol stepper"
            raise TypeError(msg)

        mixing_kernels = _option(system, "mixing_kernels")
        merged_params = {
            **(mixing_kernels if isinstance(mixing_kernels, dict) else {}),
            **params,
        }
        rhs = _rhs_from_stepper(stepper, params=merged_params, n_state=n_state)
        core = _make_core(times, y0)

        solver = CoreSolver(
            core,
            operators=operators.default if is_imex else None,
            operator_axis=operator_axis,
        )
        solver.run(rhs, config=run_cfg)

        states = _extract_states_2d(core, n_state=n_state)
        return np.asarray(np.column_stack((times, states)), dtype=np.float64)


def build(config: dict[str, Any] | ModuleModel) -> OpEngineFlepimop2Engine:
    """Build an op_engine-backed flepimop2 engine from configuration.

    Returns:
        A validated `OpEngineFlepimop2Engine` instance.
    """
    if isinstance(config, ModuleModel):
        return OpEngineFlepimop2Engine.model_validate(config.model_dump())
    return OpEngineFlepimop2Engine.model_validate(config)


__all__ = ["OpEngineEngineConfig", "OpEngineFlepimop2Engine", "build"]
