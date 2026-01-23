"""
flepimop2 Engine adapter backed by op_engine.CoreSolver.

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

from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Callable


import numpy as np
from flepimop2.configuration import IdentifierString, ModuleModel
from flepimop2.engine.abc import EngineABC
from flepimop2.system.abc import SystemABC, SystemProtocol
from pydantic import Field

from op_engine.core_solver import CoreSolver
from op_engine.model_core import ModelCore, ModelCoreOptions

from .config import OpEngineEngineConfig
from .errors import raise_unsupported_imex
from .types import (
    Float64Array,
    Float64Array2D,
    as_float64_1d_times,
    as_float64_state,
    ensure_strictly_increasing_times,
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


def _rhs_from_stepper(
    stepper: SystemProtocol,
    *,
    params: dict[IdentifierString, object],
    n_state: int,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Wrap a flepimop2 stepper into an op_engine RHS callable.

    Args:
        stepper: flepimop2 SystemProtocol stepper function.
        params: Dictionary of parameters to pass to the stepper.
        n_state: Number of state variables.

    Returns:
        Callable representing the RHS function for op_engine.
    """

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)

        expected_2d = (n_state, 1)
        if y_arr.shape != expected_2d:
            if y_arr.shape == (n_state,):
                y_arr = y_arr.reshape(expected_2d)
            else:
                msg = _RHS_BAD_SHAPE_MSG.format(
                    actual=y_arr.shape,
                    expected=expected_2d,
                )
                raise ValueError(msg)

        y1d = y_arr[:, 0]
        out1d = stepper(np.float64(t), y1d, **params)
        out_arr = np.asarray(out1d, dtype=np.float64)

        expected_1d = (n_state,)
        if out_arr.shape != expected_1d:
            msg = _STEPPER_BAD_SHAPE_MSG.format(
                actual=out_arr.shape,
                expected=expected_1d,
            )
            raise ValueError(msg)

        return out_arr.reshape(expected_2d)

    return rhs


def _extract_states_2d(core: ModelCore, *, n_state: int) -> Float64Array2D:
    """
    Extract stored trajectory from ModelCore.

    Args:
        core: ModelCore instance with stored history.
        n_state: Number of state variables.

    Raises:
        RuntimeError: If the state history is not available or has an unexpected shape.

    Returns:
        2D array of shape (T, n_state) containing the state trajectory.
    """
    state_array = getattr(core, "state_array", None)
    if state_array is None:
        raise RuntimeError(_STATE_ARRAY_MISSING_MSG)

    arr = np.asarray(state_array, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[1] == n_state and arr.shape[2] == 1:
        return arr[:, :, 0]

    if arr.ndim == 2 and arr.shape[1] == n_state:
        return arr

    msg = _STATE_ARRAY_BAD_SHAPE_MSG.format(
        actual=arr.shape,
        n_state=n_state,
    )
    raise RuntimeError(msg)


def _make_core(times: Float64Array, y0: Float64Array) -> ModelCore:
    """
    Construct ModelCore with history enabled.

    Args:
        times: 1D time evaluation grid.
        y0: 1D initial state vector.

    Returns:
        Configured ModelCore instance.
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

    core.set_initial_state(y0.reshape(n_states, 1))
    return core


class _OpEngineFlepimop2EngineImpl(ModuleModel, EngineABC):
    """
    flepimop2 engine adapter backed by op_engine.CoreSolver.

    Attributes:
        module: Fixed module identifier.
        config: op_engine adapter configuration.
    """

    module: Literal["flepimop2.engine.op_engine"] = "flepimop2.engine.op_engine"
    config: OpEngineEngineConfig = Field(default_factory=OpEngineEngineConfig)

    def run(
        self,
        system: SystemABC,
        eval_times: np.ndarray,
        initial_state: np.ndarray,
        params: dict[IdentifierString, object],
        **kwargs: object,
    ) -> np.ndarray:
        """
        Execute the system using op_engine.

        Args:
            system: flepimop2 SystemABC instance.
            eval_times: 1D evaluation time grid.
            initial_state: 1D initial state vector.
            params: Parameter mapping forwarded to the stepper.
            **kwargs: Reserved for future flepimop2 compatibility.

        Returns:
            Array of shape (T, 1 + n_state) containing time and state trajectory.

        Raises:
            TypeError: If system does not expose a valid stepper.
        """
        del kwargs

        times = as_float64_1d(eval_times, name="eval_times")
        ensure_strictly_increasing_times(times, name="eval_times")

        y0 = as_float64_state(initial_state, name="initial_state")
        n_state = int(y0.size)

        method = str(self.config.method)
        if method.startswith("imex-"):
            raise_unsupported_imex(
                method,
                reason=(
                    "Operator specifications are not yet supported in the "
                    "op_engine.flepimop2 engine adapter."
                ),
            )

        stepper = getattr(system, "_stepper", None)
        if not isinstance(stepper, SystemProtocol):
            msg = "system does not expose a valid flepimop2 SystemProtocol stepper"
            raise TypeError(msg)

        rhs = _rhs_from_stepper(stepper, params=params, n_state=n_state)

        core = _make_core(times, y0)
        solver = CoreSolver(core, operators=None, operator_axis="state")

        run_cfg = self.config.to_run_config()
        solver.run(rhs, config=run_cfg)

        states = _extract_states_2d(core, n_state=n_state)

        out = np.column_stack((times, states))
        return np.asarray(out, dtype=np.float64)
