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

"""Configuration model for op_engine provider integration."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from op_engine.batched_ssa import BatchedSSAConfig
from op_engine.core_solver import (
    AdaptiveConfig,
    DtControllerConfig,
    NonlinearMethodConfig,
    OperatorSpecs,
    RunConfig,
)
from op_engine.stochastic_solver import DirectSSAConfig


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


class SolverMethod(StrEnum):
    """Solver method identifiers for op_engine integration."""

    EULER = "euler"
    HEUN = "heun"
    RK4 = "rk4"
    DOPRI5 = "dopri5"
    IMEX_EULER = "imex-euler"
    IMEX_HEUN_TR = "imex-heun-tr"
    IMEX_TRBDF2 = "imex-trbdf2"
    IMEX_ARK3 = "imex-ark3"
    IMPLICIT_EULER = "implicit-euler"
    TRAPEZOIDAL = "trapezoidal"
    BDF2 = "bdf2"
    ROS2 = "ros2"
    SDIRK2 = "sdirk2"

    @property
    def is_imex(self) -> bool:
        """Whether this method is an IMEX method."""
        return self.value.startswith("imex-")

    @property
    def is_implicit(self) -> bool:
        """Whether this method requires a Jacobian."""
        return self in _IMPLICIT_METHODS

    @property
    def is_nonlinear(self) -> bool:
        """Whether this method requires a full nonlinear solve."""
        return self is SolverMethod.SDIRK2

    @property
    def is_explicit(self) -> bool:
        """Whether this method is an explicit Runge--Kutta method."""
        return self in {
            SolverMethod.EULER,
            SolverMethod.HEUN,
            SolverMethod.RK4,
            SolverMethod.DOPRI5,
        }


_IMPLICIT_METHODS: frozenset[SolverMethod] = frozenset(
    {
        SolverMethod.IMPLICIT_EULER,
        SolverMethod.TRAPEZOIDAL,
        SolverMethod.BDF2,
        SolverMethod.ROS2,
    },
)


class ExecutionMode(StrEnum):
    """Provider execution modes."""

    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    HYBRID = "hybrid"


class StateLayout(StrEnum):
    """State representation used inside deterministic integration."""

    FLAT = "flat"
    PYTREE = "pytree"
    BLOCK = "block"


class AdaptiveReplayMode(StrEnum):
    """Execution strategy for a frozen adaptive schedule."""

    AUTO = "auto"
    UNROLLED = "unrolled"
    COMPACT = "compact"


class ReplayCheckpoint(StrEnum):
    """Rematerialization policy for compact JAX explicit execution."""

    NONE = "none"
    STEP = "step"
    CHUNK = "chunk"


class StochasticMethod(StrEnum):
    """Discrete reaction-network methods exposed by the provider."""

    ADAPTIVE_TAU_LEAPING = "adaptive-tau-leaping"
    DIRECT_SSA = "direct-ssa"
    REPLICATE_DIRECT_SSA = "replicate-direct-ssa"
    TAU_LEAPING = "tau-leaping"
    THINNING_SSA = "thinning-ssa"


class OpEngineEngineConfig(BaseModel):
    """Configuration schema for op_engine when used as a flepimop2 engine.

    Omitted fixed-loop controls retain the provider's existing JAX scans and
    eager execution on other namespaces. General scan rematerialization uses
    a boolean distinct from the legacy DOPRI5 checkpoint enum.

    Examples:
        >>> config = OpEngineEngineConfig(
        ...     method="rk4", fixed_loop="scan", fixed_scan_checkpoint=True
        ... )
        >>> restored = OpEngineEngineConfig.model_validate_json(
        ...     config.model_dump_json()
        ... )
        >>> restored.to_run_config().fixed_loop
        'scan'
        >>> restored.to_run_config().fixed_checkpoint
        True
    """

    model_config = ConfigDict(extra="allow", ser_json_inf_nan="strings")

    mode: ExecutionMode = ExecutionMode.DETERMINISTIC
    method: SolverMethod = SolverMethod.HEUN
    stochastic_method: StochasticMethod = StochasticMethod.TAU_LEAPING
    stochastic_reactions: tuple[str, ...] = ()
    random_seed: int | None = Field(default=None, ge=0)
    replicate_axis: str | None = Field(default=None, min_length=1)
    replicate_breakpoints: tuple[tuple[float, ...], ...] = ()
    tau_max_step: float | None = Field(default=None, gt=0.0)
    stochastic_max_steps: int = Field(default=1_000_000, ge=1)
    tau_leap_tolerance: float = Field(
        default=0.03,
        gt=0.0,
        lt=1.0,
        allow_inf_nan=False,
    )
    tau_critical_threshold: int = Field(default=10, ge=0)
    tau_exact_fallback_multiplier: float = Field(
        default=10.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    tau_max_retries: int = Field(default=20, ge=0)
    ssa_max_events: int = Field(default=1_000_000, ge=1)
    thinning_rate_bound: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
    )
    thinning_max_candidates: int = Field(default=1_000_000, ge=1, strict=True)
    forcing_breakpoints: tuple[float, ...] = ()
    state_layout: StateLayout = StateLayout.FLAT
    block_axis: str | None = None
    block_schedule: Literal["shared", "independent"] = "shared"
    block_batching: Literal["length", "serial"] = "length"
    block_batch_size: int | None = Field(default=None, ge=1, strict=True)
    adaptive: bool = False
    adaptive_replay: AdaptiveReplayMode = AdaptiveReplayMode.AUTO
    replay_checkpoint: ReplayCheckpoint = ReplayCheckpoint.NONE
    fixed_checkpoint: ReplayCheckpoint = ReplayCheckpoint.NONE
    fixed_loop: Literal["auto", "scan", "unroll"] | None = Field(
        default=None, description="Fixed explicit loop; null preserves legacy dispatch."
    )
    fixed_scan_checkpoint: bool = Field(
        default=False,
        strict=True,
        description="Rematerialize the JAX scan body; requires fixed_loop='scan'.",
    )
    checkpoint_chunk_size: int = Field(default=32, ge=1)
    schedule_tag: str | None = None
    replay_error_factor: float = Field(
        default=1.25,
        ge=1.0,
        allow_inf_nan=False,
    )
    fixed_max_step: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
    )
    strict: bool = True
    rtol: float = Field(default=1e-6, ge=0.0)
    atol: float = Field(default=1e-9, ge=0.0)
    dt_init: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    max_reject: int = Field(default=25, ge=1)
    max_steps: int = Field(default=1_000_000, ge=1)
    dt_min: float = Field(default=0.0, ge=0.0)
    dt_max: float = Field(default=float("inf"), gt=0.0)
    safety: float = Field(default=0.9, gt=0.0)
    fac_min: float = Field(default=0.2, gt=0.0)
    fac_max: float = Field(default=5.0, gt=0.0)
    gamma: float | None = Field(default=None, gt=0.0, lt=1.0)
    operator_axis: str | int = "state"
    operators: dict[str, Any] = Field(default_factory=dict)

    @field_validator("forcing_breakpoints", mode="before")
    @classmethod
    def _validate_forcing_breakpoints(
        cls, points: tuple[float, ...]
    ) -> tuple[float, ...]:
        """Apply the core solver's immutable forcing-schedule contract.

        Returns:
            Validated, snapshotted forcing times.
        """
        return DirectSSAConfig(forcing_breakpoints=points).forcing_breakpoints

    @field_validator("replicate_breakpoints", mode="before")
    @classmethod
    def _validate_replicate_breakpoints(
        cls, points: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        """Return validated immutable local forcing declarations."""
        return BatchedSSAConfig(replicate_breakpoints=points).replicate_breakpoints

    @model_validator(mode="after")
    def _validate_thinning_configuration(self) -> OpEngineEngineConfig:
        """Restrict thinning to its supported mode and controls.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If thinning is requested outside its supported path.
        """
        if self.stochastic_method is StochasticMethod.THINNING_SSA:
            if self.mode is not ExecutionMode.STOCHASTIC:
                msg = "thinning-ssa requires mode='stochastic'."
                raise ValueError(msg)
        elif (
            self.thinning_rate_bound is not None
            or self.thinning_max_candidates != 1_000_000
        ):
            msg = "Thinning controls require stochastic_method='thinning-ssa'."
            raise ValueError(msg)
        if self.stochastic_method is StochasticMethod.REPLICATE_DIRECT_SSA:
            if self.mode is not ExecutionMode.STOCHASTIC or self.replicate_axis is None:
                msg = "replicate-direct-ssa requires stochastic mode and replicate_axis"
                raise ValueError(msg)
        elif self.replicate_axis is not None or self.replicate_breakpoints:
            msg = "replicate_axis/breakpoints require replicate-direct-ssa"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_execution_configuration(self) -> OpEngineEngineConfig:
        if self.forcing_breakpoints and (
            self.mode is ExecutionMode.HYBRID
            or (
                self.mode is ExecutionMode.DETERMINISTIC and not self.method.is_explicit
            )
        ):
            msg = (
                "forcing_breakpoints require stochastic mode or an explicit "
                "deterministic method."
            )
            raise ValueError(msg)
        if (
            self.mode is not ExecutionMode.STOCHASTIC
            and self.method.startswith("imex-")
            and "operators" in self.model_fields_set
            and not _has_operator_specs(_coerce_operator_specs(self.operators))
        ):
            msg = (
                f"IMEX method '{self.method}' received operators, "
                "but none were populated. Provide at least one stage "
                "or omit operators to use system options."
            )
            raise ValueError(msg)
        if self.mode is ExecutionMode.HYBRID and not self.stochastic_reactions:
            msg = "Hybrid mode requires at least one stochastic_reactions entry."
            raise ValueError(msg)
        if self.mode is ExecutionMode.HYBRID and (
            self.method.is_implicit or self.method.is_nonlinear
        ):
            msg = (
                "Hybrid residual drift cannot reuse the full-system Jacobian "
                f"required by '{self.method}'; select an explicit or IMEX method."
            )
            raise ValueError(msg)
        if (
            self.mode is not ExecutionMode.HYBRID
            and "stochastic_reactions" in self.model_fields_set
            and self.stochastic_reactions
        ):
            msg = "stochastic_reactions selects the jump partition in hybrid mode only."
            raise ValueError(msg)
        if len(self.stochastic_reactions) != len(set(self.stochastic_reactions)):
            msg = "stochastic_reactions must not contain duplicates."
            raise ValueError(msg)
        if self.fixed_max_step is not None and (
            self.adaptive or not self.method.is_explicit
        ):
            msg = "fixed_max_step is supported only for fixed-step explicit methods."
            raise ValueError(msg)
        if self.fixed_max_step is not None and self.mode is ExecutionMode.STOCHASTIC:
            msg = "fixed_max_step does not apply in stochastic mode; use tau_max_step."
            raise ValueError(msg)
        if (
            self.stochastic_method is StochasticMethod.ADAPTIVE_TAU_LEAPING
            and self.tau_max_step is not None
        ):
            msg = (
                "tau_max_step applies to fixed tau-leaping only; adaptive "
                "tau-leaping selects its own bounded step."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_state_layout(self) -> OpEngineEngineConfig:
        """Reject layouts that the selected execution path cannot preserve.

        Returns:
            This validated configuration.

        Raises:
            ValueError: If layout, mode, method, or block-axis settings conflict.
        """
        if self.state_layout is not StateLayout.FLAT:
            if self.mode is not ExecutionMode.DETERMINISTIC:
                msg = "Structured state layouts require deterministic mode."
                raise ValueError(msg)
            if not self.method.is_explicit:
                msg = "Structured state layouts support explicit methods only."
                raise ValueError(msg)
        if self.block_axis is not None and self.state_layout is not StateLayout.BLOCK:
            msg = "block_axis may be set only when state_layout='block'."
            raise ValueError(msg)
        if self.adaptive_replay is AdaptiveReplayMode.COMPACT and (
            not self.adaptive or not self.method.is_explicit
        ):
            msg = (
                "adaptive_replay='compact' requires adaptive=True and an "
                "explicit method."
            )
            raise ValueError(msg)
        if self.replay_checkpoint is not ReplayCheckpoint.NONE and (
            self.adaptive_replay is not AdaptiveReplayMode.COMPACT
        ):
            msg = "Replay checkpointing requires adaptive_replay='compact'."
            raise ValueError(msg)
        if self.fixed_checkpoint is not ReplayCheckpoint.NONE and (
            self.adaptive
            or self.method is not SolverMethod.DOPRI5
            or self.mode is not ExecutionMode.DETERMINISTIC
            or self.state_layout is not StateLayout.FLAT
        ):
            msg = (
                "Fixed checkpointing requires deterministic flat-state "
                "fixed-step DOPRI5 execution."
            )
            raise ValueError(msg)
        if self.schedule_tag is not None and not self.schedule_tag.strip():
            msg = "schedule_tag must not be empty when provided."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_block_schedules(self) -> OpEngineEngineConfig:
        """Keep independent schedules within their explicit JAX block contract.

        Returns:
            This validated configuration.

        Raises:
            ValueError: The selected scheduling or batching policy cannot apply.
        """
        if self.block_schedule == "independent":
            if self.state_layout is not StateLayout.BLOCK or not self.adaptive:
                msg = "block_schedule='independent' requires adaptive block layout."
                raise ValueError(msg)
            if self.adaptive_replay is AdaptiveReplayMode.UNROLLED:
                msg = "Independent block schedules require compact JAX replay."
                raise ValueError(msg)
        elif self.block_batching != "length" or self.block_batch_size is not None:
            msg = "Block batching controls require block_schedule='independent'."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_fixed_loop(self) -> OpEngineEngineConfig:
        """Keep explicit fixed-loop controls separate from legacy checkpointing.

        Returns:
            This validated configuration.

        Raises:
            ValueError: The requested loop or checkpoint policy cannot apply.
        """
        if self.fixed_loop is not None:
            if self.adaptive or not self.method.is_explicit:
                msg = "fixed_loop requires a fixed explicit method (adaptive=False)"
                raise ValueError(msg)
            if self.mode is not ExecutionMode.DETERMINISTIC:
                msg = "fixed_loop requires deterministic execution."
                raise ValueError(msg)
        if self.fixed_scan_checkpoint:
            if self.fixed_loop != "scan":
                msg = "fixed_scan_checkpoint requires fixed_loop='scan'."
                raise ValueError(msg)
            if self.fixed_checkpoint is not ReplayCheckpoint.NONE:
                msg = "Select fixed_scan_checkpoint or fixed_checkpoint, not both."
                raise ValueError(msg)
        if (
            self.fixed_loop == "unroll"
            and self.fixed_checkpoint is not ReplayCheckpoint.NONE
        ):
            msg = "fixed_checkpoint requires a scan; fixed_loop='unroll' conflicts."
            raise ValueError(msg)
        return self

    def to_run_config(
        self,
        *,
        nonlinear: NonlinearMethodConfig | None = None,
    ) -> RunConfig:
        """Convert this provider config to an op_engine `RunConfig`.

        Args:
            nonlinear: Full-system nonlinear method configuration resolved from
                system options when SDIRK2 is selected.

        Returns:
            `RunConfig` derived from this provider configuration.
        """
        return RunConfig(
            method=self.method,
            adaptive=self.adaptive,
            fixed_max_step=self.fixed_max_step,
            fixed_loop=self.fixed_loop or "unroll",
            fixed_checkpoint=self.fixed_scan_checkpoint,
            forcing_breakpoints=(
                self.forcing_breakpoints
                if self.mode is ExecutionMode.DETERMINISTIC
                else ()
            ),
            strict=self.strict,
            adaptive_cfg=AdaptiveConfig(
                rtol=self.rtol,
                atol=self.atol,
                dt_init=self.dt_init,
                max_reject=self.max_reject,
                max_steps=self.max_steps,
            ),
            dt_controller=DtControllerConfig(
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                safety=self.safety,
                fac_min=self.fac_min,
                fac_max=self.fac_max,
            ),
            operators=_coerce_operator_specs(self.operators) or OperatorSpecs(),
            nonlinear=nonlinear,
            gamma=self.gamma,
        )


__all__ = [
    "AdaptiveReplayMode",
    "ExecutionMode",
    "OpEngineEngineConfig",
    "ReplayCheckpoint",
    "SolverMethod",
    "StateLayout",
    "StochasticMethod",
    "_coerce_operator_specs",
    "_has_operator_specs",
]
