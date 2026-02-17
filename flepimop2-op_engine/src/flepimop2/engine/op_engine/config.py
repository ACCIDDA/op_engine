"""Configuration model for op_engine provider integration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from op_engine.core_solver import (
    AdaptiveConfig,
    DtControllerConfig,
    OperatorSpecs,
    RunConfig,
)


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
    operators: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_explicit_empty_operators(self) -> OpEngineEngineConfig:
        if (
            self.method.startswith("imex-")
            and "operators" in self.model_fields_set
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
        """Convert this provider config to an op_engine `RunConfig`.

        Returns:
            `RunConfig` derived from this provider configuration.
        """
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


__all__ = ["OpEngineEngineConfig", "_coerce_operator_specs", "_has_operator_specs"]
