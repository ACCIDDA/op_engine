"""Connector between op_engine and flepimop2 config structure."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from op_engine.core_solver import (
    AdaptiveConfig,
    DtControllerConfig,
    OperatorSpecs,
    RunConfig,
)

MethodName = Literal[
    "euler",
    "heun",
    "imex-euler",
    "imex-heun-tr",
    "imex-trbdf2",
]


class OpEngineEngineConfig(BaseModel):
    """Configuration schema for op_engine when used as a flepimop2 engine."""

    model_config = ConfigDict(extra="allow")

    method: MethodName = Field(default="heun", description="Time integration method")
    adaptive: bool = Field(
        default=False,
        description="Enable adaptive substepping between output times",
    )
    strict: bool = Field(
        default=True, description="Fail fast on invalid configurations"
    )

    # tolerances
    rtol: float = Field(default=1e-6, ge=0.0)
    atol: float = Field(default=1e-9, ge=0.0)

    # controller
    dt_min: float = Field(default=0.0, ge=0.0)
    dt_max: float = Field(default=float("inf"), gt=0.0)
    safety: float = Field(default=0.9, gt=0.0)
    fac_min: float = Field(default=0.2, gt=0.0)
    fac_max: float = Field(default=5.0, gt=0.0)

    gamma: float | None = Field(default=None, gt=0.0, lt=1.0)

    # Operator specs (default/tr/bdf2) for IMEX methods.
    operators: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Operator specifications for IMEX methods. "
            "Required when method is an IMEX variant."
        ),
    )

    operator_axis: str | int = Field(
        default="state",
        description="Axis along which implicit operators act (name or index).",
    )

    @model_validator(mode="after")
    def _validate_imex_requirements(self) -> OpEngineEngineConfig:
        method = str(self.method)
        if method.startswith("imex-") and not self._has_any_operator_specs(
            self.operators
        ):
            msg = (
                f"IMEX method '{method}' requires operator specifications, "
                "but no operators were provided in the engine config."
            )
            raise ValueError(msg)
        return self

    @staticmethod
    def _has_any_operator_specs(operators: dict[str, Any] | None) -> bool:
        """Return True if any operator spec is provided."""
        if operators is None:
            return False
        return any(
            operators.get(name) is not None for name in ("default", "tr", "bdf2")
        )

    def to_run_config(self) -> RunConfig:
        """
        Convert to op_engine RunConfig.

        Returns:
            RunConfig instance reflecting this configuration.
        """
        adaptive_cfg = AdaptiveConfig(rtol=self.rtol, atol=self.atol)
        dt_controller = DtControllerConfig(
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            safety=self.safety,
            fac_min=self.fac_min,
            fac_max=self.fac_max,
        )

        op_specs = self._coerce_operator_specs(self.operators)

        return RunConfig(
            method=self.method,
            adaptive=self.adaptive,
            strict=self.strict,
            adaptive_cfg=adaptive_cfg,
            dt_controller=dt_controller,
            operators=op_specs,
            gamma=self.gamma,
        )

    @staticmethod
    def _coerce_operator_specs(operators: dict[str, Any] | None) -> OperatorSpecs:
        """
        Normalize operator inputs into OperatorSpecs.

        Returns:
            OperatorSpecs with default/tr/bdf2 fields populated when provided.
        """
        if operators is None:
            return OperatorSpecs()
        return OperatorSpecs(
            default=operators.get("default"),
            tr=operators.get("tr"),
            bdf2=operators.get("bdf2"),
        )
