# src/op_engine/flepimop2/config.py
"""Configuration models for the op_engine.flepimop2 engine adapter.

This module defines the pydantic-facing configuration objects used by flepimop2
YAML files and translates them into native op_engine RunConfig objects.

Notes:
    - `pydantic` is required when using the `op_engine[flepimop2]` extra.
    - flepimop2 configuration parsing may include additional fields; this model
      allows and ignores unknown fields (`extra="allow"`).
    - Operator specifications are intentionally omitted for now. IMEX methods
      require operator specs/factories; the adapter should guard and raise a
      clear error message before attempting to run an IMEX method without them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    """Configuration schema for op_engine when used as a flepimop2 engine.

    This model mirrors op_engine RunConfig fields but keeps YAML-friendly defaults
    and validation behavior.

    Notes:
        - Operator specifications are intentionally omitted for now.
          IMEX support is guarded at runtime by the adapter.
        - flepimop2 passes unknown fields through; we only parse what we need.
    """

    model_config = ConfigDict(extra="allow")

    method: MethodName = Field(
        default="heun",
        description="Time integration method",
    )

    adaptive: bool = Field(
        default=False,
        description="Enable adaptive substepping between output times",
    )

    strict: bool = Field(
        default=True,
        description="Fail fast on invalid configurations",
    )

    # Adaptive stepping controls
    rtol: float = Field(default=1e-6, ge=0.0)
    atol: float = Field(default=1e-9, ge=0.0)

    # dt controller controls
    dt_min: float = Field(default=0.0, ge=0.0)
    dt_max: float = Field(default=float("inf"), gt=0.0)
    safety: float = Field(default=0.9, gt=0.0)
    fac_min: float = Field(default=0.2, gt=0.0)
    fac_max: float = Field(default=5.0, gt=0.0)

    # TR-BDF2 parameter (only used when method == "imex-trbdf2")
    gamma: float | None = Field(default=None, gt=0.0, lt=1.0)

    def to_run_config(self) -> RunConfig:
        """Convert this config to a native op_engine RunConfig.

        Returns:
            Fully constructed RunConfig instance.
        """
        adaptive_cfg = AdaptiveConfig(
            rtol=self.rtol,
            atol=self.atol,
        )

        dt_controller = DtControllerConfig(
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            safety=self.safety,
            fac_min=self.fac_min,
            fac_max=self.fac_max,
        )

        return RunConfig(
            method=self.method,
            adaptive=self.adaptive,
            strict=self.strict,
            adaptive_cfg=adaptive_cfg,
            dt_controller=dt_controller,
            operators=OperatorSpecs(),
            gamma=self.gamma,
        )
