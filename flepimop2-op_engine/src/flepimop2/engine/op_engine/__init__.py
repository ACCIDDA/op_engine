"""flepimop2 engine integration for op_engine.

This package intentionally defines the public Engine class in this module so that
flepimop2's dynamic loader can auto-inject a default `build()` function.

Why:
- flepimop2 resolves `module: op_engine` to `flepimop2.engine.op_engine`
- if that module has no `build`, it looks for a pydantic BaseModel subclass
  defined *in this module* and generates `build()` automatically.
"""

from __future__ import annotations

from typing import Literal

from .engine import _OpEngineFlepimop2EngineImpl


class OpEngineFlepimop2Engine(_OpEngineFlepimop2EngineImpl):  # noqa: RUF067
    """Public op_engine-backed flepimop2 Engine (default-build enabled)."""

    module: Literal["flepimop2.engine.op_engine"] = "flepimop2.engine.op_engine"


__all__ = ["OpEngineFlepimop2Engine"]
