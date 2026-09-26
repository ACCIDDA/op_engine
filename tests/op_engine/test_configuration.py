"""Construction-time validation for public op_engine configuration objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_engine.core_solver import (
    AdaptiveConfig,
    DtControllerConfig,
    RunConfig,
)


def test_run_config_normalizes_method_alias_at_construction() -> None:
    """Method aliases are canonical before a solver consumes the config."""
    config = RunConfig(method=" CN ")

    assert config.method == "trapezoidal"


def test_run_config_rejects_unknown_method_at_construction() -> None:
    """Unknown methods fail where the user constructs the configuration."""
    with pytest.raises(ValueError, match="Unknown method"):
        RunConfig(method="not-a-method")


@pytest.mark.parametrize("gamma", [0.0, 1.0, np.nan, np.inf])
def test_run_config_rejects_invalid_trbdf2_gamma(gamma: float) -> None:
    """TR-BDF2 gamma is bounded at configuration construction."""
    with pytest.raises(ValueError, match="gamma must be in"):
        RunConfig(method="imex-trbdf2", gamma=gamma)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dt_min": -1.0}, "dt_min"),
        ({"dt_max": 0.0}, "dt_max"),
        ({"dt_min": 2.0, "dt_max": 1.0}, "dt_max"),
        ({"safety": 0.0}, "safety"),
        ({"fac_min": 0.0}, "fac_min"),
        ({"fac_min": 2.0, "fac_max": 1.0}, "fac_max"),
    ],
)
def test_dt_controller_rejects_invalid_scalars(
    kwargs: dict[str, float],
    match: str,
) -> None:
    """Invalid controller scalars fail before a run starts."""
    with pytest.raises(ValueError, match=match):
        DtControllerConfig(**kwargs)


def test_dt_controller_allows_unbounded_default_maximum() -> None:
    """Positive infinity remains the supported default upper bound."""
    config = DtControllerConfig()

    assert np.isinf(config.dt_max)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rtol": -1.0}, "rtol"),
        ({"atol": -1.0}, "atol"),
        ({"dt_init": 0.0}, "dt_init"),
        ({"max_reject": 0}, "max_reject"),
        ({"max_steps": 0}, "max_steps"),
    ],
)
def test_adaptive_config_rejects_invalid_scalars(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Invalid adaptive controls fail before a run starts."""
    with pytest.raises(ValueError, match=match):
        AdaptiveConfig(**kwargs)


def test_adaptive_config_validates_numpy_atol_without_coercing_array_api() -> None:
    """Eager NumPy tolerances are checked while generic arrays stay untouched."""
    tolerance = np.asarray([1e-8, 1e-7])

    config = AdaptiveConfig(atol=tolerance)

    assert config.atol is tolerance
    with pytest.raises(ValueError, match="NumPy atol"):
        AdaptiveConfig(atol=np.asarray([1e-8, -1.0]))
