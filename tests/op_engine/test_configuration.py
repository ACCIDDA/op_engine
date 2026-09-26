"""Construction-time validation for public op_engine configuration objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from op_engine import DenseNewtonSolver, NonlinearMethodConfig
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    DtControllerConfig,
    RunConfig,
)


def test_run_config_normalizes_method_alias_at_construction() -> None:
    """Method aliases are canonical before a solver consumes the config."""
    config = RunConfig(method=" CN ")

    assert config.method == "trapezoidal"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("runge-kutta-4", "rk4"),
        ("rk45", "dopri5"),
        ("dormand-prince", "dopri5"),
        ("dormand-prince-5(4)", "dopri5"),
    ],
)
def test_run_config_normalizes_explicit_runge_kutta_aliases(
    alias: str,
    canonical: str,
) -> None:
    """Common higher-order names resolve to stable canonical method names."""
    assert RunConfig(method=alias).method == canonical


@pytest.mark.parametrize("alias", ["sdirk", "alexander-sdirk2"])
def test_run_config_normalizes_sdirk2_aliases(alias: str) -> None:
    """The public nonlinear method has stable descriptive aliases."""
    assert RunConfig(method=alias).method == "sdirk2"


@pytest.mark.parametrize("alias", ["imex-ars443", "ars443"])
def test_run_config_normalizes_imex_ark3_aliases(alias: str) -> None:
    """The higher-order additive method has stable descriptive aliases."""
    assert RunConfig(method=alias).method == "imex-ark3"


def test_nonlinear_method_config_validates_protocol_boundary() -> None:
    """Nonlinear configuration stores callbacks and a backend-neutral protocol."""
    config = NonlinearMethodConfig(
        rhs_jacobian=lambda _time, _state: np.asarray([[1.0]])
    )

    assert isinstance(config.solver, DenseNewtonSolver)
    with pytest.raises(TypeError, match="rhs_jacobian"):
        NonlinearMethodConfig(rhs_jacobian=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NonlinearSolver"):
        NonlinearMethodConfig(
            rhs_jacobian=lambda _time, _state: np.asarray([[1.0]]),
            solver=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="NonlinearMethodConfig"):
        RunConfig(nonlinear=object())  # type: ignore[arg-type]


def test_run_config_rejects_unknown_method_at_construction() -> None:
    """Unknown methods fail where the user constructs the configuration."""
    with pytest.raises(ValueError, match="Unknown method"):
        RunConfig(method="not-a-method")


@pytest.mark.parametrize("fixed_max_step", [0.0, -1.0, np.nan, np.inf])
def test_run_config_rejects_invalid_fixed_max_step(fixed_max_step: float) -> None:
    """Fixed integration steps must have a finite positive upper bound."""
    with pytest.raises(ValueError, match="fixed_max_step"):
        RunConfig(fixed_max_step=fixed_max_step)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adaptive": True, "fixed_max_step": 0.1},
        {"method": "implicit-euler", "fixed_max_step": 0.1},
    ],
)
def test_run_config_rejects_incompatible_fixed_step_modes(
    kwargs: dict[str, object],
) -> None:
    """A fixed explicit step policy cannot be silently ignored."""
    with pytest.raises(ValueError, match="fixed-step explicit"):
        RunConfig(**kwargs)  # type: ignore[arg-type]


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


def test_adaptive_schedule_normalizes_and_validates_interval_steps() -> None:
    """A schedule is immutable, normalized, and covers every output interval."""
    schedule = AdaptiveStepSchedule(
        output_times=(np.float32(0.0), np.float32(0.5), np.float32(1.0)),
        step_sizes=((0.2, 0.3), (0.5,)),
    )

    assert schedule.output_times == (0.0, 0.5, 1.0)
    assert schedule.step_sizes == ((0.2, 0.3), (0.5,))


@pytest.mark.parametrize(
    ("output_times", "step_sizes", "match"),
    [
        ((), (), "at least one output time"),
        ((0.0, 0.0), ((0.1,),), "strictly increasing"),
        ((0.0, 1.0), (), "one step group"),
        ((0.0, 1.0), ((),), "must not be empty"),
        ((0.0, 1.0), ((-1.0,),), "finite and positive"),
        ((0.0, 1.0), ((0.25, 0.5),), "sum to each output interval"),
    ],
)
def test_adaptive_schedule_rejects_invalid_meshes(
    output_times: tuple[float, ...],
    step_sizes: tuple[tuple[float, ...], ...],
    match: str,
) -> None:
    """Invalid accepted-step meshes fail when the schedule is constructed."""
    with pytest.raises(ValueError, match=match):
        AdaptiveStepSchedule(output_times=output_times, step_sizes=step_sizes)
