# op_engine/examples/biogeochemical_network.py
"""
Example: Size-structured phytoplankton-zooplankton network using op_engine.

This example is designed to be both a correctness/comparison demo (vs SciPy
solve_ivp) and an API showcase for CoreSolver.run(...).

It demonstrates all five op_engine methods with adaptive=True:
- euler
- heun
- imex-euler
- imex-heun-tr
- imex-trbdf2

It also runs SciPy comparators on the same output grid:
- solve_ivp RK45
- solve_ivp BDF

IMEX splits (A/B/C) are implemented via StageOperatorContext-aware operator
factories. Operator modes control whether the implicit operator is frozen,
time-dependent, or relinearized per stage state.

Outputs are saved under:
    examples/output/biogeochemical/

Figures:
1) biogeo_explicit_bins.png
   - Bin detail: op_engine heun vs SciPy RK45

2) biogeo_imex1_bins.png
   - Bin detail: op_engine imex-euler split B vs SciPy BDF

3) biogeo_imex2_bins.png
   - Bin detail: op_engine imex-trbdf2 split B vs SciPy BDF

4) biogeo_runtime.png
   - Wall time vs output-grid dt (log-log)

Text outputs:
- biogeo_runtime.txt
- biogeo_runtime.csv
- biogeo_manifest.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc

from op_engine.core_solver import CoreSolver
from op_engine.matrix_ops import (
    StageOperatorContext,
    make_constant_base_builder,
    make_stage_operator_factory,
)
from op_engine.model_core import ModelCore, ModelCoreOptions

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# User knobs
# =============================================================================

OperatorMode = Literal["frozen", "time", "stage_state"]
OPERATOR_MODE: OperatorMode = "stage_state"

ETA_CROSS: float = 0.10  # Split C cross-coupling strength (keep small)
CLIP_NONNEGATIVE: bool = True
FAIL_FAST_ON_NONFINITE: bool = True

# SciPy reference tolerances
SCIPY_RTOL: float = 1e-4
SCIPY_ATOL: float = 1e-6

# CoreSolver tolerances (adaptive stepping)
OPE_RTOL: float = 1e-4
OPE_ATOL: float = 1e-6

# Output layout
_OUTPUT_DIR = Path("examples") / "output" / "biogeochemical"

# Constants (exception messages)
_DT_POSITIVE_ERROR = "dt_days must be positive"
_SPLIT_REQUIRED_ERROR = "split is required for IMEX methods in this example"
_STATE_ARRAY_NONE_ERROR = "store_history=True but core.state_array is None"
_SOLVEIVP_FAILED_ERROR = "solve_ivp({method}) failed: {message}"
_STAGE_STATE_SIZE_ERROR = "Stage state size mismatch: got {got}, expected {expected}"
_EXPECTED_TRAJ_ERROR = "Expected trajectory for plot: {name}"


# =============================================================================
# Parameters / data containers
# =============================================================================


@dataclass(frozen=True, slots=True)
class ParamsA:
    """Model parameters for the phytoplankton-zooplankton network."""

    n_t: float = 10.0
    gamma: float = 0.5
    r_g: float = 10.0
    sigma_g: float = 0.5
    mu_a: float = 2.0
    mu_b: float = 0.3
    k_a: float = 0.25
    k_b: float = 0.81
    g_a: float = 17.4
    g_b: float = 0.48
    delta_a: float = 2.43
    delta_b: float = 0.48
    lam: float = 0.02
    sea: float = 0.3


SplitName = Literal["A", "B", "C"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Precomputed model objects for fast RHS/operator evaluation."""

    params: ParamsA
    prd: np.ndarray
    zrd: np.ndarray
    n_bins: int


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Run configuration (output grid + initial state + RHS callables)."""

    time_grid: np.ndarray
    y0_flat: np.ndarray
    rhs_tensor: Callable[[float, np.ndarray], np.ndarray]
    rhs_flat: Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class OperatorSpecs:
    """Operator factories passed into CoreSolver.run for IMEX methods."""

    default: object | None = None
    tr: object | None = None
    bdf2: object | None = None


@dataclass(frozen=True, slots=True)
class RuntimeReportSpec:
    """Bundle arguments for runtime report writers (keeps signatures small)."""

    operator_mode: str
    main_timings: dict[str, float]
    scipy_infos: dict[str, dict[str, object]]
    dt_days: np.ndarray
    sweep_timings: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class Manifest:
    """Machine-readable metadata for outputs and settings."""

    operator_mode: str
    eta_cross: float
    clip_nonnegative: bool
    fail_fast_on_nonfinite: bool
    scipy_rtol: float
    scipy_atol: float
    ope_rtol: float
    ope_atol: float
    n_bins: int
    dt_out_main_days: float
    total_time_days: float
    sweep_total_days: float
    dt_sweep_days: list[float]
    outputs: dict[str, str]


# =============================================================================
# Helpers
# =============================================================================


def ensure_output_dir() -> Path:
    """Ensure examples/output/biogeochemical exists.

    Returns:
        Output directory path.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def _size_bins(n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Phytoplankton and zooplankton diameter-based size bins.

    Returns:
        (prd, zrd) arrays, each shape (n_bins,).
    """
    prd = np.linspace(np.emath.logn(4, 1), np.emath.logn(4, 100), n_bins)
    prd = 4**prd
    zrd = 10.0 * prd
    return np.asarray(prd, dtype=float), np.asarray(zrd, dtype=float)


def mu_maxes(a: float, b: float, p: np.ndarray) -> np.ndarray:
    """mu_max(p) = a * p^{-b}.

    Returns:
        Array of mu_max values, shape p.shape.
    """
    return a * (p ** (-b))


def kns(a: float, b: float, p: np.ndarray) -> np.ndarray:
    """Half-saturation KN(p) = a * p^{b}.

    Returns:
        Array of KN values, shape p.shape.
    """
    return a * (p**b)


def gs(a: float, b: float, z: np.ndarray) -> np.ndarray:
    """Grazing coefficient g(z) = a * z^{-b}.

    Returns:
        Array of grazing coefficients, shape z.shape.
    """
    return a * (z ** (-b))


def deltas(a: float, b: float, z: np.ndarray) -> np.ndarray:
    """Mortality delta(z) = a * z^{-b}.

    Returns:
        Array of mortality coefficients, shape z.shape.
    """
    return a * (z ** (-b))


def rho(r: float, sigma: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Feeding kernel (vectorized), broadcasting-friendly.

    Returns:
        Kernel matrix with shape broadcast(p, z).
    """
    y = np.abs(z / p - r) / (np.sqrt(2.0) * (sigma * r))
    return erfc(y)


def unpack_flat_state(u: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Map flat u shape (2*n_bins,) -> (p, z), each shape (n_bins,).

    Returns:
        (p, z) arrays, each shape (n_bins,).
    """
    p = u[:n_bins]
    z = u[n_bins:]
    return p, z


def pack_flat_state(p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Map (p, z) each shape (n_bins,) -> flat u shape (2*n_bins,).

    Returns:
        Flat state vector, shape (2*n_bins,).
    """
    return np.concatenate(
        [np.asarray(p, dtype=float), np.asarray(z, dtype=float)],
        axis=0,
    )


def tensor_from_flat(u: np.ndarray) -> np.ndarray:
    """Convert flat (2n,) -> tensor (2n, 1) for ModelCore.

    Returns:
        Tensor state, shape (2n, 1).
    """
    return np.asarray(u, dtype=float).reshape(-1, 1)


def flat_from_tensor(state: np.ndarray) -> np.ndarray:
    """Convert tensor (2n, 1) -> flat (2n,).

    Returns:
        Flat state vector, shape (2n,).
    """
    return np.asarray(state, dtype=float).reshape(-1)


def _first_bad_index(arr: np.ndarray) -> int | None:
    bad = ~np.isfinite(arr)
    if not np.any(bad):
        return None
    return int(np.flatnonzero(bad)[0])


def _assert_finite(name: str, arr: np.ndarray, *, t: float | None = None) -> None:
    if np.all(np.isfinite(arr)):
        return
    if not FAIL_FAST_ON_NONFINITE:
        return

    idx = _first_bad_index(arr)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    t_msg = "" if t is None else f" at t={t:.6g}"
    msg = (
        f"{name} contains inf/NaN{t_msg}. min={mn:.6g}, max={mx:.6g}, "
        f"first_bad_index={idx}"
    )
    raise FloatingPointError(msg)


def _seasonal_factor(params: ParamsA, t_days: float) -> float:
    """Always-positive seasonal modulation factor.

    Returns:
        Seasonal multiplier (scalar).
    """
    return float(1.0 + params.sea * np.sin(2.0 * np.pi * t_days / 365.0))


def _sanitize_flat(u_flat: np.ndarray, n_bins: int) -> np.ndarray:
    u = np.asarray(u_flat, dtype=float).reshape(-1)
    if CLIP_NONNEGATIVE:
        p, z = unpack_flat_state(u, n_bins)
        p = np.clip(p, 0.0, None)
        z = np.clip(z, 0.0, None)
        return pack_flat_state(p, z)
    return u


def _season_scalar_for_mode(params: ParamsA, t: float) -> float:
    if OPERATOR_MODE in {"time", "stage_state"}:
        return _seasonal_factor(params, t)
    return 1.0


# =============================================================================
# Reaction RHS for op_engine (tensor form) and SciPy (flat form)
# =============================================================================


def make_reaction_tensor(
    model: ModelSpec,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build RHS for CoreSolver.run(...), tensor in/out.

    Returns:
        Callable f(t, y_tensor)->dy_tensor with y_tensor shape (2n,1).
    """
    params = model.params
    prd = model.prd
    zrd = model.zrd
    n_bins = model.n_bins

    mu_max_ = mu_maxes(params.mu_a, params.mu_b, prd)
    kn_ = kns(params.k_a, params.k_b, prd)
    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    kern = rho(params.r_g, params.sigma_g, prd[:, None], zrd[None, :])

    def reaction(t: float, state_tensor: np.ndarray) -> np.ndarray:
        u = flat_from_tensor(state_tensor)
        _assert_finite("state", u, t=t)

        p, z = unpack_flat_state(u, n_bins)
        if CLIP_NONNEGATIVE:
            p = np.clip(p, 0.0, None)
            z = np.clip(z, 0.0, None)

        total = float(p.sum() + z.sum())
        avail = params.n_t - total

        season = _seasonal_factor(params, t)
        frac = avail / (kn_ + avail)

        growth = season * mu_max_ * frac * p
        loss_lin = params.lam * p
        graze = p * (kern @ (g_ * z))
        dp = growth - loss_lin - graze

        intake = kern.T @ p
        dz = z * g_ * (params.gamma * intake) - delta_ * z

        out = pack_flat_state(dp, dz)
        _assert_finite("reaction(state)", out, t=t)
        return tensor_from_flat(out)

    return reaction


def make_rhs_flat(model: ModelSpec) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build RHS for SciPy solve_ivp, flat in/out.

    Returns:
        Callable f(t, u_flat)->du_flat with u_flat shape (2n,).
    """
    params = model.params
    prd = model.prd
    zrd = model.zrd
    n_bins = model.n_bins

    mu_max_ = mu_maxes(params.mu_a, params.mu_b, prd)
    kn_ = kns(params.k_a, params.k_b, prd)
    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    kern = rho(params.r_g, params.sigma_g, prd[:, None], zrd[None, :])

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        _assert_finite("state", u, t=t)
        p, z = unpack_flat_state(u, n_bins)
        if CLIP_NONNEGATIVE:
            p = np.clip(p, 0.0, None)
            z = np.clip(z, 0.0, None)

        total = float(p.sum() + z.sum())
        avail = params.n_t - total

        season = _seasonal_factor(params, t)
        frac = avail / (kn_ + avail)

        growth = season * mu_max_ * frac * p
        loss_lin = params.lam * p
        graze = p * (kern @ (g_ * z))
        dp = growth - loss_lin - graze

        intake = kern.T @ p
        dz = z * g_ * (params.gamma * intake) - delta_ * z

        out = np.empty_like(u)
        out[:n_bins] = dp
        out[n_bins:] = dz
        _assert_finite("rhs(u)", out, t=t)
        return out

    return rhs


# =============================================================================
# IMEX split operators A/B/C: build A(t, y_stage) as a 2n x 2n matrix
# =============================================================================


def split_a_matrix(model: ModelSpec, *, t: float, y_flat: np.ndarray) -> np.ndarray:
    """Split A: diagonal damping only.

    Returns:
        A matrix, shape (2n, 2n).
    """
    _ = y_flat
    params = model.params
    zrd = model.zrd
    n = model.n_bins

    season = _season_scalar_for_mode(params, t)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)

    lam_eff = float(params.lam * season)
    delta_eff = delta_ * season

    a_mat = np.zeros((2 * n, 2 * n), dtype=float)
    a_mat[:n, :n] = -lam_eff * np.eye(n, dtype=float)
    a_mat[n:, n:] = -np.diag(delta_eff)
    return a_mat


def split_b_matrix(model: ModelSpec, *, t: float, y_flat: np.ndarray) -> np.ndarray:
    """Split B: diagonal damping + grazing sink on P (state-dependent).

    Returns:
        A matrix, shape (2n, 2n).
    """
    params = model.params
    prd = model.prd
    zrd = model.zrd
    n = model.n_bins

    u = _sanitize_flat(y_flat, n)
    _, z = unpack_flat_state(u, n)

    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    kern = rho(params.r_g, params.sigma_g, prd[:, None], zrd[None, :])

    season = _season_scalar_for_mode(params, t)
    lam_eff = float(params.lam * season)
    delta_eff = delta_ * season

    g_sink = kern @ (g_ * z)  # (n,)

    a_mat = np.zeros((2 * n, 2 * n), dtype=float)
    a_mat[:n, :n] = -np.diag(lam_eff + g_sink)
    a_mat[n:, n:] = -np.diag(delta_eff)
    return a_mat


def _split_c_blocks(
    model: ModelSpec,
    *,
    t: float,
    y_flat: np.ndarray,
    eta_cross: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = model.params
    prd = model.prd
    zrd = model.zrd
    n = model.n_bins

    u = _sanitize_flat(y_flat, n)
    p, z = unpack_flat_state(u, n)

    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    kern = rho(params.r_g, params.sigma_g, prd[:, None], zrd[None, :])

    season = _season_scalar_for_mode(params, t)
    lam_eff = float(params.lam * season)
    delta_eff = delta_ * season

    g_sink = kern @ (g_ * z)  # (n,)

    a_pp = -lam_eff * np.eye(n, dtype=float) - np.diag(g_sink)
    a_zz = -np.diag(delta_eff)

    a_pz = -(np.diag(p) @ (kern @ np.diag(g_)))
    a_zp = np.diag(z * g_ * params.gamma) @ kern.T

    scale = float(eta_cross)
    return a_pp, a_zz, scale * a_pz, scale * a_zp


def split_c_matrix(
    model: ModelSpec,
    *,
    t: float,
    y_flat: np.ndarray,
    eta_cross: float = ETA_CROSS,
) -> np.ndarray:
    """Split C: dissipative diagonal blocks + small cross-coupling.

    Returns:
        A matrix, shape (2n, 2n).
    """
    n = model.n_bins
    a_pp, a_zz, a_pz, a_zp = _split_c_blocks(
        model, t=t, y_flat=y_flat, eta_cross=eta_cross
    )

    a_mat = np.zeros((2 * n, 2 * n), dtype=float)
    a_mat[:n, :n] = a_pp
    a_mat[n:, n:] = a_zz
    a_mat[:n, n:] = a_pz
    a_mat[n:, :n] = a_zp
    return a_mat


def make_base_builder_for_split(
    split: SplitName,
    *,
    model: ModelSpec,
    y0_flat: np.ndarray,
    eta_cross: float = ETA_CROSS,
) -> Callable[[StageOperatorContext], np.ndarray]:
    """Return base_builder(ctx)->A matrix, honoring OPERATOR_MODE.

    Returns:
        A StageOperatorContext-aware operator builder.
    """
    n_bins = model.n_bins
    y0_sanitized = _sanitize_flat(y0_flat, n_bins)

    def ctx_y_to_flat(ctx: StageOperatorContext) -> np.ndarray:
        y_tensor = np.asarray(ctx.y, dtype=float)
        y_flat_local = flat_from_tensor(y_tensor)
        expected = 2 * n_bins
        if y_flat_local.size != expected:
            msg = _STAGE_STATE_SIZE_ERROR.format(
                got=y_flat_local.size, expected=expected
            )
            raise ValueError(msg)
        return y_flat_local

    def build_for(
        split_name: SplitName, *, t: float, y_flat_local: np.ndarray
    ) -> np.ndarray:
        if split_name == "A":
            return split_a_matrix(model, t=t, y_flat=y_flat_local)
        if split_name == "B":
            return split_b_matrix(model, t=t, y_flat=y_flat_local)
        return split_c_matrix(model, t=t, y_flat=y_flat_local, eta_cross=eta_cross)

    if OPERATOR_MODE == "frozen":
        a0 = build_for(split, t=0.0, y_flat_local=y0_sanitized)
        return make_constant_base_builder(a0)

    if OPERATOR_MODE == "time":

        def builder(ctx: StageOperatorContext) -> np.ndarray:
            return build_for(split, t=float(ctx.t), y_flat_local=y0_sanitized)

        return builder

    def builder(ctx: StageOperatorContext) -> np.ndarray:
        y_flat_local = _sanitize_flat(ctx_y_to_flat(ctx), n_bins)
        return build_for(split, t=float(ctx.t), y_flat_local=y_flat_local)

    return builder


# =============================================================================
# CoreSolver / SciPy runners
# =============================================================================


def build_uniform_time_grid_days(total_time_days: float, dt_days: float) -> np.ndarray:
    """Build a uniform time grid in days.

    Args:
        total_time_days: End time in days.
        dt_days: Output-grid spacing in days.

    Returns:
        1D numpy array of output times in days.

    Raises:
        ValueError: If dt_days is not positive.
    """
    dt = float(dt_days)
    if dt <= 0.0:
        msg = _DT_POSITIVE_ERROR
        raise ValueError(msg)
    return np.arange(0.0, total_time_days + 0.5 * dt, dt, dtype=float)


def make_core(time_grid: np.ndarray, n_state: int, *, store_history: bool) -> ModelCore:
    """Construct a ModelCore for this example.

    Returns:
        Initialized ModelCore instance.
    """
    opts = ModelCoreOptions(
        other_axes=(),
        axis_names=("state", "subgroup"),
        store_history=store_history,
        dtype=np.float64,
    )
    return ModelCore(n_states=n_state, n_subgroups=1, time_grid=time_grid, options=opts)


def run_op_engine(
    *,
    method: str,
    run: RunSpec,
    model: ModelSpec,
    split: SplitName | None = None,
    store_history: bool = True,
) -> tuple[np.ndarray | None, float]:
    """Run CoreSolver.run(...) with adaptive=True.

    Args:
        method: One of "euler", "heun", "imex-euler", "imex-heun-tr", "imex-trbdf2".
        run: RunSpec containing time_grid, y0, and RHS functions.
        model: ModelSpec (params + bins).
        split: "A"|"B"|"C" for IMEX methods.
        store_history: If True, return full trajectory; if False, timing only.

    Returns:
        (states_flat_over_time, wall_s). If store_history is False, states is None.

    Raises:
        ValueError: If an IMEX method is selected without split.
        RuntimeError: If store_history=True but core.state_array is None.
    """
    n_state = int(run.y0_flat.size)
    core = make_core(run.time_grid, n_state, store_history=store_history)
    core.set_initial_state(tensor_from_flat(run.y0_flat))

    solver = CoreSolver(core, operators=None, operator_axis="state")

    operators = OperatorSpecs(default=None, tr=None, bdf2=None)

    if method in {"imex-euler", "imex-heun-tr", "imex-trbdf2"}:
        if split is None:
            msg = _SPLIT_REQUIRED_ERROR
            raise ValueError(msg)

        base_builder = make_base_builder_for_split(
            split, model=model, y0_flat=run.y0_flat
        )

        if method == "imex-euler":
            operators = OperatorSpecs(
                default=make_stage_operator_factory(
                    base_builder, scheme="implicit-euler"
                ),
            )
        elif method == "imex-heun-tr":
            operators = OperatorSpecs(
                default=make_stage_operator_factory(base_builder, scheme="trapezoidal"),
            )
        else:
            operators = OperatorSpecs(
                tr=make_stage_operator_factory(base_builder, scheme="trapezoidal"),
                bdf2=make_stage_operator_factory(base_builder, scheme="implicit-euler"),
            )

    kw: dict[str, object] = dict(
        method=method,
        adaptive=True,
        rtol=OPE_RTOL,
        atol=OPE_ATOL,
        strict=True,
    )

    # Only pass operator kwargs that exist for the selected method.
    if operators.default is not None:
        kw["operators"] = operators.default
    if operators.tr is not None:
        kw["operators_tr"] = operators.tr
    if operators.bdf2 is not None:
        kw["operators_bdf2"] = operators.bdf2

    t0 = time.perf_counter()
    solver.run(run.rhs_tensor, **kw)
    wall = time.perf_counter() - t0

    if not store_history:
        return None, wall

    if core.state_array is None:
        msg = _STATE_ARRAY_NONE_ERROR
        raise RuntimeError(msg)

    states = np.asarray(core.state_array[:, :, 0], dtype=float)  # (T, 2n)
    return states, wall


def run_scipy(
    *,
    method: str,
    run: RunSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run solve_ivp on the full RHS.

    Returns:
        (y, info) where y has shape (T, 2n) and info is a dict of solver stats.

    Raises:
        RuntimeError: If solve_ivp reports failure.
    """
    t0 = float(run.time_grid[0])
    t1 = float(run.time_grid[-1])

    start = time.perf_counter()
    sol = solve_ivp(
        fun=run.rhs_flat,
        t_span=(t0, t1),
        y0=run.y0_flat,
        method=method,
        t_eval=run.time_grid,
        rtol=SCIPY_RTOL,
        atol=SCIPY_ATOL,
        vectorized=False,
    )
    wall = time.perf_counter() - start

    info: dict[str, object] = {
        "success": sol.success,
        "message": sol.message,
        "nfev": sol.nfev,
        "njev": getattr(sol, "njev", None),
        "nlu": getattr(sol, "nlu", None),
        "wall_s": wall,
    }

    if not sol.success:
        msg = _SOLVEIVP_FAILED_ERROR.format(method=method, message=sol.message)
        raise RuntimeError(msg)

    return np.asarray(sol.y.T, dtype=float), info


# =============================================================================
# Plotting
# =============================================================================


def save_bins_figure(
    out_png: Path,
    time_days: np.ndarray,
    n_bins: int,
    *,
    main_name: str,
    main_arr: np.ndarray,
    scipy_name: str,
    scipy_arr: np.ndarray,
) -> None:
    """Bin detail plot for one op_engine run + SciPy overlay."""
    t_years = np.asarray(time_days, dtype=float) / 365.0

    fig, axes = plt.subplots(
        nrows=n_bins,
        ncols=2,
        figsize=(14.0, 1.35 * n_bins),
        sharex=True,
        constrained_layout=True,
    )

    p_main = main_arr[:, :n_bins]
    z_main = main_arr[:, n_bins:]
    p_ref = scipy_arr[:, :n_bins]
    z_ref = scipy_arr[:, n_bins:]

    for i in range(n_bins):
        axp = axes[i, 0]
        axz = axes[i, 1]

        axp.plot(t_years, p_main[:, i], label=main_name if i == 0 else None)
        axp.plot(
            t_years,
            p_ref[:, i],
            label=scipy_name if i == 0 else None,
            linewidth=1.0,
            alpha=0.5,
        )
        axp.grid(visible=True)
        axp.set_ylabel(f"bin {i}", rotation=0, labelpad=25, va="center")

        axz.plot(t_years, z_main[:, i], label=main_name if i == 0 else None)
        axz.plot(
            t_years,
            z_ref[:, i],
            label=scipy_name if i == 0 else None,
            linewidth=1.0,
            alpha=0.5,
        )
        axz.grid(visible=True)

    axes[0, 0].set_title("P bins")
    axes[0, 1].set_title("Z bins")
    axes[-1, 0].set_xlabel("Time (years)")
    axes[-1, 1].set_xlabel("Time (years)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=True)

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_runtime_sweep_plot(
    out_png: Path,
    dt_days: np.ndarray,
    runtime_by_key: dict[str, np.ndarray],
) -> None:
    """Plot wall time vs dt in log-log scale."""

    def marker_for_key(key: str) -> str:
        if "split A" in key:
            return "o"
        if "split B" in key:
            return "s"
        if "split C" in key:
            return "^"
        return "x"

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    for name, times in runtime_by_key.items():
        ax.plot(dt_days, times, marker=marker_for_key(name), label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Output dt (days)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time vs output dt (log-log)")
    ax.grid(visible=True, which="both")
    ax.legend(fontsize=8, loc="best")

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_runtime_reports(
    out_txt: Path,
    out_csv: Path,
    report: RuntimeReportSpec,
) -> None:
    """Write runtime summaries to a text report and a CSV table."""
    lines: list[str] = []
    lines.extend([
        "Timing / solver stats",
        "",
        f"Operator mode: {report.operator_mode}",
        "",
        "Main run wall times:",
    ])

    lines.extend([f"{k:45s} {v:.6f} s" for k, v in report.main_timings.items()])

    lines.extend(["", "SciPy solve_ivp details (main run):"])
    for name, info in report.scipy_infos.items():
        lines.append(f"  {name}")
        keys = ("success", "message", "nfev", "njev", "nlu", "wall_s")
        lines.extend([f"    {kk:8s}: {info.get(kk)}" for kk in keys])
        lines.append("")

    keys = list(report.sweep_timings.keys())
    header = "dt_days," + ",".join(keys)
    csv_lines: list[str] = [header]

    for i, dt in enumerate(report.dt_days):
        row = [f"{float(dt):.10g}"]
        row.extend([f"{float(report.sweep_timings[k][i]):.10g}" for k in keys])
        csv_lines.append(",".join(row))

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_csv.write_text("\n".join(csv_lines), encoding="utf-8")


def write_manifest(out_json: Path, manifest: Manifest) -> None:
    """Write a JSON manifest describing outputs and settings."""
    out_json.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )


# =============================================================================
# Canonical runs + sweep orchestration
# =============================================================================


def build_model_and_run_spec(
    *, seed: int = 123
) -> tuple[ModelSpec, RunSpec, float, float, int]:
    """Build ModelSpec and RunSpec for this example.

    Returns:
        (model, run, dt_out_main_days, total_time_days, n_bins)
    """
    n_bins = 8
    prd, zrd = _size_bins(n_bins)
    params = ParamsA()

    # Initial condition: small random perturbation
    u0 = np.full(2 * n_bins, 0.1, dtype=float)
    rng = np.random.default_rng(seed)
    perturb_scale = 0.2
    y0_flat = np.clip(
        u0 * (1.0 + perturb_scale * rng.standard_normal(u0.shape)), 0.0, None
    )

    total_time_days = 365.0 * 1.0
    dt_out_main_days = 5.0
    time_grid = build_uniform_time_grid_days(total_time_days, dt_out_main_days)

    model = ModelSpec(params=params, prd=prd, zrd=zrd, n_bins=n_bins)
    run = RunSpec(
        time_grid=time_grid,
        y0_flat=y0_flat,
        rhs_tensor=make_reaction_tensor(model),
        rhs_flat=make_rhs_flat(model),
    )
    return model, run, dt_out_main_days, total_time_days, n_bins


def run_and_save_canonical_plots(
    *,
    out_dir: Path,
    model: ModelSpec,
    run: RunSpec,
    default_split_for_plots: SplitName = "B",
) -> tuple[dict[str, float], dict[str, dict[str, object]], dict[str, Path]]:
    """Run and save the three canonical comparison plots.

    Returns:
        (timings, scipy_infos, output_paths)

    Raises:
        RuntimeError: If a required trajectory for plotting is missing.
    """
    out_bins_explicit = out_dir / "biogeo_explicit_bins.png"
    out_bins_imex1 = out_dir / "biogeo_imex1_bins.png"
    out_bins_imex2 = out_dir / "biogeo_imex2_bins.png"

    timings: dict[str, float] = {}
    scipy_infos: dict[str, dict[str, object]] = {}

    # SciPy references (needed for overlays)
    y_rk45, info_rk45 = run_scipy(method="RK45", run=run)
    y_bdf, info_bdf = run_scipy(method="BDF", run=run)
    scipy_infos["solve_ivp: RK45"] = info_rk45
    scipy_infos["solve_ivp: BDF"] = info_bdf
    timings["solve_ivp: RK45"] = float(info_rk45["wall_s"])
    timings["solve_ivp: BDF"] = float(info_bdf["wall_s"])

    # Plot 1: Explicit (heun) vs RK45
    y_heun, wall = run_op_engine(
        method="heun", run=run, model=model, store_history=True
    )
    timings["op_engine: heun (adaptive)"] = wall
    if y_heun is None:
        msg = _EXPECTED_TRAJ_ERROR.format(name="heun")
        raise RuntimeError(msg)

    save_bins_figure(
        out_bins_explicit,
        run.time_grid,
        model.n_bins,
        main_name="op_engine heun (adaptive)",
        main_arr=y_heun,
        scipy_name="SciPy RK45",
        scipy_arr=y_rk45,
    )

    # Plot 2: IMEX 1st-order (imex-euler split B) vs BDF
    y_imex_euler_b, wall = run_op_engine(
        method="imex-euler",
        run=run,
        model=model,
        split=default_split_for_plots,
        store_history=True,
    )
    timings[f"op_engine: imex-euler split {default_split_for_plots} (adaptive)"] = wall
    if y_imex_euler_b is None:
        msg = _EXPECTED_TRAJ_ERROR.format(name="imex-euler")
        raise RuntimeError(msg)

    save_bins_figure(
        out_bins_imex1,
        run.time_grid,
        model.n_bins,
        main_name=f"op_engine imex-euler split {default_split_for_plots} (adaptive)",
        main_arr=y_imex_euler_b,
        scipy_name="SciPy BDF",
        scipy_arr=y_bdf,
    )

    # Plot 3: IMEX 2nd-order (imex-trbdf2 split B) vs BDF
    y_trbdf2_b, wall = run_op_engine(
        method="imex-trbdf2",
        run=run,
        model=model,
        split=default_split_for_plots,
        store_history=True,
    )
    timings[f"op_engine: imex-trbdf2 split {default_split_for_plots} (adaptive)"] = wall
    if y_trbdf2_b is None:
        msg = _EXPECTED_TRAJ_ERROR.format(name="imex-trbdf2")
        raise RuntimeError(msg)

    save_bins_figure(
        out_bins_imex2,
        run.time_grid,
        model.n_bins,
        main_name=f"op_engine imex-trbdf2 split {default_split_for_plots} (adaptive)",
        main_arr=y_trbdf2_b,
        scipy_name="SciPy BDF",
        scipy_arr=y_bdf,
    )

    # Time (no-history) the remaining op_engine cases on the main grid
    timings["op_engine: euler (adaptive)"] = run_op_engine(
        method="euler",
        run=run,
        model=model,
        store_history=False,
    )[1]

    for split in ("A", "B", "C"):
        for meth in ("imex-euler", "imex-heun-tr", "imex-trbdf2"):
            if split == default_split_for_plots and meth in {
                "imex-euler",
                "imex-trbdf2",
            }:
                continue
            key = f"op_engine: {meth} split {split} (adaptive)"
            timings[key] = run_op_engine(
                method=meth,
                run=run,
                model=model,
                split=split,  # type: ignore[arg-type]
                store_history=False,
            )[1]

    output_paths = {
        "biogeo_explicit_bins": out_bins_explicit,
        "biogeo_imex1_bins": out_bins_imex1,
        "biogeo_imex2_bins": out_bins_imex2,
    }
    return timings, scipy_infos, output_paths


def run_dt_sweep(
    *,
    model: ModelSpec,
    run: RunSpec,
    dt_days: np.ndarray,
    sweep_total_days: float,
    default_split_for_plots: SplitName,
) -> dict[str, np.ndarray]:
    """Run a dt sweep, storing only wall times (no trajectories).

    Returns:
        Mapping key -> wall time array, each shape dt_days.shape.
    """
    sweep_timings: dict[str, np.ndarray] = {}

    keys: list[str] = [
        "op_engine: euler",
        "op_engine: heun",
        "solve_ivp: RK45",
        "solve_ivp: BDF",
    ]
    for split in ("A", "B", "C"):
        for meth in ("imex-euler", "imex-heun-tr", "imex-trbdf2"):
            keys.append(f"op_engine: {meth} split {split}")

    for k in keys:
        sweep_timings[k] = np.zeros(dt_days.shape, dtype=float)

    # Use the same y0 and RHS, but regenerate time_grid each iteration.
    for i, dt in enumerate(dt_days):
        tg = build_uniform_time_grid_days(sweep_total_days, float(dt))
        run_i = RunSpec(
            time_grid=tg,
            y0_flat=run.y0_flat,
            rhs_tensor=run.rhs_tensor,
            rhs_flat=run.rhs_flat,
        )

        sweep_timings["op_engine: euler"][i] = run_op_engine(
            method="euler",
            run=run_i,
            model=model,
            store_history=False,
        )[1]

        sweep_timings["op_engine: heun"][i] = run_op_engine(
            method="heun",
            run=run_i,
            model=model,
            store_history=False,
        )[1]

        for split in ("A", "B", "C"):
            for meth in ("imex-euler", "imex-heun-tr", "imex-trbdf2"):
                key = f"op_engine: {meth} split {split}"
                sweep_timings[key][i] = run_op_engine(
                    method=meth,
                    run=run_i,
                    model=model,
                    split=split,  # type: ignore[arg-type]
                    store_history=False,
                )[1]

        y_rk, info_rk = run_scipy(method="RK45", run=run_i)
        _ = y_rk
        sweep_timings["solve_ivp: RK45"][i] = float(info_rk["wall_s"])

        y_bdf, info_bdf = run_scipy(method="BDF", run=run_i)
        _ = y_bdf
        sweep_timings["solve_ivp: BDF"][i] = float(info_bdf["wall_s"])

    # Ensure the main plotted IMEX keys exist (nice legend consistency).
    _ = default_split_for_plots
    return sweep_timings


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the three canonical comparison plots and a dt runtime sweep."""
    out_dir = ensure_output_dir()

    model, run, dt_out_main_days, total_time_days, n_bins = build_model_and_run_spec()

    out_runtime = out_dir / "biogeo_runtime.png"
    out_txt = out_dir / "biogeo_runtime.txt"
    out_csv = out_dir / "biogeo_runtime.csv"
    out_manifest = out_dir / "biogeo_manifest.json"

    default_split_for_plots: SplitName = "B"

    timings, scipy_infos, plot_paths = run_and_save_canonical_plots(
        out_dir=out_dir,
        model=model,
        run=run,
        default_split_for_plots=default_split_for_plots,
    )

    sweep_total_days = 365.0 * 1.0
    dt_days = np.asarray(
        dt_out_main_days * np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=float),
        dtype=float,
    )

    sweep_timings = run_dt_sweep(
        model=model,
        run=run,
        dt_days=dt_days,
        sweep_total_days=sweep_total_days,
        default_split_for_plots=default_split_for_plots,
    )

    save_runtime_sweep_plot(out_runtime, dt_days, sweep_timings)

    report = RuntimeReportSpec(
        operator_mode=OPERATOR_MODE,
        main_timings=timings,
        scipy_infos=scipy_infos,
        dt_days=dt_days,
        sweep_timings=sweep_timings,
    )
    write_runtime_reports(out_txt, out_csv, report)

    outputs = {
        **{k: str(v) for k, v in plot_paths.items()},
        "biogeo_runtime": str(out_runtime),
        "biogeo_runtime_txt": str(out_txt),
        "biogeo_runtime_csv": str(out_csv),
    }

    manifest = Manifest(
        operator_mode=OPERATOR_MODE,
        eta_cross=ETA_CROSS,
        clip_nonnegative=CLIP_NONNEGATIVE,
        fail_fast_on_nonfinite=FAIL_FAST_ON_NONFINITE,
        scipy_rtol=SCIPY_RTOL,
        scipy_atol=SCIPY_ATOL,
        ope_rtol=OPE_RTOL,
        ope_atol=OPE_ATOL,
        n_bins=n_bins,
        dt_out_main_days=float(dt_out_main_days),
        total_time_days=float(total_time_days),
        sweep_total_days=float(sweep_total_days),
        dt_sweep_days=[float(x) for x in dt_days],
        outputs=outputs,
    )
    write_manifest(out_manifest, manifest)


if __name__ == "__main__":
    main()
