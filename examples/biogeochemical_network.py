"""
Example: Size-structured phytoplankton–zooplankton network
using op_engine's ModelCore/CoreSolver + matrix_ops utilities.

UPDATED to leverage:
- ModelCore axis helpers + reshape conventions
- CoreSolver TR-BDF2 staging
- matrix_ops stage-operator factories with StageOperatorContext
- optional time-varying and stage/state-dependent implicit operators

We run solver configurations and compare against SciPy solve_ivp:

A) op_engine baseline: IMEX with A=0  -> pure explicit Heun (run_imex with operators=None)
B) op_engine TR-BDF2 split A (dissipative diagonal damping)
C) op_engine TR-BDF2 split B (diagonal + grazing sink on P, damping on Z)
D) op_engine TR-BDF2 split C (dense cross-coupled, dissipative-by-construction)
E) SciPy solve_ivp: RK45
F) SciPy solve_ivp: BDF

Operator modes (choose one):
- "frozen": A built once from y0 (and t0) and reused for all stages
- "time":   A depends on stage time t (simple seasonal modulation of damping)
- "stage_state": A depends on stage time t and stage state y (relinearize per stage)

Outputs:
    - outputs/biogeochemical_network.png
    - outputs/biogeochemical_network_bins_P.png
    - outputs/biogeochemical_network_bins_Z.png
    - outputs/biogeochemical_network_runtime.png
    - outputs/biogeochemical_network_runtime.txt
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc

from op_engine.core_solver import CoreSolver, ReactionRHSFunction
from op_engine.matrix_ops import (
    StageOperatorContext,
    make_constant_base_builder,
    make_stage_operator_factory,
)
from op_engine.model_core import ModelCore, ModelCoreOptions


# =============================================================================
# Configuration knobs for the example
# =============================================================================

OperatorMode = Literal["frozen", "time", "stage_state"]

# Choose how the implicit operator A is constructed during TR-BDF2 stages:
OPERATOR_MODE: OperatorMode = "time"

# Split-C cross coupling scaling (kept small for robustness)
ETA_CROSS: float = 0.10

# Example-only hygiene
CLIP_NONNEGATIVE: bool = True
FAIL_FAST_ON_NONFINITE: bool = True

# Solver tolerances for SciPy reference
SCIPY_RTOL: float = 1e-6
SCIPY_ATOL: float = 1e-9


# =============================================================================
# Parameters / helpers
# =============================================================================


@dataclass(frozen=True)
class ParamsA:
    N_T: float = 10.0
    gamma: float = 0.5
    r_G: float = 10.0
    sigma_G: float = 0.5
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


def _size_bins(n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Phytoplankton and zooplankton diameter-based size bins."""
    prd = np.linspace(np.emath.logn(4, 1), np.emath.logn(4, 100), n_bins)
    prd = 4**prd
    zrd = 10.0 * prd
    return np.asarray(prd, dtype=float), np.asarray(zrd, dtype=float)


def mu_maxes(a: float, b: float, p: np.ndarray) -> np.ndarray:
    return a * (p ** (-b))


def KNs(a: float, b: float, p: np.ndarray) -> np.ndarray:
    return a * (p**b)


def gs(a: float, b: float, z: np.ndarray) -> np.ndarray:
    return a * (z ** (-b))


def deltas(a: float, b: float, z: np.ndarray) -> np.ndarray:
    return a * (z ** (-b))


def rho(r: float, sigma: float, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Feeding kernel (vectorized).
    Returns erfc(|z/p - r| / (sqrt(2) * sigma * r)).
    Broadcasting-friendly:
        p shape (..., np_)
        z shape (..., nz_)
    """
    y = np.abs(z / p - r) / (np.sqrt(2.0) * (sigma * r))
    return erfc(y)


def unpack_flat_state(u: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """u: shape (2*n_bins,) -> returns (P, Z) each shape (n_bins,)"""
    p = u[:n_bins]
    z = u[n_bins:]
    return p, z


def pack_flat_state(p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """p, z: shape (n_bins,) -> flat shape (2*n_bins,)"""
    return np.concatenate([np.asarray(p, dtype=float), np.asarray(z, dtype=float)], axis=0)


def tensor_from_flat(u: np.ndarray) -> np.ndarray:
    """Flat (2n,) -> tensor (2n, 1) for ModelCore."""
    return np.asarray(u, dtype=float).reshape(-1, 1)


def flat_from_tensor(state: np.ndarray) -> np.ndarray:
    """Tensor (2n, 1) -> flat (2n,)."""
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
    raise FloatingPointError(
        f"{name} contains inf/NaN{t_msg}. min={mn:.6g}, max={mx:.6g}, first_bad_index={idx}"
    )


def _seasonal_factor(params: ParamsA, t_days: float) -> float:
    """Always-positive seasonal modulation factor."""
    return float(1.0 + params.sea * np.sin(2.0 * np.pi * t_days / 365.0))


# =============================================================================
# Model A reaction term (tensor wrapper for op_engine)
# =============================================================================


def make_reaction_A_flat_tensor(
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
) -> ReactionRHSFunction:
    """
    Reaction RHS for op_engine, operating on ModelCore tensor state (2*n_bins, 1),
    returning tensor RHS (2*n_bins, 1).
    """
    mu_max_ = mu_maxes(params.mu_a, params.mu_b, prd)
    kn_ = KNs(params.k_a, params.k_b, prd)
    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    K = rho(params.r_G, params.sigma_G, prd[:, None], zrd[None, :])

    n_bins = int(prd.size)

    def reaction(t: float, state_tensor: np.ndarray) -> np.ndarray:
        u = flat_from_tensor(state_tensor)
        _assert_finite("state", u, t=t)

        p, z = unpack_flat_state(u, n_bins)
        if CLIP_NONNEGATIVE:
            p = np.clip(p, 0.0, None)
            z = np.clip(z, 0.0, None)

        total = float(p.sum() + z.sum())
        avail = params.N_T - total

        season = _seasonal_factor(params, t)
        frac = avail / (kn_ + avail)

        growth = season * mu_max_ * frac * p
        loss_lin = params.lam * p
        graze = p * (K @ (g_ * z))
        dp = growth - loss_lin - graze

        intake = (K.T @ p)
        dz = z * g_ * (params.gamma * intake) - delta_ * z

        out = pack_flat_state(dp, dz)
        _assert_finite("reaction(state)", out, t=t)
        return tensor_from_flat(out)

    return reaction


# =============================================================================
# SciPy RHS (flat vector form)
# =============================================================================


def make_rhs_A_flat(
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
) -> Callable[[float, np.ndarray], np.ndarray]:
    mu_max_ = mu_maxes(params.mu_a, params.mu_b, prd)
    kn_ = KNs(params.k_a, params.k_b, prd)
    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    K = rho(params.r_G, params.sigma_G, prd[:, None], zrd[None, :])

    n = int(prd.size)

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        _assert_finite("state", u, t=t)
        p = u[:n]
        z = u[n:]
        if CLIP_NONNEGATIVE:
            p = np.clip(p, 0.0, None)
            z = np.clip(z, 0.0, None)

        total = float(p.sum() + z.sum())
        avail = params.N_T - total

        season = _seasonal_factor(params, t)
        frac = avail / (kn_ + avail)

        growth = season * mu_max_ * frac * p
        loss_lin = params.lam * p
        graze = p * (K @ (g_ * z))
        dp = growth - loss_lin - graze

        intake = (K.T @ p)
        dz = z * g_ * (params.gamma * intake) - delta_ * z

        out = np.empty_like(u)
        out[:n] = dp
        out[n:] = dz
        _assert_finite("rhs(u)", out, t=t)
        return out

    return rhs


# =============================================================================
# Split operators A / B / C as functions of (t, y_stage)
# =============================================================================


def _sanitize_stage_state(u_flat: np.ndarray, n_bins: int) -> np.ndarray:
    u = np.asarray(u_flat, dtype=float).reshape(-1)
    if CLIP_NONNEGATIVE:
        p, z = unpack_flat_state(u, n_bins)
        p = np.clip(p, 0.0, None)
        z = np.clip(z, 0.0, None)
        return pack_flat_state(p, z)
    return u


def A_split_A(params: ParamsA, prd: np.ndarray, zrd: np.ndarray, *, t: float, y: np.ndarray) -> np.ndarray:
    """
    Split A: dissipative diagonal linear damping only:
        dp includes -lam * p
        dz includes -delta(zbin) * z

    In "time" / "stage_state" modes, we apply a mild seasonal modulation to
    damping (still dissipative).
    """
    n = int(prd.size)
    season = _seasonal_factor(params, t) if OPERATOR_MODE in ("time", "stage_state") else 1.0

    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    lam_eff = float(params.lam * season)
    delta_eff = delta_ * season

    A = np.zeros((2 * n, 2 * n), dtype=float)
    A[:n, :n] = -lam_eff * np.eye(n, dtype=float)
    A[n:, n:] = -np.diag(delta_eff)
    return A


def A_split_B(params: ParamsA, prd: np.ndarray, zrd: np.ndarray, *, t: float, y: np.ndarray) -> np.ndarray:
    """
    Split B: diagonal-only, stronger damping using grazing sink on P:

      dp has -lam*p - G*p   where G_i = (K @ (g*z))_i
      dz has -delta*z       (kept purely damping for robustness)

    Time modulation (if enabled) scales lam/delta; grazing sink is state-derived.
    """
    n = int(prd.size)
    u = _sanitize_stage_state(y, n)

    p, z = unpack_flat_state(u, n)
    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    K = rho(params.r_G, params.sigma_G, prd[:, None], zrd[None, :])

    season = _seasonal_factor(params, t) if OPERATOR_MODE in ("time", "stage_state") else 1.0
    lam_eff = float(params.lam * season)
    delta_eff = delta_ * season

    G = K @ (g_ * z)  # (n,)

    A = np.zeros((2 * n, 2 * n), dtype=float)
    A[:n, :n] = -np.diag(lam_eff + G)
    A[n:, n:] = -np.diag(delta_eff)
    return A


def A_split_C(
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
    *,
    t: float,
    y: np.ndarray,
    eta_cross: float = ETA_CROSS,
) -> np.ndarray:
    """
    Split C: dense cross-coupled frozen/dynamic linear part, dissipative by construction.

    Key robustness choices:
      - Keep Z block as pure damping: A_zz = -diag(delta)
      - Keep P block dissipative: A_pp = -lam*I - diag(G)
      - Scale cross blocks by eta_cross

    If time-varying is enabled, lam/delta are seasonally modulated (still positive).
    """
    n = int(prd.size)
    u = _sanitize_stage_state(y, n)

    p, z = unpack_flat_state(u, n)

    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    K = rho(params.r_G, params.sigma_G, prd[:, None], zrd[None, :])

    season = _seasonal_factor(params, t) if OPERATOR_MODE in ("time", "stage_state") else 1.0
    lam_eff = float(params.lam * season)
    delta_eff = delta_ * season

    # Damping capture for P from grazing sink
    G = K @ (g_ * z)  # (n,)

    # Dissipative diagonal blocks
    A_pp = -lam_eff * np.eye(n, dtype=float) - np.diag(G)
    A_zz = -np.diag(delta_eff)

    # Cross blocks (scaled)
    A_pz = -(np.diag(p) @ (K @ np.diag(g_)))
    A_zp = (np.diag(z * g_ * params.gamma) @ K.T)

    A = np.zeros((2 * n, 2 * n), dtype=float)
    A[:n, :n] = A_pp
    A[n:, n:] = A_zz
    A[:n, n:] = float(eta_cross) * A_pz
    A[n:, :n] = float(eta_cross) * A_zp
    return A


# =============================================================================
# op_engine runs
# =============================================================================


def _make_core_flat(time_grid: np.ndarray, n_state: int) -> ModelCore:
    opts = ModelCoreOptions(
        other_axes=(),
        axis_names=("state", "subgroup"),
        store_history=True,
        dtype=np.float64,
    )
    return ModelCore(n_states=n_state, n_subgroups=1, time_grid=time_grid, options=opts)


def run_op_engine_baseline_heun_A0(
    time_grid: np.ndarray,
    y0_flat: np.ndarray,
    reaction_tensor: ReactionRHSFunction,
) -> np.ndarray:
    n_state = int(y0_flat.size)
    core = _make_core_flat(time_grid, n_state)
    core.set_initial_state(tensor_from_flat(y0_flat))

    solver = CoreSolver(core, operators=None, operator_axis="state")
    solver.run_imex(reaction_tensor)

    assert core.state_array is not None
    return np.asarray(core.state_array[:, :, 0], dtype=float)  # (T, 2n)


def _make_base_builder_for_split(
    split: Literal["A", "B", "C"],
    *,
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
    y0_flat: np.ndarray,
    eta_cross: float = ETA_CROSS,
) -> Callable[[StageOperatorContext], np.ndarray]:
    n_bins = int(prd.size)
    y0_sanitized = _sanitize_stage_state(y0_flat, n_bins)

    def _ctx_to_y(ctx: StageOperatorContext) -> np.ndarray:
        # ctx.y is expected to already be the flattened operator-axis state (1D).
        y = np.asarray(ctx.y, dtype=float).reshape(-1)
        if y.size != 2 * n_bins:
            # Be defensive: if someone changes ModelCore layout, fail loudly.
            raise ValueError(f"Stage state size mismatch: got {y.size}, expected {2*n_bins}")
        return y

    if OPERATOR_MODE == "frozen":
        t0 = float(0.0)
        if split == "A":
            A0 = A_split_A(params, prd, zrd, t=t0, y=y0_sanitized)
        elif split == "B":
            A0 = A_split_B(params, prd, zrd, t=t0, y=y0_sanitized)
        else:
            A0 = A_split_C(params, prd, zrd, t=t0, y=y0_sanitized, eta_cross=eta_cross)
        return make_constant_base_builder(A0)

    if OPERATOR_MODE == "time":

        def _builder(ctx: StageOperatorContext) -> np.ndarray:
            t = float(ctx.t)
            # In time mode, use y0 for any state dependence.
            y = y0_sanitized
            if split == "A":
                return A_split_A(params, prd, zrd, t=t, y=y)
            if split == "B":
                return A_split_B(params, prd, zrd, t=t, y=y)
            return A_split_C(params, prd, zrd, t=t, y=y, eta_cross=eta_cross)

        return _builder

    # OPERATOR_MODE == "stage_state"
    def _builder(ctx: StageOperatorContext) -> np.ndarray:
        t = float(ctx.t)
        y = _sanitize_stage_state(_ctx_to_y(ctx), n_bins)
        if split == "A":
            return A_split_A(params, prd, zrd, t=t, y=y)
        if split == "B":
            return A_split_B(params, prd, zrd, t=t, y=y)
        return A_split_C(params, prd, zrd, t=t, y=y, eta_cross=eta_cross)

    return _builder


def run_op_engine_trbdf2(
    time_grid: np.ndarray,
    y0_flat: np.ndarray,
    reaction_tensor: ReactionRHSFunction,
    *,
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
    split: Literal["A", "B", "C"],
    eta_cross: float = ETA_CROSS,
) -> np.ndarray:
    n_state = int(y0_flat.size)
    core = _make_core_flat(time_grid, n_state)
    core.set_initial_state(tensor_from_flat(y0_flat))

    base_builder = _make_base_builder_for_split(
        split,
        params=params,
        prd=prd,
        zrd=zrd,
        y0_flat=y0_flat,
        eta_cross=eta_cross,
    )

    tr_factory = make_stage_operator_factory(base_builder, scheme="trapezoidal")
    be_factory = make_stage_operator_factory(base_builder, scheme="implicit-euler")

    solver = CoreSolver(core, operators=None, operator_axis="state")
    solver.run_imex_trbdf2(
        reaction_tensor,
        operators_tr=tr_factory,
        operators_bdf2=be_factory,
    )

    assert core.state_array is not None
    return np.asarray(core.state_array[:, :, 0], dtype=float)  # (T, 2n)


# =============================================================================
# SciPy runs
# =============================================================================


def run_scipy(
    method: str,
    time_grid: np.ndarray,
    y0_flat: np.ndarray,
    rhs: Callable[[float, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, dict]:
    t0 = float(time_grid[0])
    t1 = float(time_grid[-1])

    start = time.perf_counter()
    sol = solve_ivp(
        fun=rhs,
        t_span=(t0, t1),
        y0=y0_flat,
        method=method,
        t_eval=time_grid,
        rtol=SCIPY_RTOL,
        atol=SCIPY_ATOL,
        vectorized=False,
    )
    wall = time.perf_counter() - start

    info = {
        "success": sol.success,
        "message": sol.message,
        "nfev": sol.nfev,
        "njev": getattr(sol, "njev", None),
        "nlu": getattr(sol, "nlu", None),
        "wall_s": wall,
    }

    if not sol.success:
        raise RuntimeError(f"solve_ivp({method}) failed: {sol.message}")

    y = sol.y.T
    return np.asarray(y, dtype=float), info


# =============================================================================
# Plotting / reporting
# =============================================================================


def _totals_PZ(arr: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    p = arr[:, :n_bins].sum(axis=-1)
    z = arr[:, n_bins:].sum(axis=-1)
    return p, z


def _extract_bins(arr: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    p = arr[:, :n_bins]
    z = arr[:, n_bins:]
    return p, z


def save_bin_overlay_figures(
    time_grid_days: np.ndarray,
    results: dict[str, np.ndarray],
    n_bins: int,
    out_png_p: Path,
    out_png_z: Path,
) -> None:
    t_years = np.asarray(time_grid_days, dtype=float) / 365.0

    fig_p, axes_p = plt.subplots(
        nrows=n_bins,
        ncols=1,
        figsize=(14, 1.1 * n_bins),
        sharex=True,
        constrained_layout=True,
    )
    if n_bins == 1:
        axes_p = [axes_p]

    for bin_idx in range(n_bins):
        ax = axes_p[bin_idx]
        for name, arr in results.items():
            p_bins, _ = _extract_bins(arr, n_bins)
            ax.plot(t_years, p_bins[:, bin_idx], label=name)
        ax.grid(True)
        ax.set_ylabel(f"bin {bin_idx}", rotation=0, labelpad=25, va="center")

    axes_p[0].set_title("P by size bin (solvers overlaid)")
    axes_p[-1].set_xlabel("Time (years)")

    handles, labels = axes_p[0].get_legend_handles_labels()
    fig_p.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=True)
    fig_p.savefig(out_png_p, dpi=200)
    plt.close(fig_p)

    fig_z, axes_z = plt.subplots(
        nrows=n_bins,
        ncols=1,
        figsize=(14, 1.1 * n_bins),
        sharex=True,
        constrained_layout=True,
    )
    if n_bins == 1:
        axes_z = [axes_z]

    for bin_idx in range(n_bins):
        ax = axes_z[bin_idx]
        for name, arr in results.items():
            _, z_bins = _extract_bins(arr, n_bins)
            ax.plot(t_years, z_bins[:, bin_idx], label=name)
        ax.grid(True)
        ax.set_ylabel(f"bin {bin_idx}", rotation=0, labelpad=25, va="center")

    axes_z[0].set_title("Z by size bin (solvers overlaid)")
    axes_z[-1].set_xlabel("Time (years)")

    handles, labels = axes_z[0].get_legend_handles_labels()
    fig_z.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=True)
    fig_z.savefig(out_png_z, dpi=200)
    plt.close(fig_z)


def save_plots_one_panel_per_solver(
    time_grid_days: np.ndarray,
    results: dict[str, np.ndarray],
    n_bins: int,
    out_png: Path,
) -> None:
    t_years = np.asarray(time_grid_days, dtype=float) / 365.0

    names = list(results.keys())
    n = len(names)

    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.2 * ncols, 3.6 * nrows),
        constrained_layout=True,
        sharex=True,
        sharey=False,
    )

    if isinstance(axes, np.ndarray):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = [axes]

    for ax, name in zip(ax_list, names):
        arr = results[name]
        p, z = _totals_PZ(arr, n_bins)

        ax.plot(t_years, p, label="P total")
        ax.plot(t_years, z, label="Z total", linestyle="--")

        ax.set_title(name)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Total biomass")
        ax.grid(True)
        ax.legend(fontsize=9, loc="best")

    for ax in ax_list[len(names) :]:
        ax.set_visible(False)

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_runtime_sweep_plot(
    out_png: Path,
    dt_days: np.ndarray,
    runtime_by_method: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)

    for name, times in runtime_by_method.items():
        ax.plot(dt_days, times, marker="o", label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dt (days)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time vs dt (log-log)")
    ax.grid(True, which="both")
    ax.legend(fontsize=9, loc="best")

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_runtime_report(
    out_txt: Path,
    timings: dict[str, float],
    scipy_infos: dict[str, dict],
    *,
    operator_mode: str,
    sweep_dt_days: np.ndarray | None = None,
    sweep_results: dict[str, np.ndarray] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("Timing / solver stats")
    lines.append("")
    lines.append(f"Operator mode: {operator_mode}")
    lines.append("")

    lines.append("Main run wall times:")
    for k, v in timings.items():
        lines.append(f"{k:45s} {v:.6f} s")

    lines.append("")
    lines.append("SciPy solve_ivp details (main run):")
    for name, info in scipy_infos.items():
        lines.append(f"  {name}")
        for kk in ("success", "message", "nfev", "njev", "nlu", "wall_s"):
            lines.append(f"    {kk:8s}: {info.get(kk)}")
        lines.append("")

    if sweep_dt_days is not None and sweep_results is not None:
        lines.append("")
        lines.append("dt sweep wall times:")
        header = "dt_days," + ",".join(sweep_results.keys())
        lines.append(header)
        for i, dt in enumerate(sweep_dt_days):
            row = [f"{float(dt):.10g}"]
            for name in sweep_results.keys():
                row.append(f"{float(sweep_results[name][i]):.10g}")
            lines.append(",".join(row))

    out_txt.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Time-grid utilities
# =============================================================================


def build_uniform_time_grid_days(total_time_days: float, dt_days: float) -> np.ndarray:
    dt = float(dt_days)
    n = int(math.floor(total_time_days / dt + 0.5)) + 1
    t = dt * np.arange(n, dtype=float)
    if abs(t[-1] - total_time_days) > 0.5 * dt:
        t = np.arange(0.0, total_time_days + 0.5 * dt, dt, dtype=float)
    else:
        t[-1] = float(total_time_days)
    return t


# =============================================================================
# Main: build, perturb, run, compare, sweep
# =============================================================================


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "biogeochemical_network.png"
    out_png_p = out_dir / "biogeochemical_network_bins_P.png"
    out_png_z = out_dir / "biogeochemical_network_bins_Z.png"
    out_png_runtime = out_dir / "biogeochemical_network_runtime.png"
    out_txt = out_dir / "biogeochemical_network_runtime.txt"

    # Model setup
    n_bins = 8
    prd, zrd = _size_bins(n_bins)
    params = ParamsA()

    u0 = np.full(2 * n_bins, 0.1, dtype=float)
    rng = np.random.default_rng(123)
    perturb_scale = 0.2
    y0_flat = np.clip(u0 * (1.0 + perturb_scale * rng.standard_normal(u0.shape)), 0.0, None)

    # Horizon
    total_time_days = 365.0 * 20.0

    # Keep dt fineness consistent with prior 10y/10001 grid:
    dt_target = (365.0 * 10.0) / 10000.0
    time_grid = build_uniform_time_grid_days(total_time_days, dt_target)
    dt0 = float(time_grid[1] - time_grid[0])

    # RHS for op_engine + SciPy
    reaction_tensor = make_reaction_A_flat_tensor(params, prd, zrd)
    rhs_flat = make_rhs_A_flat(params, prd, zrd)

    # -------------------------------------------------------------------------
    # Main runs
    # -------------------------------------------------------------------------
    timings: dict[str, float] = {}
    scipy_infos: dict[str, dict] = {}

    start = time.perf_counter()
    out_heun = run_op_engine_baseline_heun_A0(time_grid, y0_flat, reaction_tensor)
    timings["op_engine: Heun (A=0)"] = time.perf_counter() - start

    start = time.perf_counter()
    out_tr_a = run_op_engine_trbdf2(
        time_grid,
        y0_flat,
        reaction_tensor,
        params=params,
        prd=prd,
        zrd=zrd,
        split="A",
    )
    timings["op_engine: TR-BDF2 split A"] = time.perf_counter() - start

    start = time.perf_counter()
    out_tr_b = run_op_engine_trbdf2(
        time_grid,
        y0_flat,
        reaction_tensor,
        params=params,
        prd=prd,
        zrd=zrd,
        split="B",
    )
    timings["op_engine: TR-BDF2 split B"] = time.perf_counter() - start

    start = time.perf_counter()
    out_tr_c = run_op_engine_trbdf2(
        time_grid,
        y0_flat,
        reaction_tensor,
        params=params,
        prd=prd,
        zrd=zrd,
        split="C",
        eta_cross=ETA_CROSS,
    )
    timings["op_engine: TR-BDF2 split C (dissipative)"] = time.perf_counter() - start

    out_rk45, info_rk45 = run_scipy("RK45", time_grid, y0_flat, rhs_flat)
    scipy_infos["solve_ivp: RK45"] = info_rk45
    timings["solve_ivp: RK45"] = float(info_rk45["wall_s"])

    out_bdf, info_bdf = run_scipy("BDF", time_grid, y0_flat, rhs_flat)
    scipy_infos["solve_ivp: BDF"] = info_bdf
    timings["solve_ivp: BDF"] = float(info_bdf["wall_s"])

    results_main = {
        "op_engine: Heun (A=0)": out_heun,
        "op_engine: TR-BDF2 split A": out_tr_a,
        "op_engine: TR-BDF2 split B": out_tr_b,
        "op_engine: TR-BDF2 split C": out_tr_c,
        "solve_ivp: RK45": out_rk45,
        "solve_ivp: BDF": out_bdf,
    }

    # Plots
    save_plots_one_panel_per_solver(time_grid, results_main, n_bins=n_bins, out_png=out_png)
    save_bin_overlay_figures(
        time_grid_days=time_grid,
        results=results_main,
        n_bins=n_bins,
        out_png_p=out_png_p,
        out_png_z=out_png_z,
    )

    # -------------------------------------------------------------------------
    # dt sweep (wall time vs dt)
    # -------------------------------------------------------------------------
    factors = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    sweep_dt_days = dt0 * factors

    sweep_methods = [
        "op_engine: Heun (A=0)",
        "op_engine: TR-BDF2 split A",
        "op_engine: TR-BDF2 split B",
        "op_engine: TR-BDF2 split C",
        "solve_ivp: RK45",
        "solve_ivp: BDF",
    ]
    sweep_results: dict[str, np.ndarray] = {name: np.zeros_like(sweep_dt_days) for name in sweep_methods}

    for i, dt_days in enumerate(sweep_dt_days):
        tg = build_uniform_time_grid_days(total_time_days, float(dt_days))

        t0 = time.perf_counter()
        _ = run_op_engine_baseline_heun_A0(tg, y0_flat, reaction_tensor)
        sweep_results["op_engine: Heun (A=0)"][i] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = run_op_engine_trbdf2(tg, y0_flat, reaction_tensor, params=params, prd=prd, zrd=zrd, split="A")
        sweep_results["op_engine: TR-BDF2 split A"][i] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = run_op_engine_trbdf2(tg, y0_flat, reaction_tensor, params=params, prd=prd, zrd=zrd, split="B")
        sweep_results["op_engine: TR-BDF2 split B"][i] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = run_op_engine_trbdf2(
            tg, y0_flat, reaction_tensor, params=params, prd=prd, zrd=zrd, split="C", eta_cross=ETA_CROSS
        )
        sweep_results["op_engine: TR-BDF2 split C"][i] = time.perf_counter() - t0

        _, info = run_scipy("RK45", tg, y0_flat, rhs_flat)
        sweep_results["solve_ivp: RK45"][i] = float(info["wall_s"])

        _, info = run_scipy("BDF", tg, y0_flat, rhs_flat)
        sweep_results["solve_ivp: BDF"][i] = float(info["wall_s"])

    save_runtime_sweep_plot(out_png_runtime, sweep_dt_days, sweep_results)

    save_runtime_report(
        out_txt,
        timings=timings,
        scipy_infos=scipy_infos,
        operator_mode=OPERATOR_MODE,
        sweep_dt_days=sweep_dt_days,
        sweep_results=sweep_results,
    )


if __name__ == "__main__":
    main()
