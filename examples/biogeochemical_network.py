"""
Example: Size-structured phytoplankton–zooplankton network
using op_engine's ModelCore/CoreSolver + matrix_ops utilities.

We run 4 configurations and compare against SciPy solve_ivp:

1) op_engine baseline: IMEX with A=0  -> pure explicit Heun (run_imex with operators=None)
2) op_engine + diffusion regularizer along size (CN operators on size axis)
3) op_engine + explicit smooth() filter along size (post-step smoothing)
4) SciPy solve_ivp: RK45 and BDF

State layout (tensor):
    state[compartment, subgroup, size_bin]
where compartment 0=P, 1=Z; subgroup is 1; size bins = 8.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc

from op_engine.core_solver import CoreSolver, ReactionRHSFunction
from op_engine.matrix_ops import build_crank_nicolson_operator, build_laplacian_tridiag, smooth
from op_engine.model_core import ModelCore, ModelCoreOptions


# ----------------------------
# Parameters / helpers
# ----------------------------

@dataclass(frozen=True)
class ParamsA:
    # Ground-truth parameters (your list)
    N_T: float = 5.0
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
    sea: float = 0.2


def _size_bins(np_: int) -> tuple[np.ndarray, np.ndarray]:
    """Phytoplankton and zooplankton diameter-based size bins."""
    # Your nominal construction
    prd = np.linspace(np.emath.logn(4, 1), np.emath.logn(4, 100), np_)
    prd = 4 ** prd
    zrd = 10.0 * prd
    return np.asarray(prd, dtype=float), np.asarray(zrd, dtype=float)


def mu_maxes(a: float, b: float, p: np.ndarray) -> np.ndarray:
    return a * (p ** (-b))


def KNs(a: float, b: float, p: np.ndarray) -> np.ndarray:
    return a * (p ** b)


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


def unpack_tensor_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    state: (2, 1, n_bins) -> returns (P, Z) each shape (n_bins,)
    """
    p = state[0, 0, :]
    z = state[1, 0, :]
    return p, z


def pack_tensor_state(p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    p, z: shape (n_bins,) -> tensor (2, 1, n_bins)
    """
    out = np.zeros((2, 1, p.size), dtype=float)
    out[0, 0, :] = p
    out[1, 0, :] = z
    return out


# ----------------------------
# Model A reaction term (tensor form)
# ----------------------------

def make_reaction_A(
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
) -> ReactionRHSFunction:
    """
    Returns F(t, state_tensor) with state_tensor shape (2, 1, n_bins),
    output same shape.
    """
    mu_max_ = mu_maxes(params.mu_a, params.mu_b, prd)          # (np_,)
    kn_ = KNs(params.k_a, params.k_b, prd)                     # (np_,)
    g_ = gs(params.g_a, params.g_b, zrd)                       # (nz_,)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)       # (nz_,)

    # Precompute feeding preference kernel matrix K[i, j] = rho(p_i, z_j)
    # Shape: (np_, nz_)
    K = rho(params.r_G, params.sigma_G, prd[:, None], zrd[None, :])

    def reaction(t: float, state: np.ndarray) -> np.ndarray:
        p, z = unpack_tensor_state(state)  # each (n_bins,)

        total = float(p.sum() + z.sum())
        # Nutrient-like availability (your term N_T - sum(u))
        avail = params.N_T - total

        # Seasonal multiplier
        season = 1.0 + params.sea * np.sin(2.0 * np.pi * t / 365.0)

        # Phytoplankton growth term per bin i:
        # season * mu_max[i] * (avail/(kn[i]+avail)) * p[i]
        # Guard against negative avail (if total exceeds N_T, fraction can go negative)
        frac = avail / (kn_ + avail)

        growth = season * mu_max_ * frac * p                       # (np_,)
        loss_lin = params.lam * p                                   # (np_,)

        # Grazing on phytoplankton i: p[i] * sum_j g[j]*K[i,j]*z[j]
        graze = p * (K @ (g_ * z))                                  # (np_,)

        dp = growth - loss_lin - graze

        # Zooplankton bin j:
        # z[j] * g[j] * (gamma * sum_i K[i,j]*p[i]) - delta[j]*z[j]
        intake = (K.T @ p)                                          # (nz_,)
        dz = z * g_ * (params.gamma * intake) - delta_ * z          # (nz_,)

        return pack_tensor_state(dp, dz)

    return reaction


# ----------------------------
# SciPy RHS (flat vector form)
# ----------------------------

def make_rhs_A_flat(
    params: ParamsA,
    prd: np.ndarray,
    zrd: np.ndarray,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Flat state u = [P_0..P_{n-1}, Z_0..Z_{n-1}] (length 2*n).
    """
    mu_max_ = mu_maxes(params.mu_a, params.mu_b, prd)
    kn_ = KNs(params.k_a, params.k_b, prd)
    g_ = gs(params.g_a, params.g_b, zrd)
    delta_ = deltas(params.delta_a, params.delta_b, zrd)
    K = rho(params.r_G, params.sigma_G, prd[:, None], zrd[None, :])  # (n,n)

    n = prd.size

    def rhs(t: float, u: np.ndarray) -> np.ndarray:
        p = u[:n]
        z = u[n:]
        total = float(p.sum() + z.sum())
        avail = params.N_T - total
        season = 1.0 + params.sea * np.sin(2.0 * np.pi * t / 365.0)

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
        return out

    return rhs


# ----------------------------
# op_engine runs
# ----------------------------

def run_op_engine_baseline(
    time_grid: np.ndarray,
    y0_tensor: np.ndarray,
    reaction: ReactionRHSFunction,
) -> np.ndarray:
    """IMEX with A=0 => explicit Heun only (operators=None)."""
    opts = ModelCoreOptions(
        other_axes=(y0_tensor.shape[-1],),
        axis_names=("compartment", "subgroup", "size"),
        store_history=True,
        dtype=np.float64,
    )
    core = ModelCore(n_states=2, n_subgroups=1, time_grid=time_grid, options=opts)
    core.set_initial_state(y0_tensor)

    solver = CoreSolver(core, operators=None)
    solver.run_imex(reaction)

    assert core.state_array is not None
    return core.state_array


def run_op_engine_with_size_diffusion(
    time_grid: np.ndarray,
    y0_tensor: np.ndarray,
    reaction: ReactionRHSFunction,
    epsilon: float,
) -> np.ndarray:
    """
    Add tiny 'diffusion' regularizer along size bins via CN operators.
    Operators act along axis 'size'.
    """
    n_bins = y0_tensor.shape[-1]
    dt = float(time_grid[1] - time_grid[0])

    # Laplacian along size bins; dx=1 is fine for synthetic axis
    # We build CN operators for A = epsilon * Laplacian.
    geom_n = n_bins
    geom_dx = 1.0

    # We can reuse your CN operator builder by treating size bins as 1D grid.
    # build_crank_nicolson_operator internally uses build_laplacian_tridiag.
    from op_engine.matrix_ops import GridGeometry, DiffusionConfig

    geom = GridGeometry(n=geom_n, dx=geom_dx)
    cfg = DiffusionConfig(coeff=epsilon, bc="neumann")

    left_op, right_op = build_crank_nicolson_operator(geom, cfg, dt)

    opts = ModelCoreOptions(
        other_axes=(n_bins,),
        axis_names=("compartment", "subgroup", "size"),
        store_history=True,
        dtype=np.float64,
    )
    core = ModelCore(n_states=2, n_subgroups=1, time_grid=time_grid, options=opts)
    core.set_initial_state(y0_tensor)

    solver = CoreSolver(core, operators=(left_op, right_op), operator_axis="size")
    solver.run_imex(reaction)

    assert core.state_array is not None
    return core.state_array


def run_op_engine_with_smooth_filter(
    time_grid: np.ndarray,
    y0_tensor: np.ndarray,
    reaction: ReactionRHSFunction,
    alpha: float,
) -> np.ndarray:
    """
    Explicit Heun IMEX (A=0), plus post-step smooth() along size axis.
    This uses the same mathematics as CoreSolver.run_imex, but we insert
    a filter before committing each step.
    """
    opts = ModelCoreOptions(
        other_axes=(y0_tensor.shape[-1],),
        axis_names=("compartment", "subgroup", "size"),
        store_history=True,
        dtype=np.float64,
    )
    core = ModelCore(n_states=2, n_subgroups=1, time_grid=time_grid, options=opts)
    core.set_initial_state(y0_tensor)

    # Buffers
    state_pred = np.zeros_like(y0_tensor)
    f_n = np.zeros_like(y0_tensor)
    f_pred = np.zeros_like(y0_tensor)
    x = np.zeros_like(y0_tensor)

    for idx in range(core.n_timesteps - 1):
        t_n = float(time_grid[idx])
        t_np1 = float(time_grid[idx + 1])
        dt_n = float(core.get_dt(idx))
        y_n = core.get_current_state()

        np.copyto(f_n, reaction(t_n, y_n))
        np.copyto(state_pred, y_n + dt_n * f_n)
        np.copyto(f_pred, reaction(t_np1, state_pred))

        # Heun combination
        np.copyto(x, y_n + 0.5 * dt_n * (f_n + f_pred))

        # Apply size smoothing on the last axis for each compartment.
        # smooth() operates along the last axis; state is (2,1,n_bins).
        x_sm = smooth(x, alpha=alpha)
        core.advance_timestep(x_sm)

    assert core.state_array is not None
    return core.state_array


# ----------------------------
# SciPy runs
# ----------------------------

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
        rtol=1e-6,
        atol=1e-9,
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

    # sol.y shape: (n_state, n_times)
    y = sol.y.T  # (n_times, n_state)
    return y, info


# ----------------------------
# Plotting
# ----------------------------

def plot_summary(time_grid: np.ndarray, results: dict[str, np.ndarray], n_bins: int) -> None:
    """
    results:
        - op_engine tensors: (T, 2, 1, n_bins)
        - scipy flats: (T, 2*n_bins)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Total biomass over time
    ax = axes[0, 0]
    for name, arr in results.items():
        if arr.ndim == 4:
            p = arr[:, 0, 0, :].sum(axis=-1)
            z = arr[:, 1, 0, :].sum(axis=-1)
        else:
            p = arr[:, :n_bins].sum(axis=-1)
            z = arr[:, n_bins:].sum(axis=-1)
        ax.plot(time_grid, p + z, label=name)
    ax.set_title("Total biomass (P+Z)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sum")
    ax.grid(True)
    ax.legend()

    # Total P and Z over time for a subset (to reduce clutter)
    ax = axes[0, 1]
    for name, arr in results.items():
        if arr.ndim == 4:
            p = arr[:, 0, 0, :].sum(axis=-1)
            z = arr[:, 1, 0, :].sum(axis=-1)
        else:
            p = arr[:, :n_bins].sum(axis=-1)
            z = arr[:, n_bins:].sum(axis=-1)
        ax.plot(time_grid, p, label=f"{name}: P")
        ax.plot(time_grid, z, linestyle="--", label=f"{name}: Z")
    ax.set_title("Totals by compartment")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sum")
    ax.grid(True)
    ax.legend(ncol=2)

    # Final size spectra (P)
    ax = axes[1, 0]
    for name, arr in results.items():
        if arr.ndim == 4:
            p_final = arr[-1, 0, 0, :]
        else:
            p_final = arr[-1, :n_bins]
        ax.plot(np.arange(n_bins), p_final, marker="o", label=name)
    ax.set_title("Final P size spectrum")
    ax.set_xlabel("Size bin")
    ax.set_ylabel("P")
    ax.grid(True)
    ax.legend()

    # Final size spectra (Z)
    ax = axes[1, 1]
    for name, arr in results.items():
        if arr.ndim == 4:
            z_final = arr[-1, 1, 0, :]
        else:
            z_final = arr[-1, n_bins:]
        ax.plot(np.arange(n_bins), z_final, marker="o", label=name)
    ax.set_title("Final Z size spectrum")
    ax.set_xlabel("Size bin")
    ax.set_ylabel("Z")
    ax.grid(True)
    ax.legend()

    plt.show()


# ----------------------------
# Main: build, perturb, run, compare
# ----------------------------

def main() -> None:
    np_ = 8
    prd, zrd = _size_bins(np_)
    params = ParamsA()

    # Your initial conditions (flat length 16)
    u0_A = np.array(
        [
            0.16305588, 0.15818029, 0.16480592, 0.17219893, 0.18134631,
            0.27084414, 0.0, 0.0,
            0.30808019, 0.18755108, 0.17126791, 0.13948308, 0.09182324,
            0.02681242, 0.0, 0.0,
        ],
        dtype=float,
    )

    # Optional perturbation: push some mass around to showcase equilibration
    rng = np.random.default_rng(123)
    perturb_scale = 0.05
    u0_pert = np.clip(u0_A * (1.0 + perturb_scale * rng.standard_normal(u0_A.shape)), 0.0, None)

    # Tensor initial state: (2, 1, n_bins)
    p0 = u0_pert[:np_]
    z0 = u0_pert[np_:]
    y0_tensor = pack_tensor_state(p0, z0)

    # Time grid (choose something moderate for demo; increase for “slow equilibration”)
    total_time = 365.0 * 2.0  # 2 years
    n_steps = 2001            # uniform dt
    time_grid = np.linspace(0.0, total_time, n_steps, dtype=float)

    reaction = make_reaction_A(params, prd, zrd)
    rhs_flat = make_rhs_A_flat(params, prd, zrd)
    y0_flat = u0_pert.copy()

    # --- Run op_engine variants ---
    start = time.perf_counter()
    out_baseline = run_op_engine_baseline(time_grid, y0_tensor, reaction)
    t_baseline = time.perf_counter() - start

    start = time.perf_counter()
    out_diff = run_op_engine_with_size_diffusion(
        time_grid, y0_tensor, reaction, epsilon=1e-4
    )
    t_diff = time.perf_counter() - start

    start = time.perf_counter()
    out_smooth = run_op_engine_with_smooth_filter(
        time_grid, y0_tensor, reaction, alpha=0.01
    )
    t_smooth = time.perf_counter() - start

    # --- Run SciPy ---
    out_rk45, info_rk45 = run_scipy("RK45", time_grid, y0_flat, rhs_flat)
    out_bdf, info_bdf = run_scipy("BDF", time_grid, y0_flat, rhs_flat)

    print("\nTiming / solver stats")
    print(f"op_engine baseline (Heun, A=0):      {t_baseline:.3f}s")
    print(f"op_engine + size diffusion (CN):     {t_diff:.3f}s")
    print(f"op_engine + smooth(alpha=0.01):      {t_smooth:.3f}s")
    print(f"SciPy RK45:                          {info_rk45['wall_s']:.3f}s  nfev={info_rk45['nfev']}")
    print(f"SciPy BDF:                           {info_bdf['wall_s']:.3f}s  nfev={info_bdf['nfev']}  njev={info_bdf['njev']}  nlu={info_bdf['nlu']}")

    # Collect results for plotting
    results = {
        "op_engine: baseline": out_baseline,
        "op_engine: +diffusion": out_diff,
        "op_engine: +smooth": out_smooth,
        "solve_ivp: RK45": out_rk45,
        "solve_ivp: BDF": out_bdf,
    }

    plot_summary(time_grid, results, n_bins=np_)


if __name__ == "__main__":
    main()
