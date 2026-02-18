"""
Diagnostic runner to compare op_engine IMEX splits vs fully implicit and SciPy.

Scenarios covered (seasonality disabled):
- op_engine imex-trbdf2 split B (eta_cross=0.10 and 1.0)
- op_engine imex-trbdf2 split C (eta_cross=1.0)
- op_engine bdf2 (fully implicit)
- SciPy BDF

Outputs:
- Max L2 trajectory difference vs SciPy BDF for each variant
- Final-state L2 difference vs SciPy BDF for each variant
- Wall-clock timings
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

import examples.biogeochemical_network as bio  # noqa: E402
from op_engine.core_solver import (  # noqa: E402
    AdaptiveConfig,
    DtControllerConfig,
    OperatorSpecs,
    RunConfig,
)
from op_engine.core_solver import CoreSolver as OpeCoreSolver  # noqa: E402
from op_engine.matrix_ops import make_stage_operator_factory  # noqa: E402

SplitName = Literal["A", "B", "C"]


def build_model_no_season(*, n_bins: int) -> bio.ModelSpec:
    """Build ModelSpec with seasonality disabled (sea=0).

    Returns:
        ModelSpec without seasonal forcing.
    """
    params = bio.ParamsA(sea=0.0)
    prd, zrd = bio._size_bins(n_bins)  # noqa: SLF001
    mu_max_ = bio.mu_maxes(params.mu_a, params.mu_b, prd)
    kn_ = bio.kns(params.k_a, params.k_b, prd)
    g_ = bio.gs(params.g_a, params.g_b, zrd)
    delta_ = bio.deltas(params.delta_a, params.delta_b, zrd)
    kern = bio.rho(params.r_g, params.sigma_g, prd[:, None], zrd[None, :])

    return bio.ModelSpec(
        params=params,
        prd=prd,
        zrd=zrd,
        n_bins=n_bins,
        mu_max_=mu_max_,
        kn_=kn_,
        g_=g_,
        delta_=delta_,
        kern=kern,
    )


def build_run_spec_no_season(
    *,
    model: bio.ModelSpec,
    seed: int,
    total_time_days: float,
    dt_out_days: float,
) -> bio.RunSpec:
    """Build RunSpec with randomized nonnegative initial condition.

    Returns:
        RunSpec configured for the diagnostic.
    """
    rng = np.random.default_rng(seed)
    n_bins = model.n_bins
    u0 = np.full(2 * n_bins, 0.1, dtype=float)
    perturb_scale = 0.2
    y0_flat = np.clip(
        u0 * (1.0 + perturb_scale * rng.standard_normal(u0.shape)), 0.0, None
    )

    time_grid = bio.build_uniform_time_grid_days(total_time_days, dt_out_days)

    return bio.RunSpec(
        time_grid=time_grid,
        y0_flat=y0_flat,
        rhs_tensor=bio.make_reaction_tensor(model),
        rhs_flat=bio.make_rhs_flat(model),
    )


def _make_run_config(
    *, method: str, rtol: float, atol: float, operators: OperatorSpecs
) -> RunConfig:
    """Create RunConfig with supplied tolerances and operators.

    Returns:
        RunConfig instance for CoreSolver.
    """
    return RunConfig(
        method=method,
        adaptive=True,
        strict=True,
        dt_controller=DtControllerConfig(),
        adaptive_cfg=AdaptiveConfig(rtol=rtol, atol=atol),
        operators=operators,
        gamma=None,
    )


def _make_base_builder(
    *,
    split: SplitName,
    model: bio.ModelSpec,
    y0_flat: np.ndarray,
    eta_cross: float,
    operator_mode: bio.OperatorMode,
) -> Callable[[bio.StageOperatorContext], np.ndarray]:
    n_bins = model.n_bins
    y0_sanitized = bio._sanitize_flat(y0_flat, n_bins)  # noqa: SLF001

    def ctx_y_to_flat(ctx: bio.StageOperatorContext) -> np.ndarray:
        y_tensor = np.asarray(ctx.y, dtype=float)
        y_flat_local = bio.flat_from_tensor(y_tensor)
        expected = 2 * n_bins
        if y_flat_local.size != expected:
            msg = bio._STAGE_STATE_SIZE_ERROR.format(  # noqa: SLF001
                got=y_flat_local.size, expected=expected
            )
            raise ValueError(msg)
        return y_flat_local

    def build_for(
        split_name: SplitName, *, t: float, y_flat_local: np.ndarray
    ) -> np.ndarray:
        if split_name == "A":
            return bio.split_a_matrix(model, t=t, y_flat=y_flat_local)
        if split_name == "B":
            return bio.split_b_matrix(model, t=t, y_flat=y_flat_local)
        return bio.split_c_matrix(model, t=t, y_flat=y_flat_local, eta_cross=eta_cross)

    if operator_mode == "frozen":
        a0 = build_for(split, t=0.0, y_flat_local=y0_sanitized)
        return bio.make_constant_base_builder(a0)

    if operator_mode == "time":

        def builder(ctx: bio.StageOperatorContext) -> np.ndarray:
            return build_for(split, t=float(ctx.t), y_flat_local=y0_sanitized)

        return builder

    def builder(ctx: bio.StageOperatorContext) -> np.ndarray:
        y_flat_local = bio._sanitize_flat(ctx_y_to_flat(ctx), n_bins)  # noqa: SLF001
        return build_for(split, t=float(ctx.t), y_flat_local=y_flat_local)

    return builder


def run_op_engine_custom(
    *,
    method: str,
    run: bio.RunSpec,
    model: bio.ModelSpec,
    split: SplitName | None,
    eta_cross: float,
    operator_mode: bio.OperatorMode,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, float]:
    """Run op_engine with custom split/operator settings and return states.

    Returns:
        tuple[np.ndarray, float]: (trajectory, placeholder_wall_s).

    Raises:
        ValueError: If an IMEX method is used without a split.
        RuntimeError: If state history is missing after run.
    """
    n_state = int(run.y0_flat.size)
    core = bio.make_core(run.time_grid, n_state, store_history=True)
    core.set_initial_state(bio.tensor_from_flat(run.y0_flat))

    operators = OperatorSpecs(default=None, tr=None, bdf2=None)
    if method in {"imex-euler", "imex-heun-tr", "imex-trbdf2"}:
        if split is None:
            msg = "split required for IMEX methods"
            raise ValueError(msg)
        base_builder = _make_base_builder(
            split=split,
            model=model,
            y0_flat=run.y0_flat,
            eta_cross=eta_cross,
            operator_mode=operator_mode,
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

    cfg = _make_run_config(method=method, rtol=rtol, atol=atol, operators=operators)

    solver = OpeCoreSolver(core, operators=None, operator_axis="state")
    solver.run(run.rhs_tensor, config=cfg)

    if core.state_array is None:
        msg = "state_array is None after run"
        raise RuntimeError(msg)

    states = np.asarray(core.state_array[:, :, 0], dtype=float)
    return states, 0.0


def run_scipy_custom(
    *,
    method: str,
    run: bio.RunSpec,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Run solve_ivp with specified tolerances and return trajectory.

    Returns:
        np.ndarray: Trajectory array of shape (T, 2n).

    Raises:
        RuntimeError: If SciPy reports failure.
    """
    t0 = float(run.time_grid[0])
    t1 = float(run.time_grid[-1])
    sol = solve_ivp(
        fun=run.rhs_flat,
        t_span=(t0, t1),
        y0=run.y0_flat,
        method=method,
        t_eval=run.time_grid,
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )
    if not sol.success:
        msg = f"solve_ivp({method}) failed: {sol.message}"
        raise RuntimeError(msg)
    return np.asarray(sol.y.T, dtype=float)


def _l2_max_and_final(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    per_t = np.linalg.norm(diff, axis=1)
    return float(per_t.max()), float(np.linalg.norm(diff[-1]))


def main() -> None:
    """Run diagnostic comparisons across splits and report L2 gaps."""
    rtol = 1e-6
    atol = 1e-8
    n_bins = 8
    total_time_days = 365.0 * 20.0
    dt_out_days = 5.0

    model = build_model_no_season(n_bins=n_bins)
    run = build_run_spec_no_season(
        model=model,
        seed=123,
        total_time_days=total_time_days,
        dt_out_days=dt_out_days,
    )

    # Reference trajectory (SciPy BDF)
    y_scipy = run_scipy_custom(method="BDF", run=run, rtol=rtol, atol=atol)

    def run_imex(split: SplitName, eta_cross: float) -> np.ndarray:
        y, _ = run_op_engine_custom(
            method="imex-trbdf2",
            run=run,
            model=model,
            split=split,
            eta_cross=eta_cross,
            operator_mode="stage_state",
            rtol=rtol,
            atol=atol,
        )
        return y

    y_imexb_010 = run_imex("B", 0.10)
    y_imexb_100 = run_imex("B", 1.0)
    y_imexc_100 = run_imex("C", 1.0)

    cases = {
        "imex-trbdf2 split B eta=0.10": y_imexb_010,
        "imex-trbdf2 split B eta=1.0": y_imexb_100,
        "imex-trbdf2 split C eta=1.0": y_imexc_100,
    }

    print("=== L2 differences vs SciPy BDF (seasonality off) ===")  # noqa: T201
    print(f"rtol={rtol}, atol={atol}, n_bins={n_bins}, dt_out={dt_out_days} days")  # noqa: T201
    for name, arr in cases.items():
        max_l2, final_l2 = _l2_max_and_final(arr, y_scipy)
        print(f"{name:35s}  max||diff||={max_l2:.3e}  final||diff||={final_l2:.3e}")  # noqa: T201


if __name__ == "__main__":
    main()
