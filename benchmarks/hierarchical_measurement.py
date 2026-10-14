"""Isolated optional measurements and fair Diffrax configurations for the fixture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import resource
import shutil
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from benchmarks.hierarchical_ode import (
    HierarchicalProblem,
    _flatten_mesh,
    initial_state,
    loss_function,
    nominal_parameters,
    numpy_replay_trajectory,
    objective,
    observations,
    reference_objective,
    reference_trajectory,
    replay_trajectory,
    rhs,
)
from op_engine import Array, CoreSolver, ModelCore
from op_engine.core_solver import (
    AdaptiveConfig,
    AdaptiveStepSchedule,
    DtControllerConfig,
    RunConfig,
)
from op_engine.model_core import ModelCoreOptions

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable

OUTPUT_TIMES = np.asarray([0.0, 0.2, 0.5, 1.0])
REGULARIZATION = 25.0


class _DiffraxSettings(NamedTuple):
    solver: str
    derivative: str
    max_step: float


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hardware() -> dict[str, object]:
    model = platform.processor()
    path = Path("/proc/cpuinfo")
    if path.exists():
        model = next(
            (
                line.split(":", 1)[1].strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("model name")
            ),
            model,
        )
    return {
        "machine": platform.machine(),
        "cpu": model,
        "logical_cpus": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            ]
        },
    }


def _provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    git = shutil.which("git") or "/usr/bin/git"
    commit = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(  # noqa: S603
        [git, "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    from benchmarks import hierarchical_ode  # noqa: PLC0415
    from op_engine import core_solver  # noqa: PLC0415

    files = {
        "fixture": Path(hierarchical_ode.__file__),
        "measurement": Path(__file__),
        "driver": root / "scripts/benchmark_hierarchical.py",
        "core_solver": Path(core_solver.__file__),
    }
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
        },
    }


def _discover(
    problem: HierarchicalProblem, parameters: np.ndarray, minimum_steps: int
) -> tuple[AdaptiveStepSchedule, RunConfig]:
    config = RunConfig(
        method="dopri5",
        adaptive=True,
        adaptive_cfg=AdaptiveConfig(
            rtol=2e-8, atol=2e-10, dt_init=min(0.02, 1 / minimum_steps)
        ),
        dt_controller=DtControllerConfig(dt_max=1 / minimum_steps),
        replay_model_id=problem.model_id,
        replay_reference_regime="nominal_parameters()",
    )
    core = ModelCore(
        2,
        problem.groups,
        OUTPUT_TIMES,
        options=ModelCoreOptions(
            other_axes=(problem.local_dimension,), axis_names=("pool", "group", "local")
        ),
    )
    core.set_initial_state(initial_state(problem, parameters))
    solver = CoreSolver(core)
    solver.run(rhs(problem, parameters), config=config)
    if solver.last_adaptive_schedule is None:
        msg = "Nominal discovery did not produce a mesh"
        raise RuntimeError(msg)
    return solver.last_adaptive_schedule, config


def _diffrax_functions(
    problem: HierarchicalProblem,
    schedule: AdaptiveStepSchedule,
    settings: _DiffraxSettings,
    data: Array,
) -> tuple[
    Callable[[Array], Array],
    Callable[[Array], Array],
    dict[str, object],
    Callable[[Array], Any],
]:
    import diffrax as d  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    solver_name, derivative, max_step = settings
    if derivative in {"trajectory", "grad-rev"}:
        adjoint = d.RecursiveCheckpointAdjoint()
    elif derivative == "grad-fwd":
        adjoint = d.ForwardMode()
    else:
        adjoint = d.DirectAdjoint()
    # Bounded RK-stage scans support forward and higher-order AD. The adjoint
    # selection additionally covers saved outputs and the solve iteration loop.
    if solver_name == "diffrax-fixed":
        solver = d.Dopri5(scan_kind="bounded")
        starts, sizes, _ = _flatten_mesh(schedule)
        boundaries = np.concatenate((starts, [schedule.output_times[-1]]))
        controller = d.StepTo(ts=jnp.asarray(boundaries))
        max_steps = len(sizes)
        dt0 = None
        mesh_mode = "identical frozen accepted mesh"
    else:
        solver = (
            d.Kvaerno5(scan_kind="bounded")
            if problem.stiff
            else d.Tsit5(scan_kind="bounded")
        )
        controller = d.PIDController(rtol=2e-8, atol=2e-10, dtmax=max_step)
        max_steps = max(256, 4 * sum(map(len, schedule.step_sizes)))
        dt0 = min(0.02, max_step)
        mesh_mode = "live adaptive PID; separate task from frozen replay"
    term = d.ODETerm(lambda t, y, args: rhs(problem, args)(t, y))

    def solution_at(values: Array) -> Any:  # noqa: ANN401
        return d.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=dt0,
            y0=initial_state(problem, values),
            args=values,
            saveat=d.SaveAt(ts=jnp.asarray(OUTPUT_TIMES)),
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=max_steps,
        )

    def trajectory(values: Array) -> Array:
        return solution_at(values).ys

    def loss(values: Array) -> Array:
        return objective(problem, values, trajectory(values), data)

    settings = {
        "solver": type(solver).__name__,
        "adjoint": type(adjoint).__name__,
        "scan_kind": "bounded",
        "max_steps": max_steps,
        "mesh_mode": mesh_mode,
        "controller": type(controller).__name__,
        "rtol": None if solver_name == "diffrax-fixed" else 2e-8,
        "atol": None if solver_name == "diffrax-fixed" else 2e-10,
        "implicit_root_finder": type(solver.root_finder).__name__
        if hasattr(solver, "root_finder")
        else None,
        "root_find_max_steps": getattr(solver, "root_find_max_steps", None),
    }
    return trajectory, loss, settings, solution_at


def run_case(args: Namespace) -> dict[str, object]:  # noqa: C901, PLR0912, PLR0914, PLR0915
    """Measure one correctness-gated task in a fresh CPU process.

    Returns:
        Dimensions, agreement, phase timings, warm samples, memory and provenance.

    Raises:
        ValueError: If an unsupported solver/derivative combination is selected.
        RuntimeError: If independent accuracy or positive-definiteness checks fail.
    """
    total_start = time.perf_counter()
    problem = HierarchicalProblem(
        args.groups[0],
        args.local_dimensions[0],
        coupled=args.coupling[0] == "coupled",
        stiff=args.regimes[0] == "stiff",
    )
    nominal = nominal_parameters(problem)
    solver_name, derivative = args.solvers[0], args.derivatives[0]
    setup_start = time.perf_counter()
    schedule, config = _discover(problem, nominal, args.steps[0])
    discovery_seconds = time.perf_counter() - setup_start
    reference_start = time.perf_counter()
    independent = reference_trajectory(problem, nominal, OUTPUT_TIMES)
    data_np = observations(problem, OUTPUT_TIMES)
    reference_seconds = time.perf_counter() - reference_start
    base: dict[str, object] = {
        "status": "success",
        "solver": solver_name,
        "derivative": derivative,
        "groups": problem.groups,
        "local_parameter_dimension": problem.local_dimension,
        "parameter_count": problem.parameter_count,
        "state_shape": problem.state_shape,
        "regime": "stiff" if problem.stiff else "nonstiff",
        "coupled": problem.coupled,
        "requested_minimum_steps": args.steps[0],
        "accepted_steps": sum(map(len, schedule.step_sizes)),
        "core_discovery_accepted_steps": sum(map(len, schedule.step_sizes)),
        "output_times": OUTPUT_TIMES.tolist(),
        "dtype": "float64",
        "platform": "cpu",
        "rtol": 2e-8,
        "atol": 2e-10,
        "discovery_dt_max": 1 / args.steps[0],
        "tolerances_apply_to": "Discovery only; frozen replay bypasses error control",
        "reference_method": "SciPy DOP853",
        "reference_rtol": 2e-12,
        "reference_atol": 2e-14,
        "reference_seconds": reference_seconds,
        "core_discovery_seconds": discovery_seconds,
        "regularization": REGULARIZATION,
        "timeout_seconds": args.timeout,
        "timeout_policy": "censored; no measured speedup inferred",
        "repeats": args.repeats,
        "versions": {
            name: _version(name)
            for name in ["op-engine", "numpy", "scipy", "jax", "diffrax"]
        },
        "python": sys.version.split()[0],
        "hardware": _hardware(),
        "provenance": _provenance(),
    }
    if solver_name in {"numpy", "scipy"}:
        if derivative != "trajectory":
            msg = f"{solver_name} is a trajectory reference, not an AD implementation"
            raise ValueError(msg)
        function = (
            (lambda: numpy_replay_trajectory(problem, nominal, schedule, config))
            if solver_name == "numpy"
            else (lambda: reference_trajectory(problem, nominal, OUTPUT_TIMES))
        )
        start = time.perf_counter()
        value = function()
        first_seconds = time.perf_counter() - start
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
        error = float(np.max(np.abs(value - independent)))
        if not np.allclose(value, independent, rtol=5e-7, atol=3e-9):
            msg = "Concrete trajectory failed its independent reference"
            raise RuntimeError(msg)
        base.update({
            "trace_lower_seconds": None,
            "compile_seconds": None,
            "first_execution_seconds": first_seconds,
            "warm_execution_seconds": samples,
            "trajectory_max_absolute_error": error,
            "xla_buffer_bytes": None,
            "method": "dopri5 frozen replay"
            if solver_name == "numpy"
            else "DOP853 accurate live reference",
        })
    else:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)  # noqa: FBT003
        jax.config.update("jax_enable_compilation_cache", False)  # noqa: FBT003
        base["persistent_compilation_cache_enabled"] = (
            jax.config.jax_enable_compilation_cache
        )
        backend_start = time.perf_counter()
        parameters = jnp.asarray(nominal)
        data = jnp.asarray(data_np)
        parameters.block_until_ready()
        data.block_until_ready()
        base["backend_setup_seconds"] = time.perf_counter() - backend_start
        if solver_name in {"core", "external"}:

            def trajectory(values: Array) -> Array:
                return replay_trajectory(
                    problem,
                    values,
                    schedule,
                    config,
                    external=solver_name == "external",
                )

            loss = loss_function(
                problem, schedule, config, data, external=solver_name == "external"
            )
            settings = {
                "solver": "dopri5",
                "adjoint": "native JAX AD with checkpointed scan",
                "max_steps": sum(map(len, schedule.step_sizes)),
                "mesh_mode": "identical frozen accepted mesh",
                "fsal": True,
            }
        else:
            trajectory, loss, settings, solution_at = _diffrax_functions(
                problem,
                schedule,
                _DiffraxSettings(solver_name, derivative, 1 / args.steps[0]),
                data,
            )
        base["numerical_configuration"] = settings

        def logdet(values: Array) -> Array:
            matrix = jax.hessian(loss)(values) + REGULARIZATION * jnp.eye(
                problem.parameter_count
            )
            return 0.5 * jnp.linalg.slogdet(matrix)[1]

        functions = {
            "trajectory": trajectory,
            "grad-fwd": jax.jacfwd(loss),
            "grad-rev": jax.grad(loss),
            "hessian": jax.hessian(loss),
            "logdet-grad": jax.grad(logdet),
        }
        trace_start = time.perf_counter()
        lowered = jax.jit(functions[derivative]).lower(parameters)
        lowered_at = time.perf_counter()
        compiled = lowered.compile()
        compiled_at = time.perf_counter()
        value = jax.block_until_ready(compiled(parameters))
        executed_at = time.perf_counter()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            jax.block_until_ready(compiled(parameters))
            samples.append(time.perf_counter() - start)
        base["trace_lower_seconds"] = lowered_at - trace_start
        # Keep the phase boundaries from the first compilation separate from warm runs.
        base["compile_seconds"] = compiled_at - lowered_at
        base["first_execution_seconds"] = executed_at - compiled_at
        base["warm_execution_seconds"] = samples
        actual = np.asarray(jax.jit(trajectory)(parameters))
        if solver_name.startswith("diffrax"):
            solution = solution_at(parameters)
            stats = {key: int(value) for key, value in solution.stats.items()}
            base["solver_step_statistics"] = stats
            base["accepted_steps"] = stats["num_accepted_steps"]
        error = float(np.max(np.abs(actual - independent)))
        if not np.allclose(actual, independent, rtol=5e-7, atol=3e-9):
            msg = "JAX trajectory failed the independent reference"
            raise RuntimeError(msg)
        if not np.all(np.isfinite(np.asarray(value))):
            msg = "Derivative output contains nonfinite values"
            raise RuntimeError(msg)
        base["trajectory_max_absolute_error"] = error
        if derivative in {"grad-fwd", "grad-rev"}:
            shifts = 1e-4 * np.eye(problem.parameter_count)
            expected = np.asarray([
                (
                    reference_objective(problem, nominal + shift, OUTPUT_TIMES, data_np)
                    - reference_objective(
                        problem, nominal - shift, OUTPUT_TIMES, data_np
                    )
                )
                / 2e-4
                for shift in shifts
            ])
            if not np.allclose(value, expected, rtol=3e-5, atol=3e-6):
                msg = "Gradient failed independent finite differences"
                raise RuntimeError(msg)
            base["derivative_max_absolute_error"] = float(
                np.max(np.abs(np.asarray(value) - expected))
            )
        if derivative in {"hessian", "logdet-grad"}:
            hessian = np.asarray(jax.jit(jax.hessian(loss))(parameters))
            eigenvalues = np.linalg.eigvalsh(
                hessian + REGULARIZATION * np.eye(problem.parameter_count)
            )
            if not np.all(eigenvalues > 0):
                msg = "Regularized Hessian is not positive definite"
                raise RuntimeError(msg)
            base["regularized_hessian_min_eigenvalue"] = float(eigenvalues.min())
            shifts = 1e-4 * np.eye(problem.parameter_count)
            if derivative == "hessian":
                gradient = jax.jit(jax.grad(loss))
                expected = np.column_stack([
                    (
                        np.asarray(gradient(parameters + shift))
                        - np.asarray(gradient(parameters - shift))
                    )
                    / 2e-4
                    for shift in shifts
                ])
            else:
                evaluate = jax.jit(logdet)
                expected = np.asarray([
                    (evaluate(parameters + shift) - evaluate(parameters - shift)) / 2e-4
                    for shift in shifts
                ])
            if not np.allclose(value, expected, rtol=3e-5, atol=3e-6):
                msg = "Higher derivative failed centered differences"
                raise RuntimeError(msg)
            base["derivative_max_absolute_error"] = float(
                np.max(np.abs(np.asarray(value) - expected))
            )
        analysis = compiled.memory_analysis()
        base["xla_buffer_bytes"] = {
            name: getattr(analysis, name, None)
            for name in [
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            ]
        }
    rss_divisor = 1024**2 if sys.platform == "darwin" else 1024
    base["peak_process_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor
    )
    base["total_worker_seconds"] = time.perf_counter() - total_start
    return base
