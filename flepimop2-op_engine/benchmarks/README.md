# Provider benchmarks

These opt-in benchmarks diagnose provider lifecycle cost without turning wall-clock
measurements into CI assertions.

## Solver matrix

`solver_matrix.py` integrates the analytic batched problem
`x' = rate * x` through the public Flepimop2 provider. It records accuracy beside
cost and keeps these phases separate:

- provider construction;
- adaptive controller discovery, when applicable;
- JAX tracing, lowering, and compilation;
- first and median warm execution;
- value-and-gradient tracing, lowering, compilation, first execution, and warm
  execution.

It also records accepted and rejected adaptive attempts, RHS evaluations when the
solver exposes or permits them to be recovered, JAX trace and StableHLO sizes,
temporary-memory estimates, package versions, devices, and the Git revision.
Unavailable measurements are JSON `null`; they are never replaced by a different
lifecycle boundary.

From this directory, run a small CPU comparison with:

```console
uv run python benchmarks/solver_matrix.py \
  --horizons 1 30 --batch-sizes 1 64 \
  --methods rk4 dopri5 --policies fixed adaptive \
  --references scipy --output solver-matrix.json
```

Diffrax is optional and is not a provider dependency. Install it in the benchmark
environment and request it explicitly:

```console
uv run --with diffrax python benchmarks/solver_matrix.py \
  --policies adaptive --methods dopri5 \
  --references scipy diffrax --output solver-references.json
```

The Diffrax row identifies `Tsit5`, `PIDController`,
`RecursiveCheckpointAdjoint`, and the initial step. This prevents a reference run
from silently using unmatched differentiation or controller semantics.

Set `JAX_PLATFORM_NAME` and the normal JAX device configuration to select CPU or
GPU. The output metadata records the devices that actually ran.

## Prepared execution

`prepared_execution.py` compares ordinary public-provider staging with the
stable callable returned by `engine.prepare`. It reports preparation and cache
hit time, system binding counts, trace/lower/compile phases, synchronized first
and warm execution, IR size, and analytic error.

```console
uv run python benchmarks/prepared_execution.py \
  --horizon 30 --output-count 121 --fixed-max-step 0.25 \
  --output prepared-execution.json
```

Dynamic alternate sample values are used for cache hits so the result also
checks that contents are not structural cache keys.

## Focused diagnostics

- `prepared_execution.py` compares ordinary and cached provider staging.
- `structured_blocks.py` compares flat, PyTree, and block-PyTree scaling.
- `adaptive_replay.py` compares unrolled, compact, and checkpointed frozen-mesh
  differentiation.

Keep benchmark output outside the repository unless a result is intentionally
curated for documentation. The scripts are reproducibility tools; their timings
are not test thresholds.
