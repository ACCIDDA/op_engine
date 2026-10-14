# Hierarchical scaling and comparison measurements

The [hierarchical ODE fixture](hierarchical-ode.md) supplies independent cell
references, deterministic observations, shared/local/IC parameters, four
regimes, and the statistical objective. This measurement layer invokes that
same problem in fresh serial CPU workers. Small CI cases qualify the protocol
and supported comparison configurations; larger timing sweeps are opt-in.
There are no speed or memory thresholds in CI.

## Setup and bounded commands

```shell
uv sync --dev --extra benchmark
just benchmark-hierarchical --groups 1 --local-dimensions 1 --steps 4 \
  --solvers core external diffrax-fixed numpy scipy \
  --derivatives trajectory grad-fwd grad-rev hessian logdet-grad \
  --repeats 3 --timeout 180 --require-success --output /tmp/hierarchical-small.jsonl
```

The optional `benchmark` extra installs Diffrax; it is not a numerical-core
runtime dependency. Python/JAX/Diffrax/SciPy versions are reported, so retain
the resolved environment with results. Concrete NumPy and SciPy references
produce trajectory rows, not derivative rows. A larger sweep can vary:

```shell
just benchmark-hierarchical --groups 2 4 8 --local-dimensions 1 2 \
  --steps 8 32 128 --regimes nonstiff stiff --coupling independent coupled \
  --solvers core external diffrax-fixed --derivatives grad-rev hessian logdet-grad \
  --repeats 5 --timeout 300 --output /tmp/hierarchical-scaling.jsonl
```

`--steps` sets a minimum-step policy by capping discovery/adaptive `dt_max` at
`1/steps`; it does not invent accepted steps. Actual accepted counts are
reported. Output times remain `[0, 0.2, 0.5, 1]`, so increasing step count does
not also increase observation dimension. Increasing local dimension adds rate
and IC parameters per group: total fitted dimension is
`2 + 2*groups*local_dimension`.

## Compare equivalent tasks

| Solver label | Task and configuration |
| --- | --- |
| `core` | Built-in Dormand–Prince accepted-mesh replay with JAX scan/checkpointing |
| `external` | Public `fixed_explicit_step` in an external checkpointed scan, including FSAL reuse, on the identical mesh |
| `diffrax-fixed` | Diffrax Dormand–Prince with `StepTo` on that exact frozen mesh and the same output times |
| `diffrax-adaptive` | Separate live-adaptive task: Tsit5 for nonstiff problems, Kvaerno5 for stiff problems, PID tolerances `2e-8/2e-10`, with the same maximum-step policy |
| `numpy` | Eager core replay of the identical nominal mesh; trajectory parity/reference timing |
| `scipy` | Independent cell-loop DOP853 solve at `2e-12/2e-14`; accurate trajectory reference timing |

Fixed-mesh comparisons hold step times, method, parameters, outputs and objective
constant. Adaptive comparison rows identify their own accepted/rejected steps
and the producing core mesh count separately; their workload is not identical
to frozen replay. Do not combine unlike tasks into a claimed solver speedup.
Reference-RHS loops intentionally differ from the native vectorized RHS, so
SciPy timings identify the chosen trajectory-reference implementation.

Diffrax configurations follow its
[solver recommendations](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/),
[adjoint contracts](https://docs.kidger.site/diffrax/api/adjoints/), and
[Hessian example](https://docs.kidger.site/diffrax/examples/hessian/).
Reverse gradient and trajectory rows use `RecursiveCheckpointAdjoint`, forward
rows use `ForwardMode`, and Hessian/log-determinant-gradient rows use
`DirectAdjoint`. RK stages select `scan_kind="bounded"`, and saved-output loops
use the adjoint appropriate to the transform. DirectAdjoint supports both modes
but costs can depend on `max_steps`; that bound is recorded. Fixed rows set it
to the actual mesh length. Adaptive rows use a stated bound based on discovery,
and failures remain visible rather than implying unsupported AD.

All current measurements use float64 CPU execution. GPU and other namespaces
are outside this measurement protocol. The regularized Hessian is `H+25*I`;
positive definiteness and nonfinite values are checked before a row qualifies
for interpretation. See the fixture guide for priors and nonzero residuals.

## JSON fields and timing boundaries

Each completed row is correctness-gated against the independent trajectory.
Gradient rows compare independent-reference centered differences; Hessian and
log-determinant-gradient rows compare centered differences of their relevant
native derivatives. The fixture's separate required tests also check forward/
reverse agreement and refinement. Failed agreement produces an error row.

| Fields | Meaning |
| --- | --- |
| `groups`, `local_parameter_dimension`, `parameter_count`, `state_shape` | Problem dimensions and fitted initial-condition contribution |
| `accepted_steps`, `core_discovery_accepted_steps`, `solver_step_statistics` | Actual measured-solver steps and nominal core discovery counts; Diffrax statistics distinguish rejection |
| `core_discovery_seconds`, `reference_seconds`, `backend_setup_seconds` | Preprocessing/reference/imported-backend setup, outside selected compiled numerical timing |
| `trace_lower_seconds`, `compile_seconds` | First selected transform tracing/lowering and compilation in a fresh process |
| `first_execution_seconds`, `warm_execution_seconds` | Synchronized execution after compilation and full warm samples |
| `total_worker_seconds` | Time inside the measurement function, including setup and validation; process startup/imports occur before this timer |
| `peak_process_rss_mib` | Whole worker peak RSS, including imports, setup, validation and any extra validation compilations; not isolated live solver storage |
| `xla_buffer_bytes` | Selected compiled transform's reported argument/output/temp/alias buffer requirements; may be unavailable |
| `trajectory_max_absolute_error`, `derivative_max_absolute_error`, `regularized_hessian_min_eigenvalue` | Agreement and positive-definiteness evidence |
| `versions`, `python`, `hardware`, `provenance`, `numerical_configuration` | Exact packages, CPU/thread settings, commit/dirty flag/content hashes, adjoint/stage/controller/step bounds |

A fresh worker prevents previous cases' compilation/memory caches from becoming
an advantage. Workers also disable JAX's persistent compilation cache, even if
the parent configured a cache directory, and report
`persistent_compilation_cache_enabled: false`. See JAX's
[cache setting](https://docs.jax.dev/en/latest/config_options.html#enable-compilation-cache).
First compiled-transform timing occurs before the postmeasurement
accuracy checks; every result is validated before emission. Warm calls block
until completed. RSS includes the full worker lifecycle, while XLA buffer
analysis describes only the selected compiled transform. Concrete reference
rows have no tracing/compilation or XLA-memory claims.
`backend_setup_seconds` covers array/backend initialization after JAX imports;
the process timeout and RSS cover startup/import costs too.

Timeouts are serialized with `status: timeout`, `censored: true` and no timing
result. They are not measured speedups. Errors include diagnostics and return
codes; `--require-success` makes any failed/censored case fail the command.
Retain failed rows when reporting a sweep rather than selecting only favorable
results. CI uploads its tiny JSON protocol evidence as an artifact; virtual
runner timings are not publication scaling claims.

For publication, save the JSON, environment resolution, exact source tag/commit
and content hashes with the archive. Report hardware, dimension and mesh policy,
objective/regularization, tolerances, derivative/adjoint mode, repetitions,
timeout policy, and agreement before interpreting runtime or memory trends.
