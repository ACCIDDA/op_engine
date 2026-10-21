# Representative execution baselines

This protocol starts with existing public synthetic fixtures. It measures the
current implementation before selecting Numba, Rust, Torch compilation or new
GPU work. It does not establish application performance or a release-wide speedup.
See the [execution roadmap](execution-roadmap.md) for ownership and work order.

## Initial workload matrix

| Family | Initial cases | Existing harness and correctness gate |
| --- | --- | --- |
| Compiler/evaluator | 4 and 16 groups; 2 and 8 local stages; independent and coupled; NumPy/JAX/CPU Torch; float64 | OP System `scripts.benchmark_hierarchical`: independent RHS and derivative checks, flat/structured/available block layouts. |
| Solver/derivatives | Small and larger shared/local problems; independent and coupled; nonstiff and stiff; identical-mesh comparisons | `scripts.benchmark_hierarchical`: independent trajectory and derivative references; core/external/Diffrax frozen meshes identified separately from adaptive tasks. |
| Loop compilation | 8 and 64 fixed steps; JAX scan/unroll; gradient and Hessian | `scripts/benchmark_replay.py`: finite-output and compilation/memory probe, supplemented by the required replay/fixed-scan numerical tests. |
| Stochastic banks | 16 and 64 independent replicates; fixed output horizon; bounded flat comparison | `benchmarks.replicate_ssa`: analytic drift and population invariants, plus required distribution/replay/stream-ownership tests. |
| Simulation to observation | Compiled latent observations, returned observations versus stored state history | Existing observation qualification first; complete fused/batched timing remains a separate extension rather than an invented completed result. |

Run bounded subsets first. The first receipt must state the exact subset executed;
the matrix is a coverage target, not a declaration that every combination ran.
Add float32, native Torch compilation, CuPy/CUDA, transfer costs and more demanding
operators in separately qualified increments. Missing devices remain unqualified.

## Measurement contract

- Use immutable source revisions or record dirty source content hashes, exact
  package versions, Python, CPU/device model, dtype, shapes, methods, output grid,
  derivative mode, tolerances and actual accepted steps/events.
- Run measurement workers serially in fresh processes. Disable inherited persistent
  JAX compilation caches. Record CPU affinity and requested/observed thread pools;
  a thread environment variable alone does not prove effective concurrency.
- Separate normalization/compilation, backend setup, first execution and warm
  samples. Synchronize accelerator results before stopping each timer. Do not
  include unrelated CI workloads in the timed interval.
- Retain raw repetitions, medians and dispersion, failures, unavailable cases and
  censored timeouts. Cross-implementation comparisons use reversed/alternating
  case order and separately labeled cold and warm measurements.
- Distinguish whole-worker peak RSS from live solver storage and XLA buffers.
  Report observations retained, state history policy and stochastic event counts.
- Qualify numerical values, namespaces and derivatives before interpreting timings.
  Stochastic kernels additionally need distributional and stream/replay evidence;
  deterministic drift agreement alone does not qualify their law.
- A changed adaptive mesh or random-number workload defines a different task.
  Keep those comparisons separate from equivalent-task speedup claims.
- Keep timing thresholds out of required CI. CI checks the small correctness and
  receipt protocol; dedicated hardware supplies larger advisory measurements.
- Publish synthetic fixtures and reviewed environment fields. Exclude nonpublic
  consumer inputs, local identities/paths, credentials and unrelated planning text.

## Reproduction

In an OP System checkout with the selected CPU backends installed:

```sh
just benchmark --backends numpy jax torch --groups 4 16 --stages 2 8 \
  --coupling both --dtype float64 --repeats 5 --inner-loops 50 \
  --output /tmp/compiler-baseline.json
```

In an OP Engine checkout, install the optional comparison extra and a compatible
OP System before running compiler-backed stochastic measurements:

```sh
uv sync --dev --extra benchmark
just benchmark-hierarchical --groups 1 4 --local-dimensions 1 \
  --steps 16 --regimes nonstiff --coupling independent coupled \
  --solvers core external diffrax-fixed numpy scipy \
  --derivatives trajectory grad-rev --repeats 5 --timeout 180 \
  --require-success --output /tmp/solver-baseline.jsonl
uv run --no-sync python scripts/benchmark_replay.py --execution fixed \
  --methods dopri5 --steps 8 64 --loops scan unroll \
  --derivatives grad hessian --timeout 180
uv run --no-sync python -m benchmarks.replicate_ssa --sizes 16 64 \
  --days 5 --flat-max 64 --output /tmp/ssa-baseline.json
```

The replay probe has one timed execution per worker, not a warm-sample median;
repeat it in independent processes for dispersion. The SSA harness also needs
independent process repetitions. Retain those boundaries in the summary.
The compiler harness's JAX initialization can occur during qualification before
selected evaluator compilation; report its `cold_jit_seconds` as selected-function
compilation, not whole-process cold startup. Apply an outer process timeout where
an existing harness has no worker timeout.

The initial receipt directory also retains the [resolved package versions](../assets/execution-baseline-2026-10-21/environment.json), excluding local editable-install paths.

## Initial CPU receipt (2026-10-21)

The [complete receipt](../assets/execution-baseline-2026-10-21/receipt.json) and
[derived summary](../assets/execution-baseline-2026-10-21/summary.json) retain the
exact 30-case matrix, three fresh process repetitions in alternating order, raw
warm samples, source/content identities and failed/censored attempts. The host
was an Intel Core Ultra 7 265U with singleton inherited logical-CPU affinity,
float64 and one requested BLAS/OpenMP thread. This is a shared-host initial
baseline, not a hardware-independent speedup or full backend qualification.

| Family | Cases | Completed process records | Censored | Errors |
| --- | ---: | ---: | ---: | ---: |
| compiler | 12 | 36 | 0 | 0 |
| solver | 10 | 30 | 0 | 0 |
| loop | 6 | 12 | 6 | 0 |
| ssa | 2 | 6 | 0 | 0 |

Compiler cases cover both selected shapes and coupling regimes on NumPy, JAX
and CPU Torch. Solver cases cover small independent nonstiff and larger coupled
stiff problems; only the small shape compares all selected solver strategies.
The matrix does not imply a complete Cartesian product of methods or devices.

The large unrolled JAX probes have a declared 30-second process bound. A timeout
has no completed runtime or speedup. The finite-output replay probe supplements
the required independent numerical tests; it is not its own complete accuracy
qualification. Selected-function compiler JIT costs follow backend/qualification
setup, so they are not whole-process cold startup.

An incomplete setup attempt exposed a missing declared compiler dependency. Its
32 structured records are retained separately in the receipt, along with the
interrupted-attempt status and earlier timeout policy. After correcting setup,
the complete matrix was restarted from scratch. Those setup failures are excluded
from the corrected matrix aggregates rather than hidden or counted as successful
measurements.

Actual library pool sizes were not sampled inside every existing worker; singleton
affinity bounds CPU execution separately from requested pool settings. The SSA
harness records setup/execution/events/storage but not per-worker peak RSS. GPU,
float32 expansion, device-transfer costs and complete fused observation timing
remain follow-up coverage. No new accelerator or statistical default was changed.

### Selected observations

These compare existing execution paths; no accelerator implementation changed.
For the fixed-step JAX gradient probe, selected compile times and whole-worker
peak RSS were:

| Probe | Median selected compilation | Median whole-worker peak RSS |
| --- | ---: | ---: |
| loop-scan-n64-grad | 0.506 s | 252.6 MiB |
| loop-scan-n64-hessian | 1.134 s | 282.6 MiB |
| loop-scan-n8-grad | 0.467 s | 251.7 MiB |
| loop-unroll-n8-grad | 13.435 s | 556.8 MiB |

Both 64-step unrolled probes were censored in all three process repetitions.
The 30-second bound covers the whole worker, so it is not a measured compile
time or a completed runtime. RSS includes backend startup and the full probe.

The structured RHS's median warm call latencies were:

| Shape/coupling | NumPy | JAX JIT | CPU Torch eager |
| --- | ---: | ---: | ---: |
| g4-s2, independent | 69.5 us | 23.5 us | 104.3 us |
| g4-s2, coupled | 74.4 us | 27.6 us | 125.5 us |
| g16-s8, independent | 118.4 us | 40.8 us | 137.2 us |
| g16-s8, coupled | 122.0 us | 42.1 us | 139.4 us |

Each entry aggregates nine warm samples from three fresh processes; each sample
averages 20 synchronized calls. All structured RHS value comparisons had maximum
absolute error at most `1.8e-15`. These small CPU cases motivate measuring complete
compiled simulation-to-observation next; they do not predict GPU, application or
Rust/Numba performance. Torch compilation was not measured.

Compiler normalization and `compile_rhs` repeat within each worker and can reuse caches. The summary separately retains all three process-initial samples and their median for those phases; the within-process median must not be labeled cold compilation.

## Next integration measurements

The provider measurements below extend the initial baseline through
[#261](https://github.com/ACCIDDA/op_engine/issues/261). Further integration work
should measure complete simulation-to-observation execution and retained storage. Use public synthetic multi-axis models at small and bounded
larger sizes, identical explicit-method meshes and output grids, and compare
provider loop variants, parameter gradients, graph size, compile time,
warm execution and peak memory. Include checkpoint variants only after resolving
the provider/core checkpoint configuration distinction in the roadmap. Keep
core-only probes and provider measurements separately identified.

Downstream version comparisons need isolated pinned environments, identical
inputs/seeds and a written agreement/adoption rule established before results are
read. Retain failures and changed-workload cases. Baseline publication does not
authorize changing a consumer's frozen runs or its scientific acceptance rules.

Provider fixed JAX paths already use specialized scans. The pre-change provider
graph/path is therefore a required baseline: an ignored core option does not
prove Python unrolling in the provider. Explicit selection must be qualified
through its actual dispatch, without changing its established default accidentally.

## Provider fixed-loop controls (2026-10-21)

The [provider receipt](../assets/provider-fixed-loops-2026-10-21/receipt.json) and
[summary](../assets/provider-fixed-loops-2026-10-21/summary.json) contain 20 cases
with three fresh CPU workers each: **60 successful records, zero errors and zero
timeouts** under a 45-second worker bound. The same shared laptop CPU, singleton
affinity, float64 and one requested BLAS/OpenMP thread were used. No CI workload
ran during this measurement interval; other host activity remained uncontrolled.

The matrix uses the existing public exponential-system provider harness with
rate gradients of the sum of final states. It covers ordinary and prepared
RK4 flat/PyTree/block execution and flat DOPRI5, omitted/automatic controls,
explicit scan/unroll and scan rematerialization. Most cases have eight steps and
16 states; a larger flat RK4 probe has 64 steps and 256 states. This is a bounded
subset, not a Cartesian product of all methods, shapes and namespaces.

The receipt also retains the before/after prepared reproduction and source
identities. Before the fix, accepted `unroll` and `scan` keys both produced a scan
in all three layouts. After the fix, explicit unroll has no scan, explicit scan
has one forward scan, and omitted settings retain their original dispatch.
Independent discrete-method values and parameter/initial-state derivatives are
qualified separately in provider tests, including the existing precision policy.

Selected **prepared rate-gradient** medians follow. Whole-worker RSS includes
both ordinary and prepared paths and setup; XLA temporary bytes are compiler
estimates and are a different quantity.

| RK4 flat case | Selected compile | Warm call | Worker peak RSS | XLA temporary bytes |
| --- | ---: | ---: | ---: | ---: |
| 8 steps / 16 states, scan | 0.181 s | 43.8 us | 261.7 MiB | 6,400 |
| 8 steps / 16 states, unroll | 0.098 s | 25.6 us | 252.0 MiB | 0 |
| 64 steps / 256 states, scan | 0.190 s | 112.9 us | 264.4 MiB | 665,536 |
| 64 steps / 256 states, unroll | 2.685 s | 196.0 us | 403.6 MiB | 319,488 |
| 64 steps / 256 states, checkpointed scan | 0.133 s | 106.8 us | 259.9 MiB | 401,584 |

The small case favors unroll; the larger case favors scan for selected compile
time, warm execution and whole-worker peak RSS. The examples differ in both
state count and step count, so they do not identify a universal crossover.
Rematerialization reduces the larger scan's temporary-buffer estimate by about
40%; its similar worker RSS illustrates why that estimate is not a measured
live-memory reduction for an application.

All measured rate-gradient comparisons with the analytic exponential reference
had maximum absolute error below `3.83e-10`. Each worker reports trace and warm
medians from three inner samples; individual inner samples are not retained by
this harness. The receipt retains all fresh-worker records, and the summary
retains their samples, medians and ranges.

Ordinary execution precedes prepared execution within each worker. Compilation
follows tracing/backend setup and can reuse in-process resources; it is not
whole-process cold startup. Persistent JAX caches were disabled. The source
hashes identify the measured dirty implementation against its base commit;
immutable development dependency pins and package versions are recorded. These
are execution-choice comparisons: the provider already used scans by default.
No production workload, GPU, statistical calibration or new default-path speedup
is established by this control fix.
