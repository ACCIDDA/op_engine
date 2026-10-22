# Stochastic CPU measurements

Direct SSA, bounded thinning and adaptive tau's exact critical events now update
one selected NumPy batch cell in a fresh state copy. The old operation allocated
a full one-hot firing array, multiplied it by stoichiometry and added a full
delta. Native NumPy float32/float64 use the smaller operation; other numerical
namespaces retain the portable operation. Inputs remain unmodified. Tau-leaping's
multi-event update and independent-clock bank implementation are unchanged.

## Whole-solver comparison

The [complete receipt](../assets/stochastic-cpu/workers.json.gz) contains 168
serial fresh workers: six methods, one/64 columns, float32/64 and original/new
event operations, with three process repetitions, plus allocation probes. Mode
order reverses in the middle repetition. Each solver worker times five full
solves, then profiles a separate sixth solve. Setup, imports, process peak RSS,
event/callback counts, source hashes and all samples are retained separately.
The original method is extracted from the archived base source; every other
solver operation is identical between modes. The manifest's `candidate_commit`
field records the base checkout HEAD, not a committed candidate; the exact
candidate module snapshot and its hash identify the measured implementation.

The public synthetic fixture is a four-species conservative ring with
heterogeneous column rates, a declared forcing change at 0.75 and six requested
outputs through time 2. Adaptive critical cases start with four counts per
species; the other cases start with twelve. Independent banks use stable
replicate IDs and native PCG64DXSM streams. Every paired worker matches full
trajectories, post-run RNG states, callback counts and exposed firing ledgers
exactly. Conservation and nonnegative integer counts are checked independently.
Bank comparisons include every actual replicate stream's final state.

The host was an Intel Core Ultra 7 265U, Python 3.12.14, NumPy 2.5.4 and
array-api-compat 1.15.0. Workers used one pinned CPU and one OpenBLAS/OMP thread.
No optional compiler is loaded by the solver change. Ratios below are original
latency divided by new latency, computed from the median of three process
medians. A ratio below one is a regression. These small synthetic measurements
are not timing CI thresholds or application/training performance claims.

| Method, 64 columns | Float32 ratio | Float64 ratio | Original → new float64 latency |
| --- | --- | --- | --- |
| Direct SSA | 1.31× | 1.36× | 69.73 → 51.41 ms |
| Bounded thinning | 1.09× | 1.10× | 190.99 → 173.68 ms |
| Adaptive tau, critical population | 1.12× | 1.04× | 56.06 → 53.81 ms |
| Adaptive tau, larger population | 1.09× | 1.05× | 13.22 → 12.55 ms |
| Fixed tau, unchanged control | 0.99× | 0.98× | 6.71 → 6.86 ms |
| Independent SSA bank, unchanged control | 0.97× | 0.96× | 8.56 → 8.92 ms |

Single-column direct SSA ratios were 1.21–1.24×. All methods and precisions,
including small-case controls and their variation, are in the
[summary](../assets/stochastic-cpu/summary.json). In one separately profiled
64-column float64 direct solve, the original event operation consumed about 30%
of solver time (1,302 accepted events); the raw profiles retain sampling,
validation, propensity and recorder costs. Profiled time is not pooled with
unprofiled wall timing.

## Allocation measurements

A separate prescribed-event probe uses 16 species and 128 reaction channels.
It checks the exact changed cell, input ownership and float32/64 output. Each
fresh process records seven traced per-call peaks and seven 100-call timing
batches. The peaks include the fresh returned state and exclude preallocated
inputs; they are **not process RSS**.

| Columns | Precision | Original → new traced peak |
| --- | --- | --- |
| 128 | float32 | 230,716 → 8,688 bytes |
| 128 | float64 | 312,590 → 16,880 bytes |
| 8,192 | float32 | 14,681,358 → 524,816 bytes |
| 8,192 | float64 | 19,924,238 → 1,049,104 bytes |

The large allocation probe's event operation was 106–228× faster. This is a
single-operation result on a different shape, not a whole-solver speedup. Each
event still copies the state; the change does not promise constant memory or
bound callback, history or process allocations.

## Optional native experiment and adoption decision

Independent-bank profiling still identifies Python event selection and optional
firing-ledger work as useful investigation targets. The 64-column float64 bank
accepted 1,278 events in 59 sweeps with only 55 vectorized propensity callbacks;
moving only callbacks or introducing a callback per event would miss this
existing batching benefit.

After profiling, an isolated Numba experiment moved exponential draws, channel
search and event-time checks for 256 independent generators into one typed
loop. Current Numba 0.68.0 / NumPy 2.5.4 and minimum Numba 0.67.0 / NumPy 2.0.2
matched selected event times, channels and final RNG states for float32/64 and
PCG64/PCG64DXSM. The initial float32 attempt failed exact replay by up to
4.77e-7: casting the result after division differs from NumPy's scalar arithmetic.
Casting the random draw before division/multiplication repaired this fixture.
Both the rejected and corrected sources and outputs are retained.

In the current-version feasibility process, warm selection took roughly
19–24 microseconds versus 675–817 microseconds for the Python reference. First
specialization calls took 0.36–0.65 seconds, excluding imports and typed-list
construction. These are isolated phase measurements in one process, not the
fresh-worker solver comparison above. They do not qualify a native SSA backend.

Ship the NumPy event optimization without a new dependency or API. Native bank
integration remains a separate experiment: include stream-list construction,
propensity evaluation, forcing/output handling, absorbing columns, optional
recorders, failure/event limits and final-only storage in whole-solver CPU
measurements. Require the existing independent law, stable-ID/order/shard and
boundary tests before exposing it. No Rust port is justified by a language
assumption; a coarse Rust/PyO3 kernel would face the same qualification and
callback requirements. No native cross-backend replay promise is made here.

## Qualification and reproduction

The new tests cover read-only strided inputs, all three positions of the reaction
axis in a three-dimensional state, mixed stoichiometry/state precision, ownership
and exact full-run replay against the dense reference for direct, thinning and
adaptive critical paths. The focused stochastic suite passes 246 cases on both
NumPy 2.5.4 and 2.0.2. Existing analytic Poisson/binomial and waiting/channel-law
checks remain active, alongside absorbing states, forcing, event limits, invalid
populations and independent-bank stream ownership. The required NumPy
qualification includes the new cases and rejects skipped selections.

To reproduce, extract `reproduction_files` from the gzip JSON to a new directory,
check out the implementation containing this guide and install its recorded CPU
versions in an isolated environment. The archive retains the original baseline
module, candidate module, exact worker/driver and experimental probe sources.
The relocatable driver uses a clearer `checkout_head` label for future receipts.

```sh
python reproduce.py --python /path/to/cpu-env/bin/python \
    --engine-source /path/to/op_engine/src --worker worker.py \
    --baseline-source baseline-stochastic_solver.py --cpu 0 \
    --output /path/to/new-results
```

Choose an available logical CPU. The driver runs workers serially and stops on
failed execution or exact comparisons, retaining stdout, stderr and its journal.
The early conservative-bound fixture failure and failed float32 native probe
remain separate feasibility evidence; neither is discarded or pooled into the
168 successful workers. This delivery supplies no GPU execution or estimator
training qualification.
