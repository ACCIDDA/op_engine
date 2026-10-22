# Execution and backend roadmap

Engineering decisions recorded on 2026-10-21. This is a work plan, not a claim
that planned backends or statistical interfaces are implemented. Start with
the [qualification matrix](../guides/backend-qualification.md) for current support.

## Package boundaries

| Package | Responsibility |
| --- | --- |
| OP System | Model IR, checked dependency/gauge metadata, deterministic RHS and latent-observable compilation, portable derivative/export artifacts. |
| OP Engine | Numerical methods, native execution drivers, stochastic kernels, solver observations and numerical qualification. |
| OP Infer | Parameters, priors, statistical observation models, forward adapters, native estimator adapters and statistical diagnostics. |
| trade-study | Replicate orchestration, scoring, comparison, result tables and optional execution services. |
| OP Forecast | Proposed forecast products, ensemble/fallback policies and vintage-aware evaluation, subject to its consumer/start gate. |

Keep Python public interfaces and discover numerical namespaces from input
arrays. Array portability, compilation, differentiation, sparse acceleration and
hardware qualification are separate capabilities. Preserve the MIT core and
separate GPL provider boundaries.

## Language and compilation choices

- Use existing JAX `jit`, independent-axis `vmap`, sequential `scan` and
  checkpointing contracts before introducing another numerical implementation.
  Fixed and frozen-replay execution differ from the eager live-adaptive controller.
- Prototype Numba for measured NumPy bottlenecks. The compiler already caches
  generated vectorized functions; do not assume runtime `eval` remains its hot path.
  Emit compilable functions and report setup, first-call and warm costs separately.
- Consider Rust/PyO3 for a coarse stochastic kernel only after profiling existing
  direct SSA, independent-clock banks and tau-leaping. Propensity updates and
  independent random streams are part of that kernel contract. Python callbacks
  per event can erase the benefit of moving only the loop.
- A Rust implementation of OP System's IR normalization is a deferred experiment,
  preserving Python interfaces. It needs measured compile-time or external-runtime
  demand. Exhaustive enum matching helps implementation discipline; it is not a
  correctness proof. Moving the small Python array protocol provides no comparable
  established benefit.
- Julia export and optional `juliacall` integration are interoperability work.
  Full Julia package ports have no active delivery commitment. R forecast interfaces
  use an explicitly specified interchange boundary. No new language is required by
  the numerical core.
- Torch compilation/loop work needs its own supported-operation contract. `vmap`
  batches independent computations and does not implement a sequential scan.
  Native inference adapters do not require simulator and estimator graphs to join.

## Array and device policy

| Surface | Commitment |
| --- | --- |
| NumPy, JAX, CPU Torch | Retain required operation-specific CPU qualification, including shapes, float32/float64, namespaces and supported transforms. |
| CuPy / CUDA | Next end-to-end qualification target; optional sparse adapters and skipped GPU tests do not qualify the full pipeline. |
| SciPy and CuPy sparse operators | Explicit accelerator contracts, distinct from portable dense `linalg.solve`. |
| `array-api-strict` | Useful additional conformance testing; not a replacement for native backend checks. |
| Dask, PyData/Sparse and other namespaces | Consumer-driven experiments until numerical and execution contracts are qualified. |
| Metal, ROCm and TPU | Separate device/operation qualification; CPU and CUDA evidence do not imply support. |
| xarray, Arrow/Parquet and Polars | Containers or interchange/table boundaries, not implicit solver namespaces. |

Transfers happen at declared adapter or serialization boundaries. Preserve native
arrays inside numerical kernels; an explicit host snapshot does not promise
zero-copy transfer or automatic-differentiation graph continuity.

## Work order

1. Reconcile scopes with implemented SSA, scaling gauges, compiled observables and
   existing JAX drivers. Establish [representative baselines](execution-baselines.md).
   Use the [provider fixed-loop controls](../guides/fixed-scan.md#provider-configuration)
   and retain actual prepared-path graph and memory evidence before downstream
   execution comparisons.
2. Connect the forward-model interface and structural model identity to existing
   inference provenance. Preserve semantic ordering in identity serialization.
3. Implement shared statistical observation models and declarative priors with
   NumPy/JAX/Torch conformance before exposing them as interchangeable.
4. Qualify complete simulation-to-observation JAX execution and then CUDA paths;
   bound batching and avoid storing unneeded state history. Add an optional Diffrax
   bridge for specifically qualified external adaptive/derivative workloads.
5. Develop compiler derivatives/exports alongside the interface work, then
   sensitivity equations, scoped identifiability, local stability and Fisher screens.
6. Select Numba, Torch compilation or Rust kernel experiments from measured
   remaining bottlenecks; retain the independent Python/numerical references.
7. Keep statistical-method qualification and forecast consumer gates explicit.
   Do not add an unqualified dimension penalty to predictive stacking or equate
   surrogate expected improvement with decision-theoretic EVSI.
8. Revisit Rust IR normalization and full language ports after interfaces and
   representative measurements stabilize.

## Reconciled engine scope

The external Diffrax bridge is an optional adapter, not a replacement for native
methods. `RecursiveCheckpointAdjoint` differentiates the numerical solution;
it is not a continuous backsolve adjoint. A sensitivity solve needs both
`df/dy` and `df/dtheta`, initial-condition derivatives and observation derivatives.
No universal cost advantage over native AD is assumed.

Stability output reports equilibrium residuals, eigenvalues, conservation-induced
neutral modes and local assumptions; finite numerical tests are not a global
stability certificate. Fisher screening uses sensitivities of the measured outputs
in declared parameter coordinates and observation-noise units. An RHS state
Jacobian alone is not that Fisher matrix.

Data-dependent SSA can be represented with compiled control flow and fixed-shape
state/output contracts; lack of a current driver is not an impossibility result.
Qualification still needs the stochastic law and replay/stream ownership, rather
than a speed comparison alone.

## Provider controls before additional acceleration

Provider `fixed_loop` now selects actual ordinary/prepared flat, PyTree and block
fixed explicit drivers. Omitted or `null` settings preserve the existing provider
JAX scans and eager execution elsewhere; explicit `auto` uses that same dispatch,
`scan` requires JAX, and `unroll` selects Python stepping. This differs from the
core's default unrolled loop.

The strict boolean `fixed_scan_checkpoint` requires explicit `scan`, forwards
the core's boolean checkpoint flag and rematerializes provider scan bodies. The
provider's existing `fixed_checkpoint: none|step|chunk` retains its flat DOPRI5
meaning. Unsupported modes and conflicting checkpoint selections fail at config
validation. See the [configuration guide](../guides/fixed-scan.md#provider-configuration).

The before-change reproduction showed that accepted extra loop keys were ignored
across all three prepared layouts. The fixed provider now honors them; its omitted
setting already used scans. Benchmark source identities, matched meshes, graph
structure, values/gradients, compile/warm time and memory remain necessary for any
performance comparison. This control fix alone does not establish an application
speedup.


## Delivery update (2026-10-22)

Merged producer/inference prerequisites now include portable derivative/Julia
export, forward measured sensitivities, shared iid priors/observations, scoped
geometry/calibration interfaces and the grouped JAX observation scan. The latter
retains a full CPU pipeline record, including unsuccessful baseline attempts;
its evidence is separate from estimator training and application decisions.

Optional NumPy CPU lowering landed in
[op_system #337](https://github.com/ACCIDDA/op_system/pull/337). Final fresh-worker
RHS gains were 6.4–10.0×, or 2.1–2.3× including input packing and structured return
conversion, with approximately ten seconds cold setup and approximately 210 MiB
additional worker peak RSS. It remains explicit opt-in. The
[cache audit](https://github.com/ACCIDDA/op_system/pull/338) removed the remaining
global expanded-template mapping retention and fixed shaped-exclusion context;
its traced Python retention probe is not an RSS or training-memory claim.
Minimum Torch gradient conversion and reusable scalar/observable callables landed
in [#340](https://github.com/ACCIDDA/op_system/pull/340) and
[#341](https://github.com/ACCIDDA/op_system/pull/341).

The [bounded CPU Torch sequential adapter](../guides/torch-loop.md) is separately
qualified from ordinary eager Torch and JAX constant-size scan graphs. Its cold
forward/backward, warm and memory costs must guide explicit use; it does not
change defaults or establish GPU/training performance.

All further CuPy, JAX and Torch GPU execution is on maintainer hold, pending a
workstation session. [Draft op_system #339](https://github.com/ACCIDDA/op_system/pull/339)
contains the complete isolated setup, source selection, commands, case counts,
receipt requirements and native-device gates. It stays draft and #182 stays open.
Earlier A100 records remain historical evidence. No publication or application
adoption is authorized by this engineering handoff.

The [stochastic CPU measurements](../guides/stochastic-performance.md) now cover
168 exact-replay-gated workers, a NumPy single-event optimization and a separate
Numba event-selection feasibility experiment. The 64-column direct solver gains
1.31–1.36×; allocation probes are reported separately. Full native independent-bank
integration remains unqualified and optional; Rust has no new adoption commitment.

The [optional compiled Diffrax term](../guides/diffrax.md) now uses the current
compiler's flat/PyTree evaluators and complete scalar/shaped parameter metadata.
Separate current/minimum CPU contracts and 88 fresh hierarchical workers qualify
the scoped numerical and derivative paths. Legacy consumer plugins informed the
design; their caches, global settings and application reporting policies are not
part of the core interface. Live adaptive and frozen-mesh measurements remain
separate, with cold/warm and memory evidence retained.

Next, implement local stability diagnostics (#256), checked
dependency/derivative structure (op_system #296), and the corresponding Fisher
screen (trade-study #216). Broader estimator/statistical-method qualification and
application studies retain their own gates. Rust IR normalization remains deferred
until a measured compile-time or interoperability requirement justifies it.
