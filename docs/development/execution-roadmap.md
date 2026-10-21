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
   Resolve [provider access to fixed loops](https://github.com/ACCIDDA/op_engine/issues/261)
   before downstream scan comparisons: accepting an extra config key does not
   establish that the solver receives it.
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

## Provider access before additional acceleration

Issue [#261](https://github.com/ACCIDDA/op_engine/issues/261) blocks consumers
from selecting the existing core fixed-loop driver through provider configuration.
Qualify config serialization, validation and dispatch, then compare prepared
provider loop variants and parameter gradients on the same explicit method
and static mesh. Record compile time, graph structure and peak memory separately
from warm execution; core-only measurements do not qualify the provider path.

Checkpoint names need an explicit compatibility decision: the provider currently
uses `fixed_checkpoint: none|step|chunk` for its specialized flat DOPRI5 executor,
whereas core `RunConfig.fixed_checkpoint` is a boolean for its general fixed scan.
Passing the enum directly is invalid. Preserve the existing provider policies and
defaults while defining how general scan rematerialization is selected. Retain
negative tests for unsupported methods, adaptive execution and nonstatic meshes.

Provider JAX fixed execution already has specialized scans for flat and structured
states, including prepared flat plans. These paths can bypass generic core
`RunConfig` loop dispatch. The missing configuration fields do not establish
that the current provider unrolls, and forwarding them alone cannot define all
provider behavior. Capture the actual consumer's selected namespace/layout/path
and graph first; preserve existing scan defaults while defining explicit loop
selection. The new field must not silently replace an existing scan with unroll.
