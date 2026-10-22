# Fixed explicit solves with compiled loops

For long JAX solves, set `RunConfig(fixed_loop="scan")` to run the fixed
explicit mesh through one `jax.lax.scan`. This avoids tracing a Python copy
of every step. It supports Euler, Heun, RK4 and Dormand–Prince, with the same
fixed-step updates as the default eager driver. A spatial discretization or
nonlocal model can supply the RHS just as an ODE can; the usual stability and
discretization requirements still apply.

```python
--8<-- "examples/fixed_scan.py"
```

The example checks the solution and parameter derivative against exponential
decay. Its 1,000 internal steps do not require 1,000 output records. Requested
output times, `fixed_max_step`, state shape and declared forcing breakpoints
are static under JIT. Initial states and RHS parameters can be dynamic arrays.
Changing the static mesh can trigger recompilation. Normal JAX restrictions
apply to the RHS; Python control flow depending on traced values is unsupported.

`fixed_loop="unroll"` remains the default. `"auto"` selects a registered
namespace adapter and otherwise retains the eager path. `"scan"` raises if an
adapter is unavailable; JAX is registered lazily. NumPy and CPU PyTorch retain
eager execution unless an adapter is explicitly registered. No JAX dependency
is added to ordinary core installs.

`fixed_checkpoint=True` rematerializes scan stages in reverse mode when the
adapter supports it. It can reduce stored intermediates at additional runtime
cost; the benefit depends on the model. A scan may still materialize internal
states before selecting output boundaries, so benchmark memory for large states.
`ModelCoreOptions(store_history=False)` discards the adopted output history.

This is a fixed explicit driver. Adaptive, implicit, IMEX and stochastic solves
do not use `fixed_loop`; unsupported selections raise. For a previously
discovered adaptive mesh use [adaptive replay](replay-provenance.md).
Fixed solves do not create an `AdaptiveStepSchedule` or adaptive provenance.
The FlepiMoP2 provider's prepared execution has its own loop configuration;
this option configures `CoreSolver.run` directly.

## Provider configuration

The FlepiMoP2 provider exposes `fixed_loop: auto|scan|unroll` for deterministic
fixed explicit methods on ordinary and prepared flat, PyTree and block paths.
Its default is `null`: omitted controls preserve the existing specialized JAX
scans and eager execution for other namespaces. `auto` makes that choice explicit;
`scan` requires JAX arrays, while `unroll` selects Python stepping. Block layouts
continue to require JAX for independent-axis `vmap`.

For rematerialization use `fixed_loop: scan` with
`fixed_scan_checkpoint: true`. This boolean is forwarded to core
`RunConfig.fixed_checkpoint` and applied to provider scan bodies. The provider's
existing `fixed_checkpoint: none|step|chunk` remains its separate flat DOPRI5
policy. Combining the two controls, or asking for legacy checkpointing with
`unroll`, raises at configuration. Adaptive, implicit, IMEX and stochastic modes
reject explicit fixed-loop controls. See the
[provider examples](https://github.com/ACCIDDA/op_engine/blob/main/flepimop2-op_engine/README.md).

The provider already used JAX scans before these explicit controls were added.
Compare its actual pre-change prepared graph and matched workloads before
interpreting loop timings. The change makes execution selection reliable;
omitted settings retain their established numerical and dtype behavior.

Known right-continuous forcing jumps split steps, including jumps between
requested outputs, and incoming endpoint stages use the left limit. Scan
execution preserves this contract and the `fixed_max_step` bound. It does not
add error control, positivity enforcement or a scientific accuracy guarantee.

## Compile-time measurements

The existing replay benchmark also measures live fixed solves in fresh CPU
processes, including tracing, compilation, execution and memory metadata:

```shell
uv run python scripts/benchmark_replay.py --execution fixed --methods dopri5 \
  --steps 50 100 200 --loops scan unroll --derivatives grad --timeout 120
```

These are optional measurements, not CI timing thresholds. The regression test
checks one scan and a constant traced graph size at 50 and 200 steps. A small
linear example can expose driver overhead; its timing is not a performance
claim for a larger scientific model.

The provider benchmark measures ordinary and prepared execution separately. From
the `flepimop2-op_engine` directory, with its development environment installed:

```shell
uv run --no-sync python -m benchmarks.prepared_execution --method rk4 \
  --layout flat --fixed-loop scan --rate-gradient --horizon 1 \
  --output-count 3 --fixed-max-step 0.125 --batch-size 16 --repeats 3
```

Repeat with `unroll`, with the option omitted, and with `scan` plus
`--fixed-scan-checkpoint` in fresh processes. Schema 2 reports separate
trace/lower/compile/first/warm phases, gradient error, graph/scan counts and XLA
temporary-buffer estimates. Whole-worker peak RSS includes both execution paths
and setup; it is not per-path or live solver memory. Use bounded workers and
retain censored cases when expanding the mesh.


For an explicit bounded CPU Torch option, see the
[Torch compiled-loop contract](torch-loop.md). Its static unrolling has different
cold compilation and memory costs from JAX's constant-size scan graph. Eager
execution remains the default.
