# Optional compiled Diffrax terms

`op_engine.diffrax.compiled_diffrax_term` binds a current compiler RHS to a
caller-owned Diffrax `ODETerm`. The optional adapter keeps native JAX arrays and
runtime parameters. Diffrax owns numerical integration, output storage, the
controller, adjoint and result handling. Python, NumPy and the core solver do
not acquire a Diffrax or compiler runtime dependency.

Install the `diffrax` extra from the checkout containing this adapter. The
qualified producer is the merged op_system compiler contract at
`bbde00697428f16f6cf948e7aff207f7b65231c8`; the required CI installs that source
explicitly. Consumer plugins written against older compiler releases are design
references and need fresh qualification against the current fields and outputs.

```sh
python -m pip install '.[diffrax]'
python -m pip install 'op-system @ git+https://github.com/ACCIDDA/op_system.git@bbde00697428f16f6cf948e7aff207f7b65231c8'
```

## Native input and output contract

| Surface | Qualified boundary |
| --- | --- |
| PyTree state, the default | Plain dict of native JAX arrays, keyed and shaped exactly as `compiled.template_shapes`; scalar templates use zero-dimensional leaves. Requires `pytree_eval_fn`. |
| Flat state | Select `state_layout="flat"`; shape `(len(compiled.state_names),)` in that expanded compiler ordering, using `eval_fn`. |
| Parameters | A plain dict passed as Diffrax `args`. Include scalar `param_names` **and** shaped parameters declared in metadata. Shaped leaves use declared axis sizes, including repeated axes. |
| Precision | Homogeneous float32 or float64 state and parameter leaves. Shapes, keys, namespace and output dtype are checked without host transfer or coercion. Choose compatible native time precision in the solve. |
| RHS output | The same state structure, shapes and dtype; malformed producer output raises. |
| Unsupported contributions | Declared operators, history, time-varying parameter tables and forcing breakpoints raise `NotImplementedError` before term construction. The caller declares any further smoothness assumptions. |
| Devices | Required CPU evidence below. GPU execution and additional device qualification remain on maintainer hold. |

The adapter snapshots static layout metadata and the selected compiler evaluator.
Build the term once and pass changed numeric values through `args` and `y0`.
Changed values with the same native shapes/dtypes keep the term identity; changed
models get distinct terms. Parameter values are not captured in static closures.
Module import leaves JAX precision, compilation-cache settings and backend
initialization to the caller. There is no module-global solve cache or override
dictionary. Term ownership and output retention remain explicit. JAX/Diffrax compiler caches
have their own lifetimes; dropping a term does not guarantee process-RSS release.

## Final-only solve and a parameter derivative

This example uses the compiler's scalar PyTree path and retains one requested
output. It preserves initial-state and parameter differentiation. Save policies
control returned arrays; checkpoints, compilation and stage work add separate
peak-memory costs.

```python
import diffrax as d
import jax
import jax.numpy as jnp
import numpy as np
from op_system import compile_spec
from op_engine.diffrax import compiled_diffrax_term

compiled = compile_spec({
    "kind": "expr", "state": ["x"], "equations": {"x": "-rate*x"}
})
term = compiled_diffrax_term(compiled)
dtype = jnp.float32
t0, t1, dt0 = (jnp.asarray(value, dtype=dtype) for value in (0.0, 1.0, 0.02))

def final_x(rate):
    solution = d.diffeqsolve(
        term, d.Dopri5(scan_kind="bounded"), t0=t0, t1=t1, dt0=dt0,
        y0={"x": jnp.asarray(2.0, dtype=dtype)}, args={"rate": rate},
        stepsize_controller=d.PIDController(rtol=1e-5, atol=1e-7),
        saveat=d.SaveAt(t1=True), adjoint=d.RecursiveCheckpointAdjoint(),
        max_steps=256, throw=True,
    )
    return solution.ys["x"][-1]

rate = jnp.asarray(0.7, dtype=dtype)
np.testing.assert_allclose(final_x(rate), 2*np.exp(-0.7), rtol=3e-5)
np.testing.assert_allclose(jax.grad(final_x)(rate), -2*np.exp(-0.7), rtol=3e-5)
```

For requested output times use explicit `SaveAt(ts=...)`. For measured output
reductions use an explicitly qualified Diffrax save callback. The factory does
not expand a reporting cadence, stack an unrequested state history or convert
solver failures into a statistical score. `throw=False` preserves Diffrax's
`solution.result`; the caller must inspect it before using the values. The
failure test exercises an actual step-limit failure with both throw policies.

## Adjoint and mesh choices

[Diffrax's adjoint contracts](https://docs.kidger.site/diffrax/api/adjoints/) require
separate transform choices. `RecursiveCheckpointAdjoint` supports reverse-mode
through the numerical solve with checkpointing. Select `ForwardMode` for JVP or
`jacfwd`. The bounded `DirectAdjoint` configuration has separate fixed-mesh
float64 PyTree Hessian evidence. Higher derivatives of live adaptive methods,
third derivatives and every solver/adjoint combination are outside that claim.

Use `StepTo` and an identical accepted mesh for a numerical-method comparison.
A live PID solve is a different execution task; its accepted mesh and controller
cost must be reported separately. A recursive checkpoint adjoint differentiates
the numerical solution; it is not a continuous backsolve adjoint. Controller
choices, tolerances and step limits remain explicit
[Diffrax solve arguments](https://docs.kidger.site/diffrax/api/diffeqsolve/).
The factory does not accept or silently translate an OP Engine `RunConfig`.

## Required CPU evidence

The two dedicated hosted jobs require current JAX 0.12.0 / NumPy 2.5.4 /
array-api-compat 1.15.0 and minimum JAX 0.4.38 / NumPy 2.0.2 /
array-api-compat 1.12.0, both with Diffrax 0.7.2 and the same current compiler.
Run `python scripts/qualify_backend.py jax --diffrax`; 31 selected cases must
execute with zero skips. Baseline optional skips do not qualify this contract.
The base suite separately tests a fresh import with JAX, Diffrax and the compiler
blocked, including an actionable missing-extra error.

Closed-form reversible-transfer values and derivatives cover flat/PyTree,
float32/64, fixed and live adaptive values/reverse gradients, changed runtime
parameters and parameter-dependent initial states. Forward derivatives and
independent batches use fixed meshes. A fixed float64 PyTree Hessian has an
independent analytic reference. Different models with equal names/shapes remain
isolated; invalid shapes, keys, host arrays and dtypes fail explicitly. Current
compiler aliases, shaped rates and global reductions also match independently
written nonlinear cell balances. No application training/calibration or device
speedup follows from these synthetic contracts.

## Representative measurements

The [complete receipt](../assets/compiled-diffrax-cpu/workers.json.gz) contains
88 serial fresh CPU workers: two shapes (1×1, 4×2), coupled/independent and
stiff/nonstiff regimes, native replay and handwritten/compiled Diffrax controls.
Eighty cases cover trajectories and reverse derivatives. Eight additional
small fixed nonstiff cases cover forward derivatives and Hessians. All passed
independent SciPy/refinement or derivative finite-difference checks; maximum
trajectory absolute error was 2.06e-9. Every worker used source commit
`9bf3be12b03f4628fba645d668be9ca3c344fe0a` with a clean tree. The receipt verifies
that its compiler checkout and the qualified public merge have identical Git trees.

The host was an Intel Core Ultra 7 265U, Python 3.12.14, with the current versions
listed above, one pinned logical CPU, single-threaded OMP/OpenBLAS/MKL and
persistent compilation caching disabled. Each condition has one fresh process
and five warm samples. These are descriptive engineering measurements, without
a statistical speedup, default-selection or application-adoption claim. Earlier
smoke rows remain separate. The [summary](../assets/compiled-diffrax-cpu/summary.json)
also records all dependency versions, accuracy limits and selected costs.

The coupled 4×2 nonstiff case has 16 states and 18 dynamic rate/initial-value
parameters. Fixed methods share the same 19 accepted steps; live PID uses 17.
Trace+compile excludes term/controller setup (0.30–0.34 s here). Cold first
execution, all warm samples and native/handwritten controls remain in the receipt.

| Execution | Task | Steps | Trace+compile (s) | Median warm (ms) | Worker peak RSS (MiB) |
| --- | --- | --- | --- | --- | --- |
| Fixed mesh | Trajectory | 19 | 0.686 | 0.039 | 308.2 |
| Fixed mesh | Reverse gradient | 19 | 2.360 | 0.079 | 351.7 |
| Live PID | Trajectory | 17 | 0.708 | 0.040 | 306.5 |
| Live PID | Reverse gradient | 17 | 2.623 | 0.090 | 352.8 |

Across all compiled cases worker peak RSS ranged from 302 to 611 MiB. These
peaks include imports, independent verification and auxiliary compilations;
they are not per-solve allocation or training-memory measurements. XLA argument,
output and temporary-buffer estimates are retained separately. Output shape,
adjoint choice, stiffness and compilation all affect memory.

To repeat, extract `reproduction_files` from the gzip JSON, install the recorded
CPU versions and compiler source into an isolated environment, and provide the
checkout containing this implementation:

```sh
/path/to/cpu-env/bin/python reproduce.py \
    --engine-checkout /path/to/op_engine --cpu 0 --output /path/to/new-results
```

Choose an available CPU. The coordinator reuses the existing isolated benchmark
worker and case protocol, writes each success/failure/timeout to its journal
and fails unless all 88 required conditions complete successfully.
