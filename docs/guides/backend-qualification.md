# Required CPU backend qualification

The required CI jobs `Solver qualification (numpy|jax|torch, 3.12)` install the
selected namespace explicitly and run `scripts/qualify_backend.py`. The runner
fails for an absent backend, no executed cases, any selected skip, or numerical
failure. Ordinary optional skips in baseline tests are separate evidence.
Provider integration tests run separately; provider success cannot substitute
for these core contracts.

The same required backend runner includes native [stored-time observations](observations.md).
Independent scalar/field values and error contracts run with NumPy; JAX JIT/vmap
and initial-state/observation-parameter gradients run in float64; CPU Torch
gradients run in float64 and float32. Each callback must preserve state namespace,
dtype and its declared output shape.

Separate required `Compiler observations (numpy|jax|torch, 3.12)` jobs install
op_system's merged observation contract at immutable commit
`0becde52474b48fef6b6f3c42a344498a17552a8`, then run
`scripts/qualify_backend.py BACKEND --compiler-contract`. These reject absent
backends, selected skips and empty selection as well. A cell-average linear
density on two meshes gives the same physical integral and sample at a shared
declared coordinate after fixed RK4 integration. JAX JIT/AD and CPU Torch AD
check state, dynamic-rate and observation-only gain derivatives against analytic
exponentials. The fixture uses float64 and fixed output/substep grids; live
adaptive gradients, mesh derivatives and arbitrary callbacks are not qualified
by it. `just qualify` includes these compiler checks after the core CPU matrix.
`just qualify-observations` assumes those namespaces are already installed.
The pinned compiler is a development/CI contract, not a core runtime dependency.

## Method and execution matrix

The table describes the small required structured workload, not all possible
models, versions, devices, or differentiation transforms. Each qualified fixed
method runs in float32 and float64 on all three CPU namespaces. JAX x64 is scoped
explicitly for double precision; a float32 test also runs with x64 enabled.

| Method | NumPy fixed values | JAX fixed values | CPU Torch fixed values | Required live adaptive / frozen replay evidence |
| --- | --- | --- | --- | --- |
| Euler | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| Heun | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| RK4 | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| Dormand–Prince 5(4) | Qualified | Qualified | Qualified | Qualified on all three namespaces |
| IMEX Euler | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| IMEX Heun–trapezoidal | Qualified | Qualified | Qualified | Qualified on all three namespaces |
| IMEX TR-BDF2 | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| IMEX ARK3 | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| Linearly implicit Euler | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| Linearly implicit trapezoidal | Qualified | Qualified | Qualified | Other baseline tests; unqualified here |
| BDF2 | Qualified | Qualified | Qualified | Adaptive configuration rejected |
| ROS2 | Qualified | Qualified | Qualified | Qualified on all three namespaces |
| Nonlinear SDIRK2 | Qualified | Qualified | Qualified | Qualified eager replay; forced scan rejected |

`tests/op_engine/test_backend_qualification.py` defines a state tensor with
shape `(2 compartments, 3 groups, 3 trait cells)`. Reflecting transport couples
trait cells along a named, non-default operator axis; scalar loss acts across
every cell. The independent solution uses a SciPy matrix exponential, with
separate per-batch total-decay assertions. Unequal initial cells expose axis
ordering and broadcasting errors. Tests inspect result/history shapes, dtype,
namespace, numerical values, and convergence diagnostics for nonlinear stages.

The IMEX fixture splits scalar loss from native dense transport. Its factories
use the appropriate implicit-Euler/trapezoidal stage schemes, including both
TR-BDF2 factories. Linearly implicit Jacobians act along the trait axis;
SDIRK2 receives a full flattened-system Jacobian. Geometry and method choices
remain static. This is portability evidence on a linear structured model,
with nonlinear stage machinery, rather than a stiff accuracy survey.

## AD, loops, sparse acceleration, and hardware

| Capability | Required evidence / boundary |
| --- | --- |
| Differentiation | JAX JIT/AD and CPU Torch functional AD in float64 through structured Dormand–Prince, IMEX ARK3, ROS2, and SDIRK2; initial-state derivatives agree with an exact reference. Additional transformations remain separately qualified. |
| Native matrix builders | Both precisions, namespaces and independent advection/diffusion stencils; dynamic coefficient derivatives through JAX JIT/AD and CPU Torch AD (#212). |
| Live adaptive controller | Eager Python branches extract error scalars. Preserving JAX arrays does not make the controller JIT compatible. |
| Frozen replay | A NumPy-recorded mesh is portable to each namespace. JAX `auto` uses its scan adapter except SDIRK2, whose diagnostics retain eager replay; NumPy/Torch `auto` retain eager loops. Derivatives remain conditional on the frozen mesh. |
| SciPy sparse factorization | Required NumPy sparse solve; JAX/Torch state with SciPy sparse operators is rejected explicitly. SciPy remains a core dependency, and its sparse acceleration is a separate namespace contract. Native dense operators use the state's `linalg.solve`. |
| Operator preprocessing | Host NumPy/SciPy stage-builder and Kronecker utilities are separate from namespace-aware matrices and native caller-supplied dense factories. Host factories that call `float(dt)` do not accept traced step sizes; use native factories for compiled replay. |
| CUDA / CuPy / JAX GPU / Torch GPU | No required GPU runner and no GPU release qualification established here. Required evidence is CPU only. |

CuPy is an optional runtime extra. On a suitable CUDA host, install `op-engine[cupy]`
and run the GPU contract explicitly:

```shell
uv sync --dev --extra cupy
uv run --no-sync pytest tests/op_engine/test_array_namespace.py -k cupy -rs
```

Inspect the result: a missing device or skip does not qualify GPU execution.
Record CUDA/device/package versions, dtype, executed cases and their results
before adding GPU claims to a release. This opt-in check currently covers a
CuPy sparse-solve contract; it does not establish the complete method matrix.

## Local and release commands

With a supported complete Python installation and `uv`/`just`:

```shell
just qualify-numpy
just qualify-jax
just qualify-torch
# All required CPU contracts:
just qualify
```

The Torch recipe installs CPU wheels explicitly. These targets refresh extras
in the root environment; use `uv run --no-sync` or separate environments when
retaining a namespace selection matters. The runner records Python/backend
versions in its log. Baseline Python 3.11–3.14 CI, provider tests, strict typing,
Ruff, strict docs and installed wheels remain required alongside qualification:
`just ci`, `just docs`, and `just build-all`.

Before a release, record the CPU qualification results and any separately
executed GPU evidence. A green baseline with optional skips is insufficient.
This matrix addresses numerical parity and execution boundaries; scientific
validity and accuracy outside these fixtures need independent references.
See [backend strategies](backends.md) for functional steps and loop contracts.
