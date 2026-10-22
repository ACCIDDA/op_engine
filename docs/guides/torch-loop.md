# Optional compiled Torch sequential loops

The Torch adapter is an explicit CPU execution choice. It reuses the native
fixed explicit step kernels through the existing `LoopAdapter` registry and
captures a bounded sequential loop with fullgraph Inductor. `vmap` is not a
substitute for this dependency between time steps. Ordinary solver defaults stay
`fixed_loop="unroll"`; importing op_engine or this adapter module does not import
Torch.

```python
import torch
from op_engine import array_namespace
from op_engine.loop_ops import register_loop_adapter
from op_engine.torch_loop import torch_compile_loop_adapter

adapter = torch_compile_loop_adapter(max_steps=128, max_elements=65_536)
register_loop_adapter(array_namespace(torch.ones(1)).__name__, adapter)
# Explicitly select RunConfig(fixed_loop="scan", fixed_max_step=...).
```

## Supported boundary

| Property | Qualified contract |
| --- | --- |
| Torch | CPU 2.14.1; the factory requires >=2.14. Eager >=2.6 remains separate. |
| Arithmetic | Homogeneous native CPU float32 or float64. No host/device transfer inside the adapter. |
| Methods | Fixed Euler, Heun, RK4, Dormand–Prince; original full-step kernels and FSAL carry. |
| Trees | Tensor/None leaves in standard tuple/list/dict carry and outputs; native tensor input leaves with a common leading dimension. Carry/output structure, shapes, dtype and device stay fixed. |
| Body | Functional native numerical operations; no input mutation or side effects. Independent input values remain runtime values. |
| Bounds | Positive static scan length, default/hard maximum 128; configurable input+carry+stacked-output element bound, default 65,536 and hard maximum 1,048,576. These are not process RSS or internal RHS allocation caps. |
| Derivatives | First reverse-mode gradients of the qualified consumers, including parameters and initial states. |
| Checkpoint | Not supplied; explicitly requested checkpointing raises in the consumer. |
| Empty loops | Consumers skip an empty tail after their first bootstrap step. No generic unknown-output shape inference for zero-length scans. |
| Device | CPU only. CuPy, Torch and JAX GPU work is separately on hold pending the workstation session. |

Adaptive policy differentiation, implicit replay kernels, higher derivatives,
mixed precision, custom pytree registrations, sparse states, GPU execution,
estimator training and shared simulator/estimator graphs are outside this
qualification. Separately simulated training banks remain valid consumers.
For scalar compiler/observable paths, use the merged op_system prerequisite
[PR #341](https://github.com/ACCIDDA/op_system/pull/341); the runtime callable
boundary and minimum-Torch gradient fixes are part of that reviewed source.

The adapter statically unrolls a **bounded** sequential loop into a native graph.
Its graph size and cold compilation grow with the step count. It does not have
the constant-size graph property of JAX's `lax.scan`; long meshes should continue
using the separately qualified JAX path or eager execution as appropriate.
Changing a method, body structure, shape, dtype or static grid may recompile.
Each explicit adapter owns a distinct compiler code identity, avoiding a shared
factory specialization budget. Torch's own compiler caches and generated code
still contribute to process memory.

## Required compilation and errors

`fullgraph=True`, `dynamic=False`, and the Inductor backend are selected
explicitly. Graph breaks and unsupported numerical bodies raise rather than
returning an eager fallback. The raw body also rejects disabled compilation.
Qualification exercises a real graph break even with global error suppression,
and tests the startup `TORCH_COMPILE_DISABLE` switch in a fresh process. It does
not treat FX inspection or a Python-only backend as native acceleration.
Unsupported versions, bounds, leaves, devices and dtypes fail explicitly.
The package makes no automatic backend/default selection based on these results.

```sh
# Install CPU Torch 2.14.1 explicitly, then require zero skipped cases:
python scripts/qualify_backend.py torch --torch-compile
```

The dedicated hosted job uses the CPU wheel index. Tests include independently
written stability-polynomial values/gradients for all four methods and both
precisions, dynamic parameter changes, stored/final-only results, nested
observation output trees, optional FSAL carry, positive-length boundaries,
oversized output/carry errors and disabled/broken compilation. Minimum-Torch CI
also checks that compiled support is explicitly rejected without affecting eager
state/gradient checks.

## Consumer measurements

The [complete receipt](../assets/torch-loop-cpu/workers.json.gz) retains 32 serial
fresh workers: two consumers × 8/32 steps × float32/64 × eager/compiled × two
repetitions, reversing mode order in the second repetition. Twenty-five warm
calls per phase follow independent discrete RK4 value/gradient checks for two
parameter sets. The actual observation consumer is the public op_infer scalar
fixture; the core case has two scalar states. All 32 workers passed, with native
Inductor kernels and no recorded graph breaks in each compiled worker.

The host was an Intel Core Ultra 7 265U, Python 3.12.14, Torch 2.14.1+cpu,
NumPy 2.5.4 and array-api-compat 1.15.0. Workers used one pinned CPU, one
OpenBLAS/OMP/compiler thread and separate empty compiler caches. These bounded
toy cases do not establish results across larger models or different hardware.

Ratios are eager latency divided by compiled latency, using the median of the
two process medians; values below one indicate a regression. First forward and
backward are separate process medians in seconds. Warm value+gradient includes
its forward call.

| Consumer | Steps | Precision | Warm forward ratio | Warm value+gradient ratio | First forward / backward (s) |
| --- | --- | --- | --- | --- | --- |
| core | 8 | float32 | 0.86× | 1.14× | 10.8 / 8.1 |
| core | 8 | float64 | 1.37× | 1.74× | 10.3 / 8.6 |
| core | 32 | float32 | 3.70× | 4.50× | 27.8 / 29.2 |
| core | 32 | float64 | 3.67× | 4.33× | 31.7 / 41.3 |
| observations | 8 | float32 | 2.22× | 2.34× | 19.9 / 10.8 |
| observations | 8 | float64 | 2.13× | 2.45× | 24.7 / 10.6 |
| observations | 32 | float32 | 7.40× | 8.63× | 55.3 / 49.4 |
| observations | 32 | float64 | 6.73× | 7.83× | 55.3 / 33.5 |

Compiled process peak RSS was **441–587 MiB**. Eight additional fresh ordinary
eager memory controls, one per condition and all passing, used **249–274 MiB**.
The primary timing workers import compiler diagnostic modules in both modes;
their instrumented eager RSS of 317–341 MiB must not be presented as ordinary
eager memory. The extra controls omit those unused imports and remain separate
from the timing ratios. RSS includes imports/compiler state and excludes child
compiler processes; it is distinct from output-array size or a hard memory cap.

The smallest core float32 warm forward regresses, and cold forward/backward
costs and compiler memory can outweigh repeated-call savings. Retain explicit
opt-in and existing eager defaults. These two-replicate observations support
bounded engineering decisions, not general timing CI thresholds, GPU or
estimator-training claims. Source snapshots, hashes, raw samples, compiler
diagnostics, stderr, and the eight memory controls are in the receipt; the
[summary](../assets/torch-loop-cpu/summary.json) records the protocol limits.
Only lint comments changed in the adapter during collection; all exact
snapshots and their identical executable AST digest are retained.

To rerun, extract `reproduction_files` from the gzip JSON into a new directory.
Install the receipt’s Python/backend versions in a separate CPU environment and
provide the engine, compiler and op_infer checkouts to the relocatable coordinator:

```sh
python reproduce.py --python /path/to/cpu-env/bin/python \
    --engine-checkout /path/to/op_engine --system-checkout /path/to/op_system \
    --infer-checkout /path/to/op_infer --output /path/to/new-results
```

The archived plain-eager worker runs the separate memory control by selecting
`--mode unroll`; it deliberately omits unused compiler-diagnostic imports.
Feasibility timings are not pooled into this comparison.
