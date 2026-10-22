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

A serial fresh-process comparison separately measures import/setup, first
forward, first backward, warm forward, warm value-plus-gradient and process peak
RSS for a bounded `CoreSolver` case and the actual op_infer observation adapter.
The receipt, source snapshots and limits are recorded here after the comparison
completes. Feasibility timings are not a basis for changing defaults, and none of
these CPU results qualifies GPU or training performance.
