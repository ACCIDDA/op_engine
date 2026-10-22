# Independent adaptive block schedules

The FlepiMoP2 provider can discover a separate accepted mesh for each independent
block, then replay those meshes through compact JAX scans. Select the policy
explicitly; `block_schedule: shared` preserves the existing shared-mesh default.

```yaml
engine:
  module: flepimop2.engine.op_engine
  state_change: flow
  config:
    method: dopri5
    adaptive: true
    state_layout: block
    block_axis: loc
    block_schedule: independent
    block_batching: length
    block_batch_size: 16
```

The system must publish the compiler's proven block axis, block stepper and
block-template shapes. Every state template must carry that axis. The provider
uses the published runtime parameter-axis positions: local parameters are sliced
at that position, while shared parameters are broadcast. Initial-condition
parameters participate in the same contract. A declaration of `state_layout`
alone supplies no proof. op_system rejects reductions, mixing and operators
across the proposed block axis during compilation. Supported typed operators
within a block remain part of its RHS.

This full-solve path requires real float32/float64 packed JAX states and deterministic explicit Euler,
Heun, RK4 or DOPRI5. Independent meshes require compact replay (`auto` or
`compact`); an explicit `unrolled` request is rejected. Current and minimum CPU
qualification uses JAX 0.12.0.1 and 0.4.38. The minimum joint provider/compiler
NumPy requirement is 2.2.6. Existing provider packing can promote integer or lower-precision input arrays
to its real state dtype; complex packed states are rejected before discovery
or preparation. CPU tests cover the stated first-derivative transforms.
This does not qualify Torch full-solve transforms,
GPU execution, implicit integration or controller derivatives.

## Discover and validate

```python
from flepimop2.engine.op_engine import IndependentBlockSchedule

discovered = engine.run_adaptive(
    system, times, initial_state, params, model_state=model_state
)
schedule = discovered.schedule
assert isinstance(schedule, IndependentBlockSchedule)
print(schedule.step_counts)  # accepted steps, in compiler block-axis order
print(schedule.batches(engine.config))  # original indices in each replay batch

replayed = engine.run_adaptive(
    system, times, initial_state, params,
    model_state=model_state, schedule=schedule,
)
replayed.require_accurate()  # host-side decision, outside JAX transforms
per_block = replayed.block_replay_diagnostics
```

`IndependentBlockSchedule.blocks` contains one ordinary `AdaptiveSchedule` per
block, including its interval steps, numerical controls and context. The bank
binds the structural model and each mesh's axis position. Changed controls,
output grid, structural context or a reordered mesh bank are rejected. Each
block discovers with the maximum scaled cell error within that block. Appending
an unrelated stiff block cannot change another block's controller reduction.
Declared forcing jumps are honored in every discovery and replay plan.

Replay returns per-block arrays for accuracy flags, maximum scaled local error,
inaccurate-step counts and actual accepted-step counts, plus the existing scalar
aggregate diagnostics. These arrays preserve original block order. The default
local-error threshold is `replay_error_factor: 1.25`; this is a freshness check,
not a global trajectory-error certificate. Independently check the observable
accuracy needed by the consumer.

Parameters and initial-state values remain dynamic and are not hashed. A
compatible mesh can become inaccurate when those values change. Use the
schedule-aware entry point to inspect diagnostics and refresh affected regimes
at an outer orchestration boundary. `engine.run()` and prepared executors return
only the trajectory and do not raise on a traced accuracy flag. The prepared
cache includes every block mesh; its `internal_step_count` is total accepted
work across blocks. Clear and rebuild prepared objects if system internals
change. Gradients are conditional on the frozen meshes, with shared-parameter
contributions summed across blocks.

## Batching, storage and compilation

| Policy | Replay behavior |
| --- | --- |
| `block_schedule: shared` | Discover a conservative full-state mesh, then vmap that mesh across all blocks. Appropriate when workloads are similar or staging cost dominates. |
| `independent`, `block_batching: length` | Group exactly equal total accepted-step counts. Every member carries its own step times, sizes and requested-output slots. Interval meshes need not agree. |
| `independent`, `block_batching: serial` | Replay one member per batch as a qualification/reference option. |

`block_batch_size` limits each independent bucket. There are no padded ODE steps
between unequal-length members. Chunk checkpointing can add inactive scan slots
inside a checkpoint chunk, as in shared replay. `replay_checkpoint: step|chunk`
requires `adaptive_replay: compact` and trades reverse-mode storage for
recomputation; it does not change a mesh.

Discovery currently visits blocks serially using the existing bounded compiled
controller. It includes per-block tracing/compilation and host mesh extraction;
that setup can cost more than shared discovery. Replay constructs one vmapped
scan per batch. Many distinct lengths or a small batch limit increase graph and
compilation costs. Shapes, batch membership and mesh contents are static within
a prepared executor; changed shapes/meshes require a new plan and compilation.
Changing batching without changing numerical controls can reuse the mesh bank.

Mesh metadata and replay-plan constants use storage proportional to the sum of
accepted lengths. Discovery needs one block's bounded controller buffer at a
time. Primal replay carries current block states, requested outputs and small
error summaries; it does not return the full internal-step history. Flattening
and restoring block order allocate requested-output arrays. The batch limit
bounds a batch's concurrent working state; it does not bound total output,
compiled constants or reverse-mode residual storage. Measure those separately.

## Reproduce the CPU measurements

From `flepimop2-op_engine`, after `uv sync --dev`:

```console
JAX_PLATFORMS=cpu .venv/bin/python -m benchmarks.independent_schedules \
  --blocks 4 16 --workers-per-case 3 --repeats 7 \
  --output /tmp/independent-blocks.jsonl
```

Each policy runs in a fresh CPU process, pinned to one available logical CPU on
Linux. Receipts include source hashes, dependency versions, configuration,
per-block work and buckets, all warm timings, model setup, discovery,
preparation, compilation and first-call costs. Both discovery and replay must
pass the same analytic pointwise gate before a row is marked successful; local
replay diagnostics must also pass. The workload has repeated slow-decay blocks
and one fast-decay block. It qualifies this bounded workload, without selecting
application defaults.

Compiler memory-analysis fields are estimates of argument, result, temporary
and aliased bytes for the compiled primal. Process peak RSS includes imports,
discovery and compilation; neither quantity is a measured device peak or a
reverse-mode memory bound. More block steps saved need not mean a proportional
wall-time or memory improvement. Compare the complete lifecycle and the intended
number of replays before selecting a policy.

## Recorded bounded CPU result

The retained [18 worker receipts](../assets/independent-block-schedules-cpu/workers.jsonl)
and [summary](../assets/independent-block-schedules-cpu/summary.json) pass all common
accuracy and replay gates. Measurements use an Intel Core Ultra 7 265U, one
logical CPU, float32, JAX 0.12.0.1 and NumPy 2.5.4. Each block has four state
cells; setup and dispatch are significant for this cheap RHS.

| Blocks | Policy | Block-steps | Discovery (s) | Compile (s) | Warm replay (µs) | Compiler temporary estimate (bytes) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | shared | 212 | 0.427 | 0.217 | 13.3 | 1048 |
| 4 | independent-length | 62 | 1.098 | 0.739 | 31.5 | 1496 |
| 4 | independent-serial | 62 | 1.104 | 0.666 | 32.1 | 2520 |
| 16 | shared | 848 | 0.546 | 0.342 | 43.0 | 2584 |
| 16 | independent-length | 98 | 2.737 | 0.828 | 46.5 | 3672 |
| 16 | independent-serial | 98 | 3.035 | 2.784 | 77.9 | 8664 |

Independent scheduling saves 3.4×/8.7× accepted block work at 4/16 blocks,
but increases discovery and compilation costs and does not improve warm
latency or compiler temporary estimates in this workload. Length batching is
less costly than serial replay at 16 blocks. These results support retaining
the shared default and measuring a consumer’s complete lifecycle before
adopting independent meshes; they establish no application speedup.
