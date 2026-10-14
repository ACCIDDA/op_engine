# Per-run random streams

The stochastic solvers already accept injected samplers on each run. NumPy
states can now use `rng=` instead: pass a `numpy.random.Generator` or
`SeedSequence` to `DirectSSASolver.run`, `TauLeapingSolver.run`,
`AdaptiveTauLeapingSolver.run`, or `ThinningSSASolver.run`. This creates local
NumPy convenience samplers without rebuilding the solver or changing any
independently held sampler.

```python
import numpy as np
from op_engine import DirectSSASolver, ModelCore

core = ModelCore(1, 1, np.array([0.0, 0.5, 1.0]))
initial = np.zeros((1, 1))
core.set_initial_state(initial)
solver = DirectSSASolver(core, np.array([[1.0]]))

def births(t, y):
    return np.full(y.shape, 12.0)

seed = np.random.SeedSequence([3, 7, 2, 2026])
solver.run(births, rng=seed)
first = core.state_array.copy()
core.set_initial_state(initial)
solver.run(births, rng=seed)
assert np.array_equal(core.state_array, first)
```

Reset the `ModelCore` with `set_initial_state` before rerunning it. Resetting
state/time/history and resetting random draws are separate operations. A run
does not automatically restore initial state or roll back a failing simulation.

## Stream ownership and compatibility

A supplied **Generator** is used directly and advances in place. Reusing that
object continues its stream, even after resetting model state. Construct a
fresh generator from the same recorded initialization to replay. Do not share
one mutable generator across concurrent tasks.

A supplied **SeedSequence** initializes a fresh generator per run. Reusing it
replays the same draw stream when the method, model, state, parameters, grid and
software stack are identical. The engine does not call `spawn` or mutate its
child counter. Adaptive tau's Poisson and exact-event convenience samplers share
one call-local generator, consuming it in the algorithm's draw order.

`NumpyPoissonSampler`, `NumpySSASampler`, and `NumpyThinningSampler` also accept
`rng=`. A sampler constructed from a SeedSequence initializes once and then
advances statefully when reused, just like a legacy seeded sampler. Existing
`Sampler(seed)` construction remains valid. An integer `seed`
and `rng` are mutually exclusive; solver `rng` accepts only Generator or
SeedSequence, not bare integers. Missing samplers/streams and combining `rng`
with explicit sampler(s) raise `ValueError` before integration. Non-NumPy
states reject this convenience path before sampling, including inactive runs.

Existing explicit sampler protocols retain their signatures, native namespaces
and draw-index semantics. JAX functional keys or Torch-native sampling belong
in those injected samplers. An adaptive run without `rng` still needs both its
Poisson and exact-event samplers; an explicit stream cannot silently replace
either one. There is no instance generator on the solver to override: stateful
legacy randomness lives in the injected NumPy sampler.

## Indexed replicates and reproducibility

Use stable, nonnegative identifiers to initialize a stream independently of
worker launch order:

```python
stream = np.random.SeedSequence([replicate_id, unit_id, stage_id, root_seed])
core.set_initial_state(initial)
solver.run(births, rng=stream)
```

Record the identity order and root seed. Distinct replicate/unit/stage identities
give different initializations with NumPy's probabilistic stream-independence
properties. Fixed allocation trees can alternatively use recorded `spawn`
children. Sequential spawning depends on allocation order; indexed identities
make reordered execution straightforward. These schemes follow
[NumPy's parallel random-stream guidance](https://numpy.org/doc/stable/reference/random/parallel.html).
The engine delegates PRNG selection and initialization to NumPy; a caller can
supply a Generator with its chosen BitGenerator.

Record NumPy/BitGenerator versions and type, method/configuration, output and
forcing grids, state layout, model/parameter identity and stream identity with
study artifacts. NumPy's default Generator distribution draws do not carry a
general cross-version bitwise guarantee; see its
[compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html).
No random source is serialized automatically by this engine interface.

Repeating a stream does not guarantee the same path after changing parameters,
methods or adaptive decisions: those can change draw counts/order. Shared seeds
alone do not establish event-wise common-random-number coupling, statistical
independence, or an exact likelihood target. Such consumer policies remain
separate from this per-run numerical interface.

Required NumPy qualification checks same-solver replay, generator advancement,
reordered indexed replicates, preservation of existing samplers, conflicting
sources and analytic constant-birth endpoint mean/variance. Native JAX/Torch
checks verify the NumPy convenience boundary. Existing injected-key JAX tests
remain in the baseline stochastic suites; this change adds no new native PRNG
implementation or stochastic differentiation claim.
