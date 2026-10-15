# Named observations and sampled-window summaries

`observe_solution` evaluates named instantaneous quantities from an existing
native state trajectory. `observe_model` uses a completed `ModelCore` with
`store_history=True`. Both preserve the raw trajectory API and return a separate
`ObservationResult`: static output times, a read-only mapping of named series,
and evaluation diagnostics. Each series contains its specification and a native
array shaped `(n_output_times, *output_shape)`.

```python
import numpy as np
from op_engine import ObservationSpec, observe_solution

def total(t, y, **params):
    return np.sum(y) * params["gain"] + t

times = np.array([0.0, 0.2, 0.9])
trajectory = np.arange(6.0).reshape(3, 2)
observed = observe_solution(
    times, trajectory, [ObservationSpec("total", total)], params={"gain": 2.0}
)
assert np.allclose(observed.series["total"].values, [2.0, 10.2, 18.9])
```

Times must be finite, strictly increasing, one dimensional, and nonempty. They
are the supplied trajectory's **stored output times**, not solver internal
steps. The trajectory must have one leading row per time and a real floating
dtype. Custom callbacks receive each state in its natural shape, plus time and
the supplied parameter mapping. They must return arrays in that namespace and
dtype with exactly their declared output shape. Scalar outputs have shape `()`.
An empty observation selection and a single stored output time are valid.

`ObservationMetadata` describes output shape, ordered coordinate labels, units,
direct state/parameter dependencies and already-applied physical measures.
Coordinates, when supplied, must match every output dimension. Custom callbacks
own scientific definitions and metadata; the engine checks output layout and
numerical boundaries without inferring geometry or dimensions. `time_dependent`
describes explicit time use, not all indirect evolution through a state.

## Consuming compiled op_system observables

`observation_from_compiled` reads the compiler artifact structurally. Neither
op_system nor flepimop2 is imported by engine core. Use a compiler build exposing
the named observables from [op_system #294](https://github.com/ACCIDDA/op_system/issues/294);
the pinned cross-package qualification commit is documented in
[backend qualification](backend-qualification.md).

```python
from op_engine import observation_from_compiled, observe_model

# compiled: an op_system CompiledRhs with named observables
# core: a completed ModelCore whose state order matches compiled.state_names
specs = [observation_from_compiled(item) for item in compiled.observables.values()]
observed = observe_model(core, specs, params={"gain": 2.0})
```

The adapter flattens each state without reordering its elements, so the caller
must retain `compiled.state_names` order. It delegates coordinate selection and
reductions to the compiler evaluator. Exported physical weights describe what
that evaluator already applied; the engine applies no additional weights.
Observation-only parameters remain separate from dynamics parameters. All
supplied parameters are passed to every callback; compiled evaluators select
their own required parameters.

The integration fixture uses the same definitions on meshes with cell centers
`[.25,.75]` and `[.25,.55,.8]`, with widths `[.5,.5]` and `[.5,.1,.4]`.
The exact cell averages of `rho(x,0)=1+2*x` integrate to `2` on both meshes.
For `rho'=-k*rho`, the physical total is `2*gain*exp(-k*t)` and the shared
declared-node sample at `x=.25` is `1.5*exp(-k*t)`. This checks nonuniform/refined
meshes, scientific axis metadata and preservation of observation-only gain.
A declared-node sample uses an existing coordinate; no spatial interpolation
is implied by this fixture.

## Accuracy, derivatives and storage boundaries

`ObservationDiagnostics` reports callback evaluation count and explicitly
records `temporal_interpolation="none"`, `dense_output_order=None`, and
`error_estimate=None`. Observation accuracy inherits both the supplied numerical
trajectory and callback. This interface does not estimate a separate error or
upgrade the solver's order. No solver method has a qualified continuous
extension through this interface yet. Output times do not describe all
accepted/rejected internal steps; use solver/replay diagnostics for those.

Evaluations use native indexing, reshaping and stacking. Independent CPU
qualification covers scalar/field values, JAX JIT/vmap and derivatives, CPU
Torch derivatives in both precisions, and fixed RK4/compiler state/rate/gain
derivatives in float64. For JAX transforms, extract native `.values` arrays
inside the transformed function; the static result mapping is not registered
as a JAX pytree. Times, definitions, coordinates and measures are static for
these checks. Callback-specific transforms, mesh/time derivatives, higher
derivatives, GPUs and live adaptive controller differentiation need separate
evidence.

Stored-state evaluation requires the entire supplied trajectory. Its input
storage is `O(n_times*n_state)` and named output storage is
`O(n_times*sum(output_sizes))`; evaluation retains one output's intermediate
values while stacking. There is no checkpoint/rematerialization policy in this
postprocessing path. A model without history or an incomplete model fails
explicitly. Treat native input/output arrays as immutable.

## Streaming endpoint changes and time integrals

`reduce_observations` consumes a finite iterable of `(time, native_state)` pairs
once. It returns an `ObservationSummary` containing the first/last supplied
times as `window`, a sample count, diagnostics and a read-only named mapping of
`ObservationSummaryValue` records. Each record keeps the original specification
and native `initial`, `final`, `change` (`final-initial`) and `integral` arrays.
Every array has the instantaneous output shape; there is no output-time axis.

```python
import numpy as np
from op_engine import ObservationSpec, reduce_observations

def total(t, y, **params):
    return np.sum(y)

times = [0.0, 0.1, 0.7, 2.0]
samples = ((t, np.array([2 + 3*t, 5 - t])) for t in times)
summary = reduce_observations(samples, [ObservationSpec("total", total)])
assert summary.window == (0.0, 2.0)
assert np.allclose(summary.values["total"].change, 4.0)
assert np.allclose(summary.values["total"].integral, 18.0)
```

An endpoint change is appropriate when the observable is a cumulative counter
whose value already contains the relevant accumulation. A rate is a different
quantity: integrate it over time instead. This API does not reconstruct event
times or infer which state is a cumulative counter. The `initial`, `final` and
`change` units remain those of the instantaneous observable; `integral` units
are those units multiplied by the producer's time unit. No unit conversion is
inferred.

The first/last supplied samples define the exact window. Times are static real
finite scalars and must increase strictly with finite intervals. A single sample
defines a zero-duration window with zero change and integral; an empty stream
fails. State shape, namespace and real floating dtype must stay fixed. Output
validation and observation-only parameters follow `observe_solution`. Empty
observation selections still validate the sample stream. Non-finite output
values propagate rather than being silently removed.

### Quadrature and accuracy

`quadrature="trapezoid"` identifies the
[composite trapezoidal rule](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html)
over the supplied sample times, including nonuniform grids. It integrates the
piecewise linear **sampled output**. It does not evaluate an intermediate state
or provide a solver's continuous extension; diagnostics retain
`temporal_interpolation="none"`, `dense_output_order=None` and
`error_estimate=None`. Refinement of sufficiently smooth sampled data gives the
usual second-order quadrature behavior; a fourth-order solver does not turn
this sampled integral into a fourth-order integral. Discontinuities and coarse
sampling can dominate its error. The reducer does not estimate or control it.

The qualification fixture drives public RK4 steps for `y'=-0.7*y`, `y(0)=3`
on refined nonuniform time grids. Independent exponential references separate
fourth-order endpoint convergence from second-order integral convergence.
Linear field/counter fixtures verify exact integrals and endpoint changes on
irregular grids. Simpson rules or adaptive quadrature have different sample
requirements; no general higher-order option is inferred from the solver name.

### Driving a solver without stored history

The producer owns integration. Existing public functional step kernels can
yield each completed sample without storing a trajectory:

```python
from op_engine import CoreSolver, ModelCore
from op_engine.model_core import ModelCoreOptions

times = np.linspace(0.0, 1.0, 33)
core = ModelCore(1, 1, times, options=ModelCoreOptions(store_history=False))
core.set_initial_state(np.array([[3.0]]))
solver = CoreSolver(core)

def rhs(t, y):
    return -0.7 * y

def samples():
    yield float(times[0]), core.get_current_state()
    for start, end in zip(times[:-1], times[1:]):
        state, _ = solver.fixed_explicit_step(
            rhs, method="rk4", t=float(start), dt=float(end-start),
            y=core.get_current_state(),
        )
        core.advance_timestep(state)
        yield float(end), state

summary = reduce_observations(samples(), [ObservationSpec("total", total)])
assert core.state_array is None
assert summary.sample_count == len(times)
```

This example steps once per supplied interval; it does not apply an adaptive
controller. Custom adaptive producers yield accepted samples and retain their
own controller/replay contract. The reducer does not alter `CoreSolver.run`,
`observe_model` or any raw trajectory API, and requires no provider dependency.
Compiled op_system observations can be adapted with `observation_from_compiled`
and supplied to this reducer using the same state order and measures.

### Memory, differentiation and interruptions

The reducer retains a constant number of arrays per selected output, plus one
current state reference: its numerical storage is
`O(n_state + sum(output_sizes))`, independent of the sample count. Producer
storage is additional. A forward NumPy test verifies that old generator-owned
state arrays become collectible. Native per-sample outputs are snapshotted
before requesting the next sample, so a producer can reuse its state buffer.
Forward NumPy buffer reuse is qualified; backend AD/in-place mutation rules
still apply to differentiated producers.
Treat returned arrays as immutable, and use pure evaluators and a fixed
parameter meaning for the entire window.

JAX JIT/vmap and state/observation-parameter gradients, and CPU Torch gradients
in float32/float64, are independently qualified. Native arithmetic retains
computation graphs: reverse-mode AD may retain work from every sample, so its
memory is **not** bounded by the forward numerical storage claim. There is no
automatic adjoint, checkpointing or rematerialization policy. Extract native
summary arrays inside transformed functions; the static mapping is not a
registered JAX pytree. Times, sample count and specifications stay static in
the qualified transforms.

There is no persistent or resumable reducer checkpoint. If a producer/callback
raises, the consuming call returns no summary and may have advanced the
producer. Recovery requires a fresh producer replay or its own compatible
checkpoint; an incomplete prefix is not silently reported as a full-window
result.

Off-grid times/window boundaries, interpolated physical-coordinate samples,
method-specific dense output and event-specific reconstruction remain later
focused increments of [op_engine #225](https://github.com/ACCIDDA/op_engine/issues/225).
Statistical noise, censoring and missingness belong to consuming observation
models.
