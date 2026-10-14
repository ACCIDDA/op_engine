# Named observations at stored output times

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

Off-grid observation times, interpolated physical-coordinate samples,
method-specific dense output, streaming reductions, time integrals and counter
windows remain later focused increments of
[op_engine #225](https://github.com/ACCIDDA/op_engine/issues/225). Statistical
noise, censoring and missingness belong to consuming observation models.
