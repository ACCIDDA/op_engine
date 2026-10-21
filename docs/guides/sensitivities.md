# Bounded forward parameter sensitivities

`solve_sensitivities` integrates the smooth deterministic augmented equations
`S' = (df/dy) @ S + df/dtheta`, with `S = dy/dtheta`. It reuses the existing
fixed Euler, Heun, RK4 or DOPRI5 stage equations. This is a numerical
approximation to continuous sensitivities; mesh refinement remains necessary.

The engine accepts the structural `DerivativeArtifact` protocol and imports no
compiler or statistical package. The optional `op_system.symbolic_derivatives`
artifact supplies ordered state, parameter and scalar-measurement matrices from
one retained source IR. Its own bounded supported-operation contract applies.
The caller must bind that artifact to the same RHS and coordinate definitions.

```python
import numpy as np
from op_system import compile_spec, symbolic_derivatives
from op_engine import (
    SensitivityInitial,
    SensitivityOptions,
    SensitivitySystem,
    solve_sensitivities,
)
from op_engine.core_solver import RunConfig

compiled = compile_spec({
    "kind": "expr",
    "state": ["x", "z"],
    "equations": {"x": "-rate*x", "z": "-rate*z"},
    "observables": {"measurement": {"expr": "gain*(x+2*z)"}},
})
system = SensitivitySystem(
    compiled.eval_fn,
    symbolic_derivatives(compiled),
    initial_parameters=("amplitude",),
)
assert system.parameter_names == ("rate", "gain", "amplitude")
solution = solve_sensitivities(
    system,
    {"rate": 0.7, "gain": 1.5, "amplitude": 2.0},
    SensitivityInitial(
        state=np.array([2.0, 4.0]),
        parameter_derivatives=np.array([[0., 0., 1.], [0., 0., 2.]]),
    ),
    np.array([0., 0.5, 1.]),
    options=SensitivityOptions(
        solver=RunConfig(method="rk4", fixed_max_step=0.0625),
    ),
)
assert solution.times == (1.0,)
assert solution.parameter_derivatives.shape == (1, 2, 3)
assert np.allclose(
    solution.observable_derivatives,
    np.exp(-0.7) * np.array([[[-15., 10., 7.5]]]),
    rtol=2e-6,
)
```

Initial-state derivatives are required. Supply an explicit zero matrix for a
parameter-independent initial state. Parameters used only by initialization are
appended in `initial_parameters` order, receive zero RHS/direct-measurement
columns, and enter through the supplied initial derivatives. Measurement-only
parameters retain their compiler order and direct measurement terms.

For N states, P complete parameters, M scalar measurements and T retained times,
the returned native arrays have shapes `(T,N)`, `(T,N,P)` and `(T,M,P)`.
The measured sensitivity is `dg/dy @ S + dg/dtheta`. A state Jacobian alone
does not supply it. Observation noise, prior curvature, experimental design,+nuisance treatment and Fisher definitions belong to inference consumers.

## Memory and execution scope

The augmented state has N*(1+P) entries. Final-only output is the default;
`store_history=True` retains requested output times, not internal stages.
Default bounds are 16,384 augmented entries and 64 MiB of returned numerical
arrays. They do not bound temporary work, compiler memory, retained AD graphs or
whole-process RSS. The derivative producer may additionally allocate bounded
dense matrices at every stage.

State, initial derivatives, native scalar parameters and produced matrices must
retain namespace and dtype. NumPy, JAX and CPU Torch float32/64 are qualified.
For JAX, close over a static system/mesh/options and return the needed native
array fields from the transformed function; the full provenance dataclass is
host metadata, not a registered JAX pytree. `fixed_loop="scan"` and optional
checkpointing reuse the existing engine adapter. Independent solves can be
batched with native vmap. CUDA requires its own device qualification.

The stored mesh records the solver's actual time dtype. Tolerance settings do
not define accuracy for these fixed methods; declare the mesh and maximum step,
then check refinement. Adaptive acceptance, forcing jumps, events, history
dependence, stochastic models, operator/implicit solves and automatic gauge
transformations require separately qualified sensitivity methods. Unsupported
solver configurations fail explicitly.

## Independent checks and cost comparison

Required native backend checks cover independent stability polynomials for all
four methods, shared/local states, initial-only parameters, dtype retention,
same-mesh finite differences, refinement, JAX JIT/vmap/checkpointing and native
AD. A separate mandatory compiler lane checks a coupled two-rate transfer with
transitive aliases and a direct measurement-gain term against a closed form.

With a compiler source that includes the derivative API, run each mode in a
fresh process:

```shell
python examples/forward_sensitivities.py --backend jax --mode augmented --steps 64
python examples/forward_sensitivities.py --backend jax --mode ad --steps 64
```

Both modes return the full `(3,4)` derivative of two final states and one
measurement with respect to two rates, gain and initial amplitude, on exactly
the same fixed RK4 mesh and float64 precision. The augmented system has ten
entries. The comparison uses native forward AD for JAX, native reverse AD for
Torch, and explicitly labelled central finite differences for NumPy. First-call
and synchronized warm times are separate; no universal cost or stability
advantage is assumed.

The [raw local receipt](../assets/sensitivity-costs-2026-10-21.json) retains all
12 fresh-process backend/mesh/mode cases, including Jacobians, source hashes,
continuous-reference errors and whole-worker RSS. All six paired derivative
matrices agree within `rtol=atol=1e-8`. On this shared one-core CPU run, the
64-step warm augmented/AD times were 31/37 microseconds for JAX and 50/112 ms for
Torch; first calls were approximately 0.69/0.64 s and 0.08/0.11 s respectively.
Other local test work was active, each case has only one worker, and methods
differentiate this particular small output. These advisory measurements do not
select a general sensitivity method or establish a portable speedup. The
continuous-reference tolerance scales with fourth-order mesh error; a shared
fixed-mesh derivative and a continuous derivative have different error targets.
