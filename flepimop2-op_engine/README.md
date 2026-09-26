# flepimop2-op_engine

Provider package that adapts `op_engine` to `flepimop2`.

For systems supplied by `flepimop2-op_system`, the provider consumes the
compiled `state_names`, `initial_state`, and `axis_labels` options to assemble
the flat solver state. Scalar seeds may be shared by multiple state cells, and
shaped seeds are selected by their named coordinates. Resolved
`ParameterValue` payloads are unwrapped and bound to the system once before
integration; wrappers are not forwarded through every right-hand-side
evaluation.

The numerical array namespace is selected from the initial state rather than
from an engine configuration flag. Initial-state assembly, right-hand-side
evaluation, `ModelCore` history, and the returned `(time, state...)` array stay
in that namespace. Evaluation times remain NumPy arrays because they are static
solver structure. An `op_system` stepper converts bound parameter values into
the state namespace when it evaluates the RHS, so a JAX state can safely consume
static NumPy parameters without moving the evolving state back to the host.

Install the JAX runtime for portable fixed-step JIT and differentiation:

```bash
pip install "flepimop2-op_engine[jax]"
```

Diffrax is not required for JAX differentiation of fixed-step Euler, Heun, or
dense IMEX/implicit methods. Typed dense operator descriptors are compiled in
the evolving state's namespace, so descriptor parameters remain traceable too.
The compiler supports row-source axis-kernel generators and first-order upwind
advection on uniform axes, including dynamic signed velocities.
The NumPy path applies op_system's value-dependent generator validation. A
traced non-NumPy path can validate shapes and static layout only; producers are
responsible for maintaining generator invariants in dynamic parameter values.
