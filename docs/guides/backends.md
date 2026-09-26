# Backend and solve-strategy boundaries

`op_engine` selects the numerical namespace from the state array. Explicit and
dense implicit/IMEX methods use that namespace's Array-API operations, including
`linalg.solve`. The numerical method does not change when the array namespace
changes.

Namespace portability and automatic differentiation are related but distinct.
The Array API does not define `grad`, `jit`, or traced control flow. A namespace
such as JAX can nevertheless differentiate the ordinary `op_engine` methods
because their fixed-step numerical operations remain in that namespace.

## Portable fixed-step methods

With JAX state, fixed-step Euler, Heun, RK4, Dormand--Prince, dense IMEX, and
dense linearly implicit methods can be used inside `jax.jit` and differentiated
with `jax.grad`. Dynamic values can include the initial state, RHS parameters,
dense operator values, and Jacobian values. The time grid, method selection,
state shape, and Python solver configuration are structural inputs and must
remain static during a trace.

Sparse acceleration is backend-specific. SciPy sparse solves are not a JAX
differentiation path; use dense JAX operators when gradients through a solve are
required.

The built-in adaptive controller is an eager Python controller. It preserves a
JAX array namespace, and `jax.grad` can differentiate the numerical operations
on the branch accepted by the nominal solve. Step acceptance itself extracts
scalar error values and uses Python loops and branches, however, so a live
`adaptive=True` solve is not a JAX-`jit`-compatible execution path. This
limitation belongs to the controller, not to the explicit Runge--Kutta, IMEX,
or linearly implicit formulas.

## Adaptive-controller boundary

After a successful adaptive run, `CoreSolver.last_adaptive_schedule` contains
the accepted internal step sizes. A new solver can pass that value to
`replay_adaptive_schedule`. Replay skips error norms and acceptance decisions,
but invokes the same high-order Array-API step kernels. The static schedule can
therefore be used inside `jax.jit(jax.value_and_grad(...))` while parameters,
initial state, dense operators, and Jacobians remain dynamic.

The resulting derivative is conditional on the recorded mesh. This is also the
branchwise meaning of an eager live-adaptive gradient: neither mode differentiates
the discrete accept/reject decision. Refresh the schedule as optimization
parameters move, and always refresh it after a material model or tolerance
change. A schedule is validated against the output grid, while matching the
method and other configuration is the caller's responsibility.

Schedule replay statically unrolls the accepted steps when JAX traces it. For
very long meshes, a provider may instead implement bounded, masked compiled
control flow around the same portable kernels.

Projects that require a compiled adaptive controller, checkpointed adjoints, or
other solver-specific capabilities can provide those at an external plugin or
provider boundary. A specialized integration may return a complete trajectory
and adopt it through `ModelCore.apply_trajectory`; it does not need to add a
backend-specific method to `CoreSolver`.

This keeps optional packages such as Diffrax out of the core dependency and
method surfaces. The portable fixed-step methods and their differentiation
contract remain identical across conforming array namespaces.

## Stochastic sampling boundary

`TauLeapingSolver` keeps reaction-channel arithmetic in the state array's
namespace but injects Poisson sampling through a `PoissonSampler`. This is
necessary because the Array API does not define random-number generation and
because NumPy's stateful generator and JAX's explicit keys have intentionally
different semantics. `NumpyPoissonSampler` is provided as a convenience; JAX
users can construct a fresh key for each leap from the solver's stable
`step_index`.

The current safety checks are eager, so tau-leaping is not a JIT-compatible
path. Moreover, integer Poisson samples do not have an ordinary pathwise
derivative. JAX remains useful for array execution and for differentiating a
deterministic version of the same model, but `jax.grad` through the sampled
trajectory is not part of this API contract. Score-function, reparameterized,
or other stochastic gradient estimators should be implemented explicitly by
an inference/provider layer rather than implied by the array namespace.
