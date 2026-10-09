# Backend and solve-strategy boundaries

`op_engine` selects the numerical namespace from the state array with
`array_api_compat.array_namespace()`. This supports arrays with the standard
`__array_namespace__` protocol and native arrays such as `torch.Tensor`
through compatibility namespaces. Explicit,
dense linearly implicit, and IMEX methods use that namespace's Array-API
operations, including `linalg.solve`. The numerical method does not change when
the array namespace changes.

Namespace portability and automatic differentiation are related but distinct.
The Array API does not define `grad`, `jit`, or traced control flow. A namespace
such as JAX can nevertheless differentiate the ordinary `op_engine` methods
because their fixed-step numerical operations remain in that namespace.
The optional `torch` dependency group qualifies representative eager
fixed-step kernels with PyTorch autograd. Compiler integration and complete
method qualification remain provider/backend capabilities rather than
consequences of namespace selection.

## Portable fixed-step methods

With JAX state, fixed-step Euler, Heun, RK4, Dormand--Prince, dense IMEX
(including paired ARK3), and dense linearly implicit methods can be used inside
`jax.jit` and differentiated with `jax.grad`. Dynamic values can include the
initial state, RHS parameters, dense operator values, and Jacobian values. The
time grid, method selection, state shape, and Python solver configuration are
structural inputs and must remain static during a trace.

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

Direct `CoreSolver` schedule replay statically unrolls accepted steps when JAX
traces it. The optional flepimop2 provider consumes the same functional kernels
but defaults explicit JAX replay to one compact `lax.scan`. Its forced compact
mode can also checkpoint/rematerialize each step for long reverse-mode solves;
non-JAX and nonlinear-diagnostic paths retain portable core replay.

## Public functional steps and external loops

`CoreSolver.fixed_explicit_step(rhs_func, *, method, t, dt, y,
first_stage=None)` is the supported public boundary for an external loop
driver. It returns `(y_next, fsal)` for `euler`, `heun`, `rk4`, or `dopri5`,
using the same numerical kernels as fixed-step `CoreSolver.run`. It reads the
solver's configured state shape for validation but does not update its
`ModelCore` state or history. Supply arrays with that shape and an RHS that
preserves their shape and namespace. `t` and `dt` can be traced backend
scalars; the method and state shape remain static.

For example, a non-provider JAX caller can compile a 160-step solve and
differentiate it twice using only public APIs:

```python
import jax
import jax.numpy as jnp
import numpy as np

from op_engine import CoreSolver, ModelCore

solver = CoreSolver(ModelCore(1, 1, np.asarray([0.0, 1.0])))
times = jnp.linspace(0.0, 1.0, 161)


def loss(rate):
    def rhs(t, y):
        return rate * y

    def advance(y, step):
        t, dt = step
        y_next, fsal = solver.fixed_explicit_step(
            rhs, method="dopri5", t=t, dt=dt, y=y
        )
        return y_next, None

    final, _ = jax.lax.scan(
        jax.checkpoint(advance),
        jnp.ones((1, 1)),
        (times[:-1], jnp.diff(times)),
    )
    return 0.5 * jnp.sum(final**2)


hessian = jax.jit(jax.hessian(loss))(jnp.asarray(-0.3))
reverse_over_reverse = jax.jit(
    jax.grad(lambda rate: jnp.sum(jax.grad(loss)(rate) ** 2))
)(jnp.asarray(-0.3))
```

`lax.scan` traces the step body once. `jax.checkpoint` lets reverse mode
recompute intermediate stage values rather than retaining all of them.
The example discards the optional FSAL derivative for a simple carry.
For Dormand--Prince, reuse that derivative as `first_stage` on the next step
to save one RHS evaluation. Take the first step outside the scan to obtain
an array-valued derivative, then scan the remaining steps with
`(y_next, fsal)` as the carry. A scan carry must retain its shape and structure;
starting with `None` and returning an array changes that structure. Other
explicit methods return `None`. Reuse is valid only while the next RHS
evaluation starts at the same time and state with the same model parameters.

An external driver can also flatten a recorded `AdaptiveStepSchedule` into
step start times and sizes. Heun and Dormand--Prince's fixed steps reproduce
their accepted adaptive updates. Euler and RK4's adaptive updates use two
half steps; an external fixed-step replay must split each recorded step in
two to reproduce those updates. In all cases, freeze the mesh during
differentiation and refresh it when the nominal solve changes.

Projects that require a compiled adaptive controller or other solver-specific
capabilities can provide those at an external plugin or provider boundary. A
specialized integration may return a complete trajectory and adopt it through
`ModelCore.apply_trajectory`; it does not need to add a backend-specific method
to `CoreSolver`.

This keeps optional packages such as Diffrax out of the core dependency and
method surfaces. The portable fixed-step methods and their differentiation
contract remain identical across conforming array namespaces.

## Stochastic sampling boundary

`DirectSSASolver`, `AdaptiveTauLeapingSolver`, and `TauLeapingSolver` keep
reaction-channel arithmetic in the state array's namespace. Direct SSA injects
an `SSASampler`, fixed tau injects a `PoissonSampler`, and adaptive tau uses
both for exact critical events and noncritical firing counts. This is necessary
because the Array API does not define random-number generation and because
NumPy's stateful generator and JAX's explicit keys have intentionally different
semantics.
`NumpySSASampler` and `NumpyPoissonSampler` are provided as conveniences.
JAX users can construct fresh keys from the stable direct-SSA `draw_index` or
tau-leaping `step_index`.
Direct SSA's optional forcing schedule is static Python configuration. It
preserves the state namespace and uses fresh draw indices when a forcing
boundary invalidates a pending event; observation times retain that event.

The current safety checks and stochastic event loops are eager, so these
stochastic solvers are not JIT-compatible paths. Adaptive selection also reads
drift/variance reductions, critical classifications, and accept/reject results
back to Python. Moreover, categorical reaction events and integer Poisson
samples do not have an ordinary pathwise derivative. JAX remains useful for
eager array execution and for differentiating a deterministic version of the
same model, but `jax.grad` through a sampled trajectory is not part of this API
contract. Score-function, reparameterized, or other stochastic gradient
estimators should be implemented explicitly by an inference/provider layer
rather than implied by the array namespace.
