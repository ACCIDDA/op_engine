# Choosing and configuring a solver

`ModelCore` owns output times and state storage. Deterministic systems use
`CoreSolver` and a method selected by `RunConfig`. Stochastic reaction networks
use `DirectSSASolver` or `TauLeapingSolver`, because propensities and
stoichiometry have different semantics from an ODE right-hand side. In every
case the state array selects the Array-API namespace; choosing JAX instead of
NumPy does not select a different numerical method.

## Deterministic method guide

| Method | Order | Split/operator input | Typical use |
| --- | ---: | --- | --- |
| `euler` | 1 | None | Debugging and first-order reference runs |
| `heun` | 2 | None | Default non-stiff explicit integration |
| `rk4` | 4 | None | Accurate fixed-step non-stiff integration |
| `dopri5` | 5 (embedded 4) | None | Efficient adaptive non-stiff integration |
| `imex-euler` | 1 | One implicit-Euler operator factory | Robust first-order split systems |
| `imex-heun-tr` | 2 | One trapezoidal operator factory | Second-order explicit/implicit splitting |
| `imex-trbdf2` | 2 | Trapezoidal and BDF2-stage factories | Split systems needing stronger damping |
| `imex-ark3` | 3 (embedded 2) | One implicit-Euler stage factory | Higher-order adaptive split systems |
| `implicit-euler` | 1 | Jacobian callable | One-linearization approximation with first-order damping |
| `trapezoidal` | 2 | Jacobian callable | One-linearization, second-order integration |
| `bdf2` | 2 | Jacobian callable | Uniform, fixed-step, one-linearization integration |
| `ros2` | 2 (embedded 1) | Jacobian callable | L-stable linearly implicit integration |
| `sdirk2` | 2 | Full RHS Jacobian and nonlinear solver | L-stable fully nonlinear integration |

`bdf2` currently requires a uniform output grid and `adaptive=False`. Its first
step uses linearly implicit Euler because no previous state exists yet. The
other methods support the built-in adaptive controller, subject to the
compiled-control-flow boundary described below.

The [linear-multistep design note](linear-multistep-methods.md) records the
coefficient/history contract, restart rules, and the current no-go decision on
exposing BDF3 before variable-step and rejection semantics exist.

## Stochastic reaction networks

Both stochastic solvers use one reaction contract: the stoichiometric matrix
has shape `(n_species, n_reactions)`, while the propensity function returns
the state shape with `n_reactions` replacing `n_species` on the configured
reaction axis.

### Exact direct SSA

`DirectSSASolver` implements
[Gillespie's direct method](https://doi.org/10.1021/j100540a008). It samples one
exponential waiting time and one categorical reaction/batch event, applies
exactly one stoichiometric update, and repeats. A draw beyond an output boundary
is retained rather than discarded, so adding observation times does not alter
the simulated path. Zero total propensity is absorbing.

Use direct SSA as the reference method for low-copy-number networks, rare-event
questions, or checking a tau-leaping approximation. Its cost is proportional to
the number of individual events, so tau-leaping is usually preferable when
populations and firing rates are high.

```python
import numpy as np

from op_engine import DirectSSASolver, ModelCore, NumpySSASampler

times = np.linspace(0.0, 4.0, 41)
core = ModelCore(n_states=2, n_subgroups=1, time_grid=times)
core.set_initial_state(np.asarray([[20.0], [0.0]]))

# A -> B
stoichiometry = np.asarray([[-1], [1]])


def propensity(_time, state):
    return 0.2 * state[0:1]


DirectSSASolver(core, stoichiometry).run(
    propensity,
    NumpySSASampler(seed=2026),
)
```

The direct method assumes propensities remain constant between reaction events,
as in a time-homogeneous continuous-time Markov chain. Across batch cells it
samples from the superposed event process; the result is equivalent to
independent direct-method trajectories for independent batch cells.

### Fixed-step tau-leaping

Fixed-step explicit tau-leaping approximates the number of firings in each
reaction channel over a time interval with an independent Poisson draw. Supply
a stoichiometric matrix with shape `(n_species, n_reactions)` and a propensity
function whose reaction axis has length `n_reactions`:

```python
import numpy as np

from op_engine import (
    ModelCore,
    NumpyPoissonSampler,
    TauLeapingConfig,
    TauLeapingSolver,
)

times = np.linspace(0.0, 4.0, 41)
core = ModelCore(n_states=2, n_subgroups=1, time_grid=times)
core.set_initial_state(np.asarray([[1_000.0], [0.0]]))

# A -> B
stoichiometry = np.asarray([[-1], [1]])


def propensity(_time, state):
    return 0.2 * state[0:1]


solver = TauLeapingSolver(core, stoichiometry)
solver.run(
    propensity,
    NumpyPoissonSampler(seed=2026),
    config=TauLeapingConfig(max_step=0.05),
)
```

`max_step=None` takes one leap per output interval. Setting `max_step` divides
each interval into smaller leaps and clips the final leap so that it lands on
the requested output time. The propensity is evaluated at the beginning of
each leap.

The core solver does not own random-number state. Instead, it receives a
`PoissonSampler`, keeping PRNG details outside the numerical method. A JAX run
can use an explicit key and the stable leap index:

```python
key = jax.random.key(2026)


def poisson(mean, step_index):
    return jax.random.poisson(jax.random.fold_in(key, step_index), mean)
```

This first implementation deliberately rejects a leap that produces a
negative population. It neither clips counts nor changes the process
silently. Reduce `max_step` for a better fixed-tau approximation. Bounded or
adaptive tau selection is a separate future method.

Tau-leaping is currently an eager execution path: validation reads sampled
counts and proposed populations back to Python. JAX arrays stay in their native
namespace, but a sampled stochastic trajectory does not provide an ordinary
pathwise `jax.grad` derivative. Differentiable deterministic counterparts can
continue to use `CoreSolver`; gradient estimators for stochastic paths belong
at a higher inference/provider layer.

## Explicit deterministic methods

Euler, Heun, classic RK4, and Dormand--Prince need only an RHS. The default is
fixed-step Heun, with one step per output interval:

```python
import numpy as np

from op_engine import CoreSolver, ModelCore
from op_engine.core_solver import AdaptiveConfig, RunConfig

times = np.linspace(0.0, 4.0, 41)
core = ModelCore(n_states=1, n_subgroups=1, time_grid=times)
core.set_initial_state(np.asarray([[1.0]]))


def rhs(_time, state):
    return -0.5 * state


config = RunConfig(
    method="heun",
    adaptive=True,
    adaptive_cfg=AdaptiveConfig(rtol=1e-6, atol=1e-9),
)
CoreSolver(core).run(rhs, config=config)
```

With `adaptive=False`, every adjacent pair of output times defines one solver
step. With `adaptive=True`, the controller may take multiple internal steps but
still stores only the requested output times.

Heun, RK4, and Dormand--Prince share one validated explicit Runge--Kutta
tableau kernel. `rk4` takes four RHS stages per fixed step. Under adaptivity it
uses step doubling to estimate error, so `dopri5` is normally the more efficient
adaptive choice: its embedded fourth-order formula estimates error using the
same seven stages as its fifth-order solution. Dormand--Prince also reuses its
final stage as the first stage of the next accepted step (FSAL), including
across output-time boundaries.

`dopri5`, `rk45`, and `dormand-prince` select the same canonical method. Dense
output is not currently exposed; the solver lands on and stores the requested
output times instead of interpolating between accepted internal steps.

## IMEX methods and operator factories

IMEX methods solve a split system

\[
  y' = A(t, y)y + F(t, y),
\]

where the RHS callable supplies `F` and an operator factory supplies the
implicit part. A resolved operator tuple is either `(L, R)` or
`(predictor, L, R)`, with the core solving `L @ y_next = R @ x` along the
configured operator axis.

Use a `StageOperatorFactory` whenever the output spacing is non-uniform or
adaptive stepping is enabled. The factory receives the attempted full-step
size, a stage scale, and `StageOperatorContext`, so it can rebuild operators
for the actual stage time and state.

```python
import numpy as np

from op_engine import CoreSolver, ModelCore, build_diffusion_matrix
from op_engine.core_solver import OperatorSpecs, RunConfig
from op_engine.matrix_ops import (
    StageOperatorContext,
    make_stage_operator_factory,
)

n_cells = 32
dx = 1.0 / n_cells
diffusivity = 0.02
times = np.linspace(0.0, 1.0, 21)

core = ModelCore(n_states=n_cells, n_subgroups=1, time_grid=times)
core.set_initial_state(np.ones((n_cells, 1)))


def reaction(_time, state):
    return -0.1 * state


def diffusion(ctx: StageOperatorContext):
    return build_diffusion_matrix(
        n_cells,
        dx,
        diffusivity,
        bc="neumann",
        reference=ctx.y,
    )


trapezoidal = make_stage_operator_factory(diffusion, scheme="trapezoidal")
config = RunConfig(
    method="imex-heun-tr",
    operators=OperatorSpecs(default=trapezoidal),
)
CoreSolver(core, operator_axis="state").run(reaction, config=config)
```

For `imex-trbdf2`, provide both stage schemes:

```python
implicit_euler = make_stage_operator_factory(diffusion, scheme="implicit-euler")
config = RunConfig(
    method="imex-trbdf2",
    operators=OperatorSpecs(tr=trapezoidal, bdf2=implicit_euler),
)
```

`imex-ark3` uses the five-stage ARS(4,4,3) additive Runge--Kutta
tableau from [Ascher, Ruuth, and Spiteri
(1997)](https://doi.org/10.1016/S0168-9274(97)00056-1). Its implicit
half is L-stable and stiffly accurate. A shared `c=1/2` stage supplies a
second-order explicit/implicit midpoint estimate, so adaptive error control
does not use step doubling.

```python
implicit_euler = make_stage_operator_factory(diffusion, scheme="implicit-euler")
config = RunConfig(
    method="imex-ark3",
    operators=OperatorSpecs(default=implicit_euler),
)
```

This method always requires a `StageOperatorFactory`, including on a uniform
fixed-step grid. For each non-explicit stage the factory receives the attempted
full-step size, the DIRK diagonal coefficient as `scale`, the actual stage
time, and the assembled stage base in `StageOperatorContext.y`. It must build
an implicit-Euler left operator `I - dt * scale * A`. Predictor tuples are not
accepted, and the returned right operator is not applied: the additive kernel
has already assembled all previous explicit and implicit stage contributions.
The implicit derivative is then recovered from the solved stage equation,
which avoids a duplicate callback for `A @ y` and keeps the method portable
across Array-API namespaces.

`imex-ars443` and `ars443` are aliases for `imex-ark3`.

The flepimop2 provider can compile typed `op_system` `axis_kernel`, `advection`,
and `diffusion` descriptors into these same factories. Descriptor parsing and
axis-label resolution belong to the provider; the numerical builders and
stepping methods remain in `op_engine`.

## Linearly implicit methods

The methods historically named `implicit-euler`, `trapezoidal`, and `bdf2`, as
well as the Rosenbrock method, take a Jacobian callable. The callable returns an
operator acting along `operator_axis`:

```python
def jacobian(_time, state):
    xp = state.__array_namespace__()
    return xp.multiply(xp.eye(state.shape[0], dtype=state.dtype), -0.5)


config = RunConfig(method="ros2", jacobian=jacobian)
CoreSolver(core, operator_axis="state").run(rhs, config=config)
```

`ros2` uses two increment stages and one Jacobian evaluation per attempted
step. Both stages solve with the same matrix
`I - (1 - 1/sqrt(2)) * dt * J`; the accepted second-order state and embedded
first-order state provide the adaptive error estimate. The coefficient table,
formal orders, and controller order are declared together and validated when
the module is imported.

Dense Jacobians stay in the state's namespace. SciPy sparse operators are an
optional NumPy acceleration path and are not a JAX differentiation path.

The first three methods perform a single linearization; they do not iterate a
nonlinear residual to convergence. `sdirk2` instead solves each stage through
the [nonlinear solver contract](nonlinear-solvers.md), with a distinct full
flattened Jacobian and explicit convergence diagnostics.

## Adaptivity and Array-API backends

`AdaptiveConfig` sets tolerances and attempt limits; `DtControllerConfig` sets
step-size bounds and growth factors. Their static scalar values are validated
when the configuration is constructed. Checks that depend on the time grid,
state shape, or resolved operators occur when `CoreSolver` builds its run plan.

Fixed-step explicit, dense IMEX, and dense linearly implicit methods use
ordinary Array-API operations. With JAX arrays, those same methods can run
under `jax.jit` and be differentiated with `jax.grad`; no JAX-specific solver
implementation is selected.

The built-in adaptive controller uses eager Python loops and scalar acceptance
decisions. It preserves JAX arrays, and an eager `jax.grad` differentiates the
accepted sequence of native steps, but the live controller is not itself
JIT-compatible.

For compiled differentiation, record the accepted mesh at representative
parameters and replay it:

```python
solver = CoreSolver(reference_core)
solver.run(rhs, config=config)
schedule = solver.last_adaptive_schedule
assert schedule is not None

# Construct a fresh core and solver inside the function being transformed.
solver = CoreSolver(differentiable_core)
diagnostics = solver.replay_adaptive_schedule(rhs, schedule, config=config)
```

`jax.jit(jax.value_and_grad(...))` can trace the replay because its step count
and step sizes are static, while array-valued model inputs remain dynamic. The
gradient is conditional on that mesh: replay does not differentiate the
accept/reject decisions. Refresh the schedule when parameters, tolerances, the
method, or other model structure change materially.

For SDIRK2, replay returns array-valued nonlinear diagnostics as part of the
compiled result. Inspect `diagnostics.require_converged()` after execution and
discard the replayed state and derivatives if any frozen-mesh stage failed.

A live compiled controller or specialized adjoint implementation can still
belong in an external provider and may return a trajectory through
`ModelCore.apply_trajectory`. It should reuse the portable method semantics
rather than define a JAX-only numerical method.

See [Backend and solve-strategy boundaries](backends.md) for the complete
portability contract and [the biogeochemical tutorial](biogeochemical-network.md)
for a larger IMEX splitting example.
