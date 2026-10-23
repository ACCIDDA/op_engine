# Steady States

`op_engine.steady_state` finds an equilibrium `y*` with `rhs(t, y*) = 0`, such
as the endemic state of an epidemic model, without a long burn-in
integration. It uses pseudo-transient continuation: each iteration takes an
implicit-Euler step `(I/dt - J) delta = rhs(y)` and grows `dt` as the residual
falls, so early iterations follow the dynamics and later ones become Newton
steps.

## A first solve

```python
import numpy as np

from op_engine import require_steady_state, steady_state

a = np.array([[-2.0, 1.0], [0.5, -1.0]])
b = np.array([1.0, 2.0])

result = require_steady_state(steady_state(lambda t, y: a @ y + b, np.zeros(2)))
print(result.state, int(result.iterations))
```

`steady_state` returns a `SteadyStateResult` whose diagnostics are arrays:
`converged`, `iterations`, `residual_norm`, `step_norm`, `dt`, and
`invariant_drift`. `require_steady_state` raises
`SteadyStateConvergenceError` when the solve did not converge.

## Epidemic models: sinks and conserved totals

Two features of epidemic models need declaring.

- **Absorbing states.** A cumulative counter (deaths, incidence) never reaches
  a zero derivative while its inflow is positive. Pass it as `fixed`, by index
  or mask: it keeps its starting value and is excluded from the convergence
  test. `sink_states(rhs, samples)` finds the states no derivative depends on.
- **Conserved totals.** When births balance deaths (`birth = mu * N`), the
  total population is conserved, so the steady states form a family indexed by
  it and the Jacobian is singular. Pass the conserved directions as
  `invariants`: each row `w` holds `w @ y` at its starting value exactly, and
  the solve stays well posed as `dt` grows.
  - `conserved_quantities(stoichiometry)` returns the structural laws of a
    reaction network, such as `S + I + R` for a closed SIR.
  - `linear_invariants(rhs, samples)` also finds totals conserved because
    rates balance, which births make invisible to the stoichiometry.

```python
from op_engine import linear_invariants, sink_states

beta, gamma, mu, omega = 0.4, 0.1, 1 / (70 * 365), 1 / 365


def sirs(t, y):
    s, i, r, deaths = y
    n = s + i + r
    infection = beta * s * i / n
    return np.array([
        mu * n - infection - mu * s + omega * r,
        infection - (gamma + mu) * i,
        gamma * i - (mu + omega) * r,
        mu * i,
    ])


rng = np.random.default_rng(0)
samples = [rng.uniform(10.0, 3000.0, 4) for _ in range(3)]
sinks = sink_states(sirs, samples)  # [False, False, False, True]
invariants = linear_invariants(sirs, samples, fixed=sinks)  # S + I + R

y0 = np.array([4990.0, 10.0, 0.0, 0.0])
endemic = require_steady_state(
    steady_state(sirs, y0, fixed=sinks, invariants=invariants)
)
print(endemic.state[:3], float(endemic.invariant_drift))
```

`linear_invariants` is numerical: a mode decaying more slowly than `tol`
(default `1e-6`) times the fastest rate looks conserved, and slow modes limit
the accuracy of the rows it returns. When you know an invariant, pass the
exact row (here `[[1, 1, 1, 0]]`) and use `linear_invariants` to confirm that
nothing else is conserved.

For an op_system model, use the compiled RHS. The reaction network gives the
structural laws:

```python
from op_system import compile_spec

from op_engine import conserved_quantities, from_compiled_rhs

compiled = compile_spec({
    "kind": "transitions",
    "state": ["S", "I", "R"],
    "transitions": [
        {"name": "infect", "from": "S", "to": "I", "rate": "b * I / (S + I + R)"},
        {"name": "recover", "from": "I", "to": "R", "rate": "g"},
        {"name": "wane", "from": "R", "to": "S", "rate": "w"},
    ],
})
params = {"b": 0.4, "g": 0.1, "w": 0.01}
network = from_compiled_rhs(compiled, params)
closed = conserved_quantities(network.stoichiometry)  # one row: S + I + R


def rhs(t, y):
    return np.asarray(compiled.eval_fn(t, y, **params))


sirs_state = require_steady_state(
    steady_state(rhs, np.array([990.0, 10.0, 0.0]), invariants=closed)
).state
```

## Local linear-stability evidence

`local_stability` examines the linearization at a candidate equilibrium without
changing how `steady_state` solves for that candidate. Let `y*` be an
`n`-coordinate state and let `f(t, y)` be the right-hand side, with units of
state per unit time. The diagnostic first evaluates every coordinate of
`f(t, y*)`. For the free-coordinate set `A` and positive residual tolerances
`r_i`, it accepts the candidate only when

```text
sqrt(mean((f_i(t, y*) / r_i)**2 for i in A)) <= 1.
```

Residuals at fixed coordinates do not enter this test, because a cumulative
counter can have nonzero flow at an equilibrium of the other coordinates. The
result retains both the complete residual and the tested free-coordinate
residual. An unconverged `SteadyStateResult` remains a solve failure even when
its state passes a separately chosen residual threshold.

The following two-state system has Jacobian eigenvalues `-1 +/- 2i`, in inverse
days. Its local perturbations therefore decay with a leading e-folding time of
one day.

```python
import numpy as np

from op_engine import LocalStabilityConfig, local_stability

jacobian = np.array([[-1.0, -2.0], [2.0, -1.0]])
candidate = np.zeros(2)
controls = LocalStabilityConfig(
    residual_tolerance=1e-10,
    spectral_absolute_tolerance=1e-10,
    spectral_relative_tolerance=1e-8,
    boundary_tolerance=1e-12,
)
stability = local_stability(
    lambda _t, state: jacobian @ state,
    candidate,
    config=controls,
    jacobian=lambda _t, _state: jacobian,
    jacobian_reference="analytic matrix in this example",
    coordinates=("x", "y"),
    time_unit="day",
)

assert stability.classification.value == "asymptotically_stable"
assert np.isclose(stability.leading_timescale, 1.0)
```

The configuration has no scientific defaults. `residual_tolerance` has the
units of the corresponding right-hand-side coordinate;
`spectral_absolute_tolerance` has inverse-time units;
`spectral_relative_tolerance` is dimensionless; and `boundary_tolerance` has
the units of the state coordinate. Each residual or boundary tolerance may be
a scalar or an `n`-vector. Central finite differences use the configured
`finite_difference_step`, in state units, or record the default perturbation
`cbrt(machine epsilon) * max(abs(y_i), 1)` for each coordinate.

### Jacobian sources and constraints

For a local stability interpretation, `f` must be continuously differentiable
near `y*` and autonomous. If `f` depends explicitly on time, the returned
matrix and eigenvalues describe only the frozen time supplied as `t`; they do
not establish stability of the nonautonomous system.

The diagnostic accepts one of three mutually exclusive Jacobian sources:

- an `op_system.SymbolicDerivatives` artifact, evaluated in its declared state
  order and recorded with its structural model digest;
- a dense analytic callback with a required `jacobian_reference`; or
- an explicit central finite-difference run when neither source is supplied.

A failed symbolic evaluation remains a Jacobian failure. It does not silently
switch to finite differences. All successful right-hand-side and Jacobian
evaluations must return NumPy arrays with the candidate's shape and floating
dtype. When an `op_system` artifact is supplied, the caller must also ensure
that the RHS and parameter bindings represent that same compiled model. The
diagnostic verifies the artifact's state order and structural digest, but it
cannot prove the identity of an arbitrary Python RHS callback.

Fixed coordinates and declared linear invariants restrict admissible local
perturbations. If `E_F` selects the fixed coordinates and the rows of `W`
contain the invariants, the diagnostic forms `C = [E_F; W]` and obtains an
orthonormal basis `Q` for `ker(C)` by singular value decomposition. It analyzes
the tangent Jacobian

```text
J_T = Q.T @ J @ Q,
```

where `J` is the complete `n` by `n` Jacobian. The result retains the complete,
free-coordinate and tangent spectra. It also checks that `W` annihilates the
free-coordinate Jacobian within the reported scale-aware tolerance. Dependent
constraint rows and failed invariant checks return an explicit failure instead
of removing modes. A stable tangent spectrum therefore means asymptotic
stability of the linearization on the declared constraint manifold, not in the
unrestricted state space.

For tangent eigenvalues `lambda`, the reported spectral band is

```text
tau = spectral_absolute_tolerance
      + spectral_relative_tolerance * max(abs(lambda)).
```

The result is `asymptotically_stable` only when every real part is strictly
less than `-tau`, and it is `unstable` when any real part is strictly greater
than `tau`. If every real part lies within the closed band, the result reports
`neutral_spectrum`; a mixture of critical and decaying modes reports
`nonhyperbolic`. Neither critical result establishes nonlinear stability. For
example, a zero linearization can occur at an attracting cubic root and in a
nilpotent system with growing solutions. Equality with either threshold is
treated as critical.

Eigenvalues of repeated, defective or strongly nonnormal matrices can be
sensitive to floating-point perturbations even when the reported values lie
outside the chosen band. The diagnostic does not estimate eigenvalue condition
numbers, transient amplification or a pseudospectrum. Examine those properties
separately when the model or coordinates make nonnormality plausible.

A leading e-folding time is reported only for a strictly stable or unstable
interior classification. A stable result uses
`-1 / max(real(lambda))`; an unstable result uses
`1 / max(real(lambda))`. The caller supplies the time-unit label.

### Boundaries and execution scope

Set `nonnegative=True`, or provide a boolean mask, only for coordinates with a
one-sided non-negative domain. If a free coordinate lies within its declared
boundary tolerance, the result reports the affected coordinates and suppresses
an interior classification. A supplied Jacobian can still provide descriptive
spectral evidence, but a tangent-cone or complementarity analysis is required
for a boundary conclusion. The central-difference route stops before probing
across such a boundary.

Boundary declarations refer to the supplied coordinate system. A zero boundary
in natural population or concentration coordinates is not a finite point after
a logarithmic transformation, so a result in log coordinates cannot be used as
the corresponding natural-coordinate boundary analysis.

`local_stability` is an eager, dense NumPy CPU diagnostic. It is not JIT
traceable, differentiable, or qualified as a CUDA or portable Array API
eigensolver. Transfer another backend's state explicitly at the caller
boundary, for example with `np.asarray(jax.device_get(state))`. This host-only
scope does not change the native JAX behavior of `steady_state`.

The returned `status` distinguishes residual, solve, Jacobian, constraint,
eigensolver, non-finite, boundary and zero-dimensional outcomes. Inspect the
status before using the optional classification or timescale. Numerical
eigenvalues support a local linearized conclusion under these declarations;
they do not certify nonlinear or global stability.

## Convergence

A solve converges when two conditions hold, with the scale `atol + |y|` per
state:

- the scaled residual RMS is at most `residual_tol`;
- the Newton correction, scaled the same way, is at most `step_tol`.

The second condition matters for slowly decaying modes. Near a steady state
the error is about `residual / rate`, so a residual of `1e-9` along a mode
decaying at `1e-7` per day still leaves an error of about `1e-2`. The Newton
correction measures that error directly, and the final correction is applied
to the returned state.

Pseudo-transient continuation converges to *a* steady state: started next to
an unstable one, such as the disease-free state, it can stop there. The `dt`
schedule holds the step through bounded residual growth, as it does while
an epidemic takes off. A small `dt0` (the default is 1 time unit) lets the
iterates follow the dynamics toward a stable equilibrium. Candidates with
non-finite states or residuals, or with scaled or unscaled residual growth
above `max_growth`, are rejected: the previous state and residual are retained,
and `dt` is halved for the next attempt. Rejections count toward
`max_iterations`; a smaller `dt0` or a short time integration can provide a
better starting point if the budget is insufficient. Warm starts from a nearby
solution converge fastest; raise `dt0` for those.

For compartment populations or concentrations, pass `nonnegative=True` to
reject negative candidates, including the final Newton correction. A boolean
mask constrains selected states when the model also contains signed variables:

```python
result = steady_state(rhs, y0, invariants=totals, nonnegative=True)
```

This guard is especially useful for fast take-off from a seeded disease-free
state, where a large pseudo-time step can overshoot into negative populations
or select the unstable equilibrium. Rejected steps preserve the invariant
totals; the solver does not clip states. Signs remain unrestricted by default.
The guard helps follow a physical trajectory but cannot guarantee convergence
to a particular equilibrium or avoid an iteration limit.

`SteadyStateConfig` collects the controls: `dt0`, `dt_max`, `min_growth`,
`max_growth`, `max_iterations`, `residual_tol`, `step_tol`, and `atol`. The
default tolerances assume double precision.

The safeguard retains pseudo-transient continuation for the coupled system.
[Brent's method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
and [bisection](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html)
need a scalar sign-changing bracket, which a general RHS does not provide.
The existing curvature-weighted grid utilities are spatial preprocessing tools.
For background on constraints and step control, see
[projected pseudo-transient continuation](https://ira.lib.polyu.edu.hk/bitstream/10397/6101/1/Kelley_Projected_Pseudotransient_Continuation.pdf)
and PETSc's [pseudo-time step verification](https://petsc.org/release/manualpages/TS/TSPseudoSetVerifyTimeStep/).

## JAX: jit and vmap

Pass an automatic-differentiation Jacobian and `loop=jax.lax.fori_loop`. The
loop compiles the iteration once, where the default Python loop would unroll
`max_iterations` copies of it. Invariants and the `fixed` mask are setup data,
so build them outside the traced function.

```python
import jax
import jax.numpy as jnp

from op_engine import SteadyStateConfig

jax.config.update("jax_enable_x64", True)
total = np.ones((1, 3))


def endemic_state(beta, y0):
    def rhs(t, y):
        s, i, r = y[0], y[1], y[2]
        n = s + i + r
        return jnp.stack([
            mu * n - beta * s * i / n - mu * s + omega * r,
            beta * s * i / n - (gamma + mu) * i,
            gamma * i - (mu + omega) * r,
        ])

    result = steady_state(
        rhs,
        y0,
        jacobian=lambda t, y: jax.jacfwd(lambda z: rhs(t, z))(y),
        invariants=total,
        config=SteadyStateConfig(max_iterations=60),
        loop=jax.lax.fori_loop,
    )
    return result.state, result.converged


states, converged = jax.jit(jax.vmap(endemic_state, in_axes=(0, None)))(
    jnp.array([0.3, 0.4, 0.5]), jnp.array([4990.0, 10.0, 0.0])
)
```

Without a `jacobian`, `steady_state` uses forward differences, which cost one
RHS evaluation per state per iteration.
