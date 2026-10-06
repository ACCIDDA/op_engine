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
schedule holds the step while the residual grows moderately, as it does while
an epidemic takes off, so a small `dt0` (the default is 1 time unit) lets the
iterates follow the epidemic to the endemic state. Warm starts from a nearby
solution converge fastest; raise `dt0` for those.

`SteadyStateConfig` collects the controls: `dt0`, `dt_max`, `min_growth`,
`max_growth`, `max_iterations`, `residual_tol`, `step_tol`, and `atol`. The
default tolerances assume double precision.

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
