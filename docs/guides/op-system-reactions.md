# Stochastic Simulation From An op_system Spec

An [op_system](https://github.com/ACCIDDA/op_system) spec compiles to typed
reaction artifacts, one per named transition (`CompiledRhs.reactions`).
`op_engine.from_compiled_rhs` turns them into the flat network that
`DirectSSASolver`, `TauLeapingSolver`, and `AdaptiveTauLeapingSolver`
consume. No flepimop2 installation is needed.

```shell
pip install op_engine op-system
```

## Build the network

```python
import numpy as np
from op_system import compile_spec

from op_engine import from_compiled_rhs

spec = {
    "kind": "transitions",
    "axes": [{"name": "age", "coords": ["child", "adult"]}],
    "state": ["S[age]", "I[age]", "R[age]"],
    "transitions": [
        {
            "name": "infect",
            "from": "S[age]",
            "to": "I[age]",
            "rate": "b * I[age]",
            "reactants": "auto",
        },
        {"name": "recover", "from": "I[age]", "to": "R[age]", "rate": "g"},
    ],
}
params = {"b": 0.002, "g": 0.1}

compiled = compile_spec(spec)
network = from_compiled_rhs(compiled, params)
```

Each reaction becomes one channel per source cell: `network.stoichiometry` is
`(n_state, n_channels)`, and `network.channel_names` labels the columns
(`infect[age=0]`, ...). The flat state follows `compiled.template_shapes`:
each template's cells in C order, templates in declaration order.

## Check the network against the RHS

The network's mean drift, `stoichiometry @ propensity`, must equal the
deterministic right-hand side. Check it once for each new model:

```python
rng = np.random.default_rng(0)
y = rng.uniform(1.0, 100.0, network.n_state)
np.testing.assert_allclose(
    network.mean_drift(0.0, y),
    compiled.eval_fn(0.0, y, **params),
    rtol=1e-12,
)
```

A mismatch means some dynamics have no reaction. `from_compiled_rhs` already
refuses a RHS whose `reaction_gaps` lists transitions without a reaction
artifact (an unnamed transition, for example). Pass `allow_gaps=True` only
when you intend to simulate the reactions alone.

## Simulate

```python
from op_engine import DirectSSASolver, ModelCore, NumpySSASampler

times = np.linspace(0.0, 30.0, 7)
core = ModelCore(network.n_state, 1, times)
core.set_initial_state(np.asarray([[95.0], [90.0], [5.0], [10.0], [0.0], [0.0]]))
DirectSSASolver(core, network.stoichiometry).run(
    network.propensity, NumpySSASampler(seed=1)
)
trajectory = core.state_array[:, :, 0]  # (time, state)
```

`TauLeapingSolver(core, network.stoichiometry)` runs the same network with a
`NumpyPoissonSampler`.

## Adaptive tau-leaping

`AdaptiveTauLeapingSolver` also needs each channel's molecular reactants, which
`network.reactant_stoichiometry` holds. They must be complete:
`network.reactants_complete` is false, and `network.incomplete_reactions`
names the culprits, when a reaction might read a state op_system has not
declared as a reactant. In op_system 0.7.0 and later:

- a reaction whose rate reads no state (`recover` above) is complete as is;
- `reactants: auto` infers the reactants of a rate that is a single product
  of states (`infect` above: `S` consumed, `I` catalytic);
- an explicit `reactants:` list is always authoritative.

```python
from op_engine import AdaptiveTauLeapingSolver, NumpyPoissonSampler

assert network.reactants_complete
core = ModelCore(network.n_state, 1, times)
core.set_initial_state(np.asarray([[95.0], [90.0], [5.0], [10.0], [0.0], [0.0]]))
AdaptiveTauLeapingSolver(
    core, network.stoichiometry, network.reactant_stoichiometry
).run(network.propensity, NumpyPoissonSampler(seed=2), NumpySSASampler(seed=3))
```

## Partitions and lower-level use

`from_compiled_rhs(compiled, params, reaction_names=[...])` compiles only the
named reactions, for a hybrid jump partition. `compile_reaction_network`
takes the artifacts and layout directly, for producers other than a
`CompiledRhs`:

```python
from op_engine import compile_reaction_network

network = compile_reaction_network(
    compiled.reactions,
    template_shapes=compiled.template_shapes,
    axis_sizes={"age": 2},
    params=params,
)
```

The flepimop2 provider (`flepimop2-op-engine`) builds its stochastic runs
through the same function.
