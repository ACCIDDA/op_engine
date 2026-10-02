# Bounded thinning SSA

`ThinningSSASolver` provides an exact waiting-time method for smoothly varying
propensities with user-certified bounds. This design addresses
[#173](https://github.com/ACCIDDA/op_engine/issues/173), including the flepimop2
provider's explicit `thinning-ssa` method.

## Bound contract

`ThinningSSASolver.run(propensity_func, thinning_sampler, *, rate_bound, config=None)`
uses the same stoichiometry, reaction axis, and batched propensity contract as
direct SSA. It permits arbitrary time dependence between events when a valid
upper bound is supplied.

`rate_bound` is either a finite non-negative real constant, or a
`RateBoundFunction(t, state, limit)` returning `TotalRateBound(rate, valid_until)`.
The bound covers the sum of **all** reaction and batch-cell propensities with
the supplied state held fixed, over `[t, valid_until)`. The callback chooses a
finite endpoint satisfying `t < valid_until <= limit`. `limit` is the earlier
of the next declared forcing change and the solve's final time; observations
do not shorten it. A constant bound applies to every state reached during the
solve, until the next forcing change or final time.

After an accepted event, the state changes and the callback is queried again.
A rejected candidate leaves the state and current bound unchanged. At expiry,
the solver discards a tied or later candidate, advances without firing, and
requests a new bound. The same rule applies at forcing changes, with
right-continuous propensities. No candidate fires exactly at an interval's
right endpoint, including the solve's final endpoint.

A zero bound declares the entire interval inactive. The solver checks rates
at the interval's start, advances to its endpoint without sampling, and can
resume with a positive bound. Zero instantaneous propensity with a positive
bound still produces candidates, allowing smooth rates to become active.
The caller must prove the bound over the whole interval. Checks at interval
starts and candidate times detect encountered violations but cannot certify
unsampled times. A violation fails visibly rather than clipping acceptance.
Bounds and cumulative rates use the state's floating dtype. Bounds that
overflow or become zero on conversion fail; tight bounds should allow for
floating-point rounding when summing many channels. Callback inputs must be
treated as read-only, and callbacks must not depend on observation storage.

## Sampling contract

`ThinningSampler(bound_rate, draw_index)` returns
`ThinningSample(waiting_time, uniform)`, both scalar arrays in the state array
namespace. The wait is finite and strictly positive, sampled exponentially at
`bound_rate`. The independent uniform lies in `[0, 1)`. Draw indices start at
zero and increase globally, including rejected and expired candidates.
`NumpyThinningSampler(seed)` supplies the NumPy implementation; other backends
inject their own sampler.

At a candidate time, the solver evaluates the actual channel rates. If
`uniform * bound_rate` is at least the actual total rate, it rejects the
candidate. Otherwise it selects the flattened channel/batch event whose
cumulative propensity contains that threshold. This uses one uniform for
acceptance and channel selection: each channel has probability
`channel_rate / bound_rate`, and the remaining probability rejects.

Candidates beyond an observation are retained with their sampled uniform;
their propensities are evaluated only when the solver reaches them. Inserting
observations with the same solve endpoints therefore preserves the seeded path
and callback/sampler history. `ThinningSSAConfig.max_candidates` bounds all
candidate draws during one run, including rejection and expiry. Invalid
sampler values, non-advancing clocks, invalid bounds, and impossible population
updates fail explicitly.

The construction follows
[Lewis and Shedler's thinning method](https://doi.org/10.1002/nav.3800260304).
The bound and expiry contract extends it to state-dependent reaction networks;
exactness remains conditional on the caller's bound and random sampling laws.
Existing direct SSA and tau-leaping behavior and samplers are unchanged.

## Example

For an independent birth channel in each batch cell with rate `2 * time`, the
integrated intensity over `[0, 1]` is one per cell. The total upper bound must
include every batch cell:

```python
import numpy as np

from op_engine import ModelCore, NumpyThinningSampler, ThinningSSASolver

core = ModelCore(1, 100, np.linspace(0, 1, 11))
core.set_initial_state(np.zeros((1, 100)))


def smooth_birth(time, state):
    return np.full(state.shape, 2 * time, dtype=state.dtype)


ThinningSSASolver(core, np.asarray([[1]])).run(
    smooth_birth, NumpyThinningSampler(seed=173), rate_bound=200,
)
```

A callback can instead return `TotalRateBound(2 * limit * state.shape[1], limit)`
for this example. For consuming reactions, a bound that depends on the current
state can be refreshed after each accepted event. A loose bound remains valid
but generates more rejected candidates.

The flepimop2 provider selects this method through
`mode: stochastic` and `stochastic_method: thinning-ssa`. Supply a constant
`thinning_rate_bound` in configuration, or leave it unset and pass a scalar or
`RateBoundFunction` through `run(..., rate_bound=...)`. Both interfaces cannot
be supplied together. `thinning_max_candidates` sets the per-run candidate
guard. NumPy has seeded sampling through `random_seed`; other backends inject
`thinning_sampler=`. Hybrid thinning is unsupported. See the
[provider examples](https://github.com/ACCIDDA/op_engine/tree/main/flepimop2-op_engine#bounded-thinning-ssa)
for configuration and callback usage. Existing direct SSA and tau-leaping
defaults remain unchanged.
