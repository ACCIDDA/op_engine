# Bounded thinning SSA

This implementation proposal addresses [#173](https://github.com/ACCIDDA/op_engine/issues/173)
in two increments: the core numerical method and its validation, followed by
flepimop2 provider wiring after the core PR is reviewed and merged.

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

## Delivery and validation

The first PR implements the core method, NumPy sampling, documentation, and
NumPy/JAX conformance tests. It checks accepted and rejected candidates,
state-dependent bounds, expiry and forcing ties, dormant recovery, observation
invariance, malformed contracts, analytic smooth-birth mean and variance, and
constant-rate agreement with direct SSA. A generic large-population transfer
network checks stochastic mean/drift against its deterministic solution.

The next PR exposes an explicit `thinning-ssa` provider method and its bound
and sampling inputs, retaining existing defaults. This keeps each PR near the
requested 1,000-line review limit. Issue #173 remains open until both increments
are complete.
