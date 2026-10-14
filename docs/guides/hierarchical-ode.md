# Hierarchical nonlinear ODE evidence

`benchmarks/hierarchical_ode.py` defines a synthetic generic two-pool model for
structured numerical inference. This is a correctness workload with explicit
reference equations, not a calibrated model or a new inference framework.
The fixture separates group count and local parameter dimension, with state
shape `(2 pools, groups, local cells)`.

## Problem definition and parameter layout

For group `g` and local cell `j`, write first pool `a[g,j]` and second pool
`b[g,j]`. Each pair has unit total. The forward and return flows are:

```text
q[g,j] = exp(theta_gain) * exp(theta_local[g,j])
         * a[g,j] * (a[g,j] + c * mean_group(a[:,j])) / (1 + a[g,j])
r[g,j] = exp(theta_recovery) * b[g,j]
a' = r - q
b' = q - r
```

`c=0` gives independent groups; `c=0.15` gives the coupled variant. Parameters
are ordered as two shared log rates, all group-local log-rate offsets, then all
local initial-condition coefficients. Thus there are
`2 + 2 * groups * local_dimension` fitted values. Initial values are
`a0=0.55+0.1*tanh(theta_initial)` and `b0=1-a0`; positivity and unit totals are
maintained at initialization without stopping differentiation.

Nominal shared gain is 0.4. Recovery is 0.15 in the nonstiff regime and 50 in
the stiff regime, creating a fast relaxation mode. Local rate offsets and IC
coefficients vary deterministically across cells. Stiffness is a parameter
regime rather than a change to the structural model identity.

## Independent trajectories and objective

The reference RHS uses scalar standard-library math and explicit group/cell
loops. `reference_trajectory` passes that independent RHS to SciPy DOP853 with
`rtol=2e-12`, `atol=2e-14`. Native equations are compared directly with the
reference before trajectory comparisons. NumPy replay, JAX built-in scan replay,
and an external JAX scan agree on one frozen nominal mesh.

Observations are first-pool values at noninitial output times, plus the
reproducible residual `0.025*0.2*sin(cell_index+0.5)`. The Gaussian likelihood
uses scale 0.025. Shared log rates have unit-scale priors about their nominal
centers; local log-rate offsets have scale 0.5 about `0.1*theta_gain`, and local
IC coefficients have scale 0.8 about zero. The sum of group likelihood terms
and shared/local priors defines the scalar objective up to constants.

The nonzero residual is deliberate: derivatives include curvature of the
trajectory, rather than only the squared sensitivity at a perfect fit.
The evidence checks all shared/local/IC parameters, with independent-reference
finite differences for gradients and centered differences of native gradients
for Hessians. Forward and reverse mode agree. Step halving checks gradient
stability as well as trajectory agreement.

The Laplace-style diagnostic is `0.5*logdet(H + 25*I)`, where `H` is the
objective Hessian. The fixed diagonal regularization is part of this numerical
diagnostic, not a claim that an unregularized posterior is convex. Tests verify
positive eigenvalues before interpreting the log determinant, require a
nonzero gradient, and compare that gradient with centered log-determinant
differences. Accuracy or positive definiteness outside these fixtures requires
fresh evidence.

## Public step boundary and conditional derivatives

Nominal schedules are discovered eagerly with Dormand–Prince at `rtol=2e-8`,
`atol=2e-10`. Built-in replay uses `replay_loop="scan"` and checkpointing.
The external loop uses only `CoreSolver.fixed_explicit_step`, reuses the FSAL
stage, and checkpoints its scan body. Output selections preserve the original
time grid. Both invoke the same accepted-step formulas; neither differentiates
adaptive acceptance decisions. Follow the [provenance/refresh policy](replay-provenance.md)
when parameters move. The halved comparison mesh is a manually refined mesh,
not a fresh set of controller acceptance decisions.

## Reproduce the small correctness evidence

```shell
uv sync --dev
uv run pytest tests/op_engine/test_hierarchical_ode.py -q
```

The eight cases cover independent/coupled and nonstiff/stiff regimes, direct
RHS agreement, local totals, accurate reference trajectories, NumPy/JAX parity,
built-in/external loops, forward/reverse gradients, Hessians, log-determinant
gradients, and mesh refinement. The NumPy/JAX required qualification jobs reject skipped fixture cases, and
installed core wheels execute the same evidence. Large dimension/mesh timing sweeps and fair
Diffrax comparisons are a separate opt-in measurement layer; no speedup or
memory-scaling claim follows from these correctness cases alone.
