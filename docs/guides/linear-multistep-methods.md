# Linear-multistep methods

`op_engine` represents constant-coefficient linear-multistep methods with the
convention

\[
  \sum_{j=0}^{k} \alpha_j y_{n+1-j}
  = h \sum_{j=0}^{k} \beta_j f_{n+1-j}.
\]

The representation stores the complete `alpha` and `beta` vectors, formal
order, state-history requirement, startup sequence, and uniform-step
restriction together. The executable kernel currently supports the backward
differentiation family, where only `beta[0]` is nonzero. This boundary is
intentional: an Adams-family method would also require derivative history and
must not silently reuse a state-only history contract.

## Current BDF family

| Formula | `alpha` | `beta` | Order | Older state snapshots | Startup | Step restriction | Public method |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| BDF1 | `(1, -1)` | `(1, 0)` | 1 | 0 | None | Variable step is valid | Startup only |
| BDF2 | `(3/2, -2, 1/2)` | `(1, 0, 0)` | 2 | 1 | BDF1 | Uniform step | `bdf2` |
| BDF3 | `(11/6, -3, 3/2, -1/3)` | `(1, 0, 0, 0)` | 3 | 2 | BDF1, then BDF2 | Uniform step | No |

The public `bdf2` result is unchanged by this representation. Its first step
uses the BDF1 coefficients. Every later step uses BDF2 and one stored older
state. `CoreSolver` continues to reject adaptive BDF2 and nonuniform output
spacing because a constant-coefficient formula is not a variable-step BDF
implementation.

As with the other methods historically called implicit in `CoreSolver`, the
current BDF kernel takes one Jacobian and linearizes the residual at the
current state. It is a linearly implicit BDF approximation, not a converged
nonlinear BDF solve.

## History and restart contract

Multistep history is local to one `run()` call and stores native Array-API
snapshots newest first. NumPy and JAX therefore use the same lifecycle:

- Push the pre-step state only after the proposed step is accepted.
- Never modify history for a rejected attempt.
- Clear history after a discontinuity, event reset, or external state
  replacement, then repeat the declared startup sequence.
- A new `run()` always starts with empty history. Reusing a `CoreSolver` after
  `ModelCore.set_initial_state()` cannot consume a state from the prior run.

The current public BDF2 mode cannot reject a step because it is fixed-step.
The accept-only rule is part of the reusable history boundary so a future
adaptive implementation does not have to change its meaning.

Coefficients contain no backend objects. State snapshots remain in the
state's namespace, dense linear solves use that namespace's `linalg.solve`,
and the optional SciPy sparse route remains NumPy-only. BDF2 is covered by
NumPy/JAX recurrence parity and JAX `jit`/`grad` tests.

## BDF3 decision: no-go for a public method

BDF3 is retained as a validated private coefficient table for evaluation, but
is not a `CoreSolver` method. The decision is **no-go** until the variable-step
and lifecycle work below exists.

The stability loss is material. The
[SUNDIALS BDF notes](https://sundials.readthedocs.io/en/latest/cvodes/Mathematics_link.html#ivp-solution)
state that BDF orders one and two are A-stable, while orders three through five
are not. A NASA stability study reports the familiar BDF3
\(A(86.03^\circ)\) sector rather than the full left half-plane
([NASA/TM-2010-216189](https://ntrs.nasa.gov/api/citations/20100002944/downloads/20100002944.pdf)).

The repository test uses the weakly damped oscillatory mode
\(z=h\lambda=-0.001+i\), obtainable from \(\lambda=-1+1000i\) and
\(h=0.001\). A second mode at `-10000` makes the system stiff without changing
this recurrence calculation. The largest BDF2 recurrence root is about
`0.9328`; BDF3's is about `1.0429`. Thus BDF3 grows this mode while both the
exact solution and BDF2 damp it. BDF2 is strongly dissipative here, so the
benchmark is a stability discriminator rather than an accuracy endorsement.

At a mature uniform step, BDF3 would still require one Jacobian evaluation and
one linear solve, but it needs two older state snapshots instead of one and two
startup steps instead of one. More importantly, promoting it now would provide
no error estimator, no variable-step coefficient generation, and no defined
history transition after rejection. Its nominal order is not enough to make
that incomplete interface safe.

Reconsider a public BDF3 (preferably a variable-order BDF family) only after:

1. variable-step coefficients and step-ratio validation are implemented;
2. a local-error estimator and adaptive accept/reject controller exist;
3. rejected steps leave state and history unchanged under tests;
4. events and external state modification have an explicit restart hook; and
5. stiff dissipative and weakly damped oscillatory workload benchmarks show a
   useful accuracy-per-solve region relative to BDF2, ROS2, and SDIRK2.
