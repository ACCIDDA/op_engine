# Examples

These executable examples show particular `op_engine` interfaces on small,
bounded problems. They are demonstrations of those interfaces and fixtures,
not general validation of a model, discretization, solver, or scientific
interpretation.

Run commands from the repository root after installing the development
environment with `uv sync --dev`.

## First example

- [`simple_sir.py`](simple_sir.py) introduces `ModelCore`, `CoreSolver.run`,
  stored output times, and fixed versus adaptive Heun stepping for a
  single-location SIR ODE. It reports conservation drift and writes two plots.
  It does not compare the trajectory with an independent solution. See
  [Getting started](../docs/guides/getting-started.md).

## Structured and spatial systems

- [`biogeochemical_network.py`](biogeochemical_network.py) exercises explicit
  and IMEX methods, three operator splits, and stage-time/stage-state operator
  factories for a size-structured phytoplankton--zooplankton network. It
  compares the same finite-dimensional equations with SciPy RK45/BDF and
  writes figures and timing metadata. Those comparisons are diagnostics for
  the declared splits and settings, not a universal performance result or
  validation of the scientific model. See the
  [biogeochemical tutorial](../docs/guides/biogeochemical-network.md).
- [`chemostat_pde.py`](chemostat_pde.py) couples a trait-distributed population
  to a resource on a non-uniform finite-volume grid. It demonstrates an
  implicit no-flux diffusion operator, explicit nonlinear coupling, weighted
  trait summaries, and comparison with SciPy RK45 on the same semi-discrete
  equations. The reference isolates time integration; it is not an exact
  solution of the continuous PDE and does not establish grid convergence. See
  the [chemostat tutorial](../docs/guides/chemostat-pde.md).
- [`fem_diffusion.py`](fem_diffusion.py) assembles a one-dimensional P1 mass and
  stiffness pair outside the library and maps it to the `(L, R)` stage-operator
  contract for fixed-step `imex-heun-tr`. It demonstrates the external sparse
  assembly boundary, not a general finite-element framework, nonlinear form,
  or discretization guarantee. See the
  [finite-element assembly boundary](../docs/development/fem-assembly-boundary.md).

The adjacent `test_*.py` files provide focused regression checks for these
larger examples.

## Stochastic simulation

- [`replicate_ssa.py`](replicate_ssa.py) runs two independent exact-SSA clocks
  with stable replicate identities, a declared forcing jump, and accepted
  firing counts. It checks population conservation and path replay under bank
  reordering and sharding. This NumPy example does not demonstrate stochastic
  gradients, accelerators, tau-leaping, hybrid dynamics, or compiler-certified
  independence. See the [independent SSA guide](../docs/guides/replicate-ssa.md).
- [`illustrate_replicate_ssa.py`](illustrate_replicate_ssa.py) regenerates, or
  checks with `--check`, the documented figure and retained data produced from
  the replicate example. The smooth curves are analytic means; the simulated
  markers are individual paths observed only at requested output times.

## Sensitivities and execution control

- [`forward_sensitivities.py`](forward_sensitivities.py) compares bounded
  forward sensitivity equations with derivatives of the same fixed RK4 solve
  for NumPy, JAX, or CPU Torch. It requires an `op_system` derivative artifact
  and reports advisory fresh-process timings. The fixture does not qualify
  adaptive, implicit, stochastic, event-driven, or accelerator sensitivities,
  and it does not establish a general cost advantage. See
  [Forward sensitivities](../docs/guides/sensitivities.md).
- [`fixed_scan.py`](fixed_scan.py) checks a JAX `lax.scan` fixed-step solve and
  its rate derivative against exponential decay. It demonstrates the fixed
  explicit JAX path only; it provides no adaptive error control. See
  [Fixed explicit solves](../docs/guides/fixed-scan.md).
- [`replay_mesh_refresh.py`](replay_mesh_refresh.py) shows an eager policy for
  checking a replayed adaptive mesh after a parameter change and rediscovering
  it when an independent state-error check fails. Provenance compatibility is
  not an accuracy certificate, and this one fixture does not check other
  objectives or derivative accuracy. See
  [Replay provenance](../docs/guides/replay-provenance.md).

## Diagnostics

- [`output_state_checks.py`](output_state_checks.py) records per-member
  nonnegativity and monotonicity flags at requested outputs without changing
  the numerical path. It does not inspect internal stages or prove accuracy,
  positivity between outputs, or physical validity. See
  [Output-state checks](../docs/guides/output-state-checks.md).

[`illustration_artifacts.py`](illustration_artifacts.py) is a shared helper for
reproducible documentation artifacts rather than a standalone model example.
