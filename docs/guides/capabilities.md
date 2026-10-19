# Capabilities and executable examples

`op_engine` is a framework-independent numerical core for compartmental and
spatial models. NumPy is the core dependency. The separate GPL
`flepimop2-op_engine` package adapts it to flepimop2 and shares its version.

## Choose a public interface

| Task | Implemented interface | Guide and boundary |
| --- | --- | --- |
| Store state and output times | `ModelCore` | [Getting started](getting-started.md); state is `(species, batch)` and stored history adds a leading time dimension. |
| Solve deterministic dynamics | `CoreSolver.run` | [Solver methods](solver-methods.md): explicit, IMEX and linearly implicit methods; select an accuracy policy for the problem. |
| Solve nonlinear stages | `NonlinearProblem`, `DenseNewtonSolver`, `NewtonConfig` | [Nonlinear solvers](nonlinear-solvers.md); inspect convergence evidence. |
| Integrate stiff or spatial models | matrix/operator utilities, `GridGeometry`, diffusion/advection builders | [Chemostat](chemostat-pde.md), [biogeochemistry](biogeochemical-network.md) and [linear multistep](linear-multistep-methods.md). |
| Simulate a reaction process | `DirectSSASolver`, `TauLeapingSolver`, `AdaptiveTauLeapingSolver` | [Compiled reactions](op-system-reactions.md); direct SSA has a superposed clock, tau-leaping is approximate. |
| Simulate independent replicate banks | `BatchedDirectSSASolver`, `BatchedSSAConfig`, `from_compiled_replicates` | [Independent SSA](replicate-ssa.md); eager NumPy CPU, certified local channels and one clock per replicate. |
| Handle smooth time-varying hazards | `ThinningSSASolver`, `TotalRateBound` | [Bounded thinning](bounded-thinning.md); exactness requires a valid bound throughout its declared interval. |
| Record accepted channel firings | `FiringRecorder` | Ordinary, thinning, tau and independent-bank solvers; counts do not consume randomness. |
| Replay stochastic experiments | `RandomStream`, solver `rng` and stable replicate IDs | [Random streams](random-streams.md); distinguish reproducibility within a supported runtime from cross-version bitwise identity. |
| Evaluate observations and summaries | `ObservationSpec`, `observe_model`, `observe_solution`, `reduce_observations` | [Named observations](observations.md); explicit measures, interval/point semantics and storage choices. |
| Inspect equilibria and conservation | `steady_state`, `conserved_quantities`, `linear_invariants`, `sink_states` | [Steady states](steady-states.md); convergence and conditional structural identities are separate checks. |
| Reuse an adaptive ODE schedule | `AdaptiveScheduleProvenance`, replay controls | [Replay provenance](replay-provenance.md); validate accuracy away from the producing parameters. |

The [backend matrix](backend-qualification.md) states which NumPy/JAX/CPU Torch
operations and transforms are qualified. Native adaptive solvers such as diffrax
live behind optional adapters/providers. Method availability does not imply all
methods, stochastic gradients or accelerator paths have the same guarantees.

## Run an independent-clock example

```bash
uv sync --dev
just example-replicates
just doctest
```

[`examples/replicate_ssa.py`](https://github.com/ACCIDDA/op_engine/blob/main/examples/replicate_ssa.py)
uses only the NumPy core. Two S-to-I populations have independent clocks and a
shared declared forcing jump at time 1. The script checks total population,
accepted firings and exact path replay under bank reordering and singleton runs.
Stable IDs define the streams; bank position does not.

For specification-driven wiring, use the complete example in
[independent SSA banks](replicate-ssa.md), requiring `op-system>=0.8.0`.
A compiler independence certificate is required; a plausible array shape alone
cannot prove that two processes are independent. Held rates can use breakpoint
SSA. Continuously changing rates require a suitable method such as thinning.

## What runs automatically

Plain `pytest` and `just pytest-core` execute library doctests. `just doctest`
also exercises the actual README/getting-started code and the standalone bank.
Examples check analytic decay, conserved totals and reproducible paths rather
than treating successful execution as numerical evidence. `just ci` includes
these and the core/provider tests, lint, strict typing and citation checks.

Public docstrings cover state storage, deterministic integration, firing counts,
independent clocks and steady-state analysis. Larger spatial tutorials remain
covered by their dedicated regression tests. When editing an API, keep its
complete example executable and assert an independent reference or invariant.
Before a PR, run `just ci`, `just docs`, `just qualify` and `just build-all` as
specified in [Contributing](../development/contributing.md).
