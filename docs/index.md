# Welcome to `OP Engine`

Numerical methods for spatial and trait-structured PDEs,
nonlocal integrals and mixing, compartmental models, ODEs and stochastic
reaction networks. PDE and nonlocal fields use numerical grids and operators.
The separate provider integrates with [`flepimop2`](https://github.com/ACCIDDA/flepimop2).

Start with [Getting Started](guides/getting-started.md), then use
[Choosing and configuring a solver](guides/solver-methods.md) to select a
deterministic or stochastic method and its accuracy contract. The
[biogeochemical network tutorial](guides/biogeochemical-network.md) shows
stage-dependent operator splitting in a complete example.
The [trait-structured chemostat tutorial](guides/chemostat-pde.md) combines a
non-uniform finite-volume grid, implicit diffusion, nonlinear resource
coupling, and structured trait observables.

`op_engine` numerical methods use the array namespace supplied by model state.
The [backend guide](guides/backends.md) explains portability, JAX
differentiability, sparse acceleration, and the external-provider boundary for
specialized adaptive solvers.

The [contributor guide](development/contributing.md) documents setup, required
checks and support. [Citation and archiving](development/citation-and-archiving.md)
explains attribution, the independent core and provider release lifecycles, and
the software-paper archive plan.

See [capabilities and executable examples](guides/capabilities.md) for current interfaces,
optional dependencies, limitations and the checks that run in CI.
