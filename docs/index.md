# Welcome to `OP Engine`

The next generation vectorized compartmental model engine for [`flepimop2`](https://github.com/ACCIDDA/flepimop2).

Start with [Getting Started](guides/getting-started.md), then use
[Choosing and configuring a solver](guides/solver-methods.md) to select an
explicit, IMEX, or linearly implicit method. The
[biogeochemical network tutorial](guides/biogeochemical-network.md) shows
stage-dependent operator splitting in a complete example.

`op_engine` numerical methods use the array namespace supplied by model state.
The [backend guide](guides/backends.md) explains portability, JAX
differentiability, sparse acceleration, and the external-provider boundary for
specialized adaptive solvers.
