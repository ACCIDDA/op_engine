# Finite-element assembly boundary

Finite elements are a spatial discretization and assembly system, not a new
`CoreSolver` time-method name. The current decision is **not to add a mesh,
element, or quadrature framework to `op_engine` core and not to create an
`op_engine-fem` provider yet**. Externally assembled linear finite-element
systems can already use the stage-operator interface.

## Supported semi-discrete form

Consider the linear system

\[
  M \dot y = K y + g(t, y).
\]

An operator factory can map its implicit linear part into the existing solve

\[
  L y_{n+1} = R x.
\]

For example, the factory for a trapezoidal stage of size \(h\) returns

\[
  L = M - \tfrac{h}{2}K,
  \qquad
  R = M + \tfrac{h}{2}K.
\]

An implicit-Euler stage instead returns \(L=M-hK\) and \(R=M\). This supports
fixed or variable stage sizes and permits the existing IMEX Euler,
Heun--trapezoidal, and TR-BDF2 control flow to advance an externally assembled
mass/stiffness pair.

The explicit callable still has derivative units. A finite-element load vector
`g` must therefore be supplied as `solve(M, g)`; the current API does not inject
an unscaled weak-form load directly. Likewise, `imex-ark3` currently consumes
only the left implicit solve at each additive stage, so a non-identity mass
matrix must be resolved to an operator such as `solve(M, K)` before using that
method. Changing either contract should be motivated and designed separately,
not hidden in an FEM adapter.

## Ownership

The assembly package or application owns:

- mesh topology and geometry;
- basis functions, quadrature, and element integration;
- degree-of-freedom maps and constraint elimination;
- essential and natural boundary-condition assembly;
- mass, stiffness, load, and nonlinear residual assembly;
- any matrix reassembly or preconditioner lifecycle.

`op_engine` owns time integration, stage context, operator-axis application,
accepted-step control, and the `L @ y_next = R @ x` solve boundary. This keeps
optional SciPy, PETSc, FEniCSx, and JAX FEM ecosystems from becoming core
dependencies.

## Sparse operators and differentiation

SciPy CSR pairs use the registered sparse adapter. A provider should reuse
operator objects when their values do not change so the factorization cache is
effective. Other array backends can supply dense native arrays and use the
Array-API linear solve. A JAX-native dense assembly remains differentiable
through matrix construction and solve; eager NumPy/SciPy assembly and
factorization do not. An external PETSc or FEniCSx provider would own any
custom differentiation rule for its solver.

Nonlinear forms such as \(M(y,t)\dot y=r(t,y)\), differential-algebraic
constraints, and external Krylov/preconditioner callbacks are outside the
current stage-operator contract. They require a concrete use case before a new
portable solver interface is added.

## Proof of concept

[`examples/fem_diffusion.py`](https://github.com/ACCIDDA/op_engine/blob/main/examples/fem_diffusion.py)
assembles a consistent P1 mass matrix and diffusion stiffness matrix on a 1D
mesh, eliminates homogeneous Dirichlet boundary degrees of freedom, and runs
the result with fixed-step `imex-heun-tr`. Assembly uses SciPy only because the
example is demonstrating the external sparse boundary; it is not a required
FEM representation.

The example intentionally uses fixed steps. For pure implicit diffusion,
Heun--trapezoidal's embedded low/high pair differs only in its explicit term
and therefore cannot estimate the implicit truncation error. Adaptive FEM runs
should currently use a method with a suitable estimator, such as TR-BDF2 step
doubling, or supply and validate a fixed/replayed mesh at a higher layer.

## When to reconsider a provider

Reconsider a separate provider, rather than extending core, when at least one
real unstructured-domain model identifies all of the following:

1. a specific external assembly and linear-solver ecosystem;
2. required boundary-condition and constraint semantics;
3. sparse/device and differentiation expectations;
4. nonlinear and reassembly requirements;
5. a benchmark showing that the existing `(L, R)` boundary is insufficient.

Until then, the proof of concept is the supported design pattern and a general
FEM abstraction would add dependencies and policy without a validated user.
