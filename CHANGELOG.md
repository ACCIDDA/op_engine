# Changelog

Notable user-facing changes are recorded here. Versions follow semantic
versioning while the project remains pre-1.0.

## [Unreleased]

### Added

- A machine-readable solver benchmark now separates provider construction,
  adaptive discovery, JAX trace/lower/compile, first and warm execution, and
  value-and-gradient phases while reporting analytic accuracy, work counts,
  memory, revision, and environment metadata. SciPy and optional Diffrax
  references use explicit solver, controller, and adjoint settings (#145).
- Numerical namespace discovery now uses `array-api-compat`, including native
  PyTorch tensors that do not implement `__array_namespace__`. A public
  `array_namespace` helper and optional Torch contract test preserve JAX
  JIT/grad behavior while extending eager fixed-step autograd support (#144).
- Explicit fixed-step runs can bound their internal integration step with
  `fixed_max_step` independently of requested output times, including compact
  differentiable JAX provider trajectories (#130).
- The flepimop2 provider exposes SDIRK2 with explicit full-RHS Jacobian and
  nonlinear-solver options, plus adaptive initial-step and safety limits. A
  method-surface parity test now requires explicit provider decisions for new
  core methods, and frozen schedules tolerate representational rounding when
  replayed on a lower-precision backend (#131).
- Compact JAX fixed-step provider trajectories use a single `jax.lax.scan`
  while the core retains backend-neutral functional step kernels (#123).
- The flepimop2 provider exposes eager adaptive-schedule discovery and
  validated frozen-mesh replay for conditional JAX JIT and differentiation
  workflows (#127).

### Fixed

- Explicit flepimop2 provider methods now add typed op_system axis-kernel,
  advection, and diffusion drift at every Runge--Kutta stage. Flat, PyTree,
  and block layouts share the same namespace-preserving operator path instead
  of silently omitting operators outside IMEX execution (#140).
- Typed advection and transport descriptors now honor op_system's explicit
  `increasing`/`decreasing` direction contract in both IMEX and explicit
  provider paths while preserving JAX differentiation through dynamic
  coefficients (#142).

## [0.2.0] - 2026-09-26

### Added

- Array-API-native state, explicit Runge--Kutta, dense implicit, and IMEX
  paths preserve the namespace selected by the initial state. Portable
  fixed-step methods support JAX tracing and differentiation without adding a
  backend-specific solver to the core method surface (#75, #76, #85, #87,
  #89, #90).
- Reusable explicit Runge--Kutta tableaus through Dormand--Prince 5(4),
  adaptive step control with differentiable accepted-schedule replay, and
  validated public run configuration (#95, #100, #108).
- A portable nonlinear-solver contract plus corrected ROS2, reusable
  multistep BDF2, SDIRK2 diagnostics, and higher-order paired IMEX
  Runge--Kutta integration (#113, #114, #115, #117, #118, #119).
- Backend-neutral fixed tau-leaping, exact direct SSA, and bounded adaptive
  tau-leaping with injected random samplers (#102, #120, #121).
- Portable advection and typed diffusion/advection operator compilation for
  flepimop2, including dynamic Array-API parameters (#81, #91, #92, #93).
- The flepimop2 provider consumes typed op_system reaction artifacts and can
  execute deterministic, stochastic, or hybrid reaction networks without
  parsing raw model configuration (#122).

### Changed

- Provider development and integration coverage now targets the op_system
  0.4 typed operator/reaction contract.
- `flepimop2-op-engine` declares its actual `flepimop2>=0.3.0` compatibility
  floor. Clean-wheel release validation installs the lowest published direct
  dependencies instead of installing flepimop2 from its development branch.
- The documented flepimop2 installation uses the separately distributed
  `flepimop2-op-engine` package. Its `op-system` extra installs the matching
  system provider integration.

### Fixed

- Explicit Euler once again performs one Euler update per requested output
  interval (#79).
- Solver-method documentation now matches the implemented biogeochemical
  splitting order and backend boundaries (#98).

## [0.1.1] - 2026-04-27

- Initial PyPI release of `op-engine` and `flepimop2-op-engine`.
