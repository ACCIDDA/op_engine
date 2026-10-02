# Changelog

Notable user-facing changes are recorded here. Versions follow semantic
versioning while the project remains pre-1.0.

## [Unreleased]

### Added

- Pure stochastic validation and execution reject models with dynamics that
  have no reaction artifact. These are transitions listed in the producer's
  `reaction_gaps` coverage records (op_system #249), and typed operators.
  Each uncovered transition is named by origin, selectors, and reason, so it
  can no longer be dropped silently. Hybrid mode and producers without
  coverage records are unchanged (#182).
- Stochastic execution supports reactions between templates with different
  axes, including axis-less states: scalar S→I→R models, axis-less donors
  depositing into a pinned cell, and templated sources collapsing into an
  axis-less state. Targets are indexed with op_system's `to_full_axes`,
  falling back to `full_axes` for older producers (#184).

- Stochastic execution consumes axis-wide `coord_shift` reactions from
  op_system: each source bin of an `offsets` axis becomes a channel that moves
  one unit by the published step, and off-axis destinations only deplete the
  donor. Direct SSA reproduces Poisson bin occupancy and Erlang exit times for
  pure aging chains. Artifacts without `offsets` compile unchanged (#180).

- Direct SSA accepts explicit forcing breakpoints for exact piecewise-constant
  rates, including dormant zero-rate intervals, while retaining pending events
  across observation times. The provider exposes this for pure direct SSA (#171).
- Prepared Flepimop2 execution plans now support explicit PyTree and JAX block
  layouts for fixed steps and frozen adaptive replay, including structured typed
  operators, while preserving dynamic-value cache contracts (#168).

### Fixed

- `PreparedExecution` can be passed directly to JAX transformations, including
  mixed NumPy/JAX samples whose effective namespaces and dtypes are canonicalized
  by JAX during tracing (#168).

## [0.3.0] - 2026-09-27

### Added

- A trait-structured chemostat PDE example combines a curvature-adapted
  non-uniform finite-volume grid, no-flux implicit diffusion, nonlinear
  resource coupling, structured trait moments, an RK45 reference, and a
  documented three-panel diagnostic (#47).
- The Flepimop provider compiles typed `jump_integral` descriptors into
  conservative Array-API operators for implicit/IMEX and structured explicit
  execution. It preserves dynamic rates and kernel matrices under JAX,
  applies declared direction and continuous target quadrature, and reuses
  selector/multi-axis lifting with real op_system integration coverage (#94).
- A finite-element boundary design note and tested 1D P1 diffusion proof of
  concept show how externally assembled mass/stiffness systems use the existing
  stage-operator contract, without adding mesh or element abstractions to core
  (#107).
- Curvature-weighted 1D spatial-grid generators accept vectorized profile
  callables or sampled data, with optional curvature smoothing and a feasible
  minimum-spacing constraint. They return explicit NumPy geometry for use by
  any runtime array backend (#46).
- Dense Array-API and sparse SciPy diffusion builders now accept monotone
  non-uniform cell-center grids with conservative no-flux boundaries. The
  flepimop2 provider uses the same geometry for flat, PyTree, and block
  explicit or IMEX execution while retaining JAX differentiation (#45).
- Explicit adaptive JAX discovery now keeps attempted-step acceptance and
  controller state in a bounded device loop, including conservative shared-mesh
  discovery for block layouts. Compact frozen-mesh replay reports array-valued
  local-error freshness diagnostics and can reject materially stale schedules
  at an outer orchestration boundary without compromising JIT or gradients;
  the solver matrix now benchmarks discovery in each row's actual backend
  (#148).
- Fixed flat-state JAX DOPRI5 now computes only its high-order solution, keeps
  FSAL reuse, and carries a requested-output buffer instead of materializing
  every hidden internal state. Optional `fixed_checkpoint: step` and `chunk`
  policies control reverse-mode rematerialization; RK4 retains its smaller
  compile-sensitive scan. The solver matrix records the resulting forward and
  gradient IR, compile, memory, runtime, and accuracy changes (#147).
- The flepimop2 provider can prepare and cache stable flat explicit execution
  plans for fixed stepping and frozen adaptive replay. Prepared callables keep
  array values dynamic for NumPy or caller-owned JAX transforms while reusing
  system binding, packing contracts, step grids, and execution structure;
  explicit cache invalidation and matched staging benchmarks are included (#146).
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

- Sparse implicit-solver cache entries retain and verify their exact operator
  objects, preventing stale factorizations when Python recycles an object ID
  for a different same-shape matrix (#164).
- IMEX Heun--trapezoidal now assembles its endpoint predictor and corrector as
  additive solves, so nonlinear and non-commuting explicit/implicit splits
  retain second-order convergence across NumPy and Array-API execution (#162).
- The public `Array` protocol now matches `array-api-compat` namespace
  discovery and accepts native tensors such as `torch.Tensor` without requiring
  a direct `__array_namespace__` method (#155).
- `flepimop2-op-engine` now types public trajectories with Flepimop2's
  backend-neutral `Array` protocol and returns the native NumPy or JAX result
  without a NumPy-only cast (#154).
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
