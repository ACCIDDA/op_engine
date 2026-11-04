# Changelog

Provider-specific changes are recorded here. Core changes are recorded in the
repository-root changelog and use an independent release lifecycle.

## [Unreleased]

The development source is labelled `0.9.0.dev0`; it is not part of the core
0.9.0 release and has not been published.

### Changed

- Align the development provider's exact core dependency with `op-engine==0.9.1`
  and qualify compiler state-map integration at immutable compiler revision
  `f5905e4fdb8260bef08f500ca9408609a9c1a26d`. Package versions and the published
  provider's older dependency contract are unchanged.

### Added

- Consume compiler state-map records in ordinary flat eager explicit fixed,
  adaptive and replayed execution, with explicit post-event output semantics.
  Require adaptive map identity and reject unsupported prepared, structured,
  stochastic/hybrid and scan/compact paths before executing an ordinary RHS.

- Add opt-in independent adaptive schedules for compiler-proven JAX blocks,
  with per-block context and replay diagnostics, equal-length batching without
  padded ODE steps, a batch-size limit and serial replay. Preserve shared-mesh
  defaults and qualify frozen-mesh parameter/initial-state gradients, proof
  refusal and bounded CPU discovery/compile/replay/memory measurements.

### Fixed

- Honor explicit provider fixed-loop controls on ordinary and prepared flat,
  PyTree and block solves, retaining omitted-setting defaults. Separate general
  scan rematerialization from existing DOPRI5 step/chunk checkpoint policies,
  reject conflicting or unsupported selections, and preserve the unbounded
  provider step limit through configuration JSON round-trips.

## [0.8.0] - 2026-10-19

- Last synchronized release of `flepimop2-op_engine` and `op-engine`. See the
  repository-root changelog and the exact `v0.8.0` tag for that release.
