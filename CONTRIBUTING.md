# Contributing

`src/op_engine/` contains the standalone numerical engine. The separately
released `flepimop2-op_engine/` provider adapts it to flepimop2 configuration,
compiled op_system models, parameter binding and prepared execution. Keep
numerical algorithms in the core and framework integration in the provider;
core changes still need provider compatibility checks. Runtime JAX/Torch/CuPy
extras remain optional for users.

## Clean setup

Install Git, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/man/en/), then use a complete Python installation:

```shell
git clone https://github.com/ACCIDDA/op_engine.git
cd op_engine
uv python install 3.12
export UV_MANAGED_PYTHON=true
export UV_PYTHON=3.12
export UV_PYTHON_VERSION="$UV_PYTHON"
uv sync --dev --python "$UV_PYTHON"
just provider-sync
```

The root/provider virtual environments are separate. `provider-sync` clears the
provider's `.venv`; retain personal packages elsewhere. Lockfiles are generated
locally and ignored. `UV_PYTHON` chooses uv's interpreter, while
`UV_PYTHON_VERSION` selects isolated wheel tests. Complete uv-managed Python
includes `ensurepip`, needed by wheel build isolation. Core/provider baseline
CI additionally tests Python 3.11–3.14. Root development dependencies include
JAX for the established numerical tests; this does not make JAX mandatory for
core package users. Docs API generation is an isolated uv script.

## Focused changes and submission checks

Start a branch from current `main` and keep each PR focused on an issue or one
behavior. Prefer small reviewable commits, with private helper names and
specific errors at their source. Use `NamedTuple` for small immutable records;
a frozen dataclass is appropriate for validated/computed configuration. New
behavior tests should assert independently expected equations, references, invariants
or error contracts, rather than reproducing implementation expressions.
Document changed public behavior and migration in the relevant guide and
`CHANGELOG.md` under `Unreleased`. Ordinary fixes retain the shared version.

Run these checks locally **before pushing/opening a PR**:

```shell
just format
just check
just ci
just docs
just qualify
just build-all
```

`format` applies Ruff formatting and `check` applies lint fixes: review the diff.
`ci` checks Ruff without fixes, runs core/provider tests, performs strict typing
in each package's environment, and validates citation metadata. `docs` builds
API reference pages and treats documentation warnings as errors. `build-all`
builds both wheels, validates metadata, then tests installed packages in clean
environments. Use focused checks during iteration, then the full commands
above when the change is ready.

`qualify` is the separate required NumPy/JAX/CPU Torch matrix. It fails for a
missing namespace, no executed cases or a selected skip. Baseline optional
skips and provider success cannot substitute for these core jobs. Targets
refresh extras in the root environment; use `uv run --no-sync` when retaining a
selection intentionally. Backend changes must preserve the claimed shape,
dtype, namespace and transform behavior. Read the
[capability matrix](docs/guides/backend-qualification.md) and
[replay contract](docs/guides/replay-provenance.md).

GPU checks require appropriate hardware and remain explicit opt-in evidence.
Large timing/measurement sweeps are also opt-in, with no performance thresholds
in CI. The separate required comparison job installs the benchmark extra for
tiny correctness cases. Follow [hierarchical ODE evidence](docs/guides/hierarchical-ode.md)
for independent fixtures and derivative checks, and
[hierarchical measurements](docs/guides/hierarchical-measurements.md) for the
comparison commands and protocol before using timings for publication claims.

A PR description should explain the trigger/problem, resulting behavior,
linked issue, checks run, and meaningful limitations. Core and provider checks
should be identifiable. Merge only after every CI job on the current head
passes; another commit requires new CI. Follow applicable repository review
and branch protection rules.

## Bugs and support

Use [GitHub issues](https://github.com/ACCIDDA/op_engine/issues), checking for an
existing report first. Include a minimal synthetic model, expected equations or
reference, method, adaptive/fixed/replay configuration, output grid, state and
operator shapes/dtypes/namespaces, Python/backend/package versions and traceback.
For replay, include producing provenance, the nominal versus changed parameter
regime, and an independent accuracy check. For provider problems, include both
distribution versions plus flepimop2/op_system versions and distinguish the
core step kernel from framework execution. Describe hardware for GPU reports.

## Releases, documentation and citation

`op_engine` and `flepimop2-op_engine` share one version and GitHub tag. The
provider requires the exact corresponding `op-engine==X.Y.Z`, and publication
builds/publishes core before provider. See
[creating a release](docs/development/creating-a-release.md); merging readiness
work is separate from publishing or archiving a release.

Core is MIT-licensed. Provider source grants GPL version 3 or any later version
(`GPL-3.0-or-later`). Preserve the applicable grants and existing attribution.
[CITATION.cff](CITATION.cff) retains authors/order from both package metadata
files; maintainers review attribution when it changes. `just citation` validates
CFF syntax and alignment with versions, authors, grants, dated changelog and
release URL. The CFF initially describes the already published 0.6.0 release;
unreleased main changes do not become part of 0.6.0 without a new release.

Use the [documentation version selector](https://accidda.github.io/op_engine/)
and exact release tag for reproducibility. The docs deployment groups
major/minor releases, so patch-specific behavior should use that exact tag's
source/changelog. [Citation and archiving](docs/development/citation-and-archiving.md)
describes version-specific identifiers and the ACCIDDA Zenodo follow-up. Add a
preferred paper citation only once its authors and bibliographic record exist.
