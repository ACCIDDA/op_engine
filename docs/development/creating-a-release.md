# Creating A Release

This guide covers releases of the independently usable `op_engine` core. The
repository also contains the `flepimop2-op_engine` provider, but the core release
workflow does not build or publish it. A provider release requires separate
coordination with FlepiMoP2's maintainers and a deliberately separate workflow.

## Prerequisites

- You have write access to the [`ACCIDDA/op_engine`](https://github.com/ACCIDDA/op_engine) repository.
- The core version has been updated in `pyproject.toml`.
- Your local environment is synced with a supported complete Python version.
- For a core release, `CITATION.cff` version, release date and artifact URL
  match the core release; attribution and the MIT grant have been reviewed.
- You have [GitHub's `gh` CLI](https://cli.github.com/) installed and authenticated if you plan to dispatch workflows from the command line.

## 1. Confirm The Core Version

Production releases use a canonical `X.Y.Z` version or an explicit
`X.Y.ZaN`, `X.Y.ZbN`, or `X.Y.ZrcN` version in:

- `pyproject.toml`

The provider's source version and dependency declaration are intentionally not
part of this release unit. Do not use this workflow to publish the provider.

## 2. Run The Local Release Preflight

Use the local pre-release target before dispatching the release workflow:

```shell
just release-validate-core
```

That command validates core citation and release metadata, then runs the core
clean-room build and install tests.

The release workflow runs the same validation again on GitHub Actions before publishing.

## 3. Run The Release Workflow

Releases are created through the manual GitHub Actions workflow in `.github/workflows/release.yaml`.

If you are testing the workflow before merging, add `--ref <branch-name>` to the `gh workflow run` command. Without `--ref`, GitHub dispatches the workflow definition from the repository's default branch.

### Dry Run

Use this to validate and build the core, then upload its release artifacts
without publishing anything:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_engine \
  --ref <branch-name> \
  --field publish-target=none \
  --field create-github-release=false \
  --field deploy-docs=false
```

The publish step becomes a no-op when `publish-target=none`.

### TestPyPI

Use this to publish the core to TestPyPI without creating a GitHub release or
deploying docs:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_engine \
  --ref <branch-name> \
  --field publish-target=testpypi \
  --field create-github-release=false \
  --field deploy-docs=false
```

When testing from a branch, keep `create-github-release=false` and
`deploy-docs=false`. Release documentation is built only from the exact
`v<version>` tag.

### PyPI

Use this to publish the core, create its GitHub release, and deploy versioned
documentation:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_engine \
  --field publish-target=pypi \
  --field create-github-release=true \
  --field deploy-docs=true
```

The workflow permits PyPI publication, GitHub release creation, and release-doc
deployment only from the current default-branch head. When
`create-github-release=true`, it uses `v<version>` and GitHub's generated release
notes. A zero major version is not a prerelease: only explicit alpha, beta, and
release-candidate versions receive GitHub's prerelease flag.

## 4. Documentation Deployment

The release workflow calls `.github/workflows/gh-pages.yaml` as a reusable workflow when `deploy-docs=true`.

That workflow supports two deployment modes:

- `push`: the exact `main` commit is deployed only to mutable `dev`
- `release`: the exact tag is deployed to its full immutable version; final
  releases move `latest`, and alpha, beta, or release-candidate versions move
  `pre`

The workflow refuses to overwrite an existing exact version. A manual release
deployment must provide the corresponding `v<version>` tag in `release-ref`.

## 5. Trusted Publishing Setup

The publish jobs use PyPI Trusted Publishing rather than a stored API token.

For `publish-target=testpypi`, configure the TestPyPI trusted publisher entry for:

- Owner: `ACCIDDA`
- Repository: `op_engine`
- Workflow file: `release.yaml`

For `publish-target=pypi`, configure the PyPI trusted publisher entry for the same repository and workflow file.

The publisher entry applies only to the `op-engine` distribution. Publishing
`flepimop2-op_engine` requires a separate reviewed workflow and its own trusted
publisher configuration.

## 6. Package Metadata

Published package metadata must use registry-compatible dependencies.

Required CPU qualification is a separate release gate: run `just qualify` and
retain the NumPy/JAX/CPU Torch logs alongside baseline checks. Document GPU
execution as unqualified unless explicit hardware checks have executed without
skips. See the [capability matrix](../guides/backend-qualification.md).

The [citation/archive guide](citation-and-archiving.md) records the ACCIDDA
Zenodo follow-up. Complete readiness evidence before freezing a software-paper
release, then record its verified version DOI after the deposit resolves.

After the archive ingests a new release, verify its tag, version, creators,
licenses, source archive, exact DOI, and concept DOI. Submit the published
record to the ACCIDDA community for curator review. Account linking, repository
enablement, community submission, and acceptance remain manual operations.
