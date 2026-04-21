# Creating A Release

This guide covers the `op_engine` release process for this repository's two Python distributions:

- `op_engine`
- `flepimop2-op_engine`

The release workflow treats them as a single release unit. Both package versions must match, and the workflow publishes them sequentially under one GitHub release tag.

## Prerequisites

- You have write access to the [`ACCIDDA/op_engine`](https://github.com/ACCIDDA/op_engine) repository.
- The release version has already been updated everywhere it is declared:
  - `pyproject.toml`
  - `flepimop2-op_engine/pyproject.toml`
- Your local environment is synced with a supported Python version.
- You have [GitHub's `gh` CLI](https://cli.github.com/) installed and authenticated if you plan to dispatch workflows from the command line.

## 1. Confirm The Shared Version

The release workflow validates that all version declarations match before it builds or publishes anything.

Today that means these two files must contain the same semantic version:

- `pyproject.toml`
- `flepimop2-op_engine/pyproject.toml`

If any of them differ, the `validate` job fails immediately.

## 2. Run The Local Release Preflight

Use the local pre-release target before dispatching the release workflow:

```shell
just release-validate
```

That command does two things:

- Validates that the release version matches in:
  - `pyproject.toml`
  - `flepimop2-op_engine/pyproject.toml`
- Runs `just build-all` to execute the clean-room build and install tests for both packages.

The release workflow runs the same validation again on GitHub Actions before publishing.

## 3. Run The Release Workflow

Releases are created through the manual GitHub Actions workflow in `.github/workflows/release.yaml`.

If you are testing the workflow before merging, add `--ref <branch-name>` to the `gh workflow run` command. Without `--ref`, GitHub dispatches the workflow definition from the repository's default branch.

### Dry Run

Use this to validate shared versioning, build both packages, and upload release artifacts without publishing anything:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_engine \
  --ref <branch-name> \
  --field publish-target=none \
  --field create-github-release=false \
  --field deploy-docs=false
```

This runs the `validate`, `publish-core`, and `publish-provider` jobs, but the publish jobs become no-ops when `publish-target=none`.

### TestPyPI

Use this to publish both packages to TestPyPI without creating the GitHub release or deploying docs:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_engine \
  --ref <branch-name> \
  --field publish-target=testpypi \
  --field create-github-release=false \
  --field deploy-docs=false
```

The workflow always publishes in dependency order:

1. `op_engine`
2. `flepimop2-op_engine`

When testing from a branch, keep `create-github-release=false` and `deploy-docs=false`. The reusable docs workflow checks out `main`, so it is not intended for pre-merge branch testing.

### PyPI

Use this to publish both packages, create a GitHub release, and deploy the versioned documentation:

```shell
gh workflow run release.yaml \
  --repo ACCIDDA/op_engine \
  --field publish-target=pypi \
  --field create-github-release=true \
  --field deploy-docs=true
```

When `create-github-release=true`, the workflow creates a single `vX.Y.Z` GitHub release for the shared package version and uses GitHub's generated release notes.

## 4. Documentation Deployment

The release workflow calls `.github/workflows/gh-pages.yaml` as a reusable workflow when `deploy-docs=true`.

That workflow supports two deployment modes:

- `push`: for normal `main` branch documentation updates
- `release`: for a tagged release, which updates the release alias in `mike`

If you want to deploy docs manually without running a release, you can dispatch `gh-pages.yaml` directly and choose the deploy mode in the GitHub Actions UI.

## 5. Trusted Publishing Setup

The publish jobs use PyPI Trusted Publishing rather than a stored API token.

For `publish-target=testpypi`, configure the TestPyPI trusted publisher entry for:

- Owner: `ACCIDDA`
- Repository: `op_engine`
- Workflow file: `release.yaml`

For `publish-target=pypi`, configure the PyPI trusted publisher entry for the same repository and workflow file.

Because this repository publishes two distributions from one workflow, the same trusted publisher setup must be allowed to publish both package names on the selected index.

## 6. Current Packaging Note

The release workflow now handles the mechanics of validating, building, and sequencing both distributions together. Package-index acceptance still depends on the metadata inside each built distribution, so publishing `flepimop2-op_engine` to a public index may still require follow-up packaging changes outside the workflow itself.
