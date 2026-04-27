
# Run all default tasks for local development
default: format check pytest mypy docs

# -------------------------------------------------
# Formatting
# -------------------------------------------------

format:
	uv run ruff format --preview

check:
	uv run ruff check --preview --fix


# -------------------------------------------------
# Provider venv + deps (explicit step)
# -------------------------------------------------

# Create/refresh the provider venv and install all deps (including dev group).
# Run this once before running provider pytest/mypy if you aren't using `ci`.
provider-sync:
	cd flepimop2-op_engine && uv venv --clear
	cd flepimop2-op_engine && uv sync --dev


# -------------------------------------------------
# Tests
# -------------------------------------------------

pytest-core:
	uv run pytest --doctest-modules

# Assumes `flepimop2-op_engine/.venv` already exists (run `just provider-sync` or `just ci` first).
pytest-provider: provider-sync
	#!/usr/bin/env bash
	set -euo pipefail
	CORE_DIST="$(mktemp -d)"
	trap 'rm -rf "${CORE_DIST}"' EXIT
	uv run --with build python -m build --wheel --outdir "${CORE_DIST}"
	cd flepimop2-op_engine
	export PATH="${PWD}/.venv/bin:${PATH}"
	export PIP_FIND_LINKS="${CORE_DIST}"
	.venv/bin/python -m pytest --doctest-modules

pytest: pytest-core pytest-provider


# -------------------------------------------------
# Type checking
# -------------------------------------------------

mypy-core:
	uv run mypy --strict src/op_engine

# Assumes `flepimop2-op_engine/.venv` already exists (run `just provider-sync` or `just ci` first).
mypy-provider: provider-sync
	cd flepimop2-op_engine && .venv/bin/python -m mypy --strict src/flepimop2

mypy:
	just mypy-core
	just mypy-provider


# -------------------------------------------------
# CI aggregate
# -------------------------------------------------

ci:
	uv run ruff format --preview --check
	uv run ruff check --preview --no-fix
	just provider-sync
	just pytest
	just mypy


# -------------------------------------------------
# Build validations
# -------------------------------------------------

build-check-core:
	rm -rf dist
	uv run --with build --with twine python -m build --wheel
	uv run --with twine python -m twine check --strict dist/*

build-check-provider:
	cd flepimop2-op_engine && rm -rf dist
	cd flepimop2-op_engine && uv run --no-project --with build --with twine python -m build --wheel
	cd flepimop2-op_engine && uv run --no-project --with twine python -m twine check --strict dist/*

build-check: build-check-core build-check-provider

build-test-core:
	#!/usr/bin/env bash
	set -euo pipefail
	CLEANROOM="$(mktemp -d)"
	trap 'rm -rf "${CLEANROOM}"' EXIT
	uv export --only-group dev --no-emit-project --format requirements.txt --no-hashes --output-file "${CLEANROOM}/dev-requirements.txt" >/dev/null
	uv run --with build python -m build --wheel --outdir "${CLEANROOM}/dist"
	uv venv --python "${UV_PYTHON_VERSION:-3.12}" "${CLEANROOM}/venv"
	uv pip install --python "${CLEANROOM}/venv/bin/python" "${CLEANROOM}/dist"/*.whl
	uv pip install --python "${CLEANROOM}/venv/bin/python" -r "${CLEANROOM}/dev-requirements.txt"
	cp pyproject.toml "${CLEANROOM}/pyproject.toml"
	cp -R tests "${CLEANROOM}/tests"
	cd "${CLEANROOM}"
	"${CLEANROOM}/venv/bin/pytest" --import-mode=importlib tests --quiet --exitfirst

build-test-provider:
	#!/usr/bin/env bash
	set -euo pipefail
	CLEANROOM="$(mktemp -d)"
	trap 'rm -rf "${CLEANROOM}"' EXIT
	cd flepimop2-op_engine
	uv export --only-group dev --no-emit-project --format requirements.txt --no-hashes --output-file "${CLEANROOM}/dev-requirements.txt" >/dev/null
	uv run --no-project --with build python -m build --wheel --outdir "${CLEANROOM}/provider-dist"
	cd ..
	uv run --with build python -m build --wheel --outdir "${CLEANROOM}/core-dist"
	uv venv --python "${UV_PYTHON_VERSION:-3.12}" "${CLEANROOM}/venv"
	uv pip install --python "${CLEANROOM}/venv/bin/python" "flepimop2 @ git+https://github.com/ACCIDDA/flepimop2.git@main"
	uv pip install --python "${CLEANROOM}/venv/bin/python" "${CLEANROOM}/core-dist"/*.whl
	uv pip install --python "${CLEANROOM}/venv/bin/python" --no-deps "${CLEANROOM}/provider-dist"/*.whl
	uv pip install --python "${CLEANROOM}/venv/bin/python" -r "${CLEANROOM}/dev-requirements.txt"
	cp flepimop2-op_engine/pyproject.toml "${CLEANROOM}/pyproject.toml"
	cp flepimop2-op_engine/README.md "${CLEANROOM}/README.md"
	cp flepimop2-op_engine/LICENSE "${CLEANROOM}/LICENSE"
	cp -R flepimop2-op_engine/src "${CLEANROOM}/src"
	cp -R flepimop2-op_engine/tests "${CLEANROOM}/tests"
	cd "${CLEANROOM}"
	export PATH="${CLEANROOM}/venv/bin:${PATH}"
	export PIP_FIND_LINKS="${CLEANROOM}/core-dist"
	"${CLEANROOM}/venv/bin/pytest" --import-mode=importlib tests --quiet --exitfirst

build-test: build-test-core build-test-provider

build-all-core: build-check-core build-test-core

build-all-provider: build-check-provider build-test-provider

build-all: build-all-core build-all-provider


# -------------------------------------------------
# Release validation
# -------------------------------------------------

release-check:
	uv run python scripts/release_validate.py

release-validate: release-check build-all


# -------------------------------------------------
# Utilities
# -------------------------------------------------

clean:
	rm -rf site
	rm -f uv.lock
	rm -rf .venv
	rm -rf .*_cache
	rm -f flepimop2-op_engine/uv.lock
	rm -rf flepimop2-op_engine/.venv
	rm -rf flepimop2-op_engine/.*_cache

# Build API reference for the documentation using `mkdocstrings`
api-reference:
    uv run scripts/api-reference.py

# Build the documentation using `mkdocs`
docs: api-reference
	uv run mkdocs build --verbose --strict

# Serve the documentation locally using `mkdocs`
serve: api-reference
	uv run mkdocs serve
