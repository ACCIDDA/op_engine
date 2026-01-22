
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
	cd flepimop2-op_engine && .venv/bin/python -m pytest --doctest-modules

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
