# Run all default tasks for local development
default: format check pytest mypy

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
pytest-provider:
	cd flepimop2-op_engine && .venv/bin/python -m pytest --doctest-modules

pytest:
	just pytest-core
	just pytest-provider


# -------------------------------------------------
# Type checking
# -------------------------------------------------

mypy-core:
	uv run mypy --strict src/op_engine

# Assumes `flepimop2-op_engine/.venv` already exists (run `just provider-sync` or `just ci` first).
mypy-provider:
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
	rm -f uv.lock
	rm -rf .*_cache
	rm -rf .venv
	rm -rf flepimop2-op_engine/.venv
	rm -f flepimop2-op_engine/uv.lock

docs:
	uv run mkdocs build --verbose --strict

serve:
	uv run mkdocs serve
