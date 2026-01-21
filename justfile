# Run all default tasks for local development
default: format check pytest mypy

# Optional extras to include for runs that need them
FLEPIMOP2_EXTRA := "flepimop2"

# Format code using `ruff`
format:
    uv run ruff format --preview

# Check code using `ruff`
check:
    uv run ruff check --preview --fix

# Run tests using `pytest`
# Include flepimop2 extra so pydantic is available during doctest/module collection.
pytest:
    uv run --extra {{FLEPIMOP2_EXTRA}} pytest --doctest-modules

# Type check using `mypy`
mypy:
    uv run mypy --strict .

# Run all CI checks
ci:
    uv run ruff format --preview --check
    uv run ruff check --preview --no-fix
    uv run --extra {{FLEPIMOP2_EXTRA}} pytest --doctest-modules
    uv run mypy --strict .

# Clean up generated lock files, venvs, and caches
clean:
    rm -f uv.lock
    rm -rf .*_cache
    rm -rf .venv

# Build the documentation using `mkdocs`
docs:
    uv run mkdocs build --verbose --strict

# Serve the documentation locally using `mkdocs`
serve:
    uv run mkdocs serve

