# Run all default tasks for local development
default: format check pytest mypy

# Format code using `ruff`
format:
    uv run ruff format --preview

# Check code using `ruff`
check:
    uv run ruff check --preview --fix

# Run tests using `pytest`
pytest:
    uv run pytest --doctest-modules

# Type check using `mypy`
mypy:
    uv run mypy --strict .

# Run all CI checks
ci:
    uv run ruff format --preview --check
    uv run ruff check --preview --no-fix
    uv run pytest --doctest-modules
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
