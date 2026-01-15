# gempyor2

The next generation vectorized compartmental model engine for [`flepimop2`](https://github.com/ACCIDDA/flepimop2).

## Local Development

1. Clone the repository

```bash
git clone git@github.com:ACCIDDA/gempyor2.git
cd gempyor2
```

2. Create a virtual environment and install dependencies using [`uv`](https://docs.astral.sh/uv/). To create a `.venv` with the package installed:

```bash
uv sync --dev
```

This will create a virtual environment and install the package along with development dependencies (mypy, pytest, ruff).

3. Run default checks using [`just`](https://just.systems/). To run the default development tasks:

```bash
just
```

This will run:
- `ruff format` - Format code
- `ruff check --fix` - Lint and auto-fix issues
- `pytest --doctest-modules` - Run tests including doctests
- `mypy --strict` - Type check with strict settings

4. CI runs on pull requests to `main` and tests against Python 3.10, 3.11, 3.12, and 3.13. The CI checks are defined in `just ci` and include:

- `ruff format --check` - Verify code formatting (no auto-fix)
- `ruff check --no-fix` - Lint without modifications
- `pytest --doctest-modules` - Run test suite
- `mypy --strict` - Type checking

To run the same checks locally that run in CI (say for diagnosing CI failures):

```bash
just ci
```
