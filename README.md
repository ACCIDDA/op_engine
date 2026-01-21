# OP Engine

Operator-Partitioned Engine (OP Engine) is a lightweight multiphysics
solver core for time-dependent systems, providing a unified interface
for solving ODEs and PDEs using explicit methods, operator splitting,
and IMEX (Implicit--Explicit) time integration schemes.

OP Engine can be used:

-   Directly as a standalone Python solver API, or
-   As a backend engine for the flepimop2 simulation framework via the
    optional integration layer.

The project emphasizes strong typing, explicit configuration, minimal
runtime dependencies, and clean separation between solver logic and
orchestration frameworks.

------------------------------------------------------------------------

## Installation

### Base Installation (Solver Only)

Install OP Engine without framework integrations:

    pip install op_engine

### With flepimop2 Integration

To enable the flepimop2 adapter:

    pip install "op_engine[flepimop2]"

This installs additional dependencies required for the flepimop2 engine
backend.

------------------------------------------------------------------------

## Supported Time Integration Methods

OP Engine currently provides the following solver methods:

### Explicit Methods

-   **Euler**
    -   First-order explicit forward Euler method.
    -   Fast, simple, and stable only for small time steps.
    -   Best suited for prototyping and non-stiff problems.
-   **Heun (RK2 / Improved Euler)**
    -   Second-order explicit predictor--corrector method.
    -   Improved accuracy and stability compared to Euler.
    -   Default explicit method in OP Engine.

### IMEX Methods (Operator-Based)

IMEX solvers are available internally but require operator
specifications:

-   **IMEX Euler**
    -   First-order implicit--explicit split scheme.
    -   Treats stiff linear operators implicitly.
-   **IMEX Heun-TR**
    -   Second-order IMEX scheme combining Heun explicit stages with
        trapezoidal implicit correction.
-   **IMEX TR-BDF2**
    -   Two-stage second-order stiffly accurate IMEX scheme.
    -   Designed for improved stability on stiff systems.

Note:

-   IMEX methods require operator specifications and are guarded when
    used through the flepimop2 adapter.
-   Explicit solvers work out-of-the-box without operator configuration.

------------------------------------------------------------------------

## Basic OP Engine Usage

OP Engine exposes a low-level solver API designed to integrate cleanly
with custom models.

Typical workflow:

1.  Define a right-hand side (RHS) function representing the system
    dynamics.
2.  Construct a `ModelCore` object describing state shape and time grid.
3.  Create a `CoreSolver` with the model.
4.  Run the solver using a `RunConfig` configuration.

High-level components:

-   **ModelCore**
    -   Manages state tensors, time grids, and history storage.
-   **CoreSolver**
    -   Orchestrates time stepping and solver execution.
-   **RunConfig**
    -   Controls solver method, tolerances, adaptivity, and stability
        settings.

This design allows OP Engine to remain framework-agnostic while
supporting advanced splitting and IMEX strategies.

------------------------------------------------------------------------

## flepimop2 Integration

OP Engine provides an optional adapter allowing it to act as a flepimop2
engine backend.

### Enabling the Adapter

Install with:

    pip install "op_engine[flepimop2]"

### Adapter Overview

The flepimop2 adapter:

-   Implements the flepimop2 `EngineABC` interface
-   Wraps flepimop2 System steppers into OP Engine RHS functions
-   Handles state reshaping and history extraction
-   Maps flepimop2 YAML configuration into OP Engine solver settings

### Supported Features

Currently supported through flepimop2:

-   Explicit solvers: `euler`, `heun`
-   Adaptive stepping controls
-   Strict configuration validation

IMEX methods are guarded unless operator specifications are provided.

### Adapter Expectations

The adapter expects:

-   A flepimop2 System exposing a `_stepper(t, state, **params)`
    callable
-   1D NumPy state vectors
-   Strictly increasing evaluation time grids

The adapter automatically handles shape normalization and conversion
between flepimop2 and OP Engine internal representations.

------------------------------------------------------------------------

## Local Development

### Clone Repository

    git clone git@github.com:ACCIDDA/op_engine.git
    cd op_engine

### Environment Setup

This project uses uv for dependency management.

To create a development environment:

    uv sync --dev

This installs:

-   OP Engine
-   Development tools (pytest, mypy, ruff)
-   Documentation and formatting dependencies

To include flepimop2 integration during development:

    uv sync --all-extras

or

    uv pip install ".[flepimop2]"

### Running Development Checks

Run all default development checks:

    just

This executes:

-   ruff format
-   ruff check --fix
-   pytest --doctest-modules
-   mypy --strict

### CI Parity

To reproduce CI locally:

    just ci

CI runs on Python 3.10 through 3.13 and enforces formatting, linting,
testing, and strict static typing.

------------------------------------------------------------------------

## Design Philosophy

OP Engine is designed around:

-   Strong static typing (mypy strict)
-   Clear separation between solver core and adapters
-   Explicit configuration models
-   Minimal runtime overhead
-   Predictable numerical behavior

The architecture allows OP Engine to operate as both a standalone solver
engine and a backend component for higher-level simulation frameworks
such as flepimop2.
