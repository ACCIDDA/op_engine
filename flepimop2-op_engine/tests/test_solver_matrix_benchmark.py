# flepimop2-op_engine: Operator-Partitioned Engine Provider for flepimop2
# Copyright (C) 2026  Joshua Macdonald, Carl Pearson, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for the machine-readable solver benchmark."""

# ruff: noqa: SLF001

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks import solver_matrix

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("method", "accepted", "rejected", "rhs_calls"),
    [
        ("euler", 3, 2, 8),
        ("heun", 3, 2, 10),
        ("rk4", 3, 2, 53),
        ("dopri5", 3, 2, 31),
    ],
)
def test_adaptive_rejections_are_recovered_from_rhs_calls(
    method: str,
    accepted: int,
    rejected: int,
    rhs_calls: int,
) -> None:
    """Map eager adaptive RHS calls back to rejected attempts."""
    assert solver_matrix._adaptive_rejections(method, accepted, rhs_calls) == rejected


def test_fixed_execution_rhs_counts_include_dopri_fsal() -> None:
    """Count frozen-plan RHS evaluations including DOPRI FSAL reuse."""
    assert solver_matrix._execution_rhs("euler", 4) == 4
    assert solver_matrix._execution_rhs("heun", 4) == 8
    assert solver_matrix._execution_rhs("rk4", 4) == 16
    assert solver_matrix._execution_rhs("dopri5", 4) == 25


def test_numpy_smoke_writes_versioned_json(tmp_path: Path) -> None:
    """Write a small NumPy run with schema and environment metadata."""
    output = tmp_path / "benchmark.json"

    solver_matrix.main([
        "--backends",
        "numpy",
        "--policies",
        "fixed",
        "--methods",
        "rk4",
        "--horizons",
        "0.5",
        "--batch-sizes",
        "2",
        "--output-count",
        "3",
        "--fixed-max-step",
        "0.25",
        "--repeats",
        "1",
        "--references",
        "--output",
        str(output),
    ])

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == solver_matrix.SCHEMA_VERSION
    git_metadata = document["environment"]["git"]
    assert set(git_metadata) == {"revision", "dirty"}
    assert git_metadata["revision"] is None or git_metadata["revision"]
    assert isinstance(git_metadata["dirty"], bool)
    assert document["skipped"] == []
    assert len(document["results"]) == 1
    result = document["results"][0]
    assert result["implementation"] == "op_engine"
    assert result["backend"] == "numpy"
    assert result["compile_seconds"] is None
    assert result["execution_rhs_evaluations"] == 8
    assert result["max_abs_error"] < 1e-5


def test_invalid_ranges_fail_before_benchmarking() -> None:
    """Reject invalid CLI ranges before any timed work begins."""
    options = solver_matrix._parse_args(["--repeats", "0"])
    with pytest.raises(ValueError, match="repeats"):
        solver_matrix._validate_options(options)
