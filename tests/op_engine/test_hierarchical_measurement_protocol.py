"""Measurement protocol contracts keep censored runs distinct from results."""

from __future__ import annotations

import json
import subprocess  # noqa: S404
from argparse import Namespace
from typing import TYPE_CHECKING

import pytest

from scripts.benchmark_hierarchical import _cases, _isolated, main  # noqa: PLC2701

if TYPE_CHECKING:
    from pathlib import Path


def _arguments() -> Namespace:
    return Namespace(
        groups=[1],
        local_dimensions=[1],
        steps=[4],
        regimes=["nonstiff"],
        coupling=["independent"],
        solvers=["core", "numpy", "scipy"],
        derivatives=["trajectory", "hessian"],
        repeats=1,
        timeout=0.1,
    )


def test_reference_solvers_do_not_claim_derivative_capability() -> None:
    """Concrete references emit trajectory cases while the native path emits both."""
    cases = list(_cases(_arguments()))
    assert len(cases) == 4
    assert all(
        case["derivatives"] == "trajectory"
        for case in cases
        if case["solvers"] in {"numpy", "scipy"}
    )


def test_timeout_is_censored_without_a_timing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted worker is never presented as a completed fast measurement."""

    def timeout(*_args: object, **_kwargs: object) -> None:
        command = "worker"
        raise subprocess.TimeoutExpired(command, 0.1)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = _isolated(next(_cases(_arguments())), _arguments())
    assert result["status"] == "timeout"
    assert result["censored"] is True
    assert result["timing_result"] is None


def test_required_measurement_failure_remains_visible_in_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed required case writes its diagnostic and makes the command fail."""
    monkeypatch.setattr(
        "scripts.benchmark_hierarchical._isolated",
        lambda *_args: {"status": "timeout", "censored": True},
    )
    output = tmp_path / "cases.jsonl"
    status = main([
        "--groups",
        "1",
        "--local-dimensions",
        "1",
        "--steps",
        "4",
        "--solvers",
        "core",
        "--derivatives",
        "trajectory",
        "--output",
        str(output),
        "--require-success",
    ])
    assert status == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "timeout"


def test_invalid_case_dimension_is_rejected_before_work() -> None:
    """Malformed fixture dimensions cannot produce meaningless benchmark rows."""
    with pytest.raises(SystemExit):
        main(["--groups", "0"])


def test_workers_override_inherited_persistent_compilation_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inherited disk caches cannot turn cold compilation into cached loading."""
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "true")

    def worker(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["JAX_ENABLE_COMPILATION_CACHE"] == "false"
        return subprocess.CompletedProcess("worker", 0, '{"status":"success"}', "")

    monkeypatch.setattr(subprocess, "run", worker)
    assert _isolated(next(_cases(_arguments())), _arguments())["status"] == "success"
