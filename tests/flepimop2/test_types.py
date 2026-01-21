# tests/flepimop2/test_types.py
"""Unit tests for op_engine.flepimop2.types."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np
import pytest

from op_engine.flepimop2.types import (
    EngineRunner,
    IdentifierString,
    SystemStepper,
    as_float64_1d,
    as_float64_1d_times,
    as_float64_state,
    ensure_strictly_increasing_times,
    normalize_params,
)

pytestmark = pytest.mark.flepimop2


class _GoodStepper:
    def __call__(
        self, time: np.float64, state: np.ndarray, **params: object
    ) -> np.ndarray:
        _ = time
        _ = params
        return np.asarray(state, dtype=np.float64)


class _GoodRunner:
    def __call__(
        self,
        stepper: SystemStepper,
        times: np.ndarray,
        state: np.ndarray,
        params: Mapping[IdentifierString, object],
        **engine_kwargs: object,
    ) -> np.ndarray:
        _ = engine_kwargs
        out: list[np.ndarray] = []
        for t in np.asarray(times, dtype=np.float64):
            dy = stepper(
                np.float64(t), np.asarray(state, dtype=np.float64), **dict(params)
            )
            out.append(np.asarray(dy, dtype=np.float64).reshape(-1))
        return np.asarray(out, dtype=np.float64)


def test_protocols_runtime_checkable() -> None:
    """SystemStepper and EngineRunner should be runtime checkable protocols."""
    stepper = _GoodStepper()
    runner = _GoodRunner()

    assert isinstance(stepper, SystemStepper)
    assert isinstance(runner, EngineRunner)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([1, 2, 3], np.array([1.0, 2.0, 3.0], dtype=np.float64)),
        ((1, 2, 3, 4), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)),
        (
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ),
    ],
)
def test_as_float64_1d_converts_1d_inputs(x: object, expected: np.ndarray) -> None:
    """as_float64_1d converts 1D input to float64 1D array correctly."""
    out = as_float64_1d(x)
    assert out.dtype == np.float64
    assert out.ndim == 1
    np.testing.assert_allclose(out, expected)


def test_as_float64_1d_rejects_non_1d() -> None:
    """as_float64_1d rejects non-1D inputs."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="1D"):
        as_float64_1d(x, name="x")


def test_as_float64_1d_is_contiguous() -> None:
    """as_float64_1d returns a contiguous array."""
    x = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]  # non-contiguous view
    assert not x.flags["C_CONTIGUOUS"]

    x1d = x.reshape(-1)  # still non-contiguous
    assert not x1d.flags["C_CONTIGUOUS"]

    out = as_float64_1d(x1d, name="x1d")
    assert out.flags["C_CONTIGUOUS"]
    assert out.dtype == np.float64
    assert out.shape == (6,)


def test_as_float64_1d_times_converts() -> None:
    """as_float64_1d_times converts input to float64 1D time array."""
    x = [0, 1, 2, 3]
    out = as_float64_1d_times(x, name="times")
    assert out.dtype == np.float64
    assert out.ndim == 1
    np.testing.assert_allclose(out, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))


def test_as_float64_state_converts() -> None:
    """as_float64_state converts input to float64 1D state array."""
    x = [10, 20, 30]
    out = as_float64_state(x, name="state")
    assert out.dtype == np.float64
    assert out.ndim == 1
    np.testing.assert_allclose(out, np.array([10.0, 20.0, 30.0], dtype=np.float64))


def test_ensure_strictly_increasing_times_accepts_trivial() -> None:
    """ensure_strictly_increasing_times should accept empty/length-1 times."""
    ensure_strictly_increasing_times(np.array([], dtype=np.float64), name="times")
    ensure_strictly_increasing_times(np.array([1.0], dtype=np.float64), name="times")


def test_ensure_strictly_increasing_times_accepts_increasing() -> None:
    """ensure_strictly_increasing_times should accept strictly increasing arrays."""
    times = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    ensure_strictly_increasing_times(times, name="times")


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, -1.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 1.0], dtype=np.float64),
    ],
)
def test_ensure_strictly_increasing_times_rejects_non_increasing(
    times: np.ndarray,
) -> None:
    """ensure_strictly_increasing_times should reject non-increasing arrays."""
    with pytest.raises(ValueError, match="strictly increasing"):
        ensure_strictly_increasing_times(times, name="times")


def test_normalize_params_none_returns_empty_dict() -> None:
    """normalize_params with None returns empty dict."""
    out = normalize_params(None)
    assert out == {}
    assert isinstance(out, dict)


def test_normalize_params_mapping_returns_dict_copy() -> None:
    """normalize_params returns a dict copy of the input mapping."""
    src: Mapping[str, object] = {"beta": 0.3, "gamma": 0.1}
    out = normalize_params(src)
    assert out == {"beta": 0.3, "gamma": 0.1}
    assert isinstance(out, dict)
    assert out is not src


def test_good_runner_produces_time_by_state_array() -> None:
    """A well-typed EngineRunner produces correct output shape and dtype."""
    runner = _GoodRunner()
    stepper = _GoodStepper()

    times = as_float64_1d_times([0, 1, 2], name="times")
    y0 = as_float64_state([1, 2, 3], name="y0")
    params = normalize_params({"x": 1.0})

    res = runner(stepper, times, y0, params)
    assert res.dtype == np.float64
    assert res.shape == (3, 3)  # (T, n_state)


def test_runner_protocol_signature_acceptance() -> None:
    """Structural test: a callable with compatible signature satisfies EngineRunner."""
    runner = cast("EngineRunner", _GoodRunner())
    stepper = cast("SystemStepper", _GoodStepper())

    times = np.array([0.0, 1.0], dtype=np.float64)
    y0 = np.array([1.0, 2.0], dtype=np.float64)

    out = runner(stepper, times, y0, {"beta": 0.3}, atol=1e-9)  # extra kwargs allowed
    assert out.shape == (2, 2)
