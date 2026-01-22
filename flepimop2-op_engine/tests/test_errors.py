"""Unit tests for op_engine.flepimop2.errors."""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from flepimop2.engine.op_engine import errors


def _flepimop2_is_installed() -> bool:
    """Return True if flepimop2 appears importable in this environment."""
    return find_spec("flepimop2") is not None


def test_check_flepimop2_available_matches_environment() -> None:
    """check_flepimop2_available should reflect whether flepimop2 is importable."""
    status = errors.check_flepimop2_available()
    assert status.package == "flepimop2"
    assert status.is_available == _flepimop2_is_installed()
    if status.is_available:
        assert status.detail is None
    else:
        assert isinstance(status.detail, str)
        assert "spec" in status.detail.lower() or "not found" in status.detail.lower()


def test_check_flepimop2_available_forced_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_flepimop2_available should report unavailable if find_spec returns None."""
    monkeypatch.setattr(errors, "find_spec", lambda _name: None)

    status = errors.check_flepimop2_available()
    assert status.package == "flepimop2"
    assert status.is_available is False
    assert status.detail == "Module spec not found"


def test_check_flepimop2_available_forced_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_flepimop2_available should report available if find_spec returns a spec."""
    sentinel = object()
    monkeypatch.setattr(errors, "find_spec", lambda _name: sentinel)

    status = errors.check_flepimop2_available()
    assert status.package == "flepimop2"
    assert status.is_available is True
    assert status.detail is None


def test_require_flepimop2_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """require_flepimop2 raises OptionalDependencyMissingError when unavailable."""
    monkeypatch.setattr(errors, "find_spec", lambda _name: None)

    with pytest.raises(errors.OptionalDependencyMissingError) as excinfo:
        errors.require_flepimop2()

    exc = excinfo.value
    msg = str(exc)

    assert "requires flepimop2" in msg.lower()
    assert "op_engine[flepimop2]" in msg
    assert "import detail" in msg.lower()
    assert getattr(exc, "code", None) == errors.ErrorCode.OPTIONAL_DEPENDENCY_MISSING


def test_require_flepimop2_noop_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """require_flepimop2 should not raise when flepimop2 is available."""
    monkeypatch.setattr(errors, "find_spec", lambda _name: object())
    errors.require_flepimop2()


def test_raise_unsupported_imex_raises_with_reason_and_code() -> None:
    """raise_unsupported_imex raises ValueError and chains an OpEngineFlepimop2Error."""
    method = "imex-euler"
    reason = "operators.default is missing"

    with pytest.raises(
        ValueError, match=r"Method 'imex-euler' is not supported"
    ) as excinfo:
        errors.raise_unsupported_imex(method, reason=reason)

    exc = excinfo.value
    msg = str(exc)

    assert method in msg
    assert reason in msg
    assert "imex methods require operator specifications" in msg.lower()

    cause = exc.__cause__
    assert isinstance(cause, errors.OpEngineFlepimop2Error)
    assert cause.code == errors.ErrorCode.UNSUPPORTED_METHOD


def test_raise_invalid_engine_config_includes_missing_and_detail() -> None:
    """raise_invalid_engine_config should include missing keys and detail text."""
    with pytest.raises(errors.EngineConfigError) as excinfo:
        errors.raise_invalid_engine_config(
            missing=["operators", "method", "operators"],
            detail="Bad config shape",
        )

    exc = excinfo.value
    msg = str(exc)

    assert "invalid op_engine.flepimop2 engine configuration" in msg.lower()
    assert "['method', 'operators']" in msg
    assert "detail: bad config shape" in msg.lower()
    assert getattr(exc, "code", None) == errors.ErrorCode.INVALID_ENGINE_CONFIG


def test_raise_invalid_engine_config_minimal_message() -> None:
    """raise_invalid_engine_config should still raise with a minimal message."""
    with pytest.raises(errors.EngineConfigError) as excinfo:
        errors.raise_invalid_engine_config()

    exc = excinfo.value
    msg = str(exc)

    assert "invalid op_engine.flepimop2 engine configuration" in msg.lower()
    assert getattr(exc, "code", None) == errors.ErrorCode.INVALID_ENGINE_CONFIG


def test_raise_state_shape_error_includes_name_expected_got_and_code() -> None:
    """raise_state_shape_error should surface name/expected/got clearly."""
    with pytest.raises(ValueError, match=r"y0 has an invalid shape/value") as excinfo:
        errors.raise_state_shape_error(name="y0", expected="(n_state,)", got=(3, 7))

    exc = excinfo.value
    msg = str(exc)

    assert "y0" in msg
    assert "expected (n_state,)" in msg.lower()
    assert "(3, 7)" in msg

    cause = exc.__cause__
    assert isinstance(cause, errors.OpEngineFlepimop2Error)
    assert cause.code == errors.ErrorCode.INVALID_STATE_SHAPE


def test_raise_parameter_error_raises_typeerror_and_code() -> None:
    """raise_parameter_error should raise TypeError and chain a coded cause."""
    with pytest.raises(TypeError) as excinfo:
        errors.raise_parameter_error(detail="dt_init must be positive")

    exc = excinfo.value
    msg = str(exc)

    assert "invalid parameters" in msg.lower()
    assert "dt_init must be positive" in msg

    cause = exc.__cause__
    assert isinstance(cause, errors.OpEngineFlepimop2Error)
    assert cause.code == errors.ErrorCode.INVALID_PARAMETERS
