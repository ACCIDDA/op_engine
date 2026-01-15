"""Unit tests for `op_engine.__version__`."""

from op_engine import __version__


def test_version_is_string() -> None:
    """Test that `__version__` is a string."""
    assert isinstance(__version__, str)
