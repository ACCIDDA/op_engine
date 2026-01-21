"""Global pytest configuration and shared fixtures for op_engine."""

from __future__ import annotations

import importlib.util
from typing import Final

import pytest

# -----------------------------------------------------------------------------
# Optional dependency detection
# -----------------------------------------------------------------------------

HAS_FLEPIMOP2: Final[bool] = importlib.util.find_spec("flepimop2") is not None


# -----------------------------------------------------------------------------
# Global markers registration safety (for local pytest runs)
# -----------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the test suite."""
    config.addinivalue_line(
        "markers",
        "flepimop2: mark test as requiring flepimop2 optional dependency",
    )


# -----------------------------------------------------------------------------
# Conditional skipping fixture
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def require_flepimop2() -> None:
    """
    Skip tests if flepimop2 extra is not installed.

    Usage:
        def test_x(require_flepimop2):
            ...
    """
    if not HAS_FLEPIMOP2:
        pytest.skip("flepimop2 extra not installed")
