"""Pytest configuration for integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add examples/calculator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "calculator"))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "ollama: marks tests that require Ollama to be running",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Add ollama marker to all tests in this directory."""
    for item in items:
        if "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.ollama)
