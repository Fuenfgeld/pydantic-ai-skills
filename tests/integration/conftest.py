"""
Integration test fixtures.

These fixtures are for tests that require real API keys.
Tests in this directory are automatically skipped if OPENROUTER_API_KEY is not set.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if no API keys are configured."""
    skip_real = pytest.mark.skip(reason="No API keys configured for integration tests")

    for item in items:
        # All tests in integration directory require API keys
        if "integration" in str(item.fspath):
            if not os.getenv("OPENROUTER_API_KEY"):
                item.add_marker(skip_real)


@pytest.fixture
def openrouter_api_key():
    """Get OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not configured")
    return key


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not configured")
    return key
