"""
Root conftest.py - Shared fixtures and pytest configuration.

This file provides common fixtures for mocking HTTP clients, API responses,
and blocking real API calls during testing.
"""

import os
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# Block real API calls by default in tests
os.environ.setdefault("PYDANTIC_AI_ALLOW_MODEL_REQUESTS", "false")


@dataclass
class MockAgentDeps:
    """Standard mock dependencies for testing agents."""

    user_id: str = "test_user_123"
    api_key: str = "test_api_key"
    http_client: object = None
    environment: str = "development"

    def is_dev(self) -> bool:
        return self.environment == "development"

    def get_auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient for dependency injection."""
    client = AsyncMock(spec=httpx.AsyncClient)

    # Default successful response
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"amount": 100.0, "currency": "USD"}
    response.raise_for_status = MagicMock()

    client.get.return_value = response
    client.post.return_value = response

    return client


@pytest.fixture
def mock_http_response_factory():
    """Factory for creating custom HTTP responses."""

    def _create(status_code=200, json_data=None, text=""):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.text = text
        response.raise_for_status = MagicMock()
        if status_code >= 400:
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=response
            )
        return response

    return _create


@pytest.fixture
def mock_deps(mock_http_client):
    """Create mock dependencies with mocked HTTP client."""
    return MockAgentDeps(http_client=mock_http_client)


# Marker configuration
def pytest_configure(config):
    config.addinivalue_line("markers", "real_api: mark test as requiring real API access")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Skip real API tests if no keys present
def pytest_collection_modifyitems(config, items):
    skip_real = pytest.mark.skip(reason="No API keys configured")
    for item in items:
        if "real_api" in item.keywords:
            if not os.getenv("OPENROUTER_API_KEY"):
                item.add_marker(skip_real)
