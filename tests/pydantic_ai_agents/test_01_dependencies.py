"""
Tests for 01_dependencies.py - Dependency Injection patterns.

These tests are synchronous because they test pure Python dataclasses
without any agent execution.

Note: Since the file name starts with a number (01_), we can't import it
directly. We test the patterns by recreating the dataclass structure and
verifying the expected behavior.
"""

import importlib.util
from pathlib import Path

import pytest


# Load the module dynamically since Python doesn't allow importing modules starting with numbers
def load_dependencies_module():
    """Dynamically load the 01_dependencies.py module."""
    module_path = (
        Path(__file__).parent.parent.parent
        / "skills"
        / "pydantic-ai-agents"
        / "references"
        / "01_dependencies.py"
    )
    spec = importlib.util.spec_from_file_location("dependencies_01", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the module once for all tests
dependencies_module = load_dependencies_module()
AgentDeps = dependencies_module.AgentDeps


class TestAgentDepsCreation:
    """Test AgentDeps dataclass instantiation."""

    def test_create_with_required_fields(self, mock_http_client):
        """Test AgentDeps can be instantiated with required fields."""
        deps = AgentDeps(
            user_id="user123",
            api_key="key456",
            http_client=mock_http_client,
        )

        assert deps.user_id == "user123"
        assert deps.api_key == "key456"
        assert deps.environment == "production"  # default value

    def test_create_with_all_fields(self, mock_http_client):
        """Test AgentDeps with all fields specified."""
        deps = AgentDeps(
            user_id="user123",
            api_key="key456",
            http_client=mock_http_client,
            environment="staging",
        )

        assert deps.environment == "staging"


class TestAgentDepsHelperMethods:
    """Test AgentDeps helper methods."""

    def test_is_dev_returns_true_for_development(self, mock_http_client):
        """Test is_dev() returns True when environment is development."""
        deps = AgentDeps(
            user_id="user123",
            api_key="key456",
            http_client=mock_http_client,
            environment="development",
        )

        assert deps.is_dev() is True

    def test_is_dev_returns_false_for_production(self, mock_http_client):
        """Test is_dev() returns False when environment is production."""
        deps = AgentDeps(
            user_id="user123",
            api_key="key456",
            http_client=mock_http_client,
            environment="production",
        )

        assert deps.is_dev() is False

    def test_is_dev_returns_false_for_other_environments(self, mock_http_client):
        """Test is_dev() returns False for staging or other environments."""
        deps = AgentDeps(
            user_id="user123",
            api_key="key456",
            http_client=mock_http_client,
            environment="staging",
        )

        assert deps.is_dev() is False

    def test_get_auth_header_format(self, mock_http_client):
        """Test get_auth_header() returns properly formatted Bearer token."""
        deps = AgentDeps(
            user_id="user123",
            api_key="my_secret_key",
            http_client=mock_http_client,
        )

        header = deps.get_auth_header()

        assert header == {"Authorization": "Bearer my_secret_key"}

    def test_get_auth_header_with_different_keys(self, mock_http_client):
        """Test get_auth_header() works with various API key formats."""
        test_keys = [
            "simple_key",
            "key-with-dashes",
            "key_with_underscores",
            "MixedCaseKey123",
            "key.with.dots",
        ]

        for key in test_keys:
            deps = AgentDeps(
                user_id="user123",
                api_key=key,
                http_client=mock_http_client,
            )
            header = deps.get_auth_header()
            assert header["Authorization"] == f"Bearer {key}"
