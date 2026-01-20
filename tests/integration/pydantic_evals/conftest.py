"""
Fixtures for pydantic-evals integration tests.
"""

import os

import pytest


@pytest.fixture
def openrouter_api_key():
    """Get OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not configured")
    return key


@pytest.fixture
def openrouter_model(openrouter_api_key):
    """Create an OpenRouter model using Claude Haiku 4.5 for cost-effective testing."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    return OpenAIChatModel(
        model_name="anthropic/claude-haiku-4.5",
        provider=provider,
    )
