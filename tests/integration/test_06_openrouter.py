"""
Integration tests for 06_openrouter.py - OpenRouter Provider Configuration.

Tests explicit OpenRouter provider setup with real API calls.
"""

import os

import pytest

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


@pytest.mark.asyncio
async def test_openrouter_provider_explicit(openrouter_api_key):
    """Test explicit OpenRouter provider configuration."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # Explicit provider setup (as shown in 06_openrouter.py)
    provider = OpenAIProvider(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatModel(
        model_name="anthropic/claude-haiku-4.5",
        provider=provider,
    )

    agent = Agent(
        model=model,
        system_prompt="Be brief.",
    )

    result = await agent.run("Say 'hello'")

    assert len(result.output) > 0
