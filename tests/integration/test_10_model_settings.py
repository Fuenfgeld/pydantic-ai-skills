"""
Integration tests for 10_model_settings.py - Model Settings and Configuration.

Tests model settings like temperature, max_tokens, and usage tracking.
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
async def test_model_settings_temperature(openrouter_model):
    """Test model settings like temperature and max_tokens."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant.",
    )

    # Low temperature for deterministic response
    result = await agent.run(
        "What is 2 + 2? Reply with just the number.",
        model_settings={
            "temperature": 0.0,
            "max_tokens": 10,
        },
    )

    assert "4" in result.output


@pytest.mark.asyncio
async def test_usage_tracking(openrouter_model):
    """Test that usage information is available."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="Be brief.",
    )

    result = await agent.run(
        "Say hello.",
        model_settings={"max_tokens": 20},
    )

    usage = result.usage()
    assert usage.total_tokens > 0
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
