"""
Integration tests for 07_logfire.py - Logfire Integration Pattern.

Tests the logfire span pattern with real LLM calls (without actual logfire).
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
async def test_logfire_span_pattern(openrouter_model):
    """Test the logfire span pattern (without actual logfire)."""
    from pydantic_ai import Agent

    # This test validates the pattern works even without logfire configured
    agent = Agent(
        model=openrouter_model,
        system_prompt="Be brief.",
    )

    # The pattern from 07_logfire.py
    result = await agent.run("What is 1 + 1?")

    # Simulate what logfire would capture
    output = result.output
    assert "2" in output

    # Verify we can access result attributes that logfire would log
    assert result.all_messages() is not None
    assert result.usage() is not None
