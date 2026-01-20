"""
Integration tests for 12_conversation_history.py - Conversation History.

Tests multi-turn conversation with message history persistence.
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
async def test_conversation_history(openrouter_model):
    """Test multi-turn conversation with history."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant. Remember previous messages.",
    )

    # First turn
    result1 = await agent.run("My name is Bob. Remember it.")
    history = result1.all_messages()

    # Second turn with history
    result2 = await agent.run("What is my name?", message_history=history)

    assert "Bob" in result2.output
