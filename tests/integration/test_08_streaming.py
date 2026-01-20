"""
Integration tests for 08_streaming.py - Response Streaming.

Tests streaming responses with real LLM calls.
"""

import os

import pytest
from pydantic import BaseModel, Field

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


@pytest.mark.asyncio
async def test_streaming_text(openrouter_model):
    """Test streaming text responses."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant. Keep responses brief.",
    )

    chunks = []
    async with agent.run_stream("Count from 1 to 5, one number per line.") as response:
        async for chunk in response.stream_text():
            chunks.append(chunk)

    full_response = "".join(chunks)
    # Note: Some providers may return content in fewer chunks
    assert len(chunks) >= 1, "Should receive at least one chunk"
    assert "1" in full_response
    assert "5" in full_response


class CountResult(BaseModel):
    """Result with numbers and total."""

    numbers: list[int] = Field(description="List of numbers")
    total: int = Field(description="Sum of all numbers")


@pytest.mark.asyncio
async def test_streaming_structured(openrouter_model):
    """Test streaming with structured output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=CountResult,
        system_prompt="You help with number operations.",
    )

    async with agent.run_stream("List the numbers 1, 2, 3 and their sum.") as response:
        result = await response.get_output()

    assert isinstance(result, CountResult)
    assert 1 in result.numbers
    assert result.total == sum(result.numbers)
