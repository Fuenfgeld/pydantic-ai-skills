"""
Integration tests for models.py - Pydantic models for evaluation.

Tests that the model patterns work correctly with real LLM structured outputs.
Corresponds to: skills/pydantic-evals/references/examples/models.py
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from pydantic import AwareDatetime, BaseModel, Field

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


# Models matching the pattern from models.py
class TimeRangeSuccess(BaseModel):
    """Successful time range response."""

    min_timestamp: AwareDatetime = Field(description="Start of the time range")
    max_timestamp: AwareDatetime = Field(description="End of the time range")
    explanation: Optional[str] = Field(default=None, description="Explanation of the range")


class TimeRangeError(BaseModel):
    """Error response when time range cannot be determined."""

    error_message: str = Field(description="Description of why the range couldn't be determined")


@pytest.mark.asyncio
async def test_llm_produces_success_response(openrouter_model):
    """Test LLM can produce a TimeRangeSuccess structured output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="""You are a time range parser. Given a natural language time description,
        return the appropriate time range. The current time is provided in the query.""",
    )

    now = datetime.now(timezone.utc)
    result = await agent.run(
        f"Parse 'last 7 days' relative to {now.isoformat()}. "
        f"Use {now.isoformat()} as max_timestamp and 7 days before as min_timestamp."
    )

    assert isinstance(result.output, TimeRangeSuccess)
    assert result.output.min_timestamp < result.output.max_timestamp
    # The range should be approximately 7 days
    delta = result.output.max_timestamp - result.output.min_timestamp
    assert timedelta(days=6) <= delta <= timedelta(days=8)


@pytest.mark.asyncio
async def test_llm_produces_error_response(openrouter_model):
    """Test LLM can produce a TimeRangeError structured output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeError,
        system_prompt="You are a time range parser. Return an error if the input is ambiguous or invalid.",
    )

    result = await agent.run("Parse 'sometime in the future maybe'")

    assert isinstance(result.output, TimeRangeError)
    assert len(result.output.error_message) > 0


@pytest.mark.asyncio
async def test_llm_union_type_success_or_error(openrouter_model):
    """Test LLM can choose between success and error response types."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess | TimeRangeError,  # type: ignore
        system_prompt="""You are a time range parser.
        Return TimeRangeSuccess for valid time descriptions.
        Return TimeRangeError for invalid or ambiguous inputs.""",
    )

    now = datetime.now(timezone.utc)

    # Valid input should produce success
    result = await agent.run(f"Parse 'yesterday' relative to {now.isoformat()}")
    assert isinstance(result.output, (TimeRangeSuccess, TimeRangeError))
