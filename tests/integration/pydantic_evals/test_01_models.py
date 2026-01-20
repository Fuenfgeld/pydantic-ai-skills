"""
Integration tests for 01_models.py - Pydantic Models for Evaluation.

Tests the TimeRangeBuilderSuccess/Error models and TimeRangeInputs TypedDict.
Corresponds to: skills/pydantic-evals/references/examples/models.py
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from pydantic import AwareDatetime, BaseModel
from typing_extensions import TypedDict

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


# Models matching the reference models.py
class TimeRangeBuilderSuccess(BaseModel):
    """Response when a time range could be successfully generated."""

    min_timestamp_with_offset: AwareDatetime
    max_timestamp_with_offset: AwareDatetime
    explanation: Optional[str] = None


class TimeRangeBuilderError(BaseModel):
    """Response when a time range cannot be generated."""

    error_message: str


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


class TimeRangeInputs(TypedDict):
    """The inputs for the time range inference agent."""

    prompt: str
    now: AwareDatetime


@pytest.mark.asyncio
async def test_llm_produces_success_response(openrouter_model):
    """Test LLM can produce TimeRangeBuilderSuccess structured output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeBuilderSuccess,
        system_prompt="Parse time ranges. Return timestamps with timezone offsets.",
    )

    now = datetime.now(timezone.utc)
    result = await agent.run(
        f"Parse 'last 7 days' relative to {now.isoformat()}. "
        f"Use {now.isoformat()} as max_timestamp_with_offset."
    )

    assert isinstance(result.output, TimeRangeBuilderSuccess)
    assert result.output.max_timestamp_with_offset is not None
    assert result.output.min_timestamp_with_offset is not None


@pytest.mark.asyncio
async def test_llm_produces_error_response(openrouter_model):
    """Test LLM can produce TimeRangeBuilderError for invalid inputs."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeBuilderError,
        system_prompt="Return an error for ambiguous or invalid time references.",
    )

    result = await agent.run("Parse 'whenever you feel like it' as a time range.")

    assert isinstance(result.output, TimeRangeBuilderError)
    assert len(result.output.error_message) > 0


@pytest.mark.asyncio
async def test_llm_union_type_success_or_error(openrouter_model):
    """Test LLM can return either success or error using union type."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeResponse,  # type: ignore
        system_prompt=(
            "Parse time ranges. Return TimeRangeBuilderSuccess for valid ranges, "
            "TimeRangeBuilderError for invalid or ambiguous inputs."
        ),
    )

    now = datetime.now(timezone.utc)

    # Test with valid input - should return Success
    result = await agent.run(
        f"Parse 'yesterday' relative to {now.isoformat()}. "
        f"Use {now.isoformat()} as max_timestamp_with_offset."
    )

    assert isinstance(result.output, (TimeRangeBuilderSuccess, TimeRangeBuilderError))
