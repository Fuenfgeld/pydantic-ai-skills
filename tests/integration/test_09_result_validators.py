"""
Integration tests for 09_result_validators.py - Result Validation.

Tests Pydantic field validators on LLM outputs with real API calls.
"""

import os

import pytest
from pydantic import BaseModel, Field, field_validator

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


class SentimentResult(BaseModel):
    """Sentiment analysis result with validation."""

    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        allowed = {"positive", "negative", "neutral"}
        if v.lower() not in allowed:
            raise ValueError(f"sentiment must be one of {allowed}")
        return v.lower()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


@pytest.mark.asyncio
async def test_result_validators(openrouter_model):
    """Test that Pydantic validators run on LLM output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=SentimentResult,
        system_prompt="Analyze the sentiment of the text. Be precise with confidence scores.",
    )

    result = await agent.run("I absolutely love this product! Best purchase ever!")

    assert isinstance(result.output, SentimentResult)
    assert result.output.sentiment in {"positive", "negative", "neutral"}
    assert 0 <= result.output.confidence <= 1
