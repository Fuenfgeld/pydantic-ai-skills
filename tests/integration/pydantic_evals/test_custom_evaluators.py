"""
Integration tests for custom_evaluators.py - Custom Evaluator patterns.

Tests the custom evaluator patterns from pydantic-evals.
Corresponds to: skills/pydantic-evals/references/examples/custom_evaluators.py
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from pydantic import AwareDatetime, BaseModel, Field
from typing_extensions import TypedDict

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


# Models for evaluation
class TimeRangeSuccess(BaseModel):
    """Successful time range response."""

    min_timestamp: AwareDatetime
    max_timestamp: AwareDatetime
    explanation: Optional[str] = None


class TimeRangeError(BaseModel):
    """Error response."""

    error_message: str


class TimeRangeInputs(TypedDict):
    """Inputs for time range agent."""

    prompt: str
    now: AwareDatetime


# Custom evaluator patterns (simplified versions of the reference patterns)
@dataclass
class ValidateTimeRangeEvaluator:
    """Evaluator that validates time range constraints."""

    max_window_days: int = 30

    def evaluate(self, output: TimeRangeSuccess | TimeRangeError, inputs: TimeRangeInputs) -> dict:
        """Evaluate the output against constraints."""
        if isinstance(output, TimeRangeSuccess):
            window_size = output.max_timestamp - output.min_timestamp
            return {
                "window_not_too_long": window_size <= timedelta(days=self.max_window_days),
                "window_not_in_future": output.max_timestamp <= inputs["now"],
                "valid_range": output.min_timestamp < output.max_timestamp,
            }
        return {"is_error": True}


@dataclass
class MessageConciseEvaluator:
    """Evaluator that checks if messages are concise."""

    max_words: int = 50

    def evaluate(self, output: TimeRangeSuccess | TimeRangeError) -> dict:
        """Check if the message is concise."""
        if isinstance(output, TimeRangeSuccess):
            message = output.explanation or ""
        else:
            message = output.error_message

        word_count = len(message.split())
        return {
            "is_concise": word_count < self.max_words,
            "word_count": word_count,
        }


@pytest.mark.asyncio
async def test_validate_time_range_evaluator(openrouter_model):
    """Test ValidateTimeRange evaluator pattern with real LLM output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Be precise with timestamps.",
    )

    now = datetime.now(timezone.utc)
    result = await agent.run(
        f"Parse 'last week' relative to {now.isoformat()}. "
        f"Use {now.isoformat()} as max and 7 days before as min."
    )

    # Apply evaluator
    evaluator = ValidateTimeRangeEvaluator(max_window_days=30)
    inputs: TimeRangeInputs = {"prompt": "last week", "now": now}
    evaluation = evaluator.evaluate(result.output, inputs)

    assert "window_not_too_long" in evaluation
    assert "window_not_in_future" in evaluation
    assert "valid_range" in evaluation
    assert evaluation["valid_range"] is True


@pytest.mark.asyncio
async def test_message_concise_evaluator(openrouter_model):
    """Test MessageConcise evaluator pattern with real LLM output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Keep explanations brief (under 20 words).",
    )

    now = datetime.now(timezone.utc)
    result = await agent.run(
        f"Parse 'yesterday' relative to {now.isoformat()}. Include a brief explanation."
    )

    # Apply evaluator
    evaluator = MessageConciseEvaluator(max_words=50)
    evaluation = evaluator.evaluate(result.output)

    assert "is_concise" in evaluation
    assert "word_count" in evaluation
    assert isinstance(evaluation["word_count"], int)


@pytest.mark.asyncio
async def test_multiple_evaluators_combined(openrouter_model):
    """Test applying multiple evaluators to the same output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely. Keep explanations under 30 words.",
    )

    now = datetime.now(timezone.utc)
    result = await agent.run(
        f"Parse 'last 3 days' relative to {now.isoformat()}. "
        f"Use {now.isoformat()} as max_timestamp."
    )

    inputs: TimeRangeInputs = {"prompt": "last 3 days", "now": now}

    # Apply multiple evaluators
    range_eval = ValidateTimeRangeEvaluator(max_window_days=7)
    concise_eval = MessageConciseEvaluator(max_words=30)

    range_results = range_eval.evaluate(result.output, inputs)
    concise_results = concise_eval.evaluate(result.output)

    # Combine results
    all_results = {**range_results, **concise_results}

    assert "window_not_too_long" in all_results
    assert "is_concise" in all_results
    assert all_results["valid_range"] is True
