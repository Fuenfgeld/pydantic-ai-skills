"""
Integration tests for model comparison patterns from pydantic-evals.

Tests comparing different models/configurations on the same dataset.
Corresponds to: skills/pydantic-evals/references/examples/compare_models.py
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


class TimeRangeSuccess(BaseModel):
    """Successful time range response."""

    min_timestamp: AwareDatetime
    max_timestamp: AwareDatetime
    explanation: Optional[str] = None


class TimeRangeInputs(TypedDict):
    """Inputs for time range agent."""

    prompt: str
    now: AwareDatetime


@pytest.mark.asyncio
async def test_compare_model_settings(openrouter_model):
    """Test comparing different model settings on same dataset."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    # Create base agent
    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely.",
    )

    # Create dataset
    cases = [
        Case(
            name="test_case",
            inputs=TimeRangeInputs(prompt="last week", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Evaluate with low temperature
    async def parse_low_temp(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}.",
            model_settings={"temperature": 0.0},
        )
        return result.output

    # Evaluate with higher temperature
    async def parse_high_temp(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}.",
            model_settings={"temperature": 0.7},
        )
        return result.output

    # Run both evaluations
    report_low = await dataset.evaluate(parse_low_temp, name="low_temp")
    report_high = await dataset.evaluate(parse_high_temp, name="high_temp")

    # Both should produce valid outputs
    assert report_low.cases[0].output is not None
    assert report_high.cases[0].output is not None


@pytest.mark.asyncio
async def test_compare_system_prompts(openrouter_model):
    """Test comparing different system prompts on same dataset."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    # Create dataset
    cases = [
        Case(
            name="ambiguous_input",
            inputs=TimeRangeInputs(prompt="recently", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Agent with strict interpretation
    strict_agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges strictly. 'Recently' means last 24 hours.",
    )

    # Agent with loose interpretation
    loose_agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges loosely. 'Recently' means last 7 days.",
    )

    async def parse_strict(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await strict_agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    async def parse_loose(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await loose_agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    # Compare results
    report_strict = await dataset.evaluate(parse_strict, name="strict")
    report_loose = await dataset.evaluate(parse_loose, name="loose")

    strict_output = report_strict.cases[0].output
    loose_output = report_loose.cases[0].output

    assert strict_output is not None
    assert loose_output is not None

    # The time ranges should be different due to different interpretations
    strict_delta = strict_output.max_timestamp - strict_output.min_timestamp
    loose_delta = loose_output.max_timestamp - loose_output.min_timestamp

    # We can't assert exact values, but both should be valid
    assert strict_delta > timedelta(0)
    assert loose_delta > timedelta(0)
