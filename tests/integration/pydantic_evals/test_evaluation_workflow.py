"""
Integration tests for evaluation workflow patterns from pydantic-evals.

Tests the complete evaluation workflow including assertions.
Corresponds to: skills/pydantic-evals/references/examples/unit_testing.py
"""

import os
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


# Models
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
async def test_evaluation_with_assertions(openrouter_model):
    """Test evaluation workflow with assertions on results."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import IsInstance

    now = datetime.now(timezone.utc)

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely.",
    )

    # Create dataset with evaluators
    cases = [
        Case(
            name="last_week",
            inputs=TimeRangeInputs(prompt="last 7 days", now=now),
            evaluators=[IsInstance(TimeRangeSuccess)],
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}. "
            f"Use {inputs['now'].isoformat()} as max_timestamp."
        )
        return result.output

    # Run evaluation
    report = await dataset.evaluate(parse_time_range)

    # Check report
    assert report is not None
    assert len(report.cases) == 1

    # The IsInstance evaluator should pass
    case_report = report.cases[0]
    assert case_report.output is not None


@pytest.mark.asyncio
async def test_evaluation_report_averages(openrouter_model):
    """Test accessing evaluation report averages."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Equals, IsInstance

    now = datetime.now(timezone.utc)

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Always include an explanation.",
    )

    # Multiple test cases
    cases = [
        Case(
            name="case_1",
            inputs=TimeRangeInputs(prompt="yesterday", now=now),
            evaluators=[IsInstance(TimeRangeSuccess)],
        ),
        Case(
            name="case_2",
            inputs=TimeRangeInputs(prompt="last 3 days", now=now),
            evaluators=[IsInstance(TimeRangeSuccess)],
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}. "
            f"Include a brief explanation."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    # Access averages
    averages = report.averages()
    assert averages is not None


@pytest.mark.asyncio
async def test_dataset_case_structure(openrouter_model):
    """Test dataset case structure and access patterns."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges.",
    )

    cases = [
        Case(
            name="simple_case",
            inputs=TimeRangeInputs(prompt="today", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Test dataset structure
    assert len(dataset.cases) == 1
    assert dataset.cases[0].name == "simple_case"
    assert dataset.cases[0].inputs["prompt"] == "today"

    # Test async evaluation (avoid sync evaluation in async context)
    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 1
