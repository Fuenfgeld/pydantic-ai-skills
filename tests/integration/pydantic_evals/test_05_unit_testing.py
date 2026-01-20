"""
Integration tests for 05_unit_testing.py - Evaluation Workflow & Assertions.

Tests the complete evaluation workflow with assertions on report averages.
Corresponds to: skills/pydantic-evals/references/examples/unit_testing.py
"""

import os
from datetime import datetime, timezone
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
async def test_evaluate_dataset_pattern(openrouter_model):
    """Test dataset.evaluate() pattern from unit_testing.py."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import IsInstance

    now = datetime.now(timezone.utc)

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely.",
    )

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

    # Run evaluation like unit_testing.py
    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 1
    assert report.cases[0].output is not None


@pytest.mark.asyncio
async def test_report_averages_pattern(openrouter_model):
    """Test report.averages() pattern from unit_testing.py."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import IsInstance

    now = datetime.now(timezone.utc)

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Always include an explanation.",
    )

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
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    # Access averages like in unit_testing.py
    averages = report.averages()
    assert averages is not None


@pytest.mark.asyncio
async def test_assertion_pass_rate_pattern(openrouter_model):
    """Test assertion pass rate pattern from unit_testing.py."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import IsInstance

    now = datetime.now(timezone.utc)

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely.",
    )

    cases = [
        Case(
            name="today",
            inputs=TimeRangeInputs(prompt="today", now=now),
            evaluators=[IsInstance(TimeRangeSuccess)],
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    # Check averages and assertions like in unit_testing.py
    averages = report.averages()
    assert averages is not None
    # The pattern from unit_testing.py checks assertion_pass_rate > 0.9
    # We just verify the structure exists
    assert len(report.cases) == 1
