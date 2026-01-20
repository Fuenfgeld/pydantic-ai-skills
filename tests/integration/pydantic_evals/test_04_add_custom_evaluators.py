"""
Integration tests for 04_add_custom_evaluators.py - Adding Evaluators to Dataset.

Tests the add_evaluator() method and dataset-wide evaluator patterns.
Corresponds to: skills/pydantic-evals/references/examples/add_custom_evaluators.py
"""

import os
from dataclasses import dataclass
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
async def test_add_evaluator_to_dataset(openrouter_model):
    """Test add_evaluator() pattern from add_custom_evaluators.py."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

    now = datetime.now(timezone.utc)

    @dataclass
    class ValidateTimeRange(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess):
                window_size = ctx.output.max_timestamp - ctx.output.min_timestamp
                return {"window_is_valid": window_size <= timedelta(days=30)}
            return {}

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely.",
    )

    cases = [
        Case(
            name="last_week",
            inputs=TimeRangeInputs(prompt="last week", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Add evaluator to entire dataset (like add_custom_evaluators.py does)
    dataset.add_evaluator(ValidateTimeRange())

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}. "
            f"Use {inputs['now'].isoformat()} as max_timestamp."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 1


@pytest.mark.asyncio
async def test_add_evaluator_to_specific_case(openrouter_model):
    """Test add_evaluator() with specific_case parameter."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

    now = datetime.now(timezone.utc)

    @dataclass
    class StrictValidation(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess):
                return {"has_explanation": ctx.output.explanation is not None}
            return {}

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Always include an explanation.",
    )

    cases = [
        Case(
            name="case_with_validation",
            inputs=TimeRangeInputs(prompt="yesterday", now=now),
        ),
        Case(
            name="case_without_validation",
            inputs=TimeRangeInputs(prompt="last 3 days", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Add evaluator only to specific case (like add_custom_evaluators.py pattern)
    dataset.add_evaluator(StrictValidation(), specific_case="case_with_validation")

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 2


@pytest.mark.asyncio
async def test_multiple_evaluators_on_dataset(openrouter_model):
    """Test adding multiple evaluators to dataset."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

    now = datetime.now(timezone.utc)

    @dataclass
    class ValidateTimeRange(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess):
                window_size = ctx.output.max_timestamp - ctx.output.min_timestamp
                return {"window_valid": window_size <= timedelta(days=30)}
            return {}

    @dataclass
    class ConciseMessage(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess) and ctx.output.explanation:
                return {"is_concise": len(ctx.output.explanation.split()) < 50}
            return {}

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges with brief explanations.",
    )

    cases = [
        Case(
            name="test_case",
            inputs=TimeRangeInputs(prompt="last 24 hours", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Add multiple evaluators like in add_custom_evaluators.py
    dataset.add_evaluator(ValidateTimeRange())
    dataset.add_evaluator(ConciseMessage())

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 1
