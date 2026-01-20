"""
Integration tests for 03_custom_evaluators.py - Custom Evaluator Classes.

Tests custom evaluator patterns: ValidateTimeRange, UserMessageIsConcise.
Corresponds to: skills/pydantic-evals/references/examples/custom_evaluators.py
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
async def test_validate_time_range_evaluator_pattern(openrouter_model):
    """Test ValidateTimeRange evaluator pattern from custom_evaluators.py."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

    now = datetime.now(timezone.utc)

    # Custom evaluator matching the reference pattern
    @dataclass
    class ValidateTimeRange(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess):
                window_end = ctx.output.max_timestamp
                window_size = window_end - ctx.output.min_timestamp
                return {
                    "window_is_not_too_long": window_size <= timedelta(days=30),
                    "window_is_not_in_the_future": window_end <= ctx.inputs["now"],
                }
            return {}

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges precisely. Never return future dates.",
    )

    cases = [
        Case(
            name="last_week",
            inputs=TimeRangeInputs(prompt="last 7 days", now=now),
            evaluators=[ValidateTimeRange()],
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

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
async def test_user_message_concise_evaluator_pattern(openrouter_model):
    """Test UserMessageIsConcise evaluator pattern from custom_evaluators.py."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput

    now = datetime.now(timezone.utc)

    # Custom evaluator matching the reference pattern
    @dataclass
    class UserMessageIsConcise(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        async def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess):
                user_facing_message = ctx.output.explanation
                if user_facing_message is not None:
                    return len(user_facing_message.split()) < 50
            return {}

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Include a brief explanation (under 50 words).",
    )

    cases = [
        Case(
            name="yesterday",
            inputs=TimeRangeInputs(prompt="yesterday", now=now),
            evaluators=[UserMessageIsConcise()],
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

    assert report is not None
    assert len(report.cases) == 1


@pytest.mark.asyncio
async def test_multiple_evaluators_combined(openrouter_model):
    """Test combining multiple custom evaluators like in custom_evaluators.py."""
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
                return {"window_is_not_too_long": window_size <= timedelta(days=30)}
            return {}

    @dataclass
    class UserMessageIsConcise(Evaluator[TimeRangeInputs, TimeRangeSuccess]):
        def evaluate(
            self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeSuccess]
        ) -> EvaluatorOutput:
            if isinstance(ctx.output, TimeRangeSuccess) and ctx.output.explanation:
                return len(ctx.output.explanation.split()) < 50
            return {}

    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Include brief explanations.",
    )

    cases = [
        Case(
            name="last_3_days",
            inputs=TimeRangeInputs(prompt="last 3 days", now=now),
            evaluators=[ValidateTimeRange(), UserMessageIsConcise()],
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}."
        )
        return result.output

    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 1
