"""
Integration tests for Dataset operations from pydantic-evals.

Tests Dataset creation, manipulation, and serialization patterns.
Corresponds to: skills/pydantic-evals/references/examples/add_custom_evaluators.py
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


# Models for the dataset
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


@pytest.mark.asyncio
async def test_create_dataset_with_cases():
    """Test creating a Dataset with test cases."""
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    # Create test cases
    cases = [
        Case(
            name="last_week",
            inputs=TimeRangeInputs(prompt="last week", now=now),
            expected_output=TimeRangeSuccess(
                min_timestamp=now - timedelta(days=7),
                max_timestamp=now,
                explanation="The past 7 days",
            ),
        ),
        Case(
            name="invalid_input",
            inputs=TimeRangeInputs(prompt="whenever", now=now),
            expected_output=TimeRangeError(error_message="Ambiguous time reference"),
        ),
    ]

    # Create dataset
    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess | TimeRangeError, None](
        cases=cases
    )

    assert len(dataset.cases) == 2
    assert dataset.cases[0].name == "last_week"
    assert dataset.cases[1].name == "invalid_input"


@pytest.mark.asyncio
async def test_dataset_serialization():
    """Test Dataset can be serialized to JSON and round-tripped."""
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    cases = [
        Case(
            name="yesterday",
            inputs=TimeRangeInputs(prompt="yesterday", now=now),
            expected_output=TimeRangeSuccess(
                min_timestamp=now - timedelta(days=1),
                max_timestamp=now,
                explanation="24 hours ago",
            ),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess | TimeRangeError, None](
        cases=cases
    )

    # Test model serialization (JSON round-trip)
    json_data = dataset.model_dump_json()
    assert "yesterday" in json_data

    # Verify we can reconstruct from dict
    data = dataset.model_dump()
    assert len(data["cases"]) == 1
    assert data["cases"][0]["name"] == "yesterday"


@pytest.mark.asyncio
async def test_evaluate_dataset_with_agent(openrouter_model):
    """Test running evaluation on a dataset with a real agent."""
    from pydantic_ai import Agent
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    # Create a simple agent for the test
    agent = Agent(
        model=openrouter_model,
        output_type=TimeRangeSuccess,
        system_prompt="Parse time ranges. Be precise with timestamps.",
    )

    # Create a minimal dataset
    cases = [
        Case(
            name="test_case",
            inputs=TimeRangeInputs(prompt="last 24 hours", now=now),
        ),
    ]

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess, None](cases=cases)

    # Define the task function
    async def parse_time_range(inputs: TimeRangeInputs) -> TimeRangeSuccess:
        result = await agent.run(
            f"Parse '{inputs['prompt']}' relative to {inputs['now'].isoformat()}. "
            f"Use {inputs['now'].isoformat()} as max_timestamp."
        )
        return result.output

    # Run evaluation
    report = await dataset.evaluate(parse_time_range)

    assert report is not None
    assert len(report.cases) == 1
