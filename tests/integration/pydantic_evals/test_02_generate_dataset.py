"""
Integration tests for 02_generate_dataset.py - Dataset Generation.

Tests Dataset creation and structure patterns.
Corresponds to: skills/pydantic-evals/references/examples/generate_dataset.py
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


class TimeRangeError(BaseModel):
    """Error response."""

    error_message: str


class TimeRangeInputs(TypedDict):
    """Inputs for time range agent."""

    prompt: str
    now: AwareDatetime


@pytest.mark.asyncio
async def test_create_dataset_with_cases():
    """Test creating a Dataset with test cases - mirrors generate_dataset pattern."""
    from pydantic_evals import Case, Dataset

    now = datetime.now(timezone.utc)

    # Create test cases like generate_dataset.py produces
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

    dataset = Dataset[TimeRangeInputs, TimeRangeSuccess | TimeRangeError, None](
        cases=cases
    )

    assert len(dataset.cases) == 2
    assert dataset.cases[0].name == "last_week"
    assert dataset.cases[1].name == "invalid_input"


@pytest.mark.asyncio
async def test_dataset_serialization():
    """Test Dataset can be serialized - mirrors to_file pattern."""
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

    # Test model serialization (JSON round-trip like to_file would do)
    json_data = dataset.model_dump_json()
    assert "yesterday" in json_data

    # Verify structure
    data = dataset.model_dump()
    assert len(data["cases"]) == 1
    assert data["cases"][0]["name"] == "yesterday"
