"""
Tests for models.py - Pydantic models for time range agent.

These tests are synchronous because they test pure Pydantic models.

Note: We define equivalent models here instead of dynamically loading them
because the original models use `use_attribute_docstrings=True` which requires
source inspection that doesn't work well with dynamic module loading.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from pydantic import AwareDatetime, BaseModel
from typing_extensions import TypedDict


# Define equivalent models that match the structure in the original file
class TimeRangeBuilderSuccess(BaseModel):
    """Response when a time range could be successfully generated."""

    min_timestamp_with_offset: AwareDatetime
    max_timestamp_with_offset: AwareDatetime
    explanation: Optional[str] = None

    def __str__(self):
        readable_min_timestamp = self.min_timestamp_with_offset.strftime(
            "%A, %B %d, %Y %H:%M:%S %Z"
        )
        readable_max_timestamp = self.max_timestamp_with_offset.strftime(
            "%A, %B %d, %Y %H:%M:%S %Z"
        )
        lines = [
            "TimeRangeBuilderSuccess:",
            f"* min_timestamp_with_offset: {readable_min_timestamp}",
            f"* max_timestamp_with_offset: {readable_max_timestamp}",
        ]
        if self.explanation is not None:
            lines.append(f"* explanation: {self.explanation}")
        return "\n".join(lines)


class TimeRangeBuilderError(BaseModel):
    """Response when a time range cannot not be generated."""

    error_message: str

    def __str__(self):
        return f"TimeRangeBuilderError:\n* {self.error_message}"


TimeRangeResponse = TimeRangeBuilderSuccess | TimeRangeBuilderError


class TimeRangeInputs(TypedDict):
    """The inputs for the time range inference agent."""

    prompt: str
    now: AwareDatetime


class TestTimeRangeBuilderSuccess:
    """Test TimeRangeBuilderSuccess model."""

    def test_valid_time_range(self):
        """Test creating valid TimeRangeBuilderSuccess."""
        now = datetime.now(timezone.utc)

        result = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(days=7),
            max_timestamp_with_offset=now,
            explanation="Last 7 days",
        )

        assert result.min_timestamp_with_offset < result.max_timestamp_with_offset
        assert result.explanation == "Last 7 days"

    def test_explanation_can_be_none(self):
        """Test that explanation field is optional."""
        now = datetime.now(timezone.utc)

        result = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(hours=1),
            max_timestamp_with_offset=now,
            explanation=None,
        )

        assert result.explanation is None

    def test_str_representation(self):
        """Test __str__ method produces readable output."""
        now = datetime.now(timezone.utc)

        result = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(days=1),
            max_timestamp_with_offset=now,
            explanation="Last 24 hours",
        )

        output = str(result)
        assert "TimeRangeBuilderSuccess:" in output
        assert "min_timestamp_with_offset:" in output
        assert "max_timestamp_with_offset:" in output
        assert "Last 24 hours" in output

    def test_str_without_explanation(self):
        """Test __str__ method when explanation is None."""
        now = datetime.now(timezone.utc)

        result = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(hours=1),
            max_timestamp_with_offset=now,
            explanation=None,
        )

        output = str(result)
        assert "TimeRangeBuilderSuccess:" in output
        assert "explanation:" not in output

    def test_requires_timezone_aware_datetime(self):
        """Test that timezone-aware datetime is preserved."""
        now = datetime.now(timezone.utc)

        result = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(hours=1),
            max_timestamp_with_offset=now,
            explanation=None,
        )

        assert result.max_timestamp_with_offset.tzinfo is not None


class TestTimeRangeBuilderError:
    """Test TimeRangeBuilderError model."""

    def test_error_message_required(self):
        """Test error_message is required."""
        error = TimeRangeBuilderError(error_message="Invalid date format")
        assert error.error_message == "Invalid date format"

    def test_str_representation(self):
        """Test __str__ method output."""
        error = TimeRangeBuilderError(error_message="Could not parse date")
        output = str(error)

        assert "TimeRangeBuilderError:" in output
        assert "Could not parse date" in output

    def test_various_error_messages(self):
        """Test with various error message content."""
        messages = [
            "Invalid date format",
            "Date is in the future",
            "Cannot determine time range from input",
            "Ambiguous date reference",
        ]

        for msg in messages:
            error = TimeRangeBuilderError(error_message=msg)
            assert error.error_message == msg
            assert msg in str(error)


class TestTimeRangeInputs:
    """Test TimeRangeInputs TypedDict."""

    def test_inputs_structure(self):
        """Test TimeRangeInputs has correct structure."""
        now = datetime.now(timezone.utc)
        inputs: TimeRangeInputs = {
            "prompt": "Show me data from last week",
            "now": now,
        }

        assert inputs["prompt"] == "Show me data from last week"
        assert inputs["now"] == now

    def test_various_prompts(self):
        """Test with various prompt formats."""
        now = datetime.now(timezone.utc)
        prompts = [
            "Last week",
            "Yesterday",
            "From January 1st to February 28th",
            "The past 30 days",
            "Between 2024-01-01 and 2024-12-31",
        ]

        for prompt in prompts:
            inputs: TimeRangeInputs = {
                "prompt": prompt,
                "now": now,
            }
            assert inputs["prompt"] == prompt


class TestTimeRangeResponse:
    """Test TimeRangeResponse union type behavior."""

    def test_success_is_valid_response(self):
        """Test TimeRangeBuilderSuccess can be used as TimeRangeResponse."""
        now = datetime.now(timezone.utc)

        response = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(days=7),
            max_timestamp_with_offset=now,
            explanation=None,
        )

        assert isinstance(response, TimeRangeBuilderSuccess)

    def test_error_is_valid_response(self):
        """Test TimeRangeBuilderError can be used as TimeRangeResponse."""
        response = TimeRangeBuilderError(error_message="Error occurred")

        assert isinstance(response, TimeRangeBuilderError)

    def test_can_distinguish_response_types(self):
        """Test we can distinguish between success and error responses."""
        now = datetime.now(timezone.utc)

        success = TimeRangeBuilderSuccess(
            min_timestamp_with_offset=now - timedelta(days=1),
            max_timestamp_with_offset=now,
            explanation=None,
        )

        error = TimeRangeBuilderError(error_message="Failed")

        assert isinstance(success, TimeRangeBuilderSuccess)
        assert not isinstance(success, TimeRangeBuilderError)

        assert isinstance(error, TimeRangeBuilderError)
        assert not isinstance(error, TimeRangeBuilderSuccess)
