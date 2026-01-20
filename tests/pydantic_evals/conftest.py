"""
Pydantic Evals test fixtures.

Provides fixtures specific to testing evaluation patterns.
"""

from datetime import datetime, timezone

import pytest


@pytest.fixture
def sample_time_range_inputs():
    """Standard inputs for TimeRangeInputs testing."""
    return {
        "prompt": "Show me data from last week",
        "now": datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def mock_evaluator_context_factory():
    """Factory for creating EvaluatorContext mocks."""
    from unittest.mock import MagicMock

    def _create(inputs=None, output=None, expected_output=None, duration=1.0):
        ctx = MagicMock()
        ctx.inputs = inputs or {}
        ctx.output = output
        ctx.expected_output = expected_output
        ctx.duration = duration
        ctx.metadata = {}
        ctx.span_tree = MagicMock()
        return ctx

    return _create
