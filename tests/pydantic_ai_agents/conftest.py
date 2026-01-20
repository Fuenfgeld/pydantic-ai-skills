"""
Pydantic AI Agents test fixtures.

Provides fixtures specific to testing Pydantic AI agent patterns.
"""

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart


@pytest.fixture
def test_model():
    """Provide TestModel for simple agent testing."""
    return TestModel()


@pytest.fixture
def function_model_factory():
    """Factory for creating FunctionModel with custom logic."""

    def _create(response_func):
        return FunctionModel(response_func)

    return _create


@pytest.fixture
def simple_text_response_model():
    """FunctionModel that returns simple text responses."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Test response from mock model")])

    return FunctionModel(respond)
