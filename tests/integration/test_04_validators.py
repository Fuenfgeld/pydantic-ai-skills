"""
Integration tests for 04_validators.py - Structured Output Validation.

Tests Pydantic model validation on LLM outputs with real API calls.
"""

import os

import pytest
from pydantic import BaseModel, Field

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


class MathResult(BaseModel):
    """Structured output for math operations."""

    answer: int = Field(description="The numerical answer")
    explanation: str = Field(description="Brief explanation of the calculation")


@pytest.mark.asyncio
async def test_structured_output(openrouter_model):
    """Test structured output validation with real LLM."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=MathResult,
        system_prompt="You are a math assistant. Always show your work briefly.",
    )

    result = await agent.run("What is 15 + 27?")

    assert isinstance(result.output, MathResult)
    assert result.output.answer == 42


class Address(BaseModel):
    """Address component."""

    street: str
    city: str
    country: str


class Person(BaseModel):
    """Person with nested address."""

    name: str
    age: int
    address: Address


@pytest.mark.asyncio
async def test_nested_structured_output(openrouter_model):
    """Test complex nested Pydantic models."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=Person,
        system_prompt="Extract person information from text.",
    )

    result = await agent.run(
        "John Smith is 30 years old and lives at 123 Main St, New York, USA."
    )

    assert isinstance(result.output, Person)
    assert result.output.name == "John Smith"
    assert result.output.age == 30
    assert isinstance(result.output.address, Address)
    assert result.output.address.city == "New York"


class SuccessResponse(BaseModel):
    """Successful operation response."""

    result: str
    confidence: float


class ErrorResponse(BaseModel):
    """Error response."""

    error_message: str
    error_code: str


@pytest.mark.asyncio
async def test_union_output_types(openrouter_model):
    """Test union types for success/error responses."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=SuccessResponse | ErrorResponse,  # type: ignore
        system_prompt="Analyze queries. Return SuccessResponse for valid queries, ErrorResponse for invalid ones.",
    )

    result = await agent.run("Analyze: The sky is blue.")

    assert isinstance(result.output, (SuccessResponse, ErrorResponse))
