"""
Integration tests that use real API calls.

These tests are automatically skipped if OPENROUTER_API_KEY is not set.
Run with: uv run pytest tests/integration/ -v
"""

import os

import pytest


pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


@pytest.mark.asyncio
async def test_real_agent_call_openrouter(openrouter_api_key):
    """Test real API call with OpenRouter."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatModel(
        model_name="openai/gpt-4o-mini",
        provider=provider,
    )

    agent = Agent(model=model, system_prompt="Be brief. Reply with just numbers.")

    result = await agent.run("What is 2+2? Reply with just the number.")

    assert "4" in result.output


@pytest.mark.asyncio
async def test_real_structured_output(openrouter_api_key):
    """Test real API call with structured output."""
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    class MathResult(BaseModel):
        answer: int = Field(description="The numerical answer")
        explanation: str = Field(description="Brief explanation")

    provider = OpenAIProvider(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatModel(
        model_name="openai/gpt-4o-mini",
        provider=provider,
    )

    agent = Agent(
        model=model,
        output_type=MathResult,
        system_prompt="You are a math assistant.",
    )

    result = await agent.run("What is 15 + 27?")

    assert isinstance(result.output, MathResult)
    assert result.output.answer == 42


@pytest.mark.asyncio
async def test_real_tool_call(openrouter_api_key):
    """Test real API call with tool usage."""
    from dataclasses import dataclass
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    @dataclass
    class Deps:
        multiplier: int = 2

    provider = OpenAIProvider(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatModel(
        model_name="openai/gpt-4o-mini",
        provider=provider,
    )

    agent = Agent(
        model=model,
        deps_type=Deps,
        system_prompt="Use the multiply tool to answer questions.",
    )

    @agent.tool
    def multiply(ctx: RunContext[Deps], a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b * ctx.deps.multiplier

    deps = Deps(multiplier=1)
    result = await agent.run("What is 6 times 7?", deps=deps)

    assert "42" in result.output
