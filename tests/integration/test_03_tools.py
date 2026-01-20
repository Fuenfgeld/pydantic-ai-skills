"""
Integration tests for 03_tools.py - Tool Calling patterns.

Tests tool calling with real LLM calls.
"""

import os
from dataclasses import dataclass

import pytest

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


@pytest.mark.asyncio
async def test_tool_calling(openrouter_model):
    """Test tool calling with real LLM."""
    from pydantic_ai import Agent, RunContext

    @dataclass
    class CalculatorDeps:
        precision: int = 2

    agent = Agent(
        model=openrouter_model,
        deps_type=CalculatorDeps,
        system_prompt="Use the calculator tool for all math. Reply with just the result.",
    )

    tool_was_called = False

    @agent.tool
    def calculate(ctx: RunContext[CalculatorDeps], expression: str) -> str:
        """Evaluate a math expression."""
        nonlocal tool_was_called
        tool_was_called = True
        try:
            result = eval(expression)
            return f"{result:.{ctx.deps.precision}f}"
        except Exception:
            return "Error"

    deps = CalculatorDeps(precision=2)
    result = await agent.run("What is 100 divided by 3?", deps=deps)

    assert tool_was_called, "Tool should have been called"
    assert "33" in result.output


@pytest.mark.asyncio
async def test_multiple_tools(openrouter_model):
    """Test agent with multiple tools."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="Use the appropriate tool for each task. Be brief.",
    )

    tools_called = set()

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        tools_called.add("weather")
        return f"Weather in {city}: 22°C, sunny"

    @agent.tool_plain
    def get_time(timezone: str) -> str:
        """Get the current time in a timezone."""
        tools_called.add("time")
        return f"Current time in {timezone}: 14:30"

    result = await agent.run("What's the weather in Berlin and time in CET?")

    assert "weather" in tools_called, "Weather tool should be called"
    assert "time" in tools_called, "Time tool should be called"
    assert "22" in result.output or "Berlin" in result.output
