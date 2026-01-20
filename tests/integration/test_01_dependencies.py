"""
Integration tests for 01_dependencies.py - Dependency Injection patterns.

Tests the dependency injection pattern with real LLM calls.
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


@dataclass
class UserContext:
    """Dependencies for user-aware agent."""

    user_id: str
    user_name: str
    is_premium: bool = False


@pytest.mark.asyncio
async def test_dependency_injection(openrouter_model):
    """Test dependency injection pattern with real LLM."""
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        model=openrouter_model,
        deps_type=UserContext,
        system_prompt="You are a helpful assistant. Be brief.",
    )

    @agent.system_prompt
    def add_user_context(ctx: RunContext[UserContext]) -> str:
        premium_status = "premium" if ctx.deps.is_premium else "free"
        return f"User: {ctx.deps.user_name} (ID: {ctx.deps.user_id}, {premium_status} tier)"

    deps = UserContext(user_id="123", user_name="Alice", is_premium=True)
    result = await agent.run("Say hello to me by name.", deps=deps)

    assert "Alice" in result.output
