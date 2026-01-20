"""
Integration tests for 02_prompts.py - Dynamic System Prompts.

Tests dynamic system prompt generation with real LLM calls.
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
async def test_dynamic_system_prompt(openrouter_model):
    """Test dynamic system prompts based on context."""
    from pydantic_ai import Agent, RunContext

    @dataclass
    class LanguageContext:
        language: str

    agent = Agent(
        model=openrouter_model,
        deps_type=LanguageContext,
    )

    @agent.system_prompt
    def set_language(ctx: RunContext[LanguageContext]) -> str:
        return f"Always respond in {ctx.deps.language}. Keep it brief."

    # Test German response
    deps = LanguageContext(language="German")
    result = await agent.run("Say 'hello world'", deps=deps)

    # Should contain German words
    german_indicators = ["hallo", "welt", "grüß", "guten"]
    assert any(word in result.output.lower() for word in german_indicators)
