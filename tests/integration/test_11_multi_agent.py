"""
Integration tests for 11_multi_agent.py - Multi-Agent Orchestration.

Tests sequential, parallel, and routing multi-agent patterns.
"""

import asyncio
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


class Keywords(BaseModel):
    """Keywords extracted from text."""

    keywords: list[str] = Field(description="Important keywords from text")


@pytest.mark.asyncio
async def test_multi_agent_sequential(openrouter_model):
    """Test sequential multi-agent orchestration."""
    from pydantic_ai import Agent

    # Agent 1: Extract keywords
    extractor = Agent(
        model=openrouter_model,
        output_type=Keywords,
        system_prompt="Extract 3 important keywords from the text.",
    )

    # Agent 2: Generate summary using keywords
    summarizer = Agent(
        model=openrouter_model,
        system_prompt="Write a one-sentence summary based on the keywords provided.",
    )

    # Sequential pipeline
    text = "Python is a programming language known for its simplicity and readability."

    # Step 1: Extract keywords
    keywords_result = await extractor.run(f"Extract keywords: {text}")
    keywords = keywords_result.output.keywords

    # Step 2: Summarize based on keywords
    summary_result = await summarizer.run(f"Summarize using these keywords: {keywords}")

    assert len(keywords) >= 1
    assert len(summary_result.output) > 10


@pytest.mark.asyncio
async def test_multi_agent_parallel(openrouter_model):
    """Test parallel multi-agent execution."""
    from pydantic_ai import Agent

    # Create multiple agents for different tasks
    summarizer = Agent(
        model=openrouter_model,
        system_prompt="Summarize in one sentence.",
    )

    sentiment_analyzer = Agent(
        model=openrouter_model,
        system_prompt="Reply with only: positive, negative, or neutral.",
    )

    text = "I love learning new programming languages!"

    # Run both agents in parallel
    summary_task = summarizer.run(f"Summarize: {text}")
    sentiment_task = sentiment_analyzer.run(f"Sentiment: {text}")

    results = await asyncio.gather(summary_task, sentiment_task)

    summary = results[0].output
    sentiment = results[1].output.strip().lower()

    assert len(summary) > 5
    assert any(word in sentiment for word in ["positive", "negative", "neutral"])


@pytest.mark.asyncio
async def test_agent_routing(openrouter_model):
    """Test agent routing based on intent classification."""
    from pydantic_ai import Agent

    # Intent classifier
    classifier = Agent(
        model=openrouter_model,
        system_prompt="Classify as 'math' or 'general'. Reply with just the word.",
    )

    # Specialized agents
    math_agent = Agent(
        model=openrouter_model,
        system_prompt="You are a math expert. Solve the problem.",
    )

    general_agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant.",
    )

    # Math query
    query = "What is 7 times 8?"

    # Classify intent
    intent_result = await classifier.run(query)
    intent = intent_result.output.strip().lower()

    # Route to appropriate agent
    if "math" in intent:
        agent = math_agent
    else:
        agent = general_agent

    answer_result = await agent.run(query)

    assert "56" in answer_result.output
