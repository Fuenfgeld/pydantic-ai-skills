"""
Example: Model Settings and Configuration

This file demonstrates how to customize model parameters like
temperature, max_tokens, and other settings for fine-tuned control.
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

# Create OpenRouter provider
provider = OpenAIProvider(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)

model = OpenAIChatModel(
    model_name='openai/gpt-4o-mini',
    provider=provider
)

# Create a basic agent
agent = Agent(
    model=model,
    system_prompt="You are a creative assistant."
)


# ============================================================================
# 1. MODEL SETTINGS AT RUNTIME
# ============================================================================

async def demo_model_settings():
    """
    Pass model_settings to agent.run() for per-request customization.

    Common settings:
    - temperature: 0.0-2.0 (creativity/randomness)
    - max_tokens: Maximum response length
    - top_p: Nucleus sampling (alternative to temperature)
    """

    # Low temperature for deterministic, factual responses
    print("=== Low Temperature (Factual) ===")
    result = await agent.run(
        "What is 2 + 2?",
        model_settings={
            'temperature': 0.0,  # Deterministic
            'max_tokens': 50,
        }
    )
    print(f"Result: {result.output}")

    # High temperature for creative responses
    print("\n=== High Temperature (Creative) ===")
    result = await agent.run(
        "Write a haiku about coding",
        model_settings={
            'temperature': 0.9,  # More creative
            'max_tokens': 100,
        }
    )
    print(f"Result: {result.output}")

    # Using top_p instead of temperature
    print("\n=== Using top_p ===")
    result = await agent.run(
        "Generate a random fantasy character name",
        model_settings={
            'top_p': 0.95,  # Consider top 95% probability tokens
            'max_tokens': 20,
        }
    )
    print(f"Result: {result.output}")


# ============================================================================
# 2. DIFFERENT SETTINGS FOR DIFFERENT TASKS
# ============================================================================

class CodeAnalysis(BaseModel):
    """Structured code analysis."""
    quality_score: int
    issues: list[str]
    suggestions: list[str]


code_agent = Agent(
    model=model,
    output_type=CodeAnalysis,
    system_prompt="You are a code reviewer. Be precise and thorough."
)


class CreativeStory(BaseModel):
    """Creative story output."""
    title: str
    story: str
    genre: str


story_agent = Agent(
    model=model,
    output_type=CreativeStory,
    system_prompt="You are a creative storyteller."
)


async def task_specific_settings():
    """Different tasks require different temperature settings."""

    # Code analysis: Low temperature for consistency
    print("=== Code Analysis (Low Temp) ===")
    code_result = await code_agent.run(
        "Review: def add(a, b): return a + b",
        model_settings={
            'temperature': 0.1,  # Low for consistent analysis
            'max_tokens': 500,
        }
    )
    print(f"Quality: {code_result.output.quality_score}")
    print(f"Issues: {code_result.output.issues}")

    # Story writing: High temperature for creativity
    print("\n=== Story Writing (High Temp) ===")
    story_result = await story_agent.run(
        "Write a very short story about a robot learning to paint",
        model_settings={
            'temperature': 0.8,  # High for creativity
            'max_tokens': 300,
        }
    )
    print(f"Title: {story_result.output.title}")
    print(f"Story: {story_result.output.story[:100]}...")


# ============================================================================
# 3. USAGE TRACKING
# ============================================================================

async def track_usage():
    """
    Access token usage and cost information from results.

    The result object provides methods to track API usage.
    """
    result = await agent.run(
        "Explain Python list comprehensions in one sentence.",
        model_settings={'max_tokens': 100}
    )

    print("=== Usage Tracking ===")
    print(f"Response: {result.output}")

    # Access usage information
    usage = result.usage()
    print(f"\nUsage Info:")
    print(f"  Input tokens: {usage.input_tokens}")
    print(f"  Output tokens: {usage.output_tokens}")
    print(f"  Total tokens: {usage.total_tokens}")


# Best Practices for Model Settings:
# 1. Use temperature=0.0-0.3 for factual, deterministic tasks
# 2. Use temperature=0.7-1.0 for creative tasks
# 3. Set max_tokens to prevent runaway costs
# 4. Use top_p as alternative to temperature (don't use both)
# 5. Track usage to monitor costs
# 6. Consider per-task defaults based on use case


if __name__ == "__main__":
    print("Model Settings Demo\n" + "=" * 50)
    asyncio.run(demo_model_settings())

    print("\n\nTask-Specific Settings\n" + "=" * 50)
    asyncio.run(task_specific_settings())

    print("\n\nUsage Tracking\n" + "=" * 50)
    asyncio.run(track_usage())
