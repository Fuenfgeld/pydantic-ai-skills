"""OpenRouter Integration with Pydantic AI (v2)

OpenRouter is an API gateway that provides access to multiple LLM models through
a unified API. Since Pydantic AI v2, OpenRouter has FIRST-CLASS support via
OpenRouterModel and OpenRouterProvider - no more OpenAI-compatible workaround.

Prerequisites:
- Set OPENROUTER_API_KEY in your .env file
- Install required packages: pydantic-ai, python-dotenv
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# OPTION 1 (SIMPLEST): the 'openrouter:' model prefix
# ============================================================================
# With OPENROUTER_API_KEY set in the environment, just prefix the OpenRouter
# model id (always '<vendor>/<model>', see https://openrouter.ai/models):
#
#   agent = Agent('openrouter:anthropic/claude-sonnet-4.6')


# ============================================================================
# OPTION 2 (EXPLICIT): OpenRouterModel + OpenRouterProvider
# ============================================================================
# Use this when you want to pass the API key explicitly or configure
# app attribution / model settings.

model = OpenRouterModel(
    "openai/gpt-5.2",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key")),
)


# Optional: Define a Pydantic model for structured output
class Answer(BaseModel):
    """Structured response from the AI"""

    response: str


# Create the agent with the OpenRouter model
agent = Agent(
    model=model,
    name="openrouter_agent",
    output_type=Answer,  # Use 'str' for plain text output
    instructions="You are a helpful assistant.",
)


# ============================================================================
# OPTION 3 (ADVANCED): OpenRouter-specific model settings
# ============================================================================
# OpenRouterModelSettings exposes provider routing, reasoning effort, and
# usage accounting:
#
#   from pydantic_ai.models.openrouter import OpenRouterModelSettings
#
#   settings = OpenRouterModelSettings(
#       openrouter_reasoning={'effort': 'high'},
#       openrouter_usage={'include': True},
#   )
#   agent = Agent(model, model_settings=settings)


# Example usage
if __name__ == "__main__":
    # Synchronous usage
    result = agent.run_sync("What's the capital of France?")
    print(result.output.response)  # Access structured output

    # For plain text output (output_type=str):
    # print(result.output)  # Always use result.output, NOT result.data
