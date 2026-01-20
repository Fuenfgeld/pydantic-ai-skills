"""OpenRouter Integration with Pydantic AI

OpenRouter is an API gateway that provides access to multiple LLM models through a unified API.
This example shows how to configure a Pydantic AI agent to use OpenRouter models.

Prerequisites:
- Set OPENROUTER_API_KEY in your .env file
- Install required packages: pydantic-ai, python-dotenv
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Load environment variables from .env file
load_dotenv()

# Get your OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1'

# Define the provider for OpenRouter
# OpenRouter uses the same API format as OpenAI, so we use OpenAIProvider
provider = OpenAIProvider(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_URL
)

# Choose your OpenRouter model
# Available models: gpt-4o-mini, claude-3-5-sonnet, llama-3.1-70b, etc.
# See https://openrouter.ai/models for full list
model = OpenAIChatModel(
    model_name='gpt-4o-mini',
    provider=provider
)

# Optional: Define a Pydantic model for structured output
class Answer(BaseModel):
    """Structured response from the AI"""
    response: str

# Create the agent with OpenRouter model
agent = Agent(
    model=model,
    output_type=Answer,  # Use 'str' for plain text output
    system_prompt="You are a helpful assistant."
)

# Example usage
if __name__ == "__main__":
    # Synchronous usage
    result = agent.run_sync("What's the capital of France?")
    print(result.output.response)  # Access structured output

    # For plain text output (output_type=str):
    # print(result.output)  # Always use result.output, NOT result.data
