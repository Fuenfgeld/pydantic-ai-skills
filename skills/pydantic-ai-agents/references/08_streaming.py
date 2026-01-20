"""
Example: Streaming Responses

This file demonstrates how to stream agent responses in real-time
for better user experience with long outputs.
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

# Simple agent for streaming demo
agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant. Provide detailed explanations."
)


async def stream_text_example():
    """
    Stream text responses in real-time.

    Key Points:
    - Use `agent.run_stream()` for streaming
    - Use `async with` context manager
    - Iterate with `async for chunk in response.stream_text()`
    - Get final result with `await response.get_output()`
    """
    print("Streaming response:")
    print("-" * 40)

    async with agent.run_stream("Explain how photosynthesis works in 3 sentences.") as response:
        # Stream text chunks as they arrive
        async for chunk in response.stream_text():
            print(chunk, end='', flush=True)

        print("\n" + "-" * 40)

        # Get the complete final result
        final_result = await response.get_output()
        print(f"Final result type: {type(final_result)}")

    return final_result


# Streaming with structured output
class ExplanationResponse(BaseModel):
    """Structured response for explanations."""
    topic: str
    explanation: str
    key_points: list[str]


structured_agent = Agent(
    model=model,
    output_type=ExplanationResponse,
    system_prompt="You explain topics clearly with key points."
)


async def stream_structured_example():
    """
    Stream responses even with structured output.

    Note: When using output_type, streaming still works but the final
    structured result is only available after streaming completes.
    """
    print("\nStreaming with structured output:")
    print("-" * 40)

    async with structured_agent.run_stream("Explain machine learning") as response:
        # Stream the text representation
        async for chunk in response.stream_text():
            print(chunk, end='', flush=True)

        print("\n" + "-" * 40)

        # Get the structured result
        final_result = await response.get_output()
        print(f"\nStructured Result:")
        print(f"  Topic: {final_result.topic}")
        print(f"  Key Points: {final_result.key_points}")

    return final_result


# Best Practices for Streaming:
# 1. Use streaming for long responses to improve perceived latency
# 2. Use `flush=True` in print() for immediate output
# 3. Remember that `get_output()` returns the complete result after streaming
# 4. Streaming works with both plain text and structured outputs
# 5. Use streaming for chat interfaces and real-time feedback


if __name__ == "__main__":
    print("=== Text Streaming Example ===")
    asyncio.run(stream_text_example())

    print("\n\n=== Structured Streaming Example ===")
    asyncio.run(stream_structured_example())
