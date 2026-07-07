"""
Example: Streaming Responses

This file demonstrates how to stream agent responses in real-time
for better user experience with long outputs.
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

model = OpenRouterModel(
    "openai/gpt-5.2",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key")),
)

# Simple agent for streaming demo
agent = Agent(
    model=model,
    name="streaming_agent",
    instructions="You are a helpful assistant. Provide detailed explanations.",
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
            print(chunk, end="", flush=True)

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
    name="structured_streaming_agent",
    output_type=ExplanationResponse,
    instructions="You explain topics clearly with key points.",
)


async def stream_structured_example():
    """
    Stream responses even with structured output.

    Key Points (v2):
    - Use `response.stream_output()` to iterate progressively-validated
      partial output objects (renamed from `.stream()` in v1)
    - The final structured result is available via `await response.get_output()`
      after streaming completes
    """
    print("\nStreaming with structured output:")
    print("-" * 40)

    async with structured_agent.run_stream("Explain machine learning") as response:
        # Stream partial validated output as it builds up
        async for partial in response.stream_output():
            # `partial` is an ExplanationResponse with fields filled as available
            # (fields not yet streamed may be unset - print the whole object)
            print(partial)

        print("-" * 40)

        # Get the final structured result
        final_result = await response.get_output()
        print("\nStructured Result:")
        print(f"  Topic: {final_result.topic}")
        print(f"  Key Points: {final_result.key_points}")

    return final_result


# Best Practices for Streaming:
# 1. Use streaming for long responses to improve perceived latency
# 2. Use `flush=True` in print() for immediate output
# 3. Remember that `get_output()` returns the complete result after streaming
# 4. Streaming works with both plain text and structured outputs
#    (v2 renames: .stream() -> .stream_output(), .stream_structured() -> .stream_response())
# 5. Use streaming for chat interfaces and real-time feedback
# 6. For token-by-token events across a whole run (including tool calls), use
#    `async with agent.run_stream_events(...) as events` (v2: context manager only)


if __name__ == "__main__":
    print("=== Text Streaming Example ===")
    asyncio.run(stream_text_example())

    print("\n\n=== Structured Streaming Example ===")
    asyncio.run(stream_structured_example())
