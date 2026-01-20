"""
Example: Conversation History (Persistent Memory)

This file demonstrates how to maintain conversation context across multiple
agent.run() calls using message_history parameter.
"""

from datetime import datetime, timezone
from dataclasses import dataclass

from pydantic_ai import Agent, ModelMessage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)


# =============================================================================
# BASIC PATTERN: Using message_history
# =============================================================================

def basic_conversation_example():
    """
    Basic pattern: Pass message_history to maintain context.

    By default, each agent.run() call is stateless. To maintain conversation
    context, capture result.all_messages() and pass it to the next call.
    """
    agent = Agent(model="openai:gpt-4o-mini", system_prompt="You are helpful.")

    # First turn - no history
    result1 = agent.run_sync("My name is Alice")
    messages: list[ModelMessage] = result1.all_messages()

    # Second turn - pass history so agent remembers
    result2 = agent.run_sync(
        "What is my name?",
        message_history=messages  # Agent now knows "Alice"
    )
    messages = result2.all_messages()  # Updated history

    # Third turn - continue the conversation
    result3 = agent.run_sync(
        "Tell me a joke about my name",
        message_history=messages
    )

    print(result3.output)


# =============================================================================
# FUNCTION SIGNATURE PATTERN: Return output + history
# =============================================================================

def run_agent_with_history(
    agent: Agent,
    user_input: str,
    message_history: list[ModelMessage] | None = None,
) -> tuple[str, list[ModelMessage]]:
    """
    Run agent and return output + updated history.

    This pattern is useful for building conversation loops where you need
    to maintain state across multiple function calls.
    """
    result = agent.run_sync(
        user_input,
        message_history=message_history or [],
    )
    return result.output, result.all_messages()


def conversation_loop_example():
    """Example of a conversation loop using the pattern above."""
    agent = Agent(model="openai:gpt-4o-mini", system_prompt="You are helpful.")

    history: list[ModelMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break

        response, history = run_agent_with_history(agent, user_input, history)
        print(f"Agent: {response}")


# =============================================================================
# CUSTOM MESSAGE CONVERSION: From your own format to ModelMessage
# =============================================================================

@dataclass
class MyMessage:
    """Example custom message format (e.g., from database)."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


def convert_to_model_messages(my_messages: list[MyMessage]) -> list[ModelMessage]:
    """
    Convert custom message format to Pydantic AI format.

    Use this when you store conversation history in your own format
    (e.g., database, custom objects) and need to convert it for the agent.

    IMPORTANT:
    - ModelMessage is a union type: ModelRequest | ModelResponse
    - User messages → ModelRequest with UserPromptPart
    - Assistant messages → ModelResponse with TextPart
    - Timestamps MUST be timezone-aware (use timezone.utc)
    """
    result: list[ModelMessage] = []

    for msg in my_messages:
        # Ensure timezone-aware timestamp
        ts = msg.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if msg.role == "user":
            result.append(ModelRequest(
                parts=[UserPromptPart(content=msg.content, timestamp=ts)],
                kind="request",
            ))
        elif msg.role == "assistant":
            result.append(ModelResponse(
                parts=[TextPart(content=msg.content)],
                kind="response",
                timestamp=ts,
            ))

    return result


def custom_message_example():
    """Example using custom message conversion."""
    # Simulating messages loaded from database
    stored_messages = [
        MyMessage(role="user", content="Hi, I'm Bob", timestamp=datetime.now()),
        MyMessage(role="assistant", content="Hello Bob! How can I help?", timestamp=datetime.now()),
        MyMessage(role="user", content="What's my name?", timestamp=datetime.now()),
    ]

    # Convert to Pydantic AI format
    history = convert_to_model_messages(stored_messages)

    # Continue conversation with the agent
    agent = Agent(model="openai:gpt-4o-mini", system_prompt="You are helpful.")
    result = agent.run_sync(
        "Remind me what my name is again?",
        message_history=history
    )

    print(result.output)  # Agent knows the user is Bob


# =============================================================================
# KEY NOTES
# =============================================================================

"""
Important Notes:

1. ModelMessage is a UNION TYPE (ModelRequest | ModelResponse), not a class
   you instantiate directly.

2. System prompt is NOT stored in message_history - it's set on the Agent
   and injected automatically on each run.

3. Messages are IMMUTABLE - each run() returns a NEW list containing all
   previous messages plus the new exchange.

4. For async usage, the pattern is identical:

   result = await agent.run(user_input, message_history=history)
   history = result.all_messages()

5. When mixing sync and async in tests, keep them in separate files to avoid
   httpx connection pool issues.
"""
