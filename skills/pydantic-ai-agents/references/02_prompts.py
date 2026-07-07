"""
Example: Dynamic System Prompts

This file demonstrates the Golden Standard for creating dynamic, context-aware
system prompts in Pydantic AI agents.
"""

from dataclasses import dataclass

import httpx
from pydantic_ai import Agent, RunContext


# Example dependency container (would import from 01_dependencies.py in real code)
@dataclass
class AgentDeps:
    user_id: str
    api_key: str
    http_client: httpx.AsyncClient
    environment: str = "production"

    def is_dev(self) -> bool:
        return self.environment == "development"


# Create the agent with dependency type
# NOTE: In v2, 'openai:' uses the OpenAI Responses API. Use 'openai-chat:' for
# the Chat Completions API. See 06_openrouter.py for the OpenRouter pattern.
agent = Agent("openai:gpt-5.2", deps_type=AgentDeps)


# v2 RECOMMENDED: @agent.instructions - instructions are NOT retained in message
# history, so they always reflect the CURRENT agent when history is reused.
# (@agent.system_prompt still exists; its text is carried inside the message
# history, which matters when replaying history across different agents.)
@agent.instructions
async def generate_instructions(ctx: RunContext[AgentDeps]) -> str:
    """
    Builds the instructions dynamically at runtime.

    Key Principles:
    - Access dependencies via ctx.deps
    - Build context-aware prompts that reference available data
    - Include role definition, operational constraints, and tool usage guidance
    - Define failure modes and edge case handling
    """
    # Access dependencies to customize the prompt
    user_segment = "Developer" if ctx.deps.is_dev() else "Standard User"

    return f"""
    You are a helpful assistant serving a user in the '{user_segment}' segment.
    User ID: {ctx.deps.user_id}

    Your operational constraints:
    1. Answer concisely and professionally.
    2. Since the environment is {ctx.deps.environment}, be cautious with write operations.
    3. Always verify user permissions before executing sensitive actions.

    When you encounter errors:
    - If a tool fails, explain the issue and suggest alternatives.
    - If data is missing, ask the user for clarification.
    - Never make assumptions about user intent.
    """


# Alternative: Static instructions (for simple cases) - or pass
# instructions="..." directly to the Agent constructor.
@agent.instructions
def simple_instructions() -> str:
    """Use this pattern when you don't need dynamic context."""
    return "You are a helpful assistant. Always be concise and accurate."


# Best Practices for Instructions / System Prompts:
# 1. Role Definition: Start with "You are a specialized agent for..."
# 2. Context Awareness: Reference data from ctx.deps
#    Bad: "I help users."
#    Good: "I help user {ctx.deps.user_name} (ID: {ctx.deps.user_id})"
# 3. Tool Coercion: Tell the model when to use tools
#    "Use lookup_order immediately if the user provides an order ID."
# 4. Failure Modes: Define error handling
#    "If the database returns no results, ask for clarification."
