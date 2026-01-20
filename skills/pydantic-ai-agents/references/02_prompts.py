"""
Example: Dynamic System Prompts

This file demonstrates the Golden Standard for creating dynamic, context-aware
system prompts in Pydantic AI agents.
"""

from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import httpx


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
# NOTE: For OpenRouter, use OpenAIProvider with model object instead of string format
# See 06_openrouter.py for the OpenRouter pattern
agent = Agent('openai:gpt-4o', deps_type=AgentDeps)  # String format works for direct OpenAI API


@agent.system_prompt
async def generate_system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """
    Builds the system prompt dynamically at runtime.

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


# Alternative: Static system prompt (for simple cases)
@agent.system_prompt
def simple_prompt() -> str:
    """Use this pattern when you don't need dynamic context."""
    return "You are a helpful assistant. Always be concise and accurate."


# Best Practices for System Prompts:
# 1. Role Definition: Start with "You are a specialized agent for..."
# 2. Context Awareness: Reference data from ctx.deps
#    Bad: "I help users."
#    Good: "I help user {ctx.deps.user_name} (ID: {ctx.deps.user_id})"
# 3. Tool Coercion: Tell the model when to use tools
#    "Use lookup_order immediately if the user provides an order ID."
# 4. Failure Modes: Define error handling
#    "If the database returns no results, ask for clarification."
