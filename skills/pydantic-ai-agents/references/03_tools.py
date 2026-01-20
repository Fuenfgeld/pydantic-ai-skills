"""
Example: Tools (Function Calling)

This file demonstrates the Golden Standard for creating tools that agents can call
to interact with external systems, databases, or APIs.
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


# NOTE: For OpenRouter, use OpenAIProvider with model object instead of string format
# See 06_openrouter.py for the OpenRouter pattern
agent = Agent('openai:gpt-4o', deps_type=AgentDeps)  # String format works for direct OpenAI API


@agent.tool
async def get_user_balance(ctx: RunContext[AgentDeps], currency: str) -> str:
    """
    Fetch the current balance for the user.

    Key Principles for Tools:
    - First parameter MUST be ctx: RunContext if you need access to dependencies
    - Provide clear docstrings - the model uses these to understand tool purpose
    - Use type hints for all parameters - helps the model call correctly
    - Return strings or JSON-serializable types
    - Handle errors gracefully

    Args:
        ctx: The context containing the HTTP client and API key.
        currency: The ISO currency code (e.g., USD, EUR).

    Returns:
        The user's balance as a formatted string.
    """
    # Access injected dependencies via ctx.deps
    response = await ctx.deps.http_client.get(
        f"https://api.example.com/users/{ctx.deps.user_id}/balance",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        params={"currency": currency}
    )
    response.raise_for_status()
    balance_data = response.json()
    return f"{balance_data['amount']} {currency}"


@agent.tool
async def transfer_funds(
    ctx: RunContext[AgentDeps],
    recipient_id: str,
    amount: float,
    currency: str
) -> dict:
    """
    Transfer funds from the current user to another user.

    This demonstrates a tool with multiple parameters and error handling.

    Args:
        ctx: The context with authentication and HTTP client.
        recipient_id: The ID of the user receiving the funds.
        amount: The amount to transfer (must be positive).
        currency: The currency code (USD, EUR, etc.).

    Returns:
        A dictionary with transaction details.
    """
    if amount <= 0:
        return {"success": False, "error": "Amount must be positive"}

    # Check environment before dangerous operations
    if ctx.deps.environment == "production":
        # Add extra validation in production
        if amount > 10000:
            return {
                "success": False,
                "error": "Transfers over 10,000 require manual approval"
            }

    response = await ctx.deps.http_client.post(
        "https://api.example.com/transactions",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        json={
            "from_user": ctx.deps.user_id,
            "to_user": recipient_id,
            "amount": amount,
            "currency": currency
        }
    )

    if response.status_code != 200:
        return {"success": False, "error": response.text}

    return {"success": True, "transaction_id": response.json()["id"]}


@agent.tool
def simple_calculation(ctx: RunContext[AgentDeps], a: int, b: int) -> int:
    """
    Example of a tool that doesn't use dependencies directly.

    IMPORTANT: When deps_type is set on the agent, ALL tools MUST have
    ctx: RunContext as the first parameter - even if you don't use it.

    Args:
        ctx: The context (required when deps_type is set, even if unused)
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    # ctx is required but we don't need to access ctx.deps here
    return a + b


# Best Practices for Tools:
# 1. ALWAYS include ctx: RunContext as first parameter when deps_type is set
# 2. Write comprehensive docstrings - the AI reads these
# 3. Use type hints for all parameters
# 4. Handle errors gracefully and return meaningful messages
# 5. Validate inputs before making external calls
# 6. Consider environment-specific logic (dev vs prod)
# 7. Return structured data when possible
