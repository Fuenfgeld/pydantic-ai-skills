"""
Example: Complete Agent (Putting It All Together)

This file demonstrates the Golden Standard for building a complete Pydantic AI agent
by combining Dependencies, System Prompts, Tools, and Validators.
"""

import asyncio
import httpx
import os
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# ============================================================================
# 1. DEPENDENCIES - State and Configuration
# ============================================================================

@dataclass
class AgentDeps:
    """
    Dependency container holding all runtime state.
    Import from 01_dependencies.py in production code.
    """
    user_id: str
    api_key: str
    http_client: httpx.AsyncClient
    environment: str = "production"

    def is_dev(self) -> bool:
        return self.environment == "development"


# ============================================================================
# 2. OUTPUT VALIDATORS - Structured Output Schema
# ============================================================================

class RiskAnalysis(BaseModel):
    """
    The structure the agent MUST return.
    Import from 04_validators.py in production code.
    """
    risk_score: float = Field(
        description="A score between 0.0 (safe) and 1.0 (risky)"
    )
    reasoning: str = Field(
        description="Detailed explanation for the score"
    )
    requires_human_review: bool = Field(
        description="Whether manual review is needed"
    )

    @field_validator('risk_score')
    @classmethod
    def check_score_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Risk score must be between 0.0 and 1.0")
        return v


# ============================================================================
# 3. AGENT DEFINITION - Create Agent with Dependencies and Output Type
# ============================================================================

# Create OpenRouter provider (or use OpenAI directly)
provider = OpenAIProvider(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)

# Create model
model = OpenAIChatModel(
    model_name='openai/gpt-4o',
    provider=provider
)

# Create agent with model, output type, and dependencies
agent = Agent(
    model=model,
    deps_type=AgentDeps,
    output_type=RiskAnalysis   # Enforce structured output
)


# ============================================================================
# 4. SYSTEM PROMPT - Dynamic, Context-Aware Instructions
# ============================================================================

@agent.system_prompt
async def generate_system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """
    Build dynamic system prompt with context from dependencies.
    See 02_prompts.py for more examples.
    """
    env_note = "DEVELOPMENT MODE - Be cautious" if ctx.deps.is_dev() else "Production"

    return f"""
    You are a specialized fraud detection agent for user {ctx.deps.user_id}.
    Environment: {env_note}

    Your responsibilities:
    1. Analyze transaction patterns for suspicious activity
    2. Calculate a risk score between 0.0 (safe) and 1.0 (high risk)
    3. Provide clear reasoning for your assessment
    4. Flag transactions requiring human review

    Guidelines:
    - Use the get_transaction_history tool to fetch recent transactions
    - Consider unusual amounts, frequencies, and patterns
    - Be conservative: when in doubt, flag for review
    - Scores above 0.8 MUST trigger human review

    If tools fail or data is unavailable, explain the issue and suggest next steps.
    """


# ============================================================================
# 5. TOOLS - Functions the Agent Can Call
# ============================================================================

@agent.tool
async def get_transaction_history(
    ctx: RunContext[AgentDeps],
    days: int = 7
) -> dict:
    """
    Fetch recent transaction history for the user.
    See 03_tools.py for more tool examples.

    Args:
        ctx: Runtime context with HTTP client and auth
        days: Number of days of history to fetch (default: 7)

    Returns:
        Dictionary with transaction data
    """
    response = await ctx.deps.http_client.get(
        f"https://api.example.com/users/{ctx.deps.user_id}/transactions",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        params={"days": days}
    )

    if response.status_code != 200:
        return {
            "error": f"Failed to fetch transactions: {response.status_code}",
            "transactions": []
        }

    return response.json()


@agent.tool
async def check_user_reputation(ctx: RunContext[AgentDeps]) -> dict:
    """
    Check the user's account reputation score.

    Returns:
        Dictionary with reputation data
    """
    response = await ctx.deps.http_client.get(
        f"https://api.example.com/users/{ctx.deps.user_id}/reputation",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"}
    )

    if response.status_code != 200:
        return {"reputation_score": 0.5, "error": "Could not fetch reputation"}

    return response.json()


# ============================================================================
# 6. MAIN EXECUTION - Running the Agent
# ============================================================================

async def main():
    """
    Main execution flow demonstrating the complete agent lifecycle.

    Steps:
    1. Setup external resources (HTTP client)
    2. Initialize dependencies with configuration
    3. Run agent with user input
    4. Handle structured, validated output
    """

    # 1. Setup Resources
    async with httpx.AsyncClient() as client:
        # 2. Initialize Dependencies
        deps = AgentDeps(
            user_id="u_12345",
            api_key=os.getenv("API_KEY", "demo_key"),
            http_client=client,
            environment=os.getenv("ENVIRONMENT", "production")
        )

        # 3. Run Agent
        result = await agent.run(
            "Analyze recent transaction activity for suspicious patterns.",
            deps=deps
        )

        # 4. Handle Validated Output
        # result.output is guaranteed to be a RiskAnalysis instance
        # All validators have already run successfully
        print(f"Risk Score: {result.output.risk_score}")
        print(f"Reasoning: {result.output.reasoning}")
        print(f"Requires Review: {result.output.requires_human_review}")

        # You can also access the raw messages
        print(f"\nTotal messages exchanged: {len(result.messages())}")


# Alternative: Streaming Results
async def streaming_example():
    """
    Example of streaming agent responses for real-time feedback.
    """
    async with httpx.AsyncClient() as client:
        deps = AgentDeps(
            user_id="u_12345",
            api_key="demo_key",
            http_client=client
        )

        # Use agent.run_stream for streaming responses
        async with agent.run_stream(
            "Analyze the last transaction for fraud.",
            deps=deps
        ) as response:
            # Stream text as it's generated
            async for message in response.stream_text():
                print(message, end='', flush=True)

            # Get final validated result
            final_result = await response.get_output()
            print(f"\n\nFinal Risk Score: {final_result.risk_score}")


# ============================================================================
# BEST PRACTICES SUMMARY
# ============================================================================

# 1. Dependencies: Use dataclasses, never globals
# 2. System Prompts: Make them dynamic, reference ctx.deps
# 3. Tools: First param is ctx if you need state, use type hints
# 4. Validators: Use Pydantic models with @field_validator
# 5. Main: Setup resources, init deps, run agent, handle output
# 6. Error Handling: Validate inputs, handle API failures gracefully
# 7. Environment: Different behavior for dev/prod
# 8. Documentation: Clear docstrings for tools and models

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Uncomment to try streaming:
    # asyncio.run(streaming_example())
