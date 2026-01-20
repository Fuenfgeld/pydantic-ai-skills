Pydantic AI Reference SkillThis skill provides a modular reference for building Pydantic AI agents. It does not generate code automatically; instead, it provides the "Golden Standards" for Claude to read and copy from.Directory StructureCreate a folder .claude/skills/pydantic-ai-guide/ and add these files.1. The Skill Manifest (SKILL.md)name: pydantic-ai-guide
description: Expert guide for building Pydantic AI agents. Use this to reference best practices for Dependency Injection, Dynamic System Prompts, Tools, and Structured Output validation.
allowed-tools: ReadPydantic AI Developer GuideUse this skill when the user asks to write, debug, or structure a Pydantic AI agent.1. Core ArchitecturePydantic AI agents are defined by four key components. Read the corresponding example files to understand the implementation details:Dependencies (deps):Reference: examples/01_dependencies.pyUse dataclasses to hold API keys, database connections, and user context.Never use global variables for state.System Prompts (system_prompt):Reference: examples/02_prompts.pyPrompts should be dynamic. Use the @agent.system_prompt decorator to inject data from ctx.deps into the prompt string.Tools (@agent.tool):Reference: examples/03_tools.pyTools must accept ctx: RunContext as the first argument if they need access to state.Validators (output_type):Reference: examples/04_validators.pyUse Pydantic models to enforce structured output (JSON).Use @field_validator for logic checks (e.g., ensuring a score is between 0 and 1).2. Promoting Instructions (System Prompt Engineering)When writing the system_prompt for a generated agent, adhere to these rules:Role Definition: Start with "You are a specialized agent for."Context Awareness: Explicitly mention the data available in the dependencies.Bad: "I help users."Good: "I help user {ctx.deps.user_name} (ID: {ctx.deps.user_id}) manage their account."Tool Coercion: If tools are defined, instruct the model when to use them.Example: "Use the lookup_order tool immediately if the user provides an order ID."Failure Modes: Define what to do if a tool fails or data is missing.Example: "If the database returns no results, politely ask the user for clarification."3. UsageTo build a new agent for a user:Read the requirement.Select the relevant components from the examples/ directory.Combine them into a single file following the pattern in examples/05_main.py.2. Example: Dependencies (examples/01_dependencies.py)Pythonfrom dataclasses import dataclass
from typing import Optional
import httpx

@dataclass
class AgentDeps:
    """
    The Dependency Container.
    
    This class holds all runtime state, configuration, and connections 
    required by the agent. It is injected into every tool and prompt.
    """
    user_id: str
    api_key: str
    http_client: httpx.AsyncClient
    environment: str = "production"

    # You can add helper methods here
    def is_dev(self) -> bool:
        return self.environment == "development"
3. Example: Dynamic Prompts (examples/02_prompts.py)Pythonfrom pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
# Assuming AgentDeps is imported from 01_dependencies.py
# from.01_dependencies import AgentDeps

# Create OpenRouter provider and model
provider = OpenAIProvider(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)
model = OpenAIChatModel(model_name='openai/gpt-4o', provider=provider)

agent = Agent(model=model, deps_type=AgentDeps)

@agent.system_prompt
async def generate_system_prompt(ctx: RunContext) -> str:
    """
    Builds the system prompt dynamically at runtime.
    """
    # Access dependencies via ctx.deps
    user_segment = "Developer" if ctx.deps.is_dev() else "Standard User"
    
    return f"""
    You are a helpful assistant serving a user in the '{user_segment}' segment.
    User ID: {ctx.deps.user_id}
    
    Your operational constraints:
    1. Answer concisely.
    2. Since the environment is {ctx.deps.environment}, be cautious with write operations.
    """
4. Example: Tools (examples/03_tools.py)Pythonfrom pydantic_ai import RunContext
# from.01_dependencies import AgentDeps

@agent.tool
async def get_user_balance(ctx: RunContext, currency: str) -> str:
    """
    Fetch the current balance for the user.
    
    Args:
        ctx: The context containing the HTTP client and API key.
        currency: The ISO currency code (e.g., USD, EUR).
    """
    # Usage of the injected dependency
    response = await ctx.deps.http_client.get(
        f"https://api.example.com/users/{ctx.deps.user_id}/balance",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        params={"currency": currency}
    )
    response.raise_for_status()
    return response.json()['amount']
5. Example: Output Validators (examples/04_validators.py)Pythonfrom pydantic import BaseModel, Field, field_validator

class RiskAnalysis(BaseModel):
    """
    The structure the agent MUST return.
    """
    risk_score: float = Field(description="A score between 0.0 (safe) and 1.0 (risky)")
    reasoning: str = Field(description="Explanation for the score")
    requires_human_review: bool

    @field_validator('risk_score')
    @classmethod
    def check_score_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Risk score must be between 0.0 and 1.0")
        return v

# Configure the agent to enforce this output structure
# agent = Agent(model=model, output_type=RiskAnalysis)
6. Example: Main Glue (examples/05_main.py)Pythonimport asyncio
import httpx
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import components (simulated for this single file view)
# from.01_dependencies import AgentDeps
# from.04_validators import RiskAnalysis

# Create OpenRouter provider
provider = OpenAIProvider(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)

# Create model
model = OpenAIChatModel(
    model_name='openai/gpt-4o',
    provider=provider
)

# Define the Agent
agent = Agent(
    model=model,
    deps_type=AgentDeps,
    output_type=RiskAnalysis  # Enforce structured output
)

async def main():
    # 1. Setup Resources
    async with httpx.AsyncClient() as client:
        # 2. Initialize Dependencies
        deps = AgentDeps(
            user_id="u_999",
            api_key=os.getenv("API_KEY", "secret"),
            http_client=client
        )
        
        # 3. Run Agent
        result = await agent.run(
            "Analyze the transaction history for suspicious activity.",
            deps=deps
        )
        
        # 4. Handle Validated Output
        print(f"Risk Score: {result.output.risk_score}")
        print(f"Reasoning: {result.output.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())
