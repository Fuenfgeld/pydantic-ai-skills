"""
Example: Result Validators and Retry Logic

This file demonstrates how to use @agent.output_validator and ModelRetry
to validate agent outputs and automatically retry on validation failures.
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, ModelRetry
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


# ============================================================================
# 1. BASIC RESULT VALIDATOR
# ============================================================================

class PersonInfo(BaseModel):
    """Structured person information with validation."""
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str

    @field_validator('email')
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower()


# Create agent with retries enabled
# The `retries` parameter specifies how many times to retry on validation failure
person_agent = Agent(
    model=model,
    output_type=PersonInfo,
    retries=3,  # Retry up to 3 times on validation errors
    system_prompt="""You extract person information from text.
    Return valid data: name (non-empty), age (0-150), email (valid format)."""
)


@person_agent.output_validator
async def validate_person_result(ctx: RunContext, result: PersonInfo) -> PersonInfo:
    """
    Custom result validator that runs after Pydantic validation.

    Key Points:
    - Runs AFTER the model generates output and Pydantic validates it
    - Can perform additional business logic validation
    - Raise ModelRetry to trigger a retry with feedback to the model
    - Return the result to accept it

    Args:
        ctx: The run context
        result: The validated Pydantic model

    Returns:
        The validated result

    Raises:
        ModelRetry: To retry with feedback
    """
    # Example: Business rule - age must be 18+ for adult profiles
    if result.age < 18:
        raise ModelRetry(
            "The person must be 18 or older. Please extract adult information only."
        )

    # Example: Check for placeholder data
    if result.name.lower() in ['test', 'john doe', 'jane doe']:
        raise ModelRetry(
            "Please extract the actual name, not a placeholder."
        )

    return result


# ============================================================================
# 2. RESULT VALIDATOR WITH CONTEXT
# ============================================================================

from dataclasses import dataclass


@dataclass
class ValidationDeps:
    """Dependencies for validation."""
    allowed_domains: list[str]
    min_age: int = 18


domain_agent = Agent(
    model=model,
    output_type=PersonInfo,
    deps_type=ValidationDeps,
    retries=2,
    system_prompt="Extract person information from the given text."
)


@domain_agent.output_validator
async def validate_with_deps(
    ctx: RunContext[ValidationDeps],
    result: PersonInfo
) -> PersonInfo:
    """
    Validator that uses dependencies for context-aware validation.

    This allows dynamic validation rules based on runtime configuration.
    """
    # Check age against configurable minimum
    if result.age < ctx.deps.min_age:
        raise ModelRetry(
            f"Person must be at least {ctx.deps.min_age} years old."
        )

    # Check email domain against allowed list
    email_domain = result.email.split('@')[-1]
    if ctx.deps.allowed_domains and email_domain not in ctx.deps.allowed_domains:
        allowed = ", ".join(ctx.deps.allowed_domains)
        raise ModelRetry(
            f"Email must be from allowed domains: {allowed}"
        )

    return result


# ============================================================================
# 3. MULTIPLE VALIDATORS (CHAINED)
# ============================================================================

class ContentReview(BaseModel):
    """Content review result."""
    content: str
    rating: int = Field(ge=1, le=5)
    is_appropriate: bool


review_agent = Agent(
    model=model,
    output_type=ContentReview,
    retries=3,
    system_prompt="Review the content and rate it 1-5."
)


@review_agent.output_validator
async def check_rating_consistency(ctx: RunContext, result: ContentReview) -> ContentReview:
    """First validator: Check rating consistency with appropriateness."""
    if not result.is_appropriate and result.rating > 2:
        raise ModelRetry(
            "Inappropriate content should not have a rating higher than 2."
        )
    return result


@review_agent.output_validator
async def check_content_length(ctx: RunContext, result: ContentReview) -> ContentReview:
    """Second validator: Ensure content is substantive."""
    if len(result.content.strip()) < 10:
        raise ModelRetry(
            "Content review must be at least 10 characters long."
        )
    return result


# Best Practices for Result Validators:
# 1. Use `retries` parameter to control max retry attempts
# 2. Provide clear, actionable feedback in ModelRetry messages
# 3. Order validators from most likely to fail first
# 4. Use dependencies for configurable validation rules
# 5. Keep validators focused on single concerns
# 6. Consider performance - each retry costs tokens/time


async def main():
    """Demonstrate result validators."""

    # Example 1: Basic validation
    print("=== Basic Result Validator ===")
    try:
        result = await person_agent.run(
            "John Smith, age 25, email: john.smith@example.com"
        )
        print(f"Extracted: {result.output}")
    except Exception as e:
        print(f"Validation failed after retries: {e}")

    # Example 2: Validation with dependencies
    print("\n=== Validator with Dependencies ===")
    deps = ValidationDeps(
        allowed_domains=["company.com", "partner.org"],
        min_age=21
    )
    try:
        result = await domain_agent.run(
            "Alice Brown, 25 years old, alice@company.com",
            deps=deps
        )
        print(f"Extracted: {result.output}")
    except Exception as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
