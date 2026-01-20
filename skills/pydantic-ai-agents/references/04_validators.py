"""
Example: Output Validators (Structured Output)

This file demonstrates the Golden Standard for enforcing structured output
from Pydantic AI agents using Pydantic models and validators.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal
from datetime import datetime


class RiskAnalysis(BaseModel):
    """
    Example of a structured output model with validation.

    Key Principles:
    - Use Pydantic BaseModel for all structured outputs
    - Add Field descriptions - the agent uses these to understand requirements
    - Use @field_validator for single-field validation logic
    - Use @model_validator for cross-field validation
    - Provide clear error messages in validators
    """
    risk_score: float = Field(
        description="A score between 0.0 (safe) and 1.0 (risky)"
    )
    reasoning: str = Field(
        description="Detailed explanation for the risk score"
    )
    requires_human_review: bool = Field(
        description="Whether this transaction needs manual review"
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence in the assessment (0.0 to 1.0)"
    )

    @field_validator('risk_score')
    @classmethod
    def check_score_range(cls, v: float) -> float:
        """Validate that risk_score is within valid range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Risk score must be between 0.0 and 1.0")
        return v

    @field_validator('confidence')
    @classmethod
    def check_confidence_range(cls, v: float) -> float:
        """Validate that confidence is within valid range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @field_validator('reasoning')
    @classmethod
    def check_reasoning_length(cls, v: str) -> str:
        """Ensure reasoning is substantial."""
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters")
        return v

    @model_validator(mode='after')
    def check_high_risk_review(self):
        """Ensure high risk scores always trigger human review."""
        if self.risk_score > 0.8 and not self.requires_human_review:
            raise ValueError(
                "Transactions with risk_score > 0.8 must require human review"
            )
        return self


class UserProfile(BaseModel):
    """
    Example of a more complex structured output.

    Demonstrates:
    - Nested models
    - Enums (via Literal)
    - Optional fields with defaults
    - Multiple validators
    """
    user_id: str = Field(description="Unique user identifier")
    full_name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    account_status: Literal["active", "suspended", "closed"] = Field(
        description="Current account status"
    )
    created_at: datetime = Field(description="Account creation timestamp")
    preferences: dict[str, str] = Field(
        default_factory=dict,
        description="User preferences as key-value pairs"
    )

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError("Invalid email format")
        return v.lower()

    @field_validator('full_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not empty."""
        if not v.strip():
            raise ValueError("Full name cannot be empty")
        return v.strip()


class TransactionSummary(BaseModel):
    """
    Example showing list fields and computed properties.
    """
    total_amount: float = Field(description="Total transaction amount")
    currency: str = Field(description="ISO currency code")
    transaction_count: int = Field(description="Number of transactions")
    transaction_ids: list[str] = Field(
        description="List of transaction IDs included in this summary"
    )

    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Ensure currency is uppercase ISO code."""
        valid_currencies = {'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD'}
        v_upper = v.upper()
        if v_upper not in valid_currencies:
            raise ValueError(f"Currency must be one of {valid_currencies}")
        return v_upper

    @model_validator(mode='after')
    def check_transaction_consistency(self):
        """Ensure transaction count matches list length."""
        if len(self.transaction_ids) != self.transaction_count:
            raise ValueError(
                f"transaction_count ({self.transaction_count}) must match "
                f"length of transaction_ids ({len(self.transaction_ids)})"
            )
        return self


# Usage Pattern:
# Configure the agent to enforce structured output:
# agent = Agent(model=model, output_type=RiskAnalysis)
#
# The agent will be forced to return data matching the RiskAnalysis schema,
# and all validators will run automatically.
# Access the result via: result.output (NOT result.data)

# Best Practices for Validators:
# 1. Use Field descriptions - the agent reads these
# 2. Add @field_validator for single-field checks
# 3. Add @model_validator for cross-field logic
# 4. Raise ValueError with clear messages
# 5. Return the validated value from validators
# 6. Use Literal for enum-like fields
# 7. Provide sensible defaults where appropriate
