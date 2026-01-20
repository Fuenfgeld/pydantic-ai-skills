"""
Tests for 04_validators.py - Pydantic model validation patterns.

These tests are synchronous because they test pure Pydantic models
without any agent execution.
"""

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError


def load_validators_module():
    """Dynamically load the 04_validators.py module."""
    module_path = (
        Path(__file__).parent.parent.parent
        / "skills"
        / "pydantic-ai-agents"
        / "references"
        / "04_validators.py"
    )
    spec = importlib.util.spec_from_file_location("validators_04", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the module once for all tests
validators_module = load_validators_module()
RiskAnalysis = validators_module.RiskAnalysis
UserProfile = validators_module.UserProfile
TransactionSummary = validators_module.TransactionSummary


class TestRiskAnalysis:
    """Test RiskAnalysis Pydantic model and validators."""

    def test_valid_risk_analysis(self):
        """Test creating a valid RiskAnalysis instance."""
        result = RiskAnalysis(
            risk_score=0.75,
            reasoning="This transaction shows unusual patterns that warrant attention.",
            requires_human_review=False,
            confidence=0.9,
        )

        assert result.risk_score == 0.75
        assert result.requires_human_review is False
        assert result.confidence == 0.9

    def test_risk_score_zero_is_valid(self):
        """Test that risk_score of 0.0 is valid (edge case)."""
        result = RiskAnalysis(
            risk_score=0.0,
            reasoning="This transaction is completely safe.",
            requires_human_review=False,
        )

        assert result.risk_score == 0.0

    def test_risk_score_one_is_valid(self):
        """Test that risk_score of 1.0 is valid (edge case)."""
        result = RiskAnalysis(
            risk_score=1.0,
            reasoning="This transaction is extremely risky.",
            requires_human_review=True,  # Required for high risk
        )

        assert result.risk_score == 1.0

    def test_risk_score_above_one_raises_error(self):
        """Test that risk_score > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalysis(
                risk_score=1.5,
                reasoning="Some reasoning here.",
                requires_human_review=True,
            )

        assert "Risk score must be between 0.0 and 1.0" in str(exc_info.value)

    def test_risk_score_below_zero_raises_error(self):
        """Test that risk_score < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalysis(
                risk_score=-0.1,
                reasoning="Some reasoning here.",
                requires_human_review=False,
            )

        assert "Risk score must be between 0.0 and 1.0" in str(exc_info.value)

    def test_reasoning_too_short_raises_error(self):
        """Test that reasoning < 10 characters raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalysis(
                risk_score=0.5,
                reasoning="Short",  # Too short
                requires_human_review=False,
            )

        assert "Reasoning must be at least 10 characters" in str(exc_info.value)

    def test_high_risk_without_review_raises_error(self):
        """Test that risk_score > 0.8 without human review raises error."""
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalysis(
                risk_score=0.85,
                reasoning="High risk transaction detected.",
                requires_human_review=False,  # Should be True for high risk
            )

        assert "must require human review" in str(exc_info.value)

    def test_high_risk_with_review_is_valid(self):
        """Test that risk_score > 0.8 with human review is valid."""
        result = RiskAnalysis(
            risk_score=0.9,
            reasoning="Very high risk transaction detected.",
            requires_human_review=True,
        )

        assert result.risk_score == 0.9
        assert result.requires_human_review is True

    def test_default_confidence_value(self):
        """Test that confidence defaults to 0.5."""
        result = RiskAnalysis(
            risk_score=0.5,
            reasoning="Moderate risk assessment.",
            requires_human_review=False,
        )

        assert result.confidence == 0.5


class TestUserProfile:
    """Test UserProfile Pydantic model and validators."""

    def test_valid_user_profile(self):
        """Test creating a valid UserProfile instance."""
        result = UserProfile(
            user_id="u_12345",
            full_name="John Doe",
            email="john.doe@example.com",
            account_status="active",
            created_at=datetime.now(),
        )

        assert result.user_id == "u_12345"
        assert result.full_name == "John Doe"
        assert result.email == "john.doe@example.com"

    def test_email_is_lowercased(self):
        """Test that email is automatically converted to lowercase."""
        result = UserProfile(
            user_id="u_12345",
            full_name="John Doe",
            email="John.DOE@EXAMPLE.COM",
            account_status="active",
            created_at=datetime.now(),
        )

        assert result.email == "john.doe@example.com"

    def test_invalid_email_raises_error(self):
        """Test that invalid email format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="u_12345",
                full_name="John Doe",
                email="invalid_email",
                account_status="active",
                created_at=datetime.now(),
            )

        assert "Invalid email format" in str(exc_info.value)

    def test_empty_name_raises_error(self):
        """Test that empty full_name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UserProfile(
                user_id="u_12345",
                full_name="   ",  # Only whitespace
                email="john@example.com",
                account_status="active",
                created_at=datetime.now(),
            )

        assert "Full name cannot be empty" in str(exc_info.value)

    def test_name_is_stripped(self):
        """Test that full_name is stripped of leading/trailing whitespace."""
        result = UserProfile(
            user_id="u_12345",
            full_name="  John Doe  ",
            email="john@example.com",
            account_status="active",
            created_at=datetime.now(),
        )

        assert result.full_name == "John Doe"

    def test_account_status_literals(self):
        """Test that only valid account_status values are accepted."""
        valid_statuses = ["active", "suspended", "closed"]

        for status in valid_statuses:
            result = UserProfile(
                user_id="u_12345",
                full_name="John Doe",
                email="john@example.com",
                account_status=status,
                created_at=datetime.now(),
            )
            assert result.account_status == status

    def test_invalid_account_status_raises_error(self):
        """Test that invalid account_status raises ValidationError."""
        with pytest.raises(ValidationError):
            UserProfile(
                user_id="u_12345",
                full_name="John Doe",
                email="john@example.com",
                account_status="invalid_status",
                created_at=datetime.now(),
            )

    def test_default_preferences(self):
        """Test that preferences defaults to empty dict."""
        result = UserProfile(
            user_id="u_12345",
            full_name="John Doe",
            email="john@example.com",
            account_status="active",
            created_at=datetime.now(),
        )

        assert result.preferences == {}


class TestTransactionSummary:
    """Test TransactionSummary Pydantic model and validators."""

    def test_valid_transaction_summary(self):
        """Test creating a valid TransactionSummary instance."""
        result = TransactionSummary(
            total_amount=150.75,
            currency="USD",
            transaction_count=3,
            transaction_ids=["tx1", "tx2", "tx3"],
        )

        assert result.total_amount == 150.75
        assert result.currency == "USD"
        assert result.transaction_count == 3

    def test_currency_is_uppercased(self):
        """Test that currency is automatically converted to uppercase."""
        result = TransactionSummary(
            total_amount=100.0,
            currency="usd",
            transaction_count=1,
            transaction_ids=["tx1"],
        )

        assert result.currency == "USD"

    def test_invalid_currency_raises_error(self):
        """Test that invalid currency code raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TransactionSummary(
                total_amount=100.0,
                currency="XYZ",  # Not a valid currency
                transaction_count=1,
                transaction_ids=["tx1"],
            )

        assert "Currency must be one of" in str(exc_info.value)

    def test_valid_currencies(self):
        """Test that all valid currencies are accepted."""
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]

        for currency in valid_currencies:
            result = TransactionSummary(
                total_amount=100.0,
                currency=currency,
                transaction_count=1,
                transaction_ids=["tx1"],
            )
            assert result.currency == currency

    def test_count_mismatch_raises_error(self):
        """Test that transaction_count must match list length."""
        with pytest.raises(ValidationError) as exc_info:
            TransactionSummary(
                total_amount=100.0,
                currency="USD",
                transaction_count=5,  # Says 5
                transaction_ids=["tx1", "tx2"],  # But only 2
            )

        assert "transaction_count" in str(exc_info.value)
        assert "must match" in str(exc_info.value)

    def test_empty_transactions_with_zero_count(self):
        """Test that empty transaction list with count 0 is valid."""
        result = TransactionSummary(
            total_amount=0.0,
            currency="USD",
            transaction_count=0,
            transaction_ids=[],
        )

        assert result.transaction_count == 0
        assert result.transaction_ids == []
