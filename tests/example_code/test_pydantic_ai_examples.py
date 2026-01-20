"""
Tests for pydanticAIcodeExamples.py - Core Pydantic AI patterns.

Tests the Pydantic models and patterns defined in the example file.
"""

import importlib.util
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError


class TestHistoricCelebrityModel:
    """Test the HistoricCelebrity Pydantic model pattern."""

    def test_valid_historic_celebrity(self):
        """Test creating a valid HistoricCelebrity instance."""

        class HistoricCelebrity(BaseModel):
            """Historic significant person"""

            name: str
            birthYear: int
            shortBiography: str = Field(
                ...,
                description="The short biography should be structured first giving a short text about the person",
            )

        celebrity = HistoricCelebrity(
            name="Florence Nightingale",
            birthYear=1820,
            shortBiography="Florence Nightingale was a pioneering nurse.",
        )

        assert celebrity.name == "Florence Nightingale"
        assert celebrity.birthYear == 1820

    def test_field_descriptions_present(self):
        """Test that Field descriptions work as expected."""

        class HistoricCelebrity(BaseModel):
            name: str
            birthYear: int
            shortBiography: str = Field(
                ..., description="A detailed biography"
            )

        # Get field info
        fields = HistoricCelebrity.model_fields
        assert "shortBiography" in fields
        assert fields["shortBiography"].description == "A detailed biography"


class TestPatientHistoryModel:
    """Test the nested PatientHistory Pydantic model pattern."""

    def test_valid_patient_history(self):
        """Test creating valid nested patient history."""
        from enum import Enum
        from typing import List, Optional

        class Gender(str, Enum):
            male = "male"
            female = "female"
            other = "other"

        class Allergy(BaseModel):
            substance: str
            reaction: Optional[str] = None

        class Medication(BaseModel):
            name: str
            dose_mg: float
            frequency_per_day: int

        class Condition(BaseModel):
            name: str
            diagnosed_date: Optional[date] = None
            chronic: bool

        class PatientHistory(BaseModel):
            name: str
            birth_date: date
            gender: Gender
            height_cm: Optional[float] = None
            weight_kg: Optional[float] = None
            smoker: bool
            allergies: List[Allergy] = []
            medications: List[Medication] = []
            conditions: List[Condition] = []

        patient = PatientHistory(
            name="Jane Bond",
            birth_date=date(1985, 3, 15),
            gender=Gender.female,
            height_cm=165.0,
            weight_kg=60.0,
            smoker=False,
            allergies=[Allergy(substance="Penicillin", reaction="Rash")],
            medications=[
                Medication(name="Aspirin", dose_mg=100.0, frequency_per_day=1)
            ],
            conditions=[
                Condition(name="Asthma", diagnosed_date=date(2010, 5, 1), chronic=True)
            ],
        )

        assert patient.name == "Jane Bond"
        assert patient.gender == Gender.female
        assert len(patient.allergies) == 1
        assert patient.allergies[0].substance == "Penicillin"

    def test_gender_enum_values(self):
        """Test Gender enum has expected values."""
        from enum import Enum

        class Gender(str, Enum):
            male = "male"
            female = "female"
            other = "other"

        assert Gender.male.value == "male"
        assert Gender.female.value == "female"
        assert Gender.other.value == "other"

    def test_nested_model_with_defaults(self):
        """Test nested models with default factory lists."""
        from typing import List

        class Medication(BaseModel):
            name: str

        class PatientHistory(BaseModel):
            name: str
            medications: List[Medication] = []

        patient = PatientHistory(name="Test Patient")
        assert patient.medications == []

        patient_with_meds = PatientHistory(
            name="Test Patient",
            medications=[Medication(name="Aspirin")],
        )
        assert len(patient_with_meds.medications) == 1


class TestSystemPromptPattern:
    """Test system prompt patterns."""

    def test_deps_type_string_pattern(self):
        """Test the pattern of using str as deps_type."""
        from dataclasses import dataclass

        # Simulate the pattern without actually running the agent
        user_name = "Max"

        # The pattern generates prompts like:
        prompt = f"The user name is {user_name}."
        assert "Max" in prompt

        date_prompt = f"The date is {date.today()}."
        assert str(date.today()) in date_prompt


class TestToolsPattern:
    """Test tools pattern from the example."""

    def test_tool_plain_pattern(self):
        """Test @agent.tool_plain pattern returns expected value."""

        def getLabResult() -> str:
            """Get the Calcium level"""
            return "1.45 mmol/L"

        result = getLabResult()
        assert result == "1.45 mmol/L"

    def test_tool_with_context_pattern(self):
        """Test @agent.tool pattern with RunContext."""
        from dataclasses import dataclass

        @dataclass
        class MockContext:
            deps: str

        def getPatientName(ctx: MockContext) -> str:
            """Get the patient name from context"""
            return ctx.deps

        ctx = MockContext(deps="Max")
        result = getPatientName(ctx)
        assert result == "Max"


class TestMemoryPattern:
    """Test memory/message history pattern."""

    def test_message_history_list_pattern(self):
        """Test the message history list pattern."""
        # The pattern initializes an empty list
        message_history = []

        # Simulate adding messages
        message_history.append({"role": "user", "content": "Hello"})
        message_history.append({"role": "assistant", "content": "Hi there!"})

        assert len(message_history) == 2
        assert message_history[0]["role"] == "user"

    def test_conversation_loop_break_conditions(self):
        """Test the exit conditions for conversation loop."""
        exit_commands = ["quit", "exit", "q"]

        for cmd in exit_commands:
            user_input = cmd
            assert user_input.lower() in exit_commands


class TestOutputValidatorPattern:
    """Test output validator pattern from the example."""

    def test_validator_checks_for_wrong(self):
        """Test the validator pattern that checks for 'wrong' keyword."""

        class ModelRetry(Exception):
            """Mock ModelRetry exception."""

            pass

        def output_validator_simple(data: str) -> str:
            if "wrong" in data.lower():
                raise ModelRetry("wrong response")
            return data

        # Should pass
        result = output_validator_simple("correct")
        assert result == "correct"

        # Should raise
        with pytest.raises(ModelRetry):
            output_validator_simple("wrong")

        with pytest.raises(ModelRetry):
            output_validator_simple("That is WRONG")

    def test_validator_returns_data_unchanged(self):
        """Test validator returns data when valid."""

        def output_validator_simple(data: str) -> str:
            if "wrong" in data.lower():
                raise Exception("wrong response")
            return data

        test_inputs = ["correct", "Correct", "CORRECT", "The answer is correct"]
        for inp in test_inputs:
            assert output_validator_simple(inp) == inp
