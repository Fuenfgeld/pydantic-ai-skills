"""
Tests for 05_main.py - Complete agent pattern.

CRITICAL: All tests in this file MUST be async due to module-level agent.
We test the patterns using fresh agents to avoid module-level httpx issues.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart


def load_main_module():
    """Dynamically load the 05_main.py module to access classes."""
    module_path = (
        Path(__file__).parent.parent.parent
        / "skills"
        / "pydantic-ai-agents"
        / "references"
        / "05_main.py"
    )
    spec = importlib.util.spec_from_file_location("main_05", module_path)
    module = importlib.util.module_from_spec(spec)
    # Note: We don't exec the module to avoid creating the global agent
    # Instead, we import the classes we need
    return module_path


# Define mock versions of the classes from the module
# Note: Named "Mock*" to avoid pytest collecting them as test classes
@dataclass
class MockAgentDeps:
    """Mock version of AgentDeps from 05_main.py."""

    user_id: str
    api_key: str
    http_client: object
    environment: str = "production"

    def is_dev(self) -> bool:
        return self.environment == "development"


class MockRiskAnalysis(BaseModel):
    """Mock version of RiskAnalysis from 05_main.py."""

    risk_score: float = Field(description="A score between 0.0 (safe) and 1.0 (risky)")
    reasoning: str = Field(description="Detailed explanation for the score")
    requires_human_review: bool = Field(description="Whether manual review is needed")

    @field_validator("risk_score")
    @classmethod
    def check_score_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Risk score must be between 0.0 and 1.0")
        return v


@pytest.mark.asyncio
class TestCompleteAgentPattern:
    """Test the complete agent pattern from 05_main.py."""

    async def test_agent_with_deps_and_output_type(self):
        """Test creating an agent with deps_type and output_type."""
        agent = Agent("test", deps_type=MockAgentDeps, output_type=MockRiskAnalysis)

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="test_user",
            api_key="test_key",
            http_client=mock_client,
        )

        with agent.override(model=TestModel()):
            result = await agent.run("Analyze this transaction", deps=deps)

        assert isinstance(result.output, MockRiskAnalysis)
        assert 0.0 <= result.output.risk_score <= 1.0

    async def test_system_prompt_receives_context(self):
        """Test that @system_prompt decorator receives RunContext."""
        agent = Agent("test", deps_type=MockAgentDeps)

        prompt_received_deps = None

        @agent.system_prompt
        async def capture_prompt(ctx: RunContext[MockAgentDeps]) -> str:
            nonlocal prompt_received_deps
            prompt_received_deps = ctx.deps
            return f"Analyzing for user {ctx.deps.user_id}"

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="prompt_test_user",
            api_key="key",
            http_client=mock_client,
        )

        with agent.override(model=TestModel()):
            await agent.run("Test", deps=deps)

        assert prompt_received_deps is not None
        assert prompt_received_deps.user_id == "prompt_test_user"

    async def test_system_prompt_uses_is_dev_method(self):
        """Test that system prompt can use deps helper methods."""
        agent = Agent("test", deps_type=MockAgentDeps)

        generated_prompt = None

        @agent.system_prompt
        async def generate_prompt(ctx: RunContext[MockAgentDeps]) -> str:
            nonlocal generated_prompt
            env_note = "DEVELOPMENT MODE" if ctx.deps.is_dev() else "Production"
            generated_prompt = f"Environment: {env_note}"
            return generated_prompt

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="user",
            api_key="key",
            http_client=mock_client,
            environment="development",
        )

        with agent.override(model=TestModel()):
            await agent.run("Test", deps=deps)

        assert "DEVELOPMENT MODE" in generated_prompt


@pytest.mark.asyncio
class TestAgentTools:
    """Test tool patterns from 05_main.py."""

    async def test_tool_receives_run_context(self):
        """Test that tools receive RunContext as first parameter."""
        agent = Agent("test", deps_type=MockAgentDeps)

        received_ctx = None

        @agent.tool
        async def get_transaction_history(ctx: RunContext[MockAgentDeps], days: int = 7) -> dict:
            nonlocal received_ctx
            received_ctx = ctx
            return {"transactions": [], "days": days}

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="tool_test_user",
            api_key="key",
            http_client=mock_client,
        )

        # Use FunctionModel to force tool call
        def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart("get_transaction_history", {"days": 7})])
            return ModelResponse(parts=[TextPart("Done")])

        with agent.override(model=FunctionModel(call_tool)):
            await agent.run("Get transactions", deps=deps)

        assert received_ctx is not None
        assert received_ctx.deps.user_id == "tool_test_user"

    async def test_tool_can_use_http_client(self):
        """Test that tools can access http_client from deps."""
        agent = Agent("test", deps_type=MockAgentDeps)

        @agent.tool
        async def fetch_data(ctx: RunContext[MockAgentDeps]) -> dict:
            response = await ctx.deps.http_client.get("https://api.example.com/data")
            return response.json()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_client.get.return_value = mock_response

        deps = MockAgentDeps(
            user_id="user",
            api_key="key",
            http_client=mock_client,
        )

        def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart("fetch_data", {})])
            return ModelResponse(parts=[TextPart("Data fetched")])

        with agent.override(model=FunctionModel(call_tool)):
            await agent.run("Fetch data", deps=deps)

        mock_client.get.assert_called_once()


@pytest.mark.asyncio
class TestAgentOutput:
    """Test output handling patterns from 05_main.py."""

    async def test_result_output_is_validated(self):
        """Test that result.output is a validated Pydantic model."""
        agent = Agent("test", deps_type=MockAgentDeps, output_type=MockRiskAnalysis)

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="user",
            api_key="key",
            http_client=mock_client,
        )

        with agent.override(model=TestModel()):
            result = await agent.run("Analyze", deps=deps)

        # TestModel generates valid structured output
        assert isinstance(result.output, MockRiskAnalysis)
        # All validators should have run
        assert 0.0 <= result.output.risk_score <= 1.0

    async def test_result_messages_accessible(self):
        """Test that result.all_messages() returns conversation history."""
        agent = Agent("test", deps_type=MockAgentDeps)

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="user",
            api_key="key",
            http_client=mock_client,
        )

        with agent.override(model=TestModel()):
            result = await agent.run("Hello", deps=deps)

        messages = result.all_messages()
        assert len(messages) > 0


@pytest.mark.asyncio
class TestAgentLifecycle:
    """Test agent lifecycle patterns from 05_main.py."""

    async def test_deps_initialization_pattern(self):
        """Test the pattern of initializing deps with resources."""
        agent = Agent("test", deps_type=MockAgentDeps)

        # Simulate the pattern from 05_main.py
        mock_client = AsyncMock()

        deps = MockAgentDeps(
            user_id="lifecycle_user",
            api_key="test_key",
            http_client=mock_client,
            environment="development",
        )

        assert deps.is_dev() is True
        assert deps.http_client is mock_client

    async def test_multiple_agent_runs_with_same_deps(self):
        """Test running agent multiple times with same deps."""
        agent = Agent("test", deps_type=MockAgentDeps)

        mock_client = AsyncMock()
        deps = MockAgentDeps(
            user_id="user",
            api_key="key",
            http_client=mock_client,
        )

        with agent.override(model=TestModel()):
            result1 = await agent.run("First query", deps=deps)
            result2 = await agent.run("Second query", deps=deps)

        assert result1.output is not None
        assert result2.output is not None
