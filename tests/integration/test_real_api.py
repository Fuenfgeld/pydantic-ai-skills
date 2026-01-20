"""
Integration tests that use real API calls via OpenRouter with Claude Haiku 4.5.

These tests validate that all skill patterns work with a real LLM.
Tests are automatically skipped if OPENROUTER_API_KEY is not set.

Run with: uv run pytest tests/integration/ -v
"""

import os
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, Field, field_validator

pytestmark = [
    pytest.mark.real_api,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not configured",
    ),
]


# =============================================================================
# Pattern 1: Basic Agent with Structured Output (04_validators.py)
# =============================================================================


class MathResult(BaseModel):
    """Structured output for math operations."""

    answer: int = Field(description="The numerical answer")
    explanation: str = Field(description="Brief explanation of the calculation")


@pytest.mark.asyncio
async def test_structured_output(openrouter_model):
    """Test structured output validation with real LLM."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=MathResult,
        system_prompt="You are a math assistant. Always show your work briefly.",
    )

    result = await agent.run("What is 15 + 27?")

    assert isinstance(result.output, MathResult)
    assert result.output.answer == 42


# =============================================================================
# Pattern 2: Dependency Injection (01_dependencies.py, 05_main.py)
# =============================================================================


@dataclass
class UserContext:
    """Dependencies for user-aware agent."""

    user_id: str
    user_name: str
    is_premium: bool = False


@pytest.mark.asyncio
async def test_dependency_injection(openrouter_model):
    """Test dependency injection pattern with real LLM."""
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        model=openrouter_model,
        deps_type=UserContext,
        system_prompt="You are a helpful assistant. Be brief.",
    )

    @agent.system_prompt
    def add_user_context(ctx: RunContext[UserContext]) -> str:
        premium_status = "premium" if ctx.deps.is_premium else "free"
        return f"User: {ctx.deps.user_name} (ID: {ctx.deps.user_id}, {premium_status} tier)"

    deps = UserContext(user_id="123", user_name="Alice", is_premium=True)
    result = await agent.run("Say hello to me by name.", deps=deps)

    assert "Alice" in result.output


# =============================================================================
# Pattern 3: Tool Calling (03_tools.py, 05_main.py)
# =============================================================================


@pytest.mark.asyncio
async def test_tool_calling(openrouter_model):
    """Test tool calling with real LLM."""
    from pydantic_ai import Agent, RunContext

    @dataclass
    class CalculatorDeps:
        precision: int = 2

    agent = Agent(
        model=openrouter_model,
        deps_type=CalculatorDeps,
        system_prompt="Use the calculator tool for all math. Reply with just the result.",
    )

    tool_was_called = False

    @agent.tool
    def calculate(ctx: RunContext[CalculatorDeps], expression: str) -> str:
        """Evaluate a math expression."""
        nonlocal tool_was_called
        tool_was_called = True
        # Simple eval for demo (safe because LLM can only pass strings)
        try:
            result = eval(expression)
            return f"{result:.{ctx.deps.precision}f}"
        except Exception:
            return "Error"

    deps = CalculatorDeps(precision=2)
    result = await agent.run("What is 100 divided by 3?", deps=deps)

    assert tool_was_called, "Tool should have been called"
    assert "33" in result.output


# =============================================================================
# Pattern 4: Result Validators (09_result_validators.py)
# =============================================================================


class SentimentResult(BaseModel):
    """Sentiment analysis result with validation."""

    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        allowed = {"positive", "negative", "neutral"}
        if v.lower() not in allowed:
            raise ValueError(f"sentiment must be one of {allowed}")
        return v.lower()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


@pytest.mark.asyncio
async def test_result_validators(openrouter_model):
    """Test that Pydantic validators run on LLM output."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=SentimentResult,
        system_prompt="Analyze the sentiment of the text. Be precise with confidence scores.",
    )

    result = await agent.run("I absolutely love this product! Best purchase ever!")

    assert isinstance(result.output, SentimentResult)
    assert result.output.sentiment in {"positive", "negative", "neutral"}
    assert 0 <= result.output.confidence <= 1


# =============================================================================
# Pattern 5: Streaming Responses (08_streaming.py)
# =============================================================================


@pytest.mark.asyncio
async def test_streaming_text(openrouter_model):
    """Test streaming text responses."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant. Keep responses brief.",
    )

    chunks = []
    async with agent.run_stream("Count from 1 to 5, one number per line.") as response:
        async for chunk in response.stream_text():
            chunks.append(chunk)

    full_response = "".join(chunks)
    # Note: Some providers may return content in fewer chunks
    assert len(chunks) >= 1, "Should receive at least one chunk"
    assert "1" in full_response
    assert "5" in full_response


@pytest.mark.asyncio
async def test_streaming_structured(openrouter_model):
    """Test streaming with structured output."""
    from pydantic_ai import Agent

    class CountResult(BaseModel):
        numbers: list[int] = Field(description="List of numbers")
        total: int = Field(description="Sum of all numbers")

    agent = Agent(
        model=openrouter_model,
        output_type=CountResult,
        system_prompt="You help with number operations.",
    )

    async with agent.run_stream("List the numbers 1, 2, 3 and their sum.") as response:
        result = await response.get_output()

    assert isinstance(result, CountResult)
    assert 1 in result.numbers
    assert result.total == sum(result.numbers)


# =============================================================================
# Pattern 6: Conversation History (12_conversation_history.py)
# =============================================================================


@pytest.mark.asyncio
async def test_conversation_history(openrouter_model):
    """Test multi-turn conversation with history."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant. Remember previous messages.",
    )

    # First turn
    result1 = await agent.run("My name is Bob. Remember it.")
    history = result1.all_messages()

    # Second turn with history
    result2 = await agent.run("What is my name?", message_history=history)

    assert "Bob" in result2.output


# =============================================================================
# Pattern 7: Dynamic System Prompts (02_prompts.py)
# =============================================================================


@pytest.mark.asyncio
async def test_dynamic_system_prompt(openrouter_model):
    """Test dynamic system prompts based on context."""
    from pydantic_ai import Agent, RunContext

    @dataclass
    class LanguageContext:
        language: str

    agent = Agent(
        model=openrouter_model,
        deps_type=LanguageContext,
    )

    @agent.system_prompt
    def set_language(ctx: RunContext[LanguageContext]) -> str:
        return f"Always respond in {ctx.deps.language}. Keep it brief."

    # Test German response
    deps = LanguageContext(language="German")
    result = await agent.run("Say 'hello world'", deps=deps)

    # Should contain German words
    german_indicators = ["hallo", "welt", "grüß", "guten"]
    assert any(word in result.output.lower() for word in german_indicators)


# =============================================================================
# Pattern 8: Multiple Tools (03_tools.py)
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_tools(openrouter_model):
    """Test agent with multiple tools."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="Use the appropriate tool for each task. Be brief.",
    )

    tools_called = set()

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        tools_called.add("weather")
        return f"Weather in {city}: 22°C, sunny"

    @agent.tool_plain
    def get_time(timezone: str) -> str:
        """Get the current time in a timezone."""
        tools_called.add("time")
        return f"Current time in {timezone}: 14:30"

    result = await agent.run("What's the weather in Berlin and time in CET?")

    assert "weather" in tools_called, "Weather tool should be called"
    assert "time" in tools_called, "Time tool should be called"
    assert "22" in result.output or "Berlin" in result.output


# =============================================================================
# Pattern 9: Union Output Types (pydantic-evals patterns)
# =============================================================================


class SuccessResponse(BaseModel):
    """Successful operation response."""

    result: str
    confidence: float


class ErrorResponse(BaseModel):
    """Error response."""

    error_message: str
    error_code: str


@pytest.mark.asyncio
async def test_union_output_types(openrouter_model):
    """Test union types for success/error responses."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=SuccessResponse | ErrorResponse,  # type: ignore
        system_prompt="Analyze queries. Return SuccessResponse for valid queries, ErrorResponse for invalid ones.",
    )

    # Valid query should return success
    result = await agent.run("Analyze: The sky is blue.")

    assert isinstance(result.output, (SuccessResponse, ErrorResponse))


# =============================================================================
# Pattern 10: Complex Nested Structured Output
# =============================================================================


class Address(BaseModel):
    """Address component."""

    street: str
    city: str
    country: str


class Person(BaseModel):
    """Person with nested address."""

    name: str
    age: int
    address: Address


@pytest.mark.asyncio
async def test_nested_structured_output(openrouter_model):
    """Test complex nested Pydantic models."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        output_type=Person,
        system_prompt="Extract person information from text.",
    )

    result = await agent.run(
        "John Smith is 30 years old and lives at 123 Main St, New York, USA."
    )

    assert isinstance(result.output, Person)
    assert result.output.name == "John Smith"
    assert result.output.age == 30
    assert isinstance(result.output.address, Address)
    assert result.output.address.city == "New York"


# =============================================================================
# Pattern 11: Model Settings (10_model_settings.py)
# =============================================================================


@pytest.mark.asyncio
async def test_model_settings_temperature(openrouter_model):
    """Test model settings like temperature and max_tokens."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant.",
    )

    # Low temperature for deterministic response
    result = await agent.run(
        "What is 2 + 2? Reply with just the number.",
        model_settings={
            "temperature": 0.0,
            "max_tokens": 10,
        },
    )

    assert "4" in result.output


@pytest.mark.asyncio
async def test_usage_tracking(openrouter_model):
    """Test that usage information is available."""
    from pydantic_ai import Agent

    agent = Agent(
        model=openrouter_model,
        system_prompt="Be brief.",
    )

    result = await agent.run(
        "Say hello.",
        model_settings={"max_tokens": 20},
    )

    usage = result.usage()
    assert usage.total_tokens > 0
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


# =============================================================================
# Pattern 12: Multi-Agent Orchestration (11_multi_agent.py)
# =============================================================================


@pytest.mark.asyncio
async def test_multi_agent_sequential(openrouter_model):
    """Test sequential multi-agent orchestration."""
    from pydantic_ai import Agent

    # Agent 1: Extract keywords
    class Keywords(BaseModel):
        keywords: list[str] = Field(description="Important keywords from text")

    extractor = Agent(
        model=openrouter_model,
        output_type=Keywords,
        system_prompt="Extract 3 important keywords from the text.",
    )

    # Agent 2: Generate summary using keywords
    summarizer = Agent(
        model=openrouter_model,
        system_prompt="Write a one-sentence summary based on the keywords provided.",
    )

    # Sequential pipeline
    text = "Python is a programming language known for its simplicity and readability."

    # Step 1: Extract keywords
    keywords_result = await extractor.run(f"Extract keywords: {text}")
    keywords = keywords_result.output.keywords

    # Step 2: Summarize based on keywords
    summary_result = await summarizer.run(f"Summarize using these keywords: {keywords}")

    assert len(keywords) >= 1
    assert len(summary_result.output) > 10


@pytest.mark.asyncio
async def test_multi_agent_parallel(openrouter_model):
    """Test parallel multi-agent execution."""
    import asyncio

    from pydantic_ai import Agent

    # Create multiple agents for different tasks
    summarizer = Agent(
        model=openrouter_model,
        system_prompt="Summarize in one sentence.",
    )

    sentiment_analyzer = Agent(
        model=openrouter_model,
        system_prompt="Reply with only: positive, negative, or neutral.",
    )

    text = "I love learning new programming languages!"

    # Run both agents in parallel
    summary_task = summarizer.run(f"Summarize: {text}")
    sentiment_task = sentiment_analyzer.run(f"Sentiment: {text}")

    results = await asyncio.gather(summary_task, sentiment_task)

    summary = results[0].output
    sentiment = results[1].output.strip().lower()

    assert len(summary) > 5
    assert any(word in sentiment for word in ["positive", "negative", "neutral"])


@pytest.mark.asyncio
async def test_agent_routing(openrouter_model):
    """Test agent routing based on intent classification."""
    from pydantic_ai import Agent

    # Intent classifier
    classifier = Agent(
        model=openrouter_model,
        system_prompt="Classify as 'math' or 'general'. Reply with just the word.",
    )

    # Specialized agents
    math_agent = Agent(
        model=openrouter_model,
        system_prompt="You are a math expert. Solve the problem.",
    )

    general_agent = Agent(
        model=openrouter_model,
        system_prompt="You are a helpful assistant.",
    )

    # Math query
    query = "What is 7 times 8?"

    # Classify intent
    intent_result = await classifier.run(query)
    intent = intent_result.output.strip().lower()

    # Route to appropriate agent
    if "math" in intent:
        agent = math_agent
    else:
        agent = general_agent

    answer_result = await agent.run(query)

    assert "56" in answer_result.output


# =============================================================================
# Pattern 13: OpenRouter Provider Configuration (06_openrouter.py)
# =============================================================================


@pytest.mark.asyncio
async def test_openrouter_provider_explicit(openrouter_api_key):
    """Test explicit OpenRouter provider configuration."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # Explicit provider setup (as shown in 06_openrouter.py)
    provider = OpenAIProvider(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = OpenAIChatModel(
        model_name="anthropic/claude-haiku-4.5",
        provider=provider,
    )

    agent = Agent(
        model=model,
        system_prompt="Be brief.",
    )

    result = await agent.run("Say 'hello'")

    assert len(result.output) > 0


# =============================================================================
# Pattern 14: Logfire Integration Pattern (07_logfire.py)
# =============================================================================


@pytest.mark.asyncio
async def test_logfire_span_pattern(openrouter_model):
    """Test the logfire span pattern (without actual logfire)."""
    from pydantic_ai import Agent

    # This test validates the pattern works even without logfire configured
    agent = Agent(
        model=openrouter_model,
        system_prompt="Be brief.",
    )

    # The pattern from 07_logfire.py
    result = await agent.run("What is 1 + 1?")

    # Simulate what logfire would capture
    output = result.output
    assert "2" in output

    # Verify we can access result attributes that logfire would log
    assert result.all_messages() is not None
    assert result.usage() is not None
