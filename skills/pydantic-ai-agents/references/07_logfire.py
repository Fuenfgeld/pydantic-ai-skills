"""Logfire Integration for Debugging and Tracing Pydantic AI Agents

Logfire is a platform that helps you debug Agent systems through logging, tracing, and observability.
It is tightly integrated with Pydantic AI and provides detailed insights into agent execution.

Prerequisites:
- Set LOGFIRE_API_KEY in your .env file (get from https://logfire.pydantic.dev/)
- Install required packages: logfire, pydantic-ai
"""

import os

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent

# Load environment variables
load_dotenv()

# Configure logfire with your API key
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
logfire.configure(token=LOGFIRE_API_KEY)

# RECOMMENDED: Instrument Pydantic AI so every agent run, model request, and
# tool call is traced automatically. In v2 this uses instrumentation format
# version 5: agent run spans report token usage under gen_ai.aggregated_usage.*
# (model request spans keep gen_ai.usage.*), avoiding double-counting.
logfire.instrument_pydantic_ai()

# Example: Basic logging
logfire.info("System check, thanks good its {date}", date="Friday")


# --- Spans: Building Blocks of Traces ---
"""
Logfire spans are the building blocks of traces in the Logfire observability platform.
A span represents a single unit of work or operation within your application, allowing you
to measure the execution time of code and add contextual information to your logs. Each span
can be thought of as a log entry with extra capabilities—specifically, timing and context.
"""


def example_span(agent: Agent):
    """Demonstrate using spans to trace agent execution"""
    with logfire.span("Calling Agent") as span:
        result = agent.run_sync("What are the official languages in Luxembourg")
        span.set_attribute("result", result.output)
        logfire.info("{result=}", result=result.output)


# --- Logging Levels ---
"""
Logfire supports multiple log levels to categorize the importance of messages:

- notice: For significant, but not urgent, events
- info: For routine information about the application's operation
- debug: For detailed diagnostic messages during development
- warn: For potentially harmful situations
- error: For serious problems that prevent operations
- fatal: For very severe errors that may cause termination
"""


def example_logging_levels(agent: Agent):
    """Demonstrate different logging levels"""
    with logfire.span("Calling Agent"):
        result = agent.run_sync("What are the official languages in Luxembourg")

        # Log at different severity levels
        logfire.notice("{result=}", result=result.output)
        logfire.info("{result=}", result=result.output)
        logfire.debug("{result=}", result=result.output)
        logfire.warn("{result=}", result=result.output)
        logfire.error("{result=}", result=result.output)
        logfire.fatal("{result=}", result=result.output)


# --- Logging Exceptions ---
"""
The record_exception function attaches details of an exception (such as its type, message,
and stack trace) to a Logfire span. This makes it easy to see errors and their context
within the application's traces when monitoring or debugging.
"""


def example_exception_logging(agent: Agent):
    """Demonstrate exception logging with Logfire"""
    with logfire.span("Calling Agent") as span:
        try:
            result = agent.run_sync("what is LOINC")
            # Simulating an error for demonstration
            raise ValueError(result.output)
        except ValueError as e:
            span.record_exception(e)
            logfire.error("Agent call failed with error", exc_info=e)


# --- Complete Example ---

if __name__ == "__main__":
    from pydantic_ai.models.openai import OpenAIChatModel

    # Create a simple agent for demonstration
    model = OpenAIChatModel("gpt-5.2")
    agent = Agent(
        model=model, name="logfire_demo_agent", instructions="You are a knowledgeable assistant."
    )

    # Run examples
    print("Running span example...")
    example_span(agent)

    print("\nRunning logging levels example...")
    example_logging_levels(agent)

    print("\nRunning exception logging example...")
    example_exception_logging(agent)

    print("\nCheck your Logfire dashboard at https://logfire.pydantic.dev/ to see the traces!")


# --- Reading traces back (needs a *read* token, not the write token) ---


def query_recent_errors(read_token: str, minutes: int = 60) -> list[dict]:
    """Most recent exception records, e.g. failed tool calls with their arguments."""
    from datetime import UTC, datetime, timedelta

    from logfire.query_client import LogfireQueryClient

    sql = (
        "SELECT start_timestamp, span_name, message, exception_message, "
        "attributes->>'gen_ai.tool.name' AS tool, "
        "attributes->>'tool_arguments' AS tool_arguments "
        "FROM records WHERE is_exception OR level >= 17 "
        "ORDER BY start_timestamp DESC LIMIT 50"
    )
    since = datetime.now(UTC) - timedelta(minutes=minutes)
    with LogfireQueryClient(read_token=read_token) as client:
        return client.query_json_rows(sql, min_timestamp=since)["rows"]
