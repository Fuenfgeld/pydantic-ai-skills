"""Logfire Integration for Debugging and Tracing Pydantic AI Agents

Logfire is a platform that helps you debug Agent systems through logging, tracing, and observability.
It is tightly integrated with Pydantic AI and provides detailed insights into agent execution.

Prerequisites:
- Set LOGFIRE_API_KEY in your .env file (get from https://logfire.pydantic.dev/)
- Install required packages: logfire, pydantic-ai
"""

import os
from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent

# Load environment variables
load_dotenv()

# Configure logfire with your API key
LOGFIRE_API_KEY = os.getenv('LOGFIRE_API_KEY')
logfire.configure(token=LOGFIRE_API_KEY)

# Example: Basic logging
logfire.info('System check, thanks good its {date}', date='Friday')


# --- Spans: Building Blocks of Traces ---
"""
Logfire spans are the building blocks of traces in the Logfire observability platform.
A span represents a single unit of work or operation within your application, allowing you
to measure the execution time of code and add contextual information to your logs. Each span
can be thought of as a log entry with extra capabilities—specifically, timing and context.
"""

def example_span(agent: Agent):
    """Demonstrate using spans to trace agent execution"""
    with logfire.span('Calling Agent') as span:
        result = agent.run_sync("What are the official languages in Luxembourg")
        span.set_attribute('result', result.output)
        logfire.info('{result=}', result=result.output)


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
    with logfire.span('Calling Agent') as span:
        result = agent.run_sync("What are the official languages in Luxembourg")

        # Log at different severity levels
        logfire.notice('{result=}', result=result.output)
        logfire.info('{result=}', result=result.output)
        logfire.debug('{result=}', result=result.output)
        logfire.warn('{result=}', result=result.output)
        logfire.error('{result=}', result=result.output)
        logfire.fatal('{result=}', result=result.output)


# --- Logging Exceptions ---
"""
The record_exception function attaches details of an exception (such as its type, message,
and stack trace) to a Logfire span. This makes it easy to see errors and their context
within the application's traces when monitoring or debugging.
"""

def example_exception_logging(agent: Agent):
    """Demonstrate exception logging with Logfire"""
    with logfire.span('Calling Agent') as span:
        try:
            result = agent.run_sync("what is LOINC")
            # Simulating an error for demonstration
            raise ValueError(result.output)
        except ValueError as e:
            span.record_exception(e)
            logfire.error('Agent call failed with error', exc_info=e)


# --- Complete Example ---

if __name__ == "__main__":
    from pydantic_ai.models.openai import OpenAIChatModel

    # Create a simple agent for demonstration
    model = OpenAIChatModel('gpt-4o-mini')
    agent = Agent(
        model=model,
        system_prompt="You are a knowledgeable assistant."
    )

    # Run examples
    print("Running span example...")
    example_span(agent)

    print("\nRunning logging levels example...")
    example_logging_levels(agent)

    print("\nRunning exception logging example...")
    example_exception_logging(agent)

    print("\nCheck your Logfire dashboard at https://logfire.pydantic.dev/ to see the traces!")
