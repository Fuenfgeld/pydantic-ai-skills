
"""Developing and Debugging with logfire

logfire is a platform that helps you debug Agent systems. It is tightly integrated with pydanticAI and helps you with logging, tracing, and debugging.
"""
import logfire

# Configure logfire
logfire.configure(token=keyLogFire)

# Send a log
logfire.info('System check, thanks good its {date}', date='Friday')
"""     
Spans

Logfire spans are the building blocks of traces in the Logfire observability platform. A span represents a single unit of work or operation within your application, allowing you to measure the execution time of code and add contextual information to your logs. Each span can be thought of as a log entry with extra capabilities-specifically, timing and context
"""
with logfire.span('Calling Agent') as span:
    result = agent.run_sync("What are the official languages in Luxembourg")
    span.set_attribute('result', result.output)
    logfire.info('{result=}', result=result.output)

"""     
Logging Levels

The next example uses the Logfire API to log the same result at several different severity levels. Each line is a logging statement, but the log level determines how important or critical the message is in the context of your application. Logfire supports multiple log levels, including notice, info, debug, warn, error, and fatal. Here’s what each level means and when you might use it:

    notice: For significant, but not urgent, events that should be highlighted for operators or administrators.
    info: For routine information about the application's operation, typically confirming that things are working as expected.
    debug: For detailed diagnostic messages, usually only enabled during development or troubleshooting.
    warn: For potentially harmful situations that are not immediately critical but may cause problems.
    error: For serious problems that prevent some operation from completing.
    fatal: For very severe errors that will likely cause the application to terminate.
"""

with logfire.span('Calling Agent') as span:
    result = agent.run_sync("What are the official languages in Luxembourg")
    logfire.notice('{result=}', result=result.output)
    logfire.info('{result=}', result=result.output)
    logfire.debug('{result=}', result=result.output)
    logfire.warn('{result=}', result=result.output)
    logfire.error('{result=}', result=result.output)
    logfire.fatal('{result=}', result=result.output)
     
"""
Logging Exceptions

The record_exception function attaches details of an exception (such as its type, message, and stack trace) to a Logfire span. This makes it easy to see errors and their context within the application’s traces when monitoring or debugging.
"""
with logfire.span('Calling Agent') as span:
    try:
        result = agent.run_sync("what is LOINC")
        raise ValueError(result.output)
    except ValueError as e:
        span.record_exception(e)
     
