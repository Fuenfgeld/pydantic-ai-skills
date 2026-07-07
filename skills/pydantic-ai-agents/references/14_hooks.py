"""
Example: Lifecycle Hooks (NEW in Pydantic AI v2)

Hooks intercept and modify agent behavior at every stage of a run - model
requests, tool calls, streaming events - using simple decorators. No
subclassing needed. Hooks are themselves a capability: pass the Hooks
instance via `capabilities=[...]`.

Hook families:
- run-level hooks           (before/after a whole run)
- model-request hooks       (before_model_request / after_model_response)
- tool-validation hooks     (before/after argument validation)
- tool-execution hooks      (before_tool_execute / after_tool_execute)
- event-stream hooks        (intercept streamed events)
"""

from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart

hooks = Hooks()


# ============================================================================
# 1. MODEL REQUEST HOOK - observe/modify every request to the model
# ============================================================================


@hooks.on.before_model_request
async def log_request(
    ctx: RunContext[Any], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Log (or rewrite) the messages before they reach the model."""
    print(f"Sending {len(request_context.messages)} messages to the model")
    return request_context


# ============================================================================
# 2. TOOL EXECUTION HOOK - audit or gate specific tools
# ============================================================================
# The `tools=` filter scopes the hook to specific tool names.


@hooks.on.before_tool_execute(tools=["send_email"])
async def audit_email_tool(
    ctx: RunContext[Any],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: ValidatedToolArgs,
) -> ValidatedToolArgs:
    """Audit-log every send_email call; return (possibly modified) args."""
    print(f"AUDIT: executing {call.tool_name} with {args}")
    return args


# ============================================================================
# 3. WIRE HOOKS INTO AN AGENT
# ============================================================================

agent = Agent(
    "test",  # built-in test model - replace with a real model in production
    name="hooks_agent",
    capabilities=[hooks],
)


@agent.tool_plain
def send_email(to: str, subject: str) -> str:
    """Send an email (demo stub)."""
    return f"Email sent to {to}: {subject}"


# v1 -> v2 MIGRATION NOTES:
# - Agent(event_stream_handler=...)  -> capabilities=[ProcessEventStream(...)]
# - Agent(prepare_tools=...)         -> capabilities=[PrepareTools(...)]
#   (a prepare callback returning None now raises TypeError; return [] to
#   mean "no tools this turn")
# - Agent(history_processors=...)    -> capabilities=[ProcessHistory(...)]
# - Agent(instrument=...)            -> capabilities=[Instrumentation(...)]

# Best Practices for Hooks:
# 1. Use Hooks for app-level concerns: logging, metrics, light validation
# 2. Scope tool hooks with tools=['name'] instead of if-checks inside the hook
# 3. Always return the (possibly modified) context/args from a hook
# 4. Multiple hooks for the same event fire in registration order
# 5. For reusable bundles of hooks + tools + settings, subclass
#    AbstractCapability instead (see 13_capabilities_native_tools.py)


if __name__ == "__main__":
    result = agent.run_sync("Send an email to alice@example.com about the launch")
    print(result.output)
