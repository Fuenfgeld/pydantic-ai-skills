"""
Example: Capabilities and Native Tools (NEW in Pydantic AI v2)

Capabilities are THE core v2 primitive: a single composable unit that bundles
an agent's tools, hooks, instructions, and model settings. Configuration that
was spread across Agent arguments in v1 (builtin_tools=, prepare_tools=,
history_processors=, instrument=, event_stream_handler=) now lives in
`capabilities=[...]`.
"""

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    NativeTool,
    Thinking,
    WebSearch,
)
from pydantic_ai.native_tools import WebSearchTool

# ============================================================================
# 1. PROVIDER-ADAPTIVE CAPABILITIES
# ============================================================================
# Reach for these first - they survive model/provider changes:
#   Thinking, WebSearch, WebFetch, ImageGeneration, MCP

thinking_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    name="thinking_search_agent",
    capabilities=[
        Thinking(effort="high"),  # unified reasoning across providers
        WebSearch(),  # provider-native web search
    ],
)

# NOTE (v2 behavior change): WebSearch()/WebFetch() are native-only by default
# and RAISE on models without native support. Restore the v1 local fallback
# explicitly with WebSearch(local='duckduckgo') / WebFetch(local=True).


# ============================================================================
# 2. NATIVE TOOLS (renamed from "builtin tools" in v2)
# ============================================================================
# Use NativeTool(...) when you want provider-native behavior with
# provider-specific configuration. v1's Agent(builtin_tools=[...]) is removed.

native_search_agent = Agent(
    # For OpenAI web search use the Responses API prefix ('openai:' also maps
    # to the Responses API in v2; 'openai-chat:' is Chat Completions)
    "openai-responses:gpt-5.2",
    name="native_search_agent",
    capabilities=[
        NativeTool(WebSearchTool(user_location={"city": "Berlin"})),
    ],
)


# ============================================================================
# 3. THE Capability CONVENIENCE CLASS - bundle instructions + tools
# ============================================================================
# Capability bundles instructions and function tools without subclassing.
# Add defer_loading=True to keep its instructions and tool schemas OUT of the
# prompt until the model loads the capability on demand (progressive
# disclosure - great for long-tail workflows with capable production models).

refunds = Capability(
    id="refunds",
    description="Use for refund eligibility and refund status.",
    instructions="Always confirm the order ID before issuing a refund.",
    # defer_loading=True,  # enable on-demand loading with a real model
)


@refunds.tool
def refund_status(ctx: RunContext[Any], order_id: str) -> str:
    """Look up the refund status for an order."""
    return f"Order {order_id}: refund issued on 2026-05-01."


@refunds.tool_plain
def refund_policy() -> str:
    """Return the current refund policy."""
    return "Refunds are available within 30 days of purchase."


support_agent = Agent(
    "test",  # the built-in test model - replace with a real model in production
    name="support_agent",
    capabilities=[refunds],
)


# ============================================================================
# 4. CUSTOM CAPABILITIES - subclass AbstractCapability
# ============================================================================
# Subclass when you need to bundle model settings, hooks, or native tools
# into one reusable, installable unit.


@dataclass
class DeepReasoning(AbstractCapability[Any]):
    """Raise reasoning effort - only when the model opts into the workflow."""

    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(extra_body={"reasoning_effort": "high"})


reasoning_agent = Agent(
    "openai-responses:gpt-5.2",
    name="reasoning_agent",
    capabilities=[
        DeepReasoning(
            id="deep-reasoning",
            description="Use for multi-step planning or hard analytical problems.",
            defer_loading=True,
        ),
    ],
)


# Best Practices for Capabilities:
# 1. Prefer provider-adaptive capabilities (WebSearch, Thinking) so code
#    survives provider changes
# 2. Use NativeTool(...) only when provider-specific config matters
# 3. Use Capability(id=..., description=...) to bundle related tools +
#    instructions; consider defer_loading=True for long-tail workflows
# 4. Subclass AbstractCapability for reusable bundles with settings/hooks
# 5. Keep capabilities focused - one tool or one hook doesn't need one


if __name__ == "__main__":
    # The support agent runs offline against the built-in 'test' model
    result = support_agent.run_sync("What is the refund status of order 42?")
    print(result.output)
