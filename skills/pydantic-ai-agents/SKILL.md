---
name: pydantic-ai-agents
description: "Build and debug Pydantic AI v2 agents using best practices for dependencies, instructions, tools, capabilities, hooks, and structured output validation. Use when the user wants to: (1) Create a new Pydantic AI agent, (2) Debug or fix an existing agent, (3) Add features like tools, validators, capabilities, hooks, or dynamic instructions, (4) Integrate OpenRouter for multi-model access, (5) Add Logfire for debugging/observability, (6) Structure agent architecture with dependency injection, (7) Migrate an agent from Pydantic AI v1 to v2."
---

# Pydantic AI Reference Skill

**Targets Pydantic AI v2** (v2.0.0 released 2026-06-23). See the
[v1 → v2 migration notes](#10-v1--v2-migration-notes) at the end for the
breaking changes this skill's examples already incorporate.

## Pydantic AI Developer Guide

### 0. Environment Setup

Store API keys in a `.env` file and add it to `.gitignore`:
```
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
LOGFIRE_API_KEY=your_key
```

Load with `python-dotenv`: `load_dotenv()`. Never hardcode keys in source code.

### 1. Core Architecture

Pydantic AI agents have five key components:

#### Dependencies (deps):
- **Reference**: `references/01_dependencies.py`
- Use dataclasses to hold API keys, database connections, and user context
- Never use global variables for state

#### Instructions (instructions):
- **Reference**: `references/02_prompts.py`
- **v2 recommended**: use `instructions=` / `@agent.instructions` (instructions
  are NOT carried in message history, so they always reflect the current agent)
- Make them dynamic using the `@agent.instructions` decorator
- Inject data from `ctx.deps` into the prompt string
- Legacy `system_prompt=` / `@agent.system_prompt` still work; their text IS
  carried in message history - only use when replaying history across agents

#### Tools (@agent.tool / @agent.tool_plain):
- **Reference**: `references/03_tools.py`
- **v2**: `@agent.tool` REQUIRES `ctx: RunContext` as first parameter; use
  `@agent.tool_plain` for context-free tools (it raises otherwise)
- Use `ctx.deps` to access injected dependencies

#### Validators (output_type):
- **Reference**: `references/04_validators.py`
- Use Pydantic models to enforce structured output
- Use `@field_validator` for logic checks

#### Capabilities (capabilities) - NEW in v2:
- **Reference**: `references/13_capabilities_native_tools.py`
- THE core v2 primitive: one composable unit bundling tools, hooks,
  instructions, and model settings
- Provider-adaptive capabilities: `Thinking`, `WebSearch`, `WebFetch`,
  `ImageGeneration`, `MCP`
- Replaces v1's `builtin_tools=`, `prepare_tools=`, `history_processors=`,
  `instrument=`, `event_stream_handler=` Agent arguments

### 2. Promoting Instructions (System Prompt Engineering)

Follow these rules when writing system prompts:

1. **Role Definition**: Start with "You are a specialized agent for..."
2. **Context Awareness**: Explicitly mention the data available in the dependencies.
   - **Bad**: "I help users."
   - **Good**: "I help user {ctx.deps.user_name} (ID: {ctx.deps.user_id}) manage their account."
3. **Tool Coercion**: If tools are defined, instruct the model when to use them.
   - **Example**: "Use the lookup_order tool immediately if the user provides an order ID."
4. **Failure Modes**: Define what to do if a tool fails or data is missing.
   - **Example**: "If the database returns no results, politely ask the user for clarification."

### 3. OpenRouter Integration

**Reference**: `references/06_openrouter.py`

OpenRouter is an API gateway that provides access to multiple LLM models through a unified API.
Since Pydantic AI v2 it has **first-class support** - do NOT use the old
OpenAI-compatible `base_url` workaround anymore.

#### Key Points:
- Set `OPENROUTER_API_KEY` in your `.env` file
- Simplest: model string prefix `'openrouter:<vendor>/<model>'`
- Explicit: `OpenRouterModel` + `OpenRouterProvider`
- OpenRouter model ids are always vendor-prefixed (e.g. `openai/gpt-5.2`,
  `anthropic/claude-sonnet-4.6`) - see https://openrouter.ai/models

#### Example Setup:
```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# Simplest (reads OPENROUTER_API_KEY from environment):
agent = Agent('openrouter:anthropic/claude-sonnet-4.6')

# Explicit:
model = OpenRouterModel(
    'openai/gpt-5.2',
    provider=OpenRouterProvider(api_key=os.getenv('OPENROUTER_API_KEY', '')),
)
```

`OpenRouterModelSettings` adds provider routing, reasoning effort
(`openrouter_reasoning={'effort': 'high'}`), and usage accounting.

### 4. Debugging with Logfire

**Reference**: `references/07_logfire.py`

Logfire is a platform tightly integrated with Pydantic AI for debugging and observability.

#### Key Features:
- **Spans**: Track execution time and context of operations
- **Logging Levels**: notice, info, debug, warn, error, fatal
- **Exception Tracking**: Capture stack traces and error context
- **Tracing**: Visualize agent execution flow

#### Setup:
1. Get API key from https://logfire.pydantic.dev/
2. Set `LOGFIRE_API_KEY` in your `.env` file (the SDK auto-reads only
   `LOGFIRE_TOKEN`, so pass the key explicitly)
3. Configure: `logfire.configure(token=LOGFIRE_API_KEY)`
4. Instrument: `logfire.instrument_pydantic_ai()` - traces every agent run,
   model request, and tool call automatically

**v2 note**: the default instrumentation format is version 5; agent run spans
report token usage under `gen_ai.aggregated_usage.*` (model request spans keep
`gen_ai.usage.*`). Update dashboards that read usage from run spans.

#### Usage Pattern:
```python
with logfire.span('Calling Agent') as span:
    result = agent.run_sync("user query")
    span.set_attribute('result', result.output)
    logfire.info('{result=}', result=result.output)
```

#### Reading Traces Back:
- Needs a **read token** (project settings → Read tokens, `LOGFIRE_READ_TOKEN`);
  the write token is rejected with `401`.
- `LogfireQueryClient(read_token=...).query_json_rows(sql, min_timestamp=...)`
  runs SQL against the `records` table. Failed tool calls:
  `WHERE is_exception OR level >= 17`; tool name and arguments are in
  `attributes->>'gen_ai.tool.name'` and `attributes->>'tool_arguments'`.
- See `query_recent_errors()` in `references/07_logfire.py`.

### 5. Advanced Patterns

#### Streaming Responses:
- **Reference**: `references/08_streaming.py`
- Use `agent.run_stream()` for real-time output
- Stream text with `async for chunk in response.stream_text()`
- Stream partial structured output with `response.stream_output()` (v2 rename
  of v1's `.stream()`; `.stream_structured()` is now `.stream_response()`)
- Get final result with `await response.get_output()`
- v2: `agent.run_stream_events()` is a context manager only
  (`async with agent.run_stream_events(...) as events:`)

#### Result Validators & Retry:
- **Reference**: `references/09_result_validators.py`
- Use `@agent.output_validator` for custom validation
- Raise `ModelRetry("feedback")` to trigger retry with guidance
- Set `retries=3` on Agent for auto-retry on validation failure

#### Model Settings & Usage:
- **Reference**: `references/10_model_settings.py`
- Pass `model_settings={'temperature': 0.7, 'max_tokens': 500}` to `agent.run()`
- Use low temperature (0.0-0.3) for factual tasks
- Use high temperature (0.7-1.0) for creative tasks
- **v2**: `result.usage` is a property (NOT `result.usage()`), with
  `input_tokens`/`output_tokens` (renamed from request/response tokens)

#### Multi-Agent Systems:
- **Reference**: `references/11_multi_agent.py`
- Orchestrate multiple specialized agents for complex tasks
- Use `asyncio.gather()` for parallel agent execution
- Implement routing for intent-based agent selection

#### Capabilities & Native Tools (NEW in v2):
- **Reference**: `references/13_capabilities_native_tools.py`
- Pass `capabilities=[...]` on the Agent; prefer provider-adaptive
  capabilities (`Thinking(effort='high')`, `WebSearch()`, `WebFetch()`) so
  code survives provider changes
- v2 renamed "builtin tools" to **native tools**: wrap them as
  `NativeTool(WebSearchTool(...))` when provider-specific config matters
- `WebSearch()`/`WebFetch()` are native-only by default and RAISE on
  unsupported models; restore local fallback with
  `WebSearch(local='duckduckgo')` / `WebFetch(local=True)`
- Bundle instructions + tools with `Capability(id=..., description=...)`;
  add `defer_loading=True` for on-demand loading (progressive disclosure)
- Subclass `AbstractCapability` for reusable bundles with model settings/hooks

#### Lifecycle Hooks (NEW in v2):
- **Reference**: `references/14_hooks.py`
- Create `hooks = Hooks()`, register with `@hooks.on.before_model_request`,
  `@hooks.on.before_tool_execute(tools=['name'])`, etc., then pass
  `capabilities=[hooks]`
- Use for logging, metrics, auditing, and light interception
- Always return the (possibly modified) context/args from a hook

#### Toolsets & MCP (v2 API):
- **Reference**: `references/15_toolsets_mcp.py`
- Group reusable tools in a `FunctionToolset`; share across agents via
  `Agent(toolsets=[...])`
- MCP servers: `pydantic_ai.mcp.MCPToolset` (replaces v1's `MCPServerStdio`/
  `MCPServerSSE`/`MCPServerStreamableHTTP`); manage connections with
  `async with agent:` (replaces `agent.run_mcp_servers()`)
- Load config files with `load_mcp_toolsets(...)`; provider-native MCP via the
  `MCP(url=..., native=True)` capability

### 6. Conversation History (Persistent Memory)

**Reference**: `references/12_conversation_history.py`

By default, each `agent.run()` call is stateless - the agent has no memory of previous interactions. To maintain conversation context across multiple turns, you must pass `message_history`.

#### Key Concepts:

1. **Get messages from result**: After each `run()`, call `result.all_messages()` to get the full conversation
2. **Pass history to next call**: Use `message_history=` parameter on subsequent `run()` calls
3. **Messages are immutable**: Each call returns a NEW list; the original is not modified

#### Basic Pattern:

```python
from pydantic_ai import Agent, ModelMessage

agent = Agent(model=model, instructions="You are helpful.")

# First turn - no history
result1 = agent.run_sync("My name is Alice")
messages: list[ModelMessage] = result1.all_messages()

# Second turn - pass history so agent remembers
result2 = agent.run_sync(
    "What is my name?",
    message_history=messages  # Agent now knows "Alice"
)
messages = result2.all_messages()  # Updated history

# Third turn - continue the conversation
result3 = agent.run_sync(
    "Tell me a joke about my name",
    message_history=messages
)
```

#### Function Signature Pattern:

When building conversation loops, return both the output and messages:

```python
from pydantic_ai import Agent, ModelMessage

def run_agent_with_history(
    user_input: str,
    message_history: list[ModelMessage] | None = None,
) -> tuple[str, list[ModelMessage]]:
    """Run agent and return output + updated history."""
    result = agent.run_sync(
        user_input,
        message_history=message_history or [],
    )
    return result.output, result.all_messages()

# Usage in a conversation loop
history = []
while True:
    user_input = input("You: ")
    response, history = run_agent_with_history(user_input, history)
    print(f"Agent: {response}")
```

#### Converting Custom Message Types:

If you store conversation history in your own format (e.g., database), convert to Pydantic AI format:

```python
from datetime import timezone
from pydantic_ai import ModelMessage
from pydantic_ai.messages import (
    ModelRequest, ModelResponse,
    UserPromptPart, TextPart
)

def convert_to_model_messages(my_messages: list[MyMessage]) -> list[ModelMessage]:
    """Convert custom message format to Pydantic AI format."""
    result: list[ModelMessage] = []

    for msg in my_messages:
        # Ensure timezone-aware timestamp
        ts = msg.timestamp.replace(tzinfo=timezone.utc) if msg.timestamp.tzinfo is None else msg.timestamp

        if msg.role == "user":
            result.append(ModelRequest(
                parts=[UserPromptPart(content=msg.content, timestamp=ts)],
                kind="request",
            ))
        elif msg.role == "assistant":
            result.append(ModelResponse(
                parts=[TextPart(content=msg.content)],
                kind="response",
                timestamp=ts,
            ))

    return result
```

#### Important Notes:

- **ModelMessage is a union type**: It's `ModelRequest | ModelResponse`, not a class you instantiate directly
- **User messages** → `ModelRequest` with `UserPromptPart`
- **Assistant messages** → `ModelResponse` with `TextPart`
- **Timestamps must be timezone-aware**: Use `timezone.utc`
- **Instructions are NOT in message_history**: They're set on the Agent and injected automatically (legacy `system_prompt` text IS carried in history)

### 7. Testing Best Practices

#### Async/Sync Test Separation

**Warning**: When testing Pydantic AI agents, do NOT mix sync and async tests in the same file when using module-level agents.

**Problem**: Module-level agents create httpx clients at import time. When sync tests call `run_sync()`, they create/destroy temporary event loops which can corrupt the httpx connection pool. Subsequent async tests then fail with `Connection error`.

**Solution**: Separate async and sync tests into different files:

```python
# tests/test_agent_sync.py
from src.agent import run_agent_sync

def test_sync_behavior():
    result = run_agent_sync("input")
    assert result.field == expected

# tests/test_agent_async.py (SEPARATE FILE)
# Does NOT import run_agent_sync!
import pytest
from pydantic_ai import Agent

@pytest.mark.asyncio
async def test_async_behavior():
    # Create fresh agent inside test
    agent = Agent(model=model, output_type=Response)
    result = await agent.run("input")
    assert result.output.field == expected
```

**Alternative**: Convert all tests to async to use the same event loop consistently.

### 8. Usage

Build a new agent by:
1. Reading the requirement
2. Selecting the relevant components from the `references/` directory
3. Combining them into a single file following the pattern in `references/05_main.py`
4. Using OpenRouter (`references/06_openrouter.py`) for multi-model access
5. Adding Logfire (`references/07_logfire.py`) for debugging and monitoring
6. Adding conversation history (`references/12_conversation_history.py`) for multi-turn conversations
7. Adding capabilities/native tools (`references/13_capabilities_native_tools.py`) for web search, thinking, or reusable behavior bundles
8. Adding hooks (`references/14_hooks.py`) for logging, auditing, and interception
9. Adding toolsets or MCP servers (`references/15_toolsets_mcp.py`) for reusable/external tools
10. Adding advanced patterns (streaming, validators, multi-agent) as needed

### 9. Complete Example

See `references/05_main.py` for a complete working agent that demonstrates all patterns.

### 10. v1 → v2 Migration Notes

Key breaking changes in Pydantic AI v2 (all examples in this skill are v2):

| v1 | v2 |
|---|---|
| `result.usage()` | `result.usage` (property) |
| `usage.request_tokens` / `response_tokens` | `usage.input_tokens` / `output_tokens`; `Usage` → `RunUsage` |
| `Agent('gpt-4o')` (bare model name) | Raises `UserError` - always provider-prefix: `Agent('openai:gpt-5.2')` |
| `openai:` prefix → Chat Completions | `openai:` → Responses API; use `openai-chat:` for Chat Completions |
| OpenRouter via `OpenAIProvider(base_url=...)` | `OpenRouterModel` / `'openrouter:'` prefix |
| `Agent(builtin_tools=[...])` | `capabilities=[NativeTool(...)]`; `pydantic_ai.builtin_tools` → `pydantic_ai.native_tools` |
| `Agent(prepare_tools=...)`, `history_processors=`, `event_stream_handler=`, `instrument=` | `capabilities=[PrepareTools(...)]`, `ProcessHistory(...)`, `ProcessEventStream(...)`, `Instrumentation(...)` |
| `MCPServerStdio` / `MCPServerSSE` / `MCPServerStreamableHTTP`; `agent.run_mcp_servers()` | `MCPToolset`; `async with agent:` |
| `stream.stream()` / `.stream_structured()` / `.get()` | `.stream_output()` / `.stream_response()` / `.response` |
| `for e in agent.run_stream_events(...)` | `async with agent.run_stream_events(...) as events:` (context manager only) |
| `DeferredToolCalls` / `DeferredToolset` | `DeferredToolRequests` / `ExternalToolset` |
| `@toolset.tool` on ctx-less function | Raises - use `@toolset.tool_plain` |
| Default `end_strategy='early'` | Default `'graceful'` - function tools alongside a successful output tool now RUN; set `end_strategy='early'` explicitly to skip them |
| Un-parameterized `Agent` infers `Agent[None, str]` | Infers `Agent[object, str]` - update `Agent[None, ...]`/`RunContext[None]` annotations to `object` |
| `ModelProfile` dataclass (`.update()`, `.from_profile()`) | `TypedDict` - use `merge_profile()`, `profile.get('field')`, `{**profile, 'field': v}` |
| Instrumentation format ≤4, `gen_ai.usage.*` on run spans | Format 5 default; run spans use `gen_ai.aggregated_usage.*` |
| `pip install pydantic-ai` bundles all providers | Slimmer default (openai/anthropic/google + minimal); add extras: `pydantic-ai[bedrock,groq,...]` |

Full upgrade guide: https://ai.pydantic.dev/changelog/
