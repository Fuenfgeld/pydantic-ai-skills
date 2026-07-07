"""
Example: Toolsets and MCP (Model Context Protocol) in Pydantic AI v2

Toolsets group reusable tools that can be shared across agents. MCP servers
are connected via MCPToolset (v2 replaces v1's MCPServerStdio/MCPServerSSE/
MCPServerStreamableHTTP/MCPServerHTTP and Agent.run_mcp_servers()).
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

# ============================================================================
# 1. FUNCTION TOOLSETS - reusable groups of tools
# ============================================================================

datetime_tools: FunctionToolset[str] = FunctionToolset()


@datetime_tools.tool_plain
def current_year() -> int:
    """Return the current year (demo stub)."""
    return 2026


# v2 NOTE: @toolset.tool now REQUIRES the first parameter to be a RunContext;
# use @toolset.tool_plain for context-free tools (it raises otherwise).
@datetime_tools.tool
def greet_user(ctx: RunContext[str]) -> str:
    """Greet the user by name (deps carry the name)."""
    return f"Hello {ctx.deps}!"


agent = Agent(
    "test",  # built-in test model - replace with a real model in production
    name="toolset_agent",
    deps_type=str,
    toolsets=[datetime_tools],  # v1's Agent(mcp_servers=[...]) is also toolsets= now
)


# ============================================================================
# 2. MCP SERVERS via MCPToolset (v2 API)
# ============================================================================
# Requires a running/installable MCP server, so this is illustrative:
#
#   from pydantic_ai.mcp import MCPToolset
#
#   server = MCPToolset(
#       command='uvx', args=['mcp-run-python', 'stdio'],  # stdio transport
#       # or: url='http://localhost:8000/mcp'             # HTTP transport
#   )
#   agent = Agent('openai:gpt-5.2', toolsets=[server])
#
#   async def main():
#       # v2: `async with agent:` manages MCP connections
#       # (replaces v1's `agent.run_mcp_servers()`)
#       async with agent:
#           result = await agent.run('Run some Python code to compute 2^10.')
#
# Load servers from a standard MCP config file:
#
#   from pydantic_ai.mcp import load_mcp_toolsets
#   toolsets = load_mcp_toolsets('mcp_config.json')
#
# Or use the provider-adaptive MCP *capability*, which can run a remote
# server URL through the provider's native MCP support:
#
#   from pydantic_ai.capabilities import MCP
#   agent = Agent(..., capabilities=[MCP(url='https://mcp.example.com/mcp', native=True)])
#
# v2 NOTE: MCP(url=...) runs the server LOCALLY by default; pass native=True
# for the v1 provider-native behavior.


# Best Practices for Toolsets:
# 1. Group related tools in a FunctionToolset for reuse across agents
# 2. Use MCPToolset for external MCP servers; manage lifecycle with
#    `async with agent:`
# 3. Set sampling_model on MCPToolset if the server uses MCP sampling
#    (replaces v1's Agent.set_mcp_sampling_model())
# 4. Toolsets compose with capabilities - a Capability can also bundle
#    toolsets via Capability(toolsets=[...])


if __name__ == "__main__":
    result = agent.run_sync("Greet me and tell me the current year.", deps="Alice")
    print(result.output)
