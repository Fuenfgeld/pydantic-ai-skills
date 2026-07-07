"""
Unit tests that EXECUTE every reference example in skills/pydantic-ai-agents.

Every module is imported (running all module-level code, including agent
construction), and the offline-runnable examples (built on the 'test' model)
are executed end-to-end. Examples that call real APIs in their demo functions
are covered by tests/integration/ against OpenRouter.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REFERENCES_DIR = (
    Path(__file__).parent.parent.parent / "skills" / "pydantic-ai-agents" / "references"
)

REFERENCE_FILES = sorted(p.name for p in REFERENCES_DIR.glob("*.py"))

# Reference modules construct provider clients at import time; dummy keys are
# enough since unit tests never make real requests.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
# Prevent 07_logfire.py from sending telemetry during unit tests
# (load_dotenv() inside the module won't override an existing env var).
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")


def load_reference(name: str):
    """Dynamically load a reference module by file name."""
    module_name = f"ref_{name.removesuffix('.py')}"
    spec = importlib.util.spec_from_file_location(module_name, REFERENCES_DIR / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", REFERENCE_FILES)
def test_reference_module_executes(filename):
    """Every reference file must import and run its module-level code."""
    module = load_reference(filename)
    assert module is not None


class TestCapabilitiesExample:
    """13_capabilities_native_tools.py - executes offline on the 'test' model."""

    def test_support_agent_uses_refunds_capability(self):
        module = load_reference("13_capabilities_native_tools.py")
        result = module.support_agent.run_sync("What is the refund status of order 42?")
        # The 'test' model calls all capability tools; both should show up
        assert "refund issued" in result.output
        assert "30 days" in result.output

    def test_refunds_capability_bundles_tools_and_instructions(self):
        module = load_reference("13_capabilities_native_tools.py")
        assert module.refunds.id == "refunds"

    def test_custom_capability_model_settings(self):
        module = load_reference("13_capabilities_native_tools.py")
        cap = module.DeepReasoning(id="deep-reasoning", description="test")
        settings = cap.get_model_settings()
        assert settings["extra_body"] == {"reasoning_effort": "high"}


class TestHooksExample:
    """14_hooks.py - hooks fire during an offline run on the 'test' model."""

    def test_hooks_fire_during_run(self, capsys):
        module = load_reference("14_hooks.py")
        result = module.agent.run_sync("Send an email to alice@example.com")

        captured = capsys.readouterr().out
        assert "Sending" in captured, "before_model_request hook should log"
        assert "AUDIT: executing send_email" in captured, "tool hook should log"
        assert "Email sent" in result.output


class TestToolsetsExample:
    """15_toolsets_mcp.py - FunctionToolset runs offline on the 'test' model."""

    def test_toolset_tools_execute(self):
        module = load_reference("15_toolsets_mcp.py")
        result = module.agent.run_sync("Greet me and tell me the year.", deps="Alice")
        assert "Hello Alice!" in result.output
        assert "2026" in result.output

    def test_toolset_registered_on_agent(self):
        module = load_reference("15_toolsets_mcp.py")
        assert module.datetime_tools.tools.keys() >= {"current_year", "greet_user"}


class TestUsageProperty:
    """v2: result.usage is a property with input_tokens/output_tokens."""

    def test_usage_is_property_not_method(self):
        module = load_reference("15_toolsets_mcp.py")
        result = module.agent.run_sync("hi", deps="Bob")
        usage = result.usage  # would be a bound method object if still v1
        assert usage.input_tokens >= 0
        assert usage.output_tokens >= 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens
