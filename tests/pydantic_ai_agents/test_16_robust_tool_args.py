"""
Unit tests for 16_robust_tool_args.py - tool argument schemas that survive weaker models.

Observed in production (GeneticAgent, 2026-10-17, via OpenRouter):
- z-ai/glm-5.3 sent ``null`` for enum parameters typed as ``MyEnum | None``
  (the JSON schema used ``$ref`` + ``anyOf``) and looped on an unhelpful retry.
- anthropic/claude-sonnet-5.5 sent ``'"alive"'`` (literal quote characters)
  for a Literal parameter and exhausted the retries.
"""

import json

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from tests.pydantic_ai_agents.test_reference_examples import load_reference


def _schemas() -> dict[str, dict]:
    module = load_reference("16_robust_tool_args.py")
    captured: dict[str, dict] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for tool in info.function_tools:
            captured[tool.name] = tool.parameters_json_schema
        return ModelResponse(parts=[TextPart(content="ok")])

    module.agent.run_sync("hi", model=FunctionModel(respond))
    return captured


def _call_once(tool: str, args: dict, text: str = "done") -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        responses = sum(isinstance(m, ModelResponse) for m in messages)
        if responses == 0:
            return ModelResponse(parts=[ToolCallPart(tool, args)])
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(respond)


def _retries(messages: list[ModelMessage]) -> list[str]:
    return [
        str(p.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, RetryPromptPart)
    ]


def test_enum_arguments_are_inline_enums_without_refs():
    schema = _schemas()["set_vital_status"]
    assert "$defs" not in schema
    prop = json.dumps(schema["properties"]["status"])
    assert "$ref" not in prop
    assert '"enum"' in prop


def test_quoted_and_padded_values_are_accepted():
    module = load_reference("16_robust_tool_args.py")
    result = module.agent.run_sync(
        "hi", model=_call_once("set_vital_status", {"person": "p1", "status": ' "Alive" '})
    )
    assert _retries(result.all_messages()) == []
    assert module.STATE["p1"] == "alive"


def test_all_null_update_gets_an_example_in_the_retry():
    module = load_reference("16_robust_tool_args.py")
    result = module.agent.run_sync(
        "hi", model=_call_once("update_person", {"person": "p1", "age": None, "status": None})
    )
    retries = _retries(result.all_messages())
    assert len(retries) == 1
    assert "null" in retries[0]
    assert "update_person(person='p1', age=40)" in retries[0]


def test_demo_runs_offline():
    module = load_reference("16_robust_tool_args.py")
    assert module.demo() == "alive"
