"""
Example: Robust Tool Arguments Across Models

Tool-argument schemas that look fine with one model break with another. Two
failures observed in production (OpenRouter, 2026-10-17):

1. ``z-ai/glm-5.3`` sent ``null`` for every parameter typed as ``MyEnum | None``.
   An Enum class becomes ``$ref`` + ``anyOf`` in the JSON schema, which weaker
   models handle badly. The retry message said only "no fields to update", so
   the model repeated the same call until ``retries`` was exhausted.
2. ``anthropic/claude-sonnet-5.5`` sent ``'"alive"'`` (with literal quote
   characters) for a ``Literal["alive", "deceased"]`` parameter and failed
   validation three times.

The pattern below fixes both:
- inline ``Literal[...]`` instead of an Enum class (no ``$ref`` in the schema)
- ``BeforeValidator`` that strips quotes/whitespace and fixes the case
- convert to the domain Enum inside the tool, so the domain stays typed
- ``ModelRetry`` messages that say what was received and show a correct call

Runs fully offline on FunctionModel / TestModel.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BeforeValidator
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

# ============================================================================
# 1. DOMAIN ENUM (what your code uses) vs TOOL LITERAL (what the model sees)
# ============================================================================


class VitalStatus(StrEnum):
    alive = "alive"
    deceased = "deceased"
    unknown = "unknown"


def _clean_lower(value: object) -> object:
    """Strip whitespace and quote characters, lower-case. Non-strings pass through."""
    return value.strip().strip("\"'").strip().lower() if isinstance(value, str) else value


# Inline literal + normaliser: the JSON schema is a plain string enum, no $ref.
VitalArg = Annotated[Literal["alive", "deceased", "unknown"], BeforeValidator(_clean_lower)]


# ============================================================================
# 2. AGENT AND TOOLS
# ============================================================================

STATE: dict[str, str] = {"p1": "unknown"}  # person id -> vital status (demo only)

agent = Agent(deps_type=None, output_type=str, retries=3, name="robust_args_demo")


@agent.tool
async def set_vital_status(ctx: RunContext[None], person: str, status: VitalArg) -> str:
    """Record whether a person is alive.

    Args:
        ctx: Run context (injected).
        person: Person id, e.g. "p1".
        status: "alive", "deceased" or "unknown".
    """
    if person not in STATE:
        raise ModelRetry(f"Unknown person '{person}'. Known ids: {sorted(STATE)}.")
    STATE[person] = VitalStatus(status).value  # convert literal -> domain enum
    return f"{person} is {STATE[person]}."


@agent.tool
async def update_person(
    ctx: RunContext[None],
    person: str,
    age: int | None = None,
    status: VitalArg | None = None,
) -> str:
    """Change one or more facts about a person. Only non-null arguments are applied.

    Args:
        ctx: Run context (injected).
        person: Person id, e.g. "p1".
        age: New age in years.
        status: "alive", "deceased" or "unknown".
    """
    changes = {k: v for k, v in {"age": age, "status": status}.items() if v is not None}
    if not changes:
        # Say what happened AND show a correct call: small models need the example.
        raise ModelRetry(
            "No fields to update: every field you passed was null. "
            "Pass the actual values, e.g. update_person(person='p1', age=40)."
        )
    if "status" in changes:
        STATE[person] = VitalStatus(changes["status"]).value
    return f"Updated {person}: {changes}."


# ============================================================================
# 3. OFFLINE DEMO: a scripted model that sends a quoted, padded value
# ============================================================================


def _scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if not any(isinstance(m, ModelResponse) for m in messages):
        # What Claude Sonnet 5.5 actually sent: quotes inside the string value.
        return ModelResponse(
            parts=[ToolCallPart("set_vital_status", {"person": "p1", "status": '"Alive"'})]
        )
    return ModelResponse(parts=[TextPart(content="Recorded.")])


def demo() -> str:
    """Run the quoted-value call offline; returns the stored status ("alive")."""
    agent.run_sync("Is p1 alive? Yes.", model=FunctionModel(_scripted))
    return STATE["p1"]


# ============================================================================
# BEST PRACTICES
# ============================================================================
# 1. Literal[...] for choices; never an Enum class in a tool signature.
# 2. Normalise with BeforeValidator (quotes, whitespace, case) before validation.
# 3. Convert to your domain enum inside the tool.
# 4. Every ModelRetry: what was received + one concrete correct call.
# 5. Test with FunctionModel using the exact bad payloads you saw in Logfire.

if __name__ == "__main__":
    print(demo())
