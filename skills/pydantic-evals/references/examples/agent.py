"""
The agent under evaluation: infers a structured time range from a user prompt.

This is the "task" that the evaluation examples in this folder run against.
All sibling examples import from this file (`from agent import ...`), so the
whole folder is self-contained and executable.
"""

from __future__ import annotations as _annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from models import TimeRangeInputs, TimeRangeResponse
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

model = OpenRouterModel(
    "openai/gpt-5.2",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key")),
)


@dataclass
class TimeRangeDeps:
    """Dependencies for the time range inference agent.

    Passing `now` via deps (instead of calling datetime.now() inside tools)
    makes the agent deterministic and easy to evaluate.
    """

    now: datetime = field(default_factory=lambda: datetime.now().astimezone())


time_range_agent = Agent[TimeRangeDeps, TimeRangeResponse](
    model,
    name="time_range_agent",
    output_type=TimeRangeResponse,  # type: ignore[arg-type]
    deps_type=TimeRangeDeps,
    instructions="Convert the user's request into a structured time range.",
    retries=1,
)


@time_range_agent.tool
def get_current_time(ctx: RunContext[TimeRangeDeps]) -> str:
    """Get the user's current time and timezone in the format 'Friday, November 22, 2024 11:15:14 PST'."""
    now_str = ctx.deps.now.strftime("%A, %B %d, %Y %H:%M:%S %Z")
    return f"The user's current time is {now_str}."


async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a user prompt. This is the evaluation task."""
    deps = TimeRangeDeps(now=inputs["now"])
    return (await time_range_agent.run(inputs["prompt"], deps=deps)).output
