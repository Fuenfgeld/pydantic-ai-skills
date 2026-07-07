"""
Custom evaluators for the time range agent.

Demonstrates:
- returning multiple named checks from one evaluator (dict output)
- async evaluators
- span-based evaluation (did the agent call a specific tool?)
- v2 naming: override get_default_evaluation_name() (the v1
  `evaluation_name` class attribute fallback was removed in v2)
"""

from dataclasses import dataclass
from datetime import timedelta

from models import (
    TimeRangeBuilderSuccess,
    TimeRangeInputs,
    TimeRangeResponse,
)
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluatorOutput,
)
from pydantic_evals.otel import SpanQuery


@dataclass
class ValidateTimeRange(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    """Deterministic checks on the structure of a successful time range."""

    def evaluate(
        self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]
    ) -> EvaluatorOutput:
        if isinstance(ctx.output, TimeRangeBuilderSuccess):
            window_end = ctx.output.max_timestamp_with_offset
            window_size = window_end - ctx.output.min_timestamp_with_offset
            return {
                "window_is_not_too_long": window_size <= timedelta(days=30),
                "window_is_not_in_the_future": window_end <= ctx.inputs["now"],
            }

        return {}  # No evaluation needed for errors


@dataclass
class UserMessageIsConcise(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    """Async evaluator: the user-facing message should be under 50 words."""

    async def evaluate(
        self,
        ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse],
    ) -> EvaluatorOutput:
        if isinstance(ctx.output, TimeRangeBuilderSuccess):
            user_facing_message = ctx.output.explanation
        else:
            user_facing_message = ctx.output.error_message

        if user_facing_message is not None:
            return len(user_facing_message.split()) < 50
        else:
            return {}

    # v2: to customize the name shown in reports, override this method.
    # (Setting an `evaluation_name` class attribute no longer works in v2.)
    def get_default_evaluation_name(self) -> str:
        return "concise_user_message"


@dataclass
class AgentCalledTool(Evaluator[object, object, object]):
    """Span-based evaluation: check the agent called a specific tool.

    Requires Logfire/OTel instrumentation during evaluation - see
    integration.md. Works with any input/output types.
    """

    agent_name: str
    tool_name: str

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> bool:
        return ctx.span_tree.any(
            SpanQuery(
                name_equals="agent run",
                has_attributes={"agent_name": self.agent_name},
                stop_recursing_when=SpanQuery(name_equals="agent run"),
                some_descendant_has=SpanQuery(
                    name_equals="running tool",
                    has_attributes={"gen_ai.tool.name": self.tool_name},
                ),
            )
        )


# Pass these when loading datasets that reference custom evaluators:
# Dataset[...].from_file(path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES)
CUSTOM_EVALUATOR_TYPES = (ValidateTimeRange, UserMessageIsConcise, AgentCalledTool)
