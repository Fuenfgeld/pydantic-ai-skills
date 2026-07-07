"""
Compare different models on the same dataset using agent.override().

Each evaluate_sync() call produces a named experiment; with Logfire
configured you can compare the experiments side-by-side in the dashboard.
"""

import logfire
from agent import infer_time_range, time_range_agent
from unit_testing import build_dataset

logfire.configure(
    send_to_logfire="if-token-present",
    environment="development",
    service_name="evals",
)
logfire.instrument_pydantic_ai()


def compare_models():
    dataset = build_dataset()
    with logfire.span("Comparing different models for time_range_agent"):
        # v2: model strings must be provider-prefixed; the experiment `name`
        # is keyword-only
        with time_range_agent.override(model="openrouter:openai/gpt-5.2"):
            dataset.evaluate_sync(infer_time_range, name="gpt-5.2")
        with time_range_agent.override(model="openrouter:anthropic/claude-sonnet-4.6"):
            dataset.evaluate_sync(infer_time_range, name="claude-sonnet-4.6")


if __name__ == "__main__":
    compare_models()
