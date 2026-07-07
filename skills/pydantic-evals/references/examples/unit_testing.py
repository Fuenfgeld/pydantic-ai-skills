"""
Run an evaluation suite as a unit test (CI/CD gate).

Builds a Dataset in code, evaluates the agent task, and asserts on the
report's assertion pass rate.

v2 NOTES:
- Dataset(name=...) is a REQUIRED keyword argument.
- evaluate()/evaluate_sync() extras (name, max_concurrency, progress, ...)
  are keyword-only.
"""

from datetime import UTC, datetime
from types import NoneType

import logfire
from agent import infer_time_range
from custom_evaluators import UserMessageIsConcise, ValidateTimeRange
from models import TimeRangeInputs, TimeRangeResponse
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance

logfire.configure(
    send_to_logfire="if-token-present",
    environment="development",
    service_name="evals",
)
logfire.instrument_pydantic_ai()

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def build_dataset() -> Dataset[TimeRangeInputs, TimeRangeResponse, NoneType]:
    """Build the evaluation dataset in code.

    (You can also load one from YAML:
    Dataset[...].from_file(path, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES))
    """
    return Dataset[TimeRangeInputs, TimeRangeResponse, NoneType](
        name="time_range_dataset",  # v2: name is required
        cases=[
            Case(
                name="last_week",
                inputs=TimeRangeInputs(prompt="last week", now=NOW),
            ),
            Case(
                name="yesterday",
                inputs=TimeRangeInputs(prompt="yesterday", now=NOW),
            ),
        ],
        evaluators=[
            IsInstance(type_name="TimeRangeBuilderSuccess"),
            ValidateTimeRange(),
            UserMessageIsConcise(),
        ],
    )


def evaluate_dataset():
    dataset = build_dataset()
    # v2: extra arguments like `name` are keyword-only
    report = dataset.evaluate_sync(infer_time_range, name="time-range-ci")
    print(report)

    averages = report.averages()
    assert averages is not None
    assertion_pass_rate = averages.assertions
    assert assertion_pass_rate is not None, "There should be at least one assertion"
    assert assertion_pass_rate > 0.9, (
        f"The assertion pass rate was {assertion_pass_rate:.1%}; it should be above 90%."
    )


if __name__ == "__main__":
    evaluate_dataset()
