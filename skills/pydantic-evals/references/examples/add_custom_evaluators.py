"""
Load a generated dataset, attach custom evaluators, and save a new version.

Shows dataset-wide evaluators vs. per-case evaluators (specific_case=...),
and that custom evaluator types must be passed both when loading and saving.
"""

from pathlib import Path
from types import NoneType

from custom_evaluators import (
    CUSTOM_EVALUATOR_TYPES,
    AgentCalledTool,
    UserMessageIsConcise,
    ValidateTimeRange,
)
from models import (
    TimeRangeInputs,
    TimeRangeResponse,
)
from pydantic_evals import Dataset


def main():
    dataset_path = Path(__file__).parent / "datasets" / "time_range_v1.yaml"
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, NoneType].from_file(dataset_path)
    # Dataset-wide evaluators run on every case
    dataset.add_evaluator(ValidateTimeRange())
    dataset.add_evaluator(UserMessageIsConcise())
    # Per-case evaluator: only runs on the named case
    dataset.add_evaluator(
        AgentCalledTool("time_range_agent", "get_current_time"),
        specific_case="Relative mention without date",
    )
    dataset.to_file(
        Path(__file__).parent / "datasets" / "time_range_v2.yaml",
        custom_evaluator_types=CUSTOM_EVALUATOR_TYPES,
    )


if __name__ == "__main__":
    main()
