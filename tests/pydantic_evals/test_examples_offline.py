"""
Unit tests that EXECUTE the pydantic-evals skill examples offline.

The examples in skills/pydantic-evals/references/examples/ are self-contained
sibling modules. These tests import every module (running module-level code)
and execute the evaluation workflow end-to-end with a FunctionModel, so no
API keys or network access are needed.
"""

import importlib.util
import os
import sys
from datetime import UTC
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

EXAMPLES_DIR = (
    Path(__file__).parent.parent.parent / "skills" / "pydantic-evals" / "references" / "examples"
)

# Examples construct provider clients at import time; dummy keys suffice.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")

# Sibling imports (`from models import ...`) require the examples dir on sys.path
sys.path.insert(0, str(EXAMPLES_DIR))

EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.py"))


def load_example(name: str):
    """Load an example module under its plain name so sibling imports resolve."""
    module_name = name.removesuffix(".py")
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLES_DIR / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_module_executes(filename):
    """Every example file must import and run its module-level code."""
    assert load_example(filename) is not None


def make_success_model() -> FunctionModel:
    """A FunctionModel that returns a valid TimeRangeBuilderSuccess."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output_tool = next(t for t in info.output_tools if "TimeRangeBuilderSuccess" in t.name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool.name,
                    {
                        "min_timestamp_with_offset": "2026-01-08T12:00:00Z",
                        "max_timestamp_with_offset": "2026-01-15T11:00:00Z",
                        "explanation": "Selected the week before your current time.",
                    },
                )
            ]
        )

    return FunctionModel(respond)


class TestEvaluationWorkflow:
    """unit_testing.py executes end-to-end with a deterministic model."""

    def test_build_dataset_requires_name(self):
        ut = load_example("unit_testing.py")
        dataset = ut.build_dataset()
        assert dataset.name == "time_range_dataset"
        assert len(dataset.cases) == 2

    def test_evaluate_dataset_passes_with_valid_output(self):
        ut = load_example("unit_testing.py")
        ag = load_example("agent.py")
        with ag.time_range_agent.override(model=make_success_model()):
            # Runs the example's own assertions (assertion pass rate > 90%)
            ut.evaluate_dataset()

    def test_report_contains_all_cases(self):
        ut = load_example("unit_testing.py")
        ag = load_example("agent.py")
        dataset = ut.build_dataset()
        with ag.time_range_agent.override(model=make_success_model()):
            report = dataset.evaluate_sync(ag.infer_time_range, name="offline")
        assert len(report.cases) == 2
        assert {c.name for c in report.cases} == {"last_week", "yesterday"}


class TestCustomEvaluators:
    """custom_evaluators.py behavior, including v2 naming API."""

    def test_v2_default_evaluation_name_override(self):
        ce = load_example("custom_evaluators.py")
        evaluator = ce.UserMessageIsConcise()
        assert evaluator.get_default_evaluation_name() == "concise_user_message"

    def test_serialization_name(self):
        ce = load_example("custom_evaluators.py")
        assert ce.ValidateTimeRange.get_serialization_name() == "ValidateTimeRange"

    def test_dataset_roundtrip_with_custom_evaluators(self, tmp_path):
        """add_custom_evaluators.py pattern: add evaluators, save, reload."""
        from datetime import datetime
        from types import NoneType

        from pydantic_evals import Case, Dataset

        ce = load_example("custom_evaluators.py")
        models = load_example("models.py")

        dataset = Dataset[models.TimeRangeInputs, models.TimeRangeResponse, NoneType](
            name="roundtrip_test",
            cases=[
                Case(
                    name="one",
                    inputs=models.TimeRangeInputs(
                        prompt="last week",
                        now=datetime(2026, 1, 15, tzinfo=UTC),
                    ),
                )
            ],
        )
        dataset.add_evaluator(ce.ValidateTimeRange())
        dataset.add_evaluator(ce.UserMessageIsConcise(), specific_case="one")

        path = tmp_path / "dataset.yaml"
        dataset.to_file(path, custom_evaluator_types=ce.CUSTOM_EVALUATOR_TYPES)

        loaded = Dataset[models.TimeRangeInputs, models.TimeRangeResponse, NoneType].from_file(
            path, custom_evaluator_types=ce.CUSTOM_EVALUATOR_TYPES
        )
        assert loaded.name == "roundtrip_test"
        assert len(loaded.evaluators) == 1
        assert len(loaded.cases[0].evaluators) == 1
