# Core Concepts

## The Case: Atomic Unit of Evaluation

A `Case` represents a single, discrete test scenario. It answers: "Given this specific input, how should the system behave?"

### Anatomy of a Case

| Component | Type | Description |
|-----------|------|-------------|
| **name** | str | Unique identifier for tracking (e.g., "refund_request_valid") |
| **inputs** | Any | Data fed into the system (string, dict, Pydantic model, etc.) |
| **expected_output** | Any (Optional) | Ground truth answer. Often omitted for open-ended tasks |
| **metadata** | dict | Arbitrary context (tags, difficulty, user IDs) accessible to evaluators |
| **evaluators** | List[Evaluator] | Case-specific evaluators that override dataset-level checks |

### Example: Simple Case

```python
from pydantic_evals import Case

simple_case = Case(
    name="greeting_check",
    inputs="Hello, who are you?",
    expected_output="I am an AI assistant."
)
```

### Example: Complex Case

```python
complex_case = Case(
    name="database_query_generation",
    inputs={
        "schema": "users",
        "question": "Show me active users who signed up in 2024"
    },
    expected_output="SELECT * FROM users WHERE status = 'active' AND signup_date >= '2024-01-01';",
    metadata={
        "difficulty": "hard",
        "sql_dialect": "postgresql",
        "category": "reporting"
    }
)
```

## The Dataset: Test Suite Definition

A `Dataset` aggregates `Case` objects with default evaluators. It's generic over input, output, and metadata types, ensuring type consistency across all cases.

### Key Responsibilities

1. **Aggregation**: Groups related test cases (e.g., "Safety" vs. "Functionality" datasets)
2. **Serialization**: Saves/loads from YAML or JSON with schema generation
3. **Execution**: Orchestrates evaluation runs with concurrency management

### Type Safety

```python
from pydantic_evals import Dataset
from typing import TypedDict

class MyInputs(TypedDict):
    user_query: str
    context: str

class MyOutputs(TypedDict):
    answer: str
    confidence: float

# Dataset ensures all cases conform to these types
dataset = Dataset[MyInputs, MyOutputs, None](
    cases=[...],
    evaluators=[...]
)
```

### Serialization

Save datasets to enable non-engineers to contribute test cases:

```yaml
# dataset.yaml
name: customer_support_evals
cases:
  - name: refund_policy
    inputs: "What is your refund policy?"
    expected_output: "30 days"
  - name: account_access
    inputs: "I forgot my password"
    metadata:
      intent: "recovery"
```

Load with:

```python
dataset = Dataset.from_file('dataset.yaml')
```

## The Evaluator Interface: Logic Engine

An `Evaluator` accepts execution context and returns a verdict. It's the core assessment mechanism.

### Return Types

Evaluators support three modes:

1. **Assertions (bool)**: Binary Pass/Fail (e.g., "Must be valid JSON")
2. **Scores (float/int)**: Continuous metrics (e.g., "0.85 Relevance Score")
3. **Labels (str)**: Categorical classification (e.g., "Positive", "Neutral", "Negative")

### EvaluationResult

Results are wrapped in `EvaluationResult` with optional `EvaluationReason` for debugging:

```python
# Evaluator returns
{
    "score": 0.85,
    "reason": "Response was relevant but lacked specific examples"
}
```

### Evaluator Context

Evaluators receive `EvaluatorContext` containing:

- `inputs`: The original input to the system
- `output`: The actual output produced
- `expected_output`: The ground truth (if provided)
- `duration`: Execution time
- `metadata`: Case metadata
- `span_tree`: OpenTelemetry trace (for span-based evaluation)

## The Experiment: Point-in-Time Performance

When a `Dataset` runs against a `Task` (the function being tested), it generates an `Experiment`. This captures:

- Which dataset was used
- Which task/model was evaluated
- Timestamp of evaluation
- Complete results for all cases
- Aggregated metrics (pass rates, average scores, etc.)

### Use Case: Regression Testing

Compare experiments to detect degradation:

```python
# Yesterday's deploy
experiment_a = dataset.evaluate_sync(agent_v1)

# Today's PR
experiment_b = dataset.evaluate_sync(agent_v2)

# Compare
if experiment_b.pass_rate < experiment_a.pass_rate:
    print("⚠️ Regression detected!")
```

### Integration with Logfire

Experiments are automatically logged to Logfire for visualization and comparison in the dashboard.

## Task: The System Under Test

In Pydantic Evals terminology, the "system under test" is the `Task`. For Pydantic AI agents, this is typically a wrapper function:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')

async def task(user_input: str) -> str:
    """Wrapper that executes agent and extracts output"""
    result = await agent.run(user_input)
    return result.data
```

The task function signature must match the Dataset's input/output types.

## Type Flow Example

```python
from pydantic_evals import Case, Dataset
from typing import TypedDict

# Define types
class Inputs(TypedDict):
    query: str

class Outputs(TypedDict):
    response: str

# Create case with typed inputs/outputs
case = Case[Inputs, Outputs](
    name="test1",
    inputs={"query": "Hello"},
    expected_output={"response": "Hi there!"}
)

# Dataset enforces these types
dataset = Dataset[Inputs, Outputs, None](cases=[case])

# Task must match signature
async def my_task(inputs: Inputs) -> Outputs:
    return {"response": "Hi there!"}

# Type-safe evaluation
report = await dataset.evaluate(my_task)
```
