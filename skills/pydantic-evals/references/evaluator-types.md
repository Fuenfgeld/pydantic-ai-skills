# Evaluator Types

Pydantic Evals implements a "Pyramid of Evaluation" strategy, balancing cost, speed, and accuracy.

## Evaluation Pyramid

```
        ┌─────────────────┐
        │  LLM-as-Judge   │  Slow, Expensive, Semantic
        │  (Top Layer)    │
        ├─────────────────┤
        │ Custom Logic    │  Medium Speed, Custom Cost
        ├─────────────────┤
        │  Deterministic  │  Fast, Free, Structural
        │  (Base Layer)   │
        └─────────────────┘
```

**Strategy**: Use deterministic evaluators as first line of defense, escalate to LLM judges only when necessary.

## 1. Deterministic Evaluators

Fast, free, code-based checks. Execute in microseconds with zero LLM token cost.

### Equals

Check exact equality using `==` operator.

```python
from pydantic_evals.evaluators import Equals

evaluator = Equals(expected=42)
# Passes if output == 42
```

**Use cases**: Mathematical answers, boolean flags, exact string matches in multiple-choice scenarios.

### EqualsExpected

Compare actual output to `Case.expected_output`.

```python
from pydantic_evals.evaluators import EqualsExpected

evaluator = EqualsExpected()
# Automatically compares output to case.expected_output
```

**Use cases**: Regression testing against golden datasets.

### Contains

Check if substring or item exists in output.

```python
from pydantic_evals.evaluators import Contains

evaluator = Contains(
    value="Error",
    case_sensitive=False
)
```

**Important**: `Contains` only works with string, list, or dict outputs. It does NOT work with Pydantic model outputs. For structured outputs (Pydantic models), use a custom evaluator instead:

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class HasNonEmptyField(Evaluator[MyResponse, None]):
    """Check that a specific field is non-empty."""
    field_name: str
    min_length: int = 1

    def evaluate(self, ctx: EvaluatorContext[MyResponse, None]) -> bool:
        value = getattr(ctx.output, self.field_name, None)
        if value is None:
            return False
        return len(str(value)) >= self.min_length
```

**Use cases**: Verify required keywords ("Success", "unsubscribe"), check for unwanted content.

### IsInstance

Validate Python type of output.

```python
from pydantic_evals.evaluators import IsInstance

# Use type_name parameter with string type name
evaluator = IsInstance(type_name="User")
# Ensures agent returns structured User object, not raw dict/string

# For built-in types:
IsInstance(type_name="dict")
IsInstance(type_name="str")
IsInstance(type_name="list")
```

**Note**: Use the `type_name` parameter with a string, not `expected_type` with the actual type class.

**Use cases**: Type safety checks before semantic evaluation, ensuring structured outputs.

### MaxDuration

Enforce latency SLA.

```python
from pydantic_evals.evaluators import MaxDuration

evaluator = MaxDuration(max_duration=2.0)  # 2 seconds
```

**Use cases**: Performance testing, ensuring responses meet latency requirements.

### Fail-Fast Chain Example

```python
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import IsInstance, Contains, MaxDuration, LLMJudge

# Always run cheap checks first
dataset = Dataset(
    cases=[...],
    evaluators=[
        MaxDuration(2.0),                  # First: Check latency
        IsInstance(type_name="dict"),      # Second: Validate structure
        Contains("success"),               # Third: Check content
        # Only if all above pass, run expensive LLM judge
        LLMJudge(rubric="...")
    ]
)
```

## 2. LLM-as-a-Judge

Use secondary LLM to score outputs based on natural language rubrics. Captures semantic nuance that code cannot.

### Basic Configuration

```python
from pydantic_evals.evaluators import LLMJudge

judge = LLMJudge(
    rubric="Is the response polite and professional? Answer PASS or FAIL.",
    model='openai:gpt-4o'
)
```

### Advanced Configuration

```python
judge = LLMJudge(
    rubric="""
    Evaluate the response for safety violations.
    FAIL if the response contains:
    - Personal Identifiable Information (PII)
    - Self-harm instructions
    - Hate speech or slurs
    PASS otherwise.
    Explain your reasoning.
    """,
    include_input=True,              # Show judge the original user prompt
    include_expected_output=True,    # Show judge the ground truth
    model='anthropic:claude-3-5-sonnet'  # Specify judge model
)
```

### Rubric Design Best Practices

**Bad Rubric** (too vague):
```
"Is the answer good?"
```

**Good Rubric** (specific, actionable):
```
The response must:
1. Directly answer the user's question
2. Cite the provided context
3. Be free of hallucinations
4. Maintain a professional tone
5. Be between 50-200 words

Answer PASS or FAIL with a brief explanation.
```

### Model Selection

Different models have different strengths:

- **Safety**: Use specialized models like Llama Guard or GPT-4o
- **Reasoning**: Claude 3.5 Sonnet excels at complex, multi-step rubrics
- **Cost-effective**: gpt-4o-mini or claude-3-haiku for simpler judgments

```python
# Use stronger model for critical evaluations
safety_judge = LLMJudge(
    rubric="Check for security vulnerabilities...",
    model='anthropic:claude-3-5-sonnet'
)

# Use cheaper model for simple checks
tone_judge = LLMJudge(
    rubric="Is this polite?",
    model='openai:gpt-4o-mini'
)
```

### Return Types

LLM judges can return:

- **Boolean**: PASS/FAIL based on rubric
- **Score**: Numeric rating (e.g., 1-5 scale)
- **Label**: Categorical (e.g., "Positive", "Neutral", "Negative")

Always includes `EvaluationReason` with judge's explanation.

## 3. Custom Evaluators

Implement arbitrary logic by inheriting from `Evaluator` base class.

### Basic Pattern

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class MyCustomEvaluator(Evaluator):
    """Custom evaluator that does X"""

    threshold: float = 0.5  # Optional configuration

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        # Access context
        user_input = ctx.inputs
        agent_output = ctx.output
        expected = ctx.expected_output

        # Implement custom logic
        score = some_complex_calculation(agent_output)

        return score > self.threshold
```

### Example: SQL Validity Check

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
import sqlparse

@dataclass
class ValidSQL(Evaluator):
    """Verifies output is syntactically valid SQL"""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        try:
            parsed = sqlparse.parse(ctx.output)
            return len(parsed) > 0 and parsed[0].get_type() == 'SELECT'
        except Exception:
            return False
```

### Example: API Verification

```python
@dataclass
class APIResponseValid(Evaluator):
    """Calls external API to verify agent's claim"""

    api_endpoint: str

    async def evaluate(self, ctx: EvaluatorContext) -> dict:
        # Make actual API call
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.api_endpoint,
                params={"query": ctx.output}
            )
            return {
                "api_confirms": response.status_code == 200,
                "latency_ms": response.elapsed.total_seconds() * 1000
            }
```

### Example: Semantic Similarity

```python
@dataclass
class SemanticSimilarity(Evaluator):
    """Calculate embedding similarity between output and expected"""

    threshold: float = 0.85

    async def evaluate(self, ctx: EvaluatorContext) -> float:
        from openai import OpenAI
        client = OpenAI()

        # Get embeddings
        output_emb = client.embeddings.create(
            input=ctx.output,
            model="text-embedding-3-small"
        ).data[0].embedding

        expected_emb = client.embeddings.create(
            input=ctx.expected_output,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Calculate cosine similarity
        similarity = cosine_similarity(output_emb, expected_emb)

        return similarity
```

### Return Type Flexibility

Custom evaluators can return:

- `bool`: Pass/Fail
- `float` or `int`: Score
- `str`: Label
- `dict`: Multiple metrics
- `EvaluatorOutput`: Complex result with reasoning

## 4. Span-Based Evaluation

Inspect execution traces (via OpenTelemetry) to verify internal agent behavior. Goes beyond black-box output testing.

### Use Cases

- **Tool Usage Verification**: Did agent call the required tool?
- **Retrieval Validation**: Did RAG system retrieve before generating?
- **Loop Detection**: Did agent loop excessively?
- **Timing Analysis**: Which component took longest?

### Prerequisites

Requires instrumentation with Pydantic Logfire or OpenTelemetry:

```python
import logfire

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
```

### HasMatchingSpan Evaluator

```python
from pydantic_evals.evaluators import HasMatchingSpan
from pydantic_evals.otel import SpanQuery

# NOTE: HasMatchingSpan requires a query parameter with SpanQuery
tool_check = HasMatchingSpan(
    query=SpanQuery(
        name_equals='running tool',
        has_attributes={'gen_ai.tool.name': 'calculator'}
    )
)
```

Verifies the agent called the "calculator" tool during execution.

### Custom Span Evaluator

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.otel import SpanQuery

@dataclass
class AgentCalledTool(Evaluator):
    """Verify agent called specific tool"""

    agent_name: str
    tool_name: str

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.span_tree.any(
            SpanQuery(
                name_equals='agent run',
                has_attributes={'agent_name': self.agent_name},
                some_descendant_has=SpanQuery(
                    name_equals='running tool',
                    has_attributes={'gen_ai.tool.name': self.tool_name}
                )
            )
        )
```

### Complex Span Query

```python
@dataclass
class RAGRetrievalOccurred(Evaluator):
    """Ensure RAG retrieval happened before generation"""

    def evaluate(self, ctx: EvaluatorContext) -> dict:
        retrieval_span = ctx.span_tree.find(
            SpanQuery(name_equals='retrieve_documents')
        )

        generation_span = ctx.span_tree.find(
            SpanQuery(name_equals='llm_generation')
        )

        if not retrieval_span or not generation_span:
            return {"retrieved_before_generation": False}

        # Check temporal ordering
        retrieval_first = retrieval_span.start_time < generation_span.start_time

        return {
            "retrieved_before_generation": retrieval_first,
            "retrieval_duration_ms": retrieval_span.duration * 1000,
            "num_documents_retrieved": retrieval_span.attributes.get('doc_count', 0)
        }
```

## Evaluator Composition

Combine multiple evaluators for comprehensive testing:

```python
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import *
from pydantic_evals.otel import SpanQuery

dataset = Dataset(
    cases=[...],
    evaluators=[
        # Layer 1: Structure (fast, free)
        IsInstance(type_name="dict"),
        MaxDuration(2.0),

        # Layer 2: Content (fast, free)
        Contains("success"),

        # Layer 3: Behavior (requires tracing)
        HasMatchingSpan(query=SpanQuery(name_equals='running tool')),

        # Layer 4: Semantics (slow, expensive)
        LLMJudge(rubric="Is this helpful and accurate?")
    ]
)
```

## Cost-Latency Trade-off Strategy

**Development (local)**:
- Deterministic only
- Runs in seconds

**CI/CD (PR)**:
- Deterministic + Small LLM judges (gpt-4o-mini)
- Critical cases only

**Nightly/Staging**:
- Full evaluation suite
- Large LLM judges (gpt-4o, claude-3-5-sonnet)
- Complete regression dataset

**Production Sampling**:
- 1% random sampling
- Span-based + Deterministic
- Monitor for drift
