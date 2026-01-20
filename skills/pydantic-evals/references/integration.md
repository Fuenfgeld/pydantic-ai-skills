# Integration Guide

## Integrating with Pydantic AI Agents

Pydantic Evals is agnostic—it can test any Python function. However, it has first-class integration with Pydantic AI for testing agents.

### The Task Abstraction

The "system under test" is called a `Task`. For Pydantic AI agents, create a wrapper function:

```python
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="You are a helpful customer support agent."
)

# Task wrapper
async def run_agent(user_query: str) -> str:
    """Wrapper that executes agent and extracts output"""
    result = await agent.run(user_query)
    return result.output  # Use result.output, NOT result.data
```

The task function signature must match your Dataset's input/output types.

### Handling Async Execution

Pydantic AI agents are typically async. Pydantic Evals supports this natively:

```python
# Async evaluation (recommended for parallelism)
report = await dataset.evaluate(run_agent, max_concurrency=5)

# Sync wrapper (convenience for simple scripts)
report = dataset.evaluate_sync(run_agent, max_concurrency=5)
```

### Dependency Injection for Testing

Agents often depend on external services (APIs, databases). Use dependency injection to mock these during testing.

#### Define Dependencies

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
import httpx

@dataclass
class AgentDeps:
    api_key: str
    http_client: httpx.AsyncClient

agent = Agent('openai:gpt-4o-mini', deps_type=AgentDeps)

@agent.tool
async def lookup_order(ctx: RunContext[AgentDeps], order_id: str) -> dict:
    """Fetch order details from API"""
    response = await ctx.deps.http_client.get(
        f"https://api.example.com/orders/{order_id}",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"}
    )
    return response.json()
```

#### Override Dependencies in Tests

```python
# Mock HTTP client for deterministic testing
class MockHTTPClient:
    async def get(self, url, **kwargs):
        # Return deterministic mock data
        return MockResponse(json_data={"order_id": "123", "status": "shipped"})

# Task wrapper with dependency override
async def run_agent_with_mocks(user_query: str) -> str:
    test_deps = AgentDeps(
        api_key="test_key",
        http_client=MockHTTPClient()
    )

    result = await agent.run(user_query, deps=test_deps)
    return result.output  # Use result.output, NOT result.data

# Evaluate with mocked dependencies
report = await dataset.evaluate(run_agent_with_mocks)
```

### Testing Structured Outputs

When agents return Pydantic models instead of strings:

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class OrderStatus(BaseModel):
    order_id: str
    status: str
    estimated_delivery: str

agent = Agent('openai:gpt-4o-mini', output_type=OrderStatus)  # Use output_type, NOT result_type

# Task extracts structured data
async def run_agent(query: str) -> OrderStatus:
    result = await agent.run(query)
    return result.output  # This is an OrderStatus instance (use result.output)

# Dataset with typed outputs
from pydantic_evals import Dataset, Case

dataset = Dataset[str, OrderStatus, None](
    cases=[
        Case(
            name="check_order",
            inputs="What's the status of order 123?",
            expected_output=OrderStatus(
                order_id="123",
                status="shipped",
                estimated_delivery="2024-01-15"
            )
        )
    ]
)
```

### Multi-Turn Conversations

Testing agents that maintain conversation state:

```python
from typing import List, Dict

async def run_conversation(messages: List[Dict[str, str]]) -> str:
    """Test multi-turn conversation"""
    conversation_history = []

    for msg in messages:
        result = await agent.run(
            msg['content'],
            message_history=conversation_history
        )
        conversation_history.extend(result.new_messages())  # Use new_messages()

    # Return final response
    return result.output  # Use result.output

# Case with multi-turn input
case = Case(
    name="multi_turn_refund",
    inputs=[
        {"role": "user", "content": "I want a refund"},
        {"role": "user", "content": "Order 123"},
        {"role": "user", "content": "It arrived broken"}
    ],
    expected_output="I've initiated a full refund for order 123."
)
```

## Integrating with Logfire

Logfire provides observability for Pydantic Evals, creating rich traces for every evaluation run.

### Setup

1. Install with Logfire support:

```bash
pip install 'pydantic-evals[logfire]'
```

2. Configure Logfire:

```python
import logfire

# Configure with your token
logfire.configure(send_to_logfire='if-token-present')

# Instrument Pydantic AI for automatic tracing
logfire.instrument_pydantic_ai()
```

3. Run evaluations (now automatically traced):

```python
report = await dataset.evaluate(run_agent)
```

### What Gets Traced

Every evaluation run creates a hierarchical trace:

```
Experiment: customer_support_evals
├── Case: greeting
│   ├── Task Execution: run_agent
│   │   ├── Agent Run: support_agent
│   │   │   ├── LLM Call: openai.gpt-4o-mini
│   │   │   └── Tool Call: lookup_order
│   │   └── Duration: 450ms
│   ├── Evaluator: Contains("Hello")
│   │   └── Result: PASS
│   └── Evaluator: LLMJudge(politeness)
│       ├── LLM Call: openai.gpt-4o
│       └── Result: PASS (Reason: "Response was friendly and professional")
└── Case: refund_request
    └── ...
```

### Viewing Results in Logfire

Access the Logfire dashboard to:

1. **View Pass/Fail Rates**: High-level metrics across all cases
2. **Inspect Individual Cases**: Click any case to see full trace
3. **Debug Failures**: View exact LLM calls, tool executions, and evaluator reasoning
4. **Compare Experiments**: Side-by-side comparison of different model versions
5. **Trend Analysis**: Track performance over time

### Experiment Comparison

Compare two experiments directly:

```python
# Run experiment A (current model)
report_a = await dataset.evaluate(agent_v1)

# Run experiment B (new model)
report_b = await dataset.evaluate(agent_v2)

# Both are automatically logged to Logfire
# Navigate to Logfire → Evals → Compare Experiments
```

Dashboard shows:
- Which cases improved/regressed
- Latency comparisons
- Score distributions
- Tool usage patterns

### Custom Attributes

Add custom metadata to traces:

```python
import logfire

with logfire.span('Custom Operation') as span:
    span.set_attribute('model_version', 'v2.1')
    span.set_attribute('user_segment', 'premium')

    result = await agent.run(query)

    span.set_attribute('token_count', result.usage().total_tokens)
```

These attributes become filterable in the Logfire UI.

### Sampling for Production

In production, trace a sample of live traffic:

```python
import random

async def production_handler(user_query: str):
    # Sample 1% of requests
    if random.random() < 0.01:
        with logfire.span('Production Sample'):
            result = await agent.run(user_query)
            # Evaluation happens automatically if dataset is configured
    else:
        result = await agent.run(user_query)

    return result
```

### Alerts and Monitoring

Configure Logfire alerts based on evaluation metrics:

- Alert if pass rate drops below 90%
- Alert if latency exceeds 2 seconds
- Alert if specific evaluator starts failing

## Environment Configuration

### Development Setup

```python
import os
import logfire
from pydantic_evals import Dataset

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure for development
logfire.configure(
    send_to_logfire='if-token-present',
    environment='development',
    service_name='evals'
)
```

### CI/CD Setup

```yaml
# .github/workflows/evals.yml
name: Run Evaluations

on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LOGFIRE_API_KEY: ${{ secrets.LOGFIRE_API_KEY }}
        run: python run_evals.py

      - name: Check pass rate
        run: |
          # Parse report and fail if below threshold
          python check_pass_rate.py --threshold 0.9
```

### Production Setup

```python
import logfire

# Production configuration
logfire.configure(
    send_to_logfire='always',
    environment='production',
    service_name='ai-agent',
    # Sample traces to reduce volume
    sampling_rate=0.01
)
```

## Best Practices

1. **Isolate External Dependencies**: Always mock APIs, databases, and external services in tests
2. **Use Type Hints**: Leverage Dataset generics for type safety
3. **Version Control Datasets**: Store dataset YAML files in Git
4. **Separate Test Environments**: Use different Logfire projects for dev/staging/prod
5. **Monitor Token Usage**: Track LLM judge costs via Logfire attributes
6. **Implement Retry Logic**: Use tenacity or similar for transient API failures
7. **Cache Expensive Operations**: Cache embeddings, API responses for faster re-runs
8. **Progressive Evaluation**: Run cheap checks first, expensive checks only when needed
