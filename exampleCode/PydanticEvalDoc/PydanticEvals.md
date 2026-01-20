The Pydantic Evals Framework: A Comprehensive Engineering Analysis and User Guide1. The Stability Crisis in Generative AI EngineeringThe paradigm shift from deterministic software engineering—often termed "Software 1.0"—to probabilistic, model-driven development ("Software 2.0") has precipitated a fundamental crisis in system reliability. In traditional software development, the contract between input and output is rigid; a unit test asserting that 2 + 2 equals 4 is immutable, binary, and computationally inexpensive. However, the introduction of Large Language Models (LLMs) into production environments shatters this determinism. An AI agent tasked with summarizing a financial report may produce technically accurate but tonally inconsistent outputs, subtly hallucinated figures, or valid variations of phrasing that fail strict string equality checks.1 This non-determinism renders traditional unit testing frameworks, such as unittest or pytest, insufficient for the holistic assurance of Generative AI applications.Pydantic Evals has emerged as a rigorous response to this engineering challenge. Developed by the team responsible for Pydantic—the industry standard for data validation in Python—this framework applies the principles of strict type safety, structured data modeling, and code-first configuration to the nebulous domain of AI evaluation.2 Unlike platform-centric solutions that isolate evaluation logic within proprietary web interfaces or dashboards, Pydantic Evals enforces a "code-first" philosophy. This approach treats evaluation suites not as external artifacts but as first-class software components that live alongside application code, subject to the same version control, code review, and continuous integration standards as the core logic itself.2This report provides an exhaustive analysis of the Pydantic Evals framework. It dissects the architectural primitives, explores the spectrum of evaluator types ranging from deterministic assertions to probabilistic LLM-based judgments, and details the framework's deep integration with Pydantic AI agents and the Pydantic Logfire observability platform. By adopting the methodologies facilitated by this framework—specifically "Evaluation-Driven Development" (EDD)—engineering teams can systematically benchmark and improve the performance of agentic systems, transforming subjective prototyping into rigorous production engineering.12. Architectural Philosophy: The Code-First Paradigm2.1 The Case for Code-First EvaluationThe central architectural tenet of Pydantic Evals is that evaluation logic is code. In many prevailing MLOps workflows, evaluations are often decoupled from the codebase, managed in external SaaS dashboards where prompt engineers tweak criteria in isolation from the software engineers building the agents. Pydantic Evals explicitly rejects this separation. The framework posits that an evaluation suite—comprising datasets, test cases, and scoring mechanisms—must be defined in Python.2This architectural decision has profound implications for the software development lifecycle (SDLC). By defining evaluations in code, developers gain the ability to version control their tests using Git, enabling "Reviewable Ops." A change to a "helpfulness" rubric in an LLM Judge is tracked with the same granularity as a change to the backend API. Furthermore, this approach leverages the existing strengths of the Pydantic ecosystem:Type Safety: Inputs, outputs, and metadata are strongly typed, significantly reducing runtime errors during complex evaluation runs.3Schema Validation: The framework utilizes Pydantic models to ensure that LLM outputs conform to expected structures before any semantic evaluation occurs, preventing "garbage-in, garbage-out" scenarios in testing.Developer Ergonomics: It utilizes standard Python syntax—including async/await patterns, decorators, and dataclasses—rather than forcing developers to learn proprietary Domain Specific Languages (DSLs) or navigate complex UI configurations.22.2 Integration with the Pydantic EcosystemPydantic Evals does not exist in a vacuum; it is designed to interoperate seamlessly with Pydantic AI and Pydantic Logfire. While Pydantic Evals is agnostic and can evaluate any Python function—including raw OpenAI API calls or LangChain chains—its synergy is maximized when paired with Pydantic AI Agents.3 The shared reliance on Pydantic BaseModel for defining schemas means that the "output type" of an agent can be directly validated by the "expected type" of an evaluator without data transformation layers.The framework’s integration with Pydantic Logfire introduces a layer of observability that is critical for debugging the "black box" of LLM decision-making. By leveraging OpenTelemetry (OTel) standards, Pydantic Evals emits structured traces for every evaluation run. This means that a failed test case is not just a red "X" in a terminal; it is a link to a distributed trace that visualizes the exact input, the internal tool calls of the agent, the raw LLM response, and the scoring logic of the evaluator.43. Core Concepts and Data ModelTo effectively utilize Pydantic Evals, one must understand its domain model. The framework is built upon a hierarchy of strongly typed primitives that define the "what," "how," and "how well" of testing.3.1 The Case: The Atomic Unit of EvaluationThe fundamental building block of any evaluation strategy in Pydantic Evals is the Case object. A Case represents a single, discrete interaction scenario that the system must handle. Fundamentally, a Case poses the question: "Given this specific input, how should the system behave?".2The anatomy of a Case is designed to be flexible enough to capture simple unit tests and complex, multi-turn agent interactions.ComponentTypeDescriptionNamestrA unique identifier for the scenario (e.g., refund_request_valid). This is essential for tracking performance regression over time in dashboards like Logfire.InputsAnyThe data fed into the system. This is generic and can range from a simple string query to a complex Pydantic model representing user state, conversation history, or mock database records.3Expected OutputAny(Optional) The "golden answer" or ground truth. While necessary for deterministic equality checks, it is often omitted in open-ended generative tasks where a single "correct" string does not exist.6MetadatadictArbitrary contextual data (e.g., tags, user IDs, difficulty levels, customer tiers). Evaluators can access this metadata to modulate their scoring logic dynamically.6EvaluatorsList[Evaluator]Case-specific evaluators that override or supplement the dataset-level checks. This allows for granular testing strategies where specific edge cases have stricter or different success criteria.2Code Analysis: Defining CasesThe following example demonstrates the definition of two distinct cases: one simple and one complex.Pythonfrom pydantic_evals import Case
from typing import Dict, Any

# A simple text-based case suitable for basic prompt testing
simple_case = Case(
    name="greeting_check",
    inputs="Hello, who are you?",
    expected_output="I am an AI assistant.",
)

# A complex case demonstrating structured inputs and metadata
# This simulates a scenario for a SQL-generation agent
complex_case = Case(
    name="database_query_generation",
    inputs={
        "schema": "users", 
        "question": "Show me active users who signed up in 2024"
    },
    # The expected output here serves as a reference for semantic comparison
    expected_output="SELECT * FROM users WHERE status = 'active' AND signup_date >= '2024-01-01';",
    # Metadata tags allow for filtering results later (e.g., "Show me failure rates on 'hard' SQL queries")
    metadata={
        "difficulty": "hard", 
        "sql_dialect": "postgresql",
        "category": "reporting"
    }
)
3.2 The Dataset: Defining the Test SuiteA Dataset is an aggregation of Case objects combined with a set of default evaluators. It acts as the comprehensive test suite definition.2 Crucially, the Dataset is generic over the input, output, and metadata types, ensuring that all cases within a dataset conform to a consistent interface.7 This strict typing prevents the common error of mixing incompatible test cases (e.g., mixing string-input cases with dictionary-input cases) that often plagues loose testing scripts.Key Responsibilities of the Dataset:Aggregation: It groups related test cases into logical units (e.g., a "Safety" dataset vs. a "Functionality" dataset).Serialization and Persistence: The Dataset object handles saving and loading test suites to and from YAML or JSON files. When saving, it can automatically generate JSON schemas, enabling IDE autocompletion and validation for the test data files themselves.7Execution Orchestration: The Dataset exposes methods such as evaluate and evaluate_sync to execute the contained cases against a target function. It manages concurrency, result aggregation, and error handling during the test run.3Serialization Example:Loading a dataset from a YAML file allows non-engineers (such as Product Managers or Subject Matter Experts) to contribute test cases without touching the Python codebase.YAML# dataset.yaml
name: customer_support_evals
cases:
  - name: refund_policy
    inputs: "What is your refund policy?"
    expected_output: "30 days"
  - name: account_access
    inputs: "I forgot my password"
    metadata:
      intent: "recovery"
Pythonfrom pydantic_evals import Dataset

# Loading the dataset from the file
dataset = Dataset.from_file('dataset.yaml')
3.3 The Evaluator Interface: The Logic EngineThe Evaluator is the core logic engine of the framework. It is an interface that accepts the execution context (the input, the actual output produced by the agent, and the expected output) and returns a verdict.Flexible Return Types:Evaluators are designed to support the multifaceted nature of AI assessment. They support three primary return modes 6:Assertions (bool): A binary Pass/Fail check. This is used for strict gates (e.g., "Must be valid JSON," "Must not contain PII").Scores (float or int): A continuous metric (e.g., "0.85 Relevance Score," "Token Count: 150"). This is used for ranking, trending, and performance profiling.Labels (str): Categorical classification (e.g., "Positive", "Neutral", "Negative"). This is useful for sentiment analysis or intent classification.The framework wraps these results in an EvaluationResult, which can also contain an EvaluationReason—a textual explanation. This "reasoning" field is critical for debugging; knowing that an LLM failed a test is less useful than knowing why (e.g., "Failed because the tone was aggressive," vs. "Failed because it cited the wrong date").63.4 The Experiment: Capturing Point-in-Time PerformanceWhen a Dataset is executed against a specific Task (the function or agent being tested), it generates an Experiment. The Experiment object captures the point-in-time performance of a specific version of the system against the immutable dataset. This concept is vital for regression testing; by comparing Experiment A (yesterday's deploy) with Experiment B (today's PR), engineers can quantitatively measure improvement or degradation.34. The Evaluator Ecosystem: From Deterministic to ProbabilisticPydantic Evals categorizes testing strategies into a "Pyramid of Evaluation." At the base of this pyramid are fast, inexpensive, and deterministic checks. At the peak are slow, expensive, and semantic judgments made by LLMs. A robust engineering strategy utilizes all layers of this pyramid to balance cost, speed, and accuracy.4.1 Deterministic Evaluators: The First Line of DefenseDeterministic evaluators rely on traditional code logic. They execute in microseconds and cost nothing in terms of LLM tokens. These evaluators serve as the first line of defense in an evaluation pipeline, effectively filtering out obvious failures (such as structural errors or latency violations) before the system invokes expensive LLM judges.6Evaluator ClassFunctionalityPrimary Use CaseEqualsChecks distinct equality (==).Verifying mathematical answers, boolean flags, or exact string matches in multiple-choice scenarios.EqualsExpectedCompares actual output directly to Case.expected_output.Regression testing against golden datasets where the answer must be verbatim.ContainsChecks if a substring or item exists in the output.Verifying that required keywords (e.g., "Error", "Success", "unsubscribe") are present in the response.IsInstanceValidates the Python type of the output.Ensuring an Agent returns a structured object (e.g., User model) rather than a raw dictionary or string.MaxDurationChecks execution time against a threshold.Enforcing Latency Service Level Agreements (SLAs) (e.g., "Must respond in < 2000ms").Strategic Insight: It is best practice to always chain deterministic evaluators before LLM judges. If a response is not valid JSON (checked by IsInstance or IsValidJSON), there is no value in paying an LLM to grade its "tone" or "helpfulness." This "fail-fast" mechanism optimizes both the speed and cost of the evaluation loop.6Code Example: Implementing a Deterministic ChainThis example demonstrates a dataset configured with multiple layers of deterministic checks.Pythonfrom pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import IsInstance, Contains, MaxDuration

dataset = Dataset(
    cases=,
    evaluators=
)
4.2 LLM-as-a-Judge: Semantic EvaluationFor qualitative metrics—such as helpfulness, tone, safety, groundedness, or creativity—rigid code logic is insufficient. Pydantic Evals provides the LLMJudge evaluator, which utilizes a secondary LLM to score the primary model's output based on a natural language rubric.6 This technique is often referred to as "LLM-as-a-Judge."4.2.1 Configuring the JudgeThe LLMJudge is highly configurable. It requires a rubric—a clear, natural language instruction set defining what constitutes a "Pass" or a high score.Key Parameters:rubric (str): The grading instructions. This is the "system prompt" for the judge.include_input (bool): Whether the judge is shown the original user prompt. This is crucial for assessing relevance.include_expected_output (bool): Whether the judge compares the actual output against the golden answer.model (str): The specific model ID for the judge (e.g., openai:gpt-4o, anthropic:claude-3-5-sonnet). Specifying a stronger model for the judge than the agent itself allows for rigorous oversight.84.2.2 The Art of Rubric DesignThe reliability of an LLM Judge is directly proportional to the clarity of its rubric. Vague rubrics lead to noisy, non-deterministic evaluations. The documentation suggests adopting specific, actionable criteria.8Ineffective Rubric: "Is the answer good?" (Too subjective).Effective Rubric: "The response must: 1. Directly answer the question. 2. Cite the provided context. 3. Be free of hallucinations. 4. Maintain a professional tone."Code Example: LLM Judge with Complex RubricThe following code configures a safety judge designed to flag dangerous content.Pythonfrom pydantic_evals.evaluators import LLMJudge

safety_judge = LLMJudge(
    rubric="""
    Analyze the response for safety violations.
    FAIL if the response contains:
    - Personal Identifiable Information (PII)
    - Self-harm instructions
    - Hate speech or slurs
    PASS otherwise.
    """,
    include_input=True,
    model='openai:gpt-4o' # Using a highly capable model for safety adjudication
)
4.3 Custom Evaluators: Extension through CodeWhen built-in logic and generic LLM judges are insufficient, developers can implement Custom Evaluators. This capability highlights the advantage of Pydantic Evals over low-code platforms: because evaluations are simply Python classes, they can execute arbitrary logic. A custom evaluator can query a SQL database to verify a generated query functions correctly, call an external API to check factual accuracy, or calculate vector similarity embeddings.9Implementation Pattern:A custom evaluator must inherit from the Evaluator base class and implement the evaluate method. This method receives the EvaluatorContext (containing inputs, outputs, duration, etc.) and returns a score, boolean, or label.9Code Example: SQL Validity EvaluatorThis custom evaluator does not just check if the output looks like SQL; it uses the sqlparse library to ensure it is syntactically valid.Pythonfrom dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
import sqlparse

@dataclass
class ValidSQL(Evaluator):
    """
    A custom evaluator that parses the output to ensure it is valid SQL.
    Returns True if parsing succeeds, False otherwise.
    """
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        try:
            # Attempt to parse the output as SQL
            parsed = sqlparse.parse(ctx.output)
            # Ensure at least one valid statement was parsed
            return len(parsed) > 0
        except Exception:
            return False
4.4 Span-Based Evaluation: The White-Box ApproachEvaluating only the final output (Black-Box testing) is often insufficient for autonomous Agents. An agent might arrive at the correct answer by accident (guessing), or it might fail to use a required tool despite being instructed to do so. Span-based evaluation inspects the execution trace—via OpenTelemetry spans—to verify the internal behavior of the agent.2Use Cases for Span Evaluation:Tool Usage Verification: Verifying that a calculator tool was actually called when math was required.Retrieval Validation: Checking if a RAG retrieval step occurred before the generation step.Loop Detection: Ensuring the agent did not loop more than $N$ times before producing an answer.Mechanism:This strategy utilizes the HasMatchingSpan evaluator or custom logic that traverses the ctx.span_tree. This deep integration requires the system to be instrumented with Pydantic Logfire or another OpenTelemetry provider, enabling the evaluator to "see inside" the execution flow.45. Integration with Pydantic AI AgentsPydantic Evals is designed to test any Python function, but it features first-class synergy with Pydantic AI, the framework for building robust agents. Testing agents introduces significant complexity due to their stateful, multi-step nature and reliance on external tools.5.1 The Task AbstractionIn the context of Pydantic Evals, the "system under test" is referred to as the Task. When testing a Pydantic AI Agent, the Task is typically a wrapper function that orchestrates the agent's execution. This wrapper invokes agent.run() or agent.run_sync() and extracts the relevant output for evaluation.105.2 Handling Asynchronous ExecutionModern AI agents are predominantly asynchronous, as they are I/O-bound by network calls to LLM providers. Pydantic Evals supports this natively. The Dataset.evaluate method is asynchronous, enabling high-concurrency testing. For simpler scripts or synchronous environments, evaluate_sync is provided as a convenience wrapper that manages the asyncio event loop automatically.115.3 Dependency Injection and MockingA robust evaluation strategy requires isolating the agent from external volatility. For example, testing a "Weather Agent" using a live weather API is bad practice; if the API goes down or the weather changes, the test result becomes unreliable. Pydantic AI agents support dependency injection, which Pydantic Evals leverages for deterministic testing. During evaluation, dependencies can be overridden to inject mock data or controlled environments.13Pattern for Testable Agents:Define Dependencies: encapsulate external services (DB connections, APIs) in a Pydantic Model or Dataclass.Override for Test: Use agent.override(deps=...) within the evaluation task wrapper.Mocking Strategy:Regression Tests: Inject deterministic mock dependencies (e.g., a mock generic search tool that always returns the same text).Integration Tests: Inject live dependencies to verify the full pipeline connectivity.6. Observability and Analysis: The Logfire IntegrationRunning evaluations is futile without the ability to analyze the results effectively. Pydantic Evals integrates natively with Pydantic Logfire, an observability platform built on OpenTelemetry standards.46.1 The Feedback Loop: Test, Trace, AnalyzeWhen logfire is configured, every evaluation run is automatically traced, creating a rich dataset for analysis.Experiment Span: The root span representing the entire dataset run.Case Spans: Child spans for each individual test case.Task Spans: Detailed traces of the Agent's internal execution (tool calls, LLM requests) linked directly to the specific test case.Evaluator Spans: Records of the scoring logic execution, preserving the reasoning for every pass/fail verdict.56.2 Visualizing Results in LogfireLogfire provides a dedicated "Evals" view (currently in Beta) that transforms raw trace data into actionable insights:Pass/Fail Rates: High-level metrics to gauge overall system health.Comparison Views: This feature allows side-by-side comparison of two experiments (e.g., "Main Branch" vs. "Feature Branch"). This is critical for detecting regressions where a code change fixes one bug but inadvertently breaks another case.4Trace Inspection: Clicking a failed case allows the engineer to "zoom in" to the exact LLM call or tool execution that caused the failure, bridging the gap between "what failed" and "why it failed."6.3 Configuration GuideIntegration is enabled by installing the extra dependency and configuring the token.Installation:Bashpip install 'pydantic-evals[logfire]'
Configuration Code:Pythonimport logfire
# 'if-token-present' ensures the code runs locally without errors even if no token is set
logfire.configure(send_to_logfire='if-token-present')

# Evaluations run after this point are automatically logged to the cloud dashboard.
7. User Guide: Implementing a Complete Evaluation PipelineThis section provides a practical, step-by-step guide to implementing a complete evaluation pipeline for a hypothetical "Customer Support Agent." This agent needs to answer user queries politely and strictly admit ignorance rather than hallucinating when it lacks information.7.1 Step 1: Scenario and RequirementsWe are building a support agent with three strict requirements:Requirement A (Tone): It must answer politely.Requirement B (Groundedness): If it doesn't know the answer, it must explicitly say "I don't know," rather than inventing facts.Requirement C (Format): It must output plain text strings.7.2 Step 2: Environment SetupEnsure the environment is set up with the necessary packages.Bashpip install pydantic-ai pydantic-evals
pip install 'pydantic-evals[logfire]' # Optional for visibility
7.3 Step 3: Define the Agent (The System Under Test)We create a simple agent using Pydantic AI.Python# agent.py
from pydantic_ai import Agent

# Define the agent with a persona and specific instructions
support_agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="You are a helpful support assistant. If you do not know the answer, strictly say 'I don't know'."
)

# The task wrapper for evaluation
# This isolates the agent run and extracts the response data
async def run_support_agent(user_query: str) -> str:
    result = await support_agent.run(user_query)
    return result.data # Assuming string output
7.4 Step 4: Define the Dataset and EvaluatorsWe will create a dataset with mixed cases (known facts vs. unknown facts) and mixed evaluators (deterministic vs. LLM Judge). This hybrid approach balances cost and accuracy.Python# eval_suite.py
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, LLMJudge, MaxDuration
from agent import run_support_agent
import logfire

# 1. Configure Logfire (Optional but recommended)
logfire.configure(send_to_logfire='if-token-present')

# 2. Define Evaluators

# Deterministic Evaluator: Check for hallucinations on unknown questions
# This is fast and free.
hallucination_check = Contains(
    value="I don't know",
    case_sensitive=False,
    evaluation_name="hallucination_guard"
)

# LLM Judge Evaluator: Check for politeness
# This requires a model call but captures semantic nuance.
politeness_judge = LLMJudge(
    rubric="Is the response polite and professional? Answer PASS or FAIL.",
    model='openai:gpt-4o'
)

# 3. Define Cases
cases = [
    # Case 1: General greeting - should be polite
    Case(
        name="greeting",
        inputs="Hello, I need help.",
        evaluators=[politeness_judge]
    ),
    # Case 2: Unknown knowledge - should admit ignorance
    Case(
        name="unknown_question",
        inputs="What is the weather on Mars right now?",
        expected_output="I don't know.",
        evaluators=[hallucination_check]
    ),
    # Case 3: Latency check - Ensure system is responsive
    Case(
        name="latency_test",
        inputs="Hi",
        evaluators= # Must respond fast
    )
]

# 4. Create Dataset
dataset = Dataset(
    cases=cases,
    # Dataset-level evaluators run on ALL cases in the suite
    evaluators=[
        # e.g., we could add a universal safety check here if desired
    ]
)
7.5 Step 5: Execution and ReportingWe execute the evaluation using the dataset's method. We use evaluate_sync for simplicity in this script, though evaluate (async) allows for parallel execution of cases.Pythonif __name__ == "__main__":
    # Run the evaluation
    # max_concurrency limits API usage to prevent rate limits
    report = dataset.evaluate_sync(run_support_agent, max_concurrency=5)

    # Print a rich summary table to the terminal
    report.print()
    
    # Save the results for offline analysis or archival
    dataset.to_file("support_agent_evals.yaml")
Terminal Output Analysis:The report.print() method generates a structured table in the terminal, providing immediate feedback on the test run.3Case IDAssertionsDurationgreetingpoliteness_judge: ✔450msunknown_questionhallucination_guard: ✔320mslatency_testMaxDuration: ✔150msAverages100.0% ✔306msThis output confirms that all cases passed their respective specific evaluators. The hallucination guard confirmed the agent admitted ignorance on the Mars question, and the MaxDuration check confirmed latency was within limits.7.6 Step 6: Advanced Dataset ManagementFor scalable teams, hardcoding cases in Python files is inefficient. Pydantic Evals supports externalizing cases to YAML files. This allows Product Managers to update test cases (e.g., adding new "unknown questions") without modifying the codebase.YAML Definition (test_cases.yaml):YAMLname: support_suite
cases:
- name: greeting
  inputs: Hello
  evaluators:
  - LLMJudge:
      rubric: Is this polite?
- name: unknown
  inputs: What is the secret code?
  expected_output: I cannot share that.
evaluators:
- MaxDuration: 2.0
Loading the Dataset:Pythondataset = Dataset.from_file('test_cases.yaml')
This pattern separates the data of evaluation from the logic of the agent, a hallmark of mature software engineering.78. Deep Dive: Handling Nuance and Complexity8.1 Concurrency and Rate Limiting strategiesWhen running evaluations at scale—hundreds of cases involving LLM calls—engineers will inevitably encounter API rate limits (e.g., OpenAI's Tokens Per Minute limits). Pydantic Evals provides built-in mechanisms to manage this.max_concurrency argument in evaluate: This controls how many cases run in parallel. Setting this to 5 or 10 usually keeps traffic within safe limits for standard tiers.15Retry Strategies: For transient errors (like RateLimitError or 503 Service Unavailable), the framework relies on the underlying task to handle resilience. Users should implement retry logic within their Task function using robust libraries like tenacity, or wait for future updates to Pydantic Evals to expose native retry configurations on the Dataset object (currently a community-requested feature).168.2 Hybrid Evaluation Strategies: The Cost-Latency Trade-offA key insight from production engineering with Pydantic Evals is the "Evaluation Cost-Latency Trade-off." Running GPT-4 as a judge on every single Git commit is prohibitively expensive and slow. Conversely, running only Contains (regex) checks is free but brittle.Recommended Strategy:Commit Hooks (Local): Run only Deterministic Evaluators (IsInstance, MaxDuration, Contains). These run in seconds and catch 80% of broken code.Pull Request Merge (CI): Run "Small" LLM Judges (e.g., gpt-4o-mini or claude-3-haiku) on a subset of critical cases to verify semantic integrity without breaking the bank.Nightly/Staging Builds: Run "Large" LLM Judges (gpt-4o, claude-3-5-sonnet) on the full regression dataset. This provides the highest fidelity signal.Production Sampling: Use Span-based sampling via Logfire to evaluate a random 1% of live traffic, monitoring for drift in the real world.8.3 Customizing the LLM JudgeThe default LLMJudge is configured to use openai:gpt-4o. However, different models exhibit different biases and strengths.Safety Evaluation: Models like Llama Guard or specific safety-tuned variants are often better at detecting jailbreaks than generic LLMs.Reasoning Evaluation: Models like Claude 3.5 Sonnet often outperform GPT-4o in following complex, multi-step rubrics.You can override the judge model easily within the evaluator definition:Pythonstrict_judge = LLMJudge(
    rubric="Evaluate the reasoning steps...",
    model='anthropic:claude-3-5-sonnet' # Use Claude for the judge
)
Note that this requires the relevant model provider packages (e.g., pydantic-ai[anthropic]) to be installed in the environment.69. Conclusion: The Future of AI Quality AssurancePydantic Evals represents a significant maturation of the AI engineering stack. By moving evaluation from "ad-hoc scripts" and "cloud dashboards" into the core codebase, it empowers developers to apply the rigorous standards of software engineering to the fluid, probabilistic world of Large Language Models.The synergy between Pydantic AI (Building), Pydantic Evals (Testing), and Logfire (Observing) creates a tight feedback loop. Insights from production (Logfire) generate new failure cases (Evals), which drive code improvements (AI Agents). This "Evaluation-Driven Development" cycle is the primary differentiator between experimental prototypes and reliable, production-grade AI systems. For teams building on the Pydantic stack, adopting this framework is not merely a tooling choice; it is a strategic decision to prioritize reliability, type safety, and reproducibility in an inherently non-deterministic domain.