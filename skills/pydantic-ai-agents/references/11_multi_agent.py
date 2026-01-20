"""
Example: Multi-Agent Systems

This file demonstrates how to orchestrate multiple specialized agents
that work together to accomplish complex tasks.
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

# Create OpenRouter provider
provider = OpenAIProvider(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)

model = OpenAIChatModel(
    model_name='openai/gpt-4o-mini',
    provider=provider
)


# ============================================================================
# 1. DEFINE SPECIALIZED AGENTS
# ============================================================================

# Research Agent - Gathers information
class ResearchResult(BaseModel):
    """Research findings."""
    topic: str
    key_facts: list[str]
    sources_summary: str


research_agent = Agent(
    model=model,
    output_type=ResearchResult,
    system_prompt="""You are a research specialist.
    Gather key facts and summarize information on any topic.
    Be thorough but concise."""
)


# Analysis Agent - Analyzes data
class AnalysisResult(BaseModel):
    """Analysis output."""
    insights: list[str]
    patterns: list[str]
    recommendations: list[str]


analysis_agent = Agent(
    model=model,
    output_type=AnalysisResult,
    system_prompt="""You are a data analyst.
    Identify patterns, extract insights, and provide recommendations.
    Be analytical and evidence-based."""
)


# Writer Agent - Creates final output
class WrittenReport(BaseModel):
    """Final written report."""
    title: str
    summary: str
    body: str
    conclusion: str


writer_agent = Agent(
    model=model,
    output_type=WrittenReport,
    system_prompt="""You are a professional writer.
    Create clear, well-structured reports.
    Be engaging and professional."""
)


# ============================================================================
# 2. ORCHESTRATOR PATTERN
# ============================================================================

class ReportOrchestrator:
    """
    Orchestrates multiple agents to produce a research report.

    Pattern:
    1. Research Agent gathers information
    2. Analysis Agent identifies insights
    3. Writer Agent produces final report
    """

    def __init__(self):
        self.research = research_agent
        self.analysis = analysis_agent
        self.writer = writer_agent

    async def generate_report(self, topic: str) -> dict:
        """
        Run the multi-agent pipeline.

        Args:
            topic: The research topic

        Returns:
            Dictionary with all intermediate and final results
        """
        # Step 1: Research
        print(f"📚 Researching: {topic}")
        research_result = await self.research.run(
            f"Research the topic: {topic}"
        )
        research = research_result.output
        print(f"   Found {len(research.key_facts)} key facts")

        # Step 2: Analyze
        print("🔍 Analyzing research...")
        analysis_prompt = f"""
        Analyze these research findings:
        Topic: {research.topic}
        Facts: {research.key_facts}
        Summary: {research.sources_summary}
        """
        analysis_result = await self.analysis.run(analysis_prompt)
        analysis = analysis_result.output
        print(f"   Found {len(analysis.insights)} insights")

        # Step 3: Write Report
        print("✍️  Writing final report...")
        writing_prompt = f"""
        Write a short report based on this research and analysis:

        RESEARCH:
        - Topic: {research.topic}
        - Key Facts: {research.key_facts}

        ANALYSIS:
        - Insights: {analysis.insights}
        - Patterns: {analysis.patterns}
        - Recommendations: {analysis.recommendations}
        """
        report_result = await self.writer.run(writing_prompt)
        report = report_result.output
        print(f"   Report ready: {report.title}")

        return {
            "research": research,
            "analysis": analysis,
            "report": report,
        }


# ============================================================================
# 3. PARALLEL AGENT EXECUTION
# ============================================================================

class ParallelProcessor:
    """
    Run multiple agents in parallel for independent tasks.
    """

    def __init__(self):
        self.agents = {
            "summarizer": Agent(
                model=model,
                system_prompt="Summarize text in 2-3 sentences."
            ),
            "translator": Agent(
                model=model,
                system_prompt="Translate text to Spanish."
            ),
            "sentiment": Agent(
                model=model,
                output_type=str,
                system_prompt="Analyze sentiment. Reply with: positive, negative, or neutral."
            ),
        }

    async def process_all(self, text: str) -> dict:
        """
        Process text with all agents in parallel.

        Using asyncio.gather() for concurrent execution.
        """
        # Create tasks for parallel execution
        tasks = {
            "summary": self.agents["summarizer"].run(f"Summarize: {text}"),
            "translation": self.agents["translator"].run(f"Translate: {text}"),
            "sentiment": self.agents["sentiment"].run(f"Sentiment of: {text}"),
        }

        # Run all in parallel
        results = await asyncio.gather(*tasks.values())

        # Map results back to task names
        return {
            name: result.output
            for name, result in zip(tasks.keys(), results)
        }


# ============================================================================
# 4. AGENT ROUTING (CONDITIONAL)
# ============================================================================

class AgentRouter:
    """
    Route requests to the appropriate specialized agent based on intent.
    """

    def __init__(self):
        # Intent classifier
        self.classifier = Agent(
            model=model,
            output_type=str,
            system_prompt="""Classify the user intent into one of:
            - math: Mathematical calculations
            - code: Programming questions
            - general: General knowledge
            Reply with just the category name."""
        )

        # Specialized agents
        self.math_agent = Agent(
            model=model,
            system_prompt="You are a math expert. Solve problems step by step."
        )

        self.code_agent = Agent(
            model=model,
            system_prompt="You are a coding expert. Provide clean, working code."
        )

        self.general_agent = Agent(
            model=model,
            system_prompt="You are a helpful assistant for general questions."
        )

    async def route_and_answer(self, query: str) -> dict:
        """
        Route query to appropriate agent based on classified intent.
        """
        # Step 1: Classify intent
        intent_result = await self.classifier.run(query)
        intent = intent_result.output.strip().lower()

        # Step 2: Route to specialized agent
        agent_map = {
            "math": self.math_agent,
            "code": self.code_agent,
            "general": self.general_agent,
        }

        agent = agent_map.get(intent, self.general_agent)

        # Step 3: Get answer from specialized agent
        answer_result = await agent.run(query)

        return {
            "intent": intent,
            "answer": answer_result.output,
        }


# Best Practices for Multi-Agent Systems:
# 1. Give each agent a clear, focused role
# 2. Use structured output types for agent-to-agent communication
# 3. Use parallel execution for independent tasks
# 4. Implement routing for specialized handling
# 5. Keep prompts consistent with agent roles
# 6. Log intermediate results for debugging


async def main():
    """Demonstrate multi-agent patterns."""

    # Demo 1: Sequential Orchestration
    print("=" * 60)
    print("SEQUENTIAL ORCHESTRATION")
    print("=" * 60)
    orchestrator = ReportOrchestrator()
    result = await orchestrator.generate_report("renewable energy trends")
    print(f"\nFinal Report Title: {result['report'].title}")
    print(f"Summary: {result['report'].summary}")

    # Demo 2: Parallel Processing
    print("\n" + "=" * 60)
    print("PARALLEL PROCESSING")
    print("=" * 60)
    processor = ParallelProcessor()
    text = "Artificial intelligence is transforming how we work and live."
    parallel_results = await processor.process_all(text)
    print(f"Summary: {parallel_results['summary']}")
    print(f"Spanish: {parallel_results['translation']}")
    print(f"Sentiment: {parallel_results['sentiment']}")

    # Demo 3: Intent Routing
    print("\n" + "=" * 60)
    print("INTENT ROUTING")
    print("=" * 60)
    router = AgentRouter()
    queries = [
        "What is 15 * 23?",
        "How do I write a Python function?",
        "What is the capital of France?",
    ]
    for query in queries:
        result = await router.route_and_answer(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Answer: {result['answer'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
