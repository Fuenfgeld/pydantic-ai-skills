# Core Principles

- **TDD (Test-Driven Development)**: Write tests before implementation
- **KISS (Keep It Simple, Stupid)**: Start simple, add complexity incrementally
- **SOLID Principles**: Maintain clean, maintainable architecture
- **Evaluation-Driven Development**: Treat evaluations as first-class code

## Development Workflow

```
Plan → Document → Test → Implement → Evaluate → Refine
```

### Phase 1: Plan
- Review requirements for the feature/agent
- Design high-level architecture
- Identify evaluation strategies
- **Create implementation plan document in `.claude/plan/` folder in the project folder**
- **Plan must be updated regularly throughout implementation**

### Phase 2: Document
- Write detailed specifications
- Define input/output contracts
- Document expected behavior
- Create architecture decision records

### Phase 3: Test (Write First!)
- Create Pydantic Evals dataset with test cases
- Implement custom evaluators if needed
- Write traditional unit tests for utilities
- Define success criteria

### Phase 4: Implement
- Build Pydantic models
- Implement agent with proper patterns
- Add Logfire instrumentation
- Follow SOLID principles

### Phase 5: Evaluate
- Run evaluation suite
- Analyze Logfire traces
- Identify failures and edge cases
- Iterate until tests pass

### Phase 6: Refine
- Document learnings
- Update SKILL_IMPROVEMENTS.md
- Refactor for clarity
- Update documentation

## Progress Plans

**CRITICAL**: All implementations MUST have a progress plan that is actively maintained.

### Plan Requirements

- **Location**: All plans must be stored in project directory `.claude/plan/` directory
- **Format**: Markdown files named descriptively (e.g., `user-auth-agent.md`, `data-pipeline-refactor.md`)
- **Updates**: Plans must be updated regularly as implementation progresses
- **Content**: Include:
  - Current status and phase
  - Completed tasks (with checkmarks)
  - In-progress tasks
  - Blockers or challenges
  - Next steps
  - Decision records


## Quick Start

### 1. Environment Setup

```bash
# Install dependencies with UV
uv sync

### 3. Development Commands

```bash
# Run all unit tests
uv run pytest tests/unit/

# Run all evaluations
uv run pytest tests/evals/

# Run specific evaluation
uv run python -m tests.evals.<agent_name>.run_evals

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## Observability with Logfire

RolePlayAgent integrates with [Logfire](https://logfire.pydantic.dev/) for comprehensive observability and debugging.
 **Logfire key avialable in .env**:

   LOGFIRE_API_KEY=your-token-here





## Available Skills

This project leverages Claude Code skills:

### pydantic-ai-agents
Use when building agents to reference best practices for:
- Dependencies (dependency injection)
- Dynamic system prompts
- Tools (function calling)
- Structured output validation
- OpenRouter integration
- Logfire observability

### pydantic-evals
Use when creating evaluations to understand:
- Dataset creation and management
- Evaluator types (deterministic, LLM-judge, custom, span-based)
- Integration with Pydantic AI
- Best practices for evaluation strategies

## Configuration

### Model Configuration
Edit `.env` to configure LLM models:
```bash

## Links

RolePlayAgent)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Logfire Documentation](https://logfire.pydantic.dev/)


---

**Remember**:
- Test-Driven Development means writing tests FIRST, then implementing to make them pass
- Evaluations are code, not afterthoughts
- ALL implementations require a progress plan in the project folder `.claude/plan/` that is updated regularly in the project folder
.claude/codeReview is the place for code reviews