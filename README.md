# Pydantic AI Skills for Claude Code

[![CI](https://github.com/Fuenfgeld/pydantic-ai-skills/actions/workflows/ci.yml/badge.svg)](https://github.com/Fuenfgeld/pydantic-ai-skills/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic AI](https://img.shields.io/badge/Pydantic%20AI-1.44+-green.svg)](https://ai.pydantic.dev/)

Production-ready **Claude Code skills** for building type-safe AI agents with **Pydantic AI**. Comprehensive reference implementations covering dependency injection, tool calling, structured outputs, streaming, multi-agent orchestration, and LLM evaluation patterns.

## Why Use These Skills?

- **✅ All examples are tested** - Every code example runs in CI. No broken snippets, no outdated syntax—guaranteed working code
- **Battle-tested patterns** - Real-world implementations, not toy examples
- **Type-safe by design** - Full Pydantic validation for inputs and outputs
- **Multi-model support** - Works with OpenAI, Anthropic, OpenRouter, and more
- **Evaluation-driven** - Built-in testing patterns for AI agent quality assurance
- **Production-ready** - Includes observability with Logfire integration

## Skills Included

### 1. pydantic-ai-agents — Building AI Agents

Complete reference for building production AI agents with Pydantic AI:

| Pattern | Description |
|---------|-------------|
| **Dependency Injection** | Type-safe state management with dataclasses |
| **System Prompts** | Dynamic, context-aware prompt engineering |
| **Tool Calling** | Function tools with proper context handling |
| **Structured Outputs** | Pydantic model validation for LLM responses |
| **OpenRouter Integration** | Multi-model access (GPT-4, Claude, Llama, etc.) |
| **Logfire Observability** | Debugging, tracing, and monitoring |
| **Response Streaming** | Real-time token streaming |
| **Multi-Agent Systems** | Orchestrating specialized agent teams |
| **Conversation Memory** | Persistent history across turns |

### 2. pydantic-evals — Testing AI Agents

Reference for evaluation-driven AI development:

| Feature | Description |
|---------|-------------|
| **Type-Safe Datasets** | Structured test case collections |
| **Multiple Evaluators** | Deterministic, LLM-as-Judge, custom, span-based |
| **Logfire Integration** | Trace-aware evaluation metrics |
| **Best Practices** | Evaluation-driven development (EDD) workflows |

## Installation

### Using UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pydantic-ai-skills.git
cd pydantic-ai-skills

# Install with UV
uv sync
```

### Using pip

```bash
pip install -e ".[dev]"
```

## Quick Start

### Environment Setup

Create a `.env` file with your API keys (see `.env.example`):

```bash
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
LOGFIRE_API_KEY=your_key
```

### Using as Claude Code Skills

Copy the skill directories to your Claude Code skills location:

```bash
# Copy skills to Claude Code
cp -r skills/pydantic-ai-agents ~/.claude/skills/
cp -r skills/pydantic-evals ~/.claude/skills/
```

### Project Structure

```
├── skills/
│   ├── pydantic-ai-agents/
│   │   ├── SKILL.md              # Main skill documentation
│   │   └── references/           # 12 reference implementation files
│   └── pydantic-evals/
│       ├── SKILL.md              # Main skill documentation
│       └── references/           # Evaluator examples and guides
└── tests/                        # Comprehensive test suite
```

## Development

### Running Tests

All skill examples are covered by unit tests to ensure every code pattern works correctly. Tests run automatically on every PR via GitHub Actions.

```bash
# Run all mocked tests (CI-safe, no API keys needed)
uv run pytest tests/ -v --ignore=tests/integration/

# Run integration tests (requires API keys)
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/ --cov=skills
```

**Test coverage includes:**
- All 12 reference files in `pydantic-ai-agents`
- All evaluator examples in `pydantic-evals`
- Both mocked tests (fast, CI-safe) and real API integration tests

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy skills/
```

## Use Cases

These skills help you build:

- **Chatbots & Assistants** - Customer support, internal tools, personal assistants
- **Data Processing Agents** - ETL pipelines, document analysis, data extraction
- **Code Generation** - AI-powered development tools and code review
- **Research Agents** - Information retrieval, summarization, analysis
- **Workflow Automation** - Multi-step task orchestration with LLMs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects & Resources

- [Pydantic AI](https://ai.pydantic.dev/) - The AI agent framework this skill teaches
- [Pydantic Evals](https://ai.pydantic.dev/evals/) - Evaluation framework for AI agents
- [Logfire](https://logfire.pydantic.dev/) - Observability platform for Python
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/) - AI coding assistant
- [OpenRouter](https://openrouter.ai/) - Unified API for multiple LLM providers

## Keywords

`pydantic-ai` `ai-agents` `llm` `claude-code` `evaluation` `testing` `python` `structured-output` `tool-calling` `multi-agent` `openrouter` `logfire` `dependency-injection` `type-safe`
