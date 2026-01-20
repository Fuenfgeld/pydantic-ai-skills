# Pydantic AI Skills

Claude Code skills for building and evaluating Pydantic AI agents. These skills provide comprehensive reference documentation and examples for AI agent development.

## Skills Included

### 1. pydantic-ai-agents

Reference skill for building Pydantic AI agents with best practices covering:

- **Dependencies** - Dependency injection patterns for state management
- **System Prompts** - Dynamic, context-aware prompt engineering
- **Tools** - Function calling with proper context handling
- **Validators** - Structured output validation with Pydantic models
- **OpenRouter** - Multi-model access through unified API
- **Logfire** - Debugging and observability integration
- **Streaming** - Real-time response streaming
- **Multi-Agent** - Orchestrating multiple specialized agents
- **Conversation History** - Persistent memory across turns

### 2. pydantic-evals

Reference skill for testing and evaluating AI agents with:

- **Datasets** - Type-safe test case collections
- **Evaluators** - Deterministic, LLM-as-Judge, custom, and span-based
- **Integration** - Seamless Pydantic AI and Logfire integration
- **Best Practices** - Evaluation-driven development workflows

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
├── exampleCode/
│   ├── pydanticAIcodeExamples.py # Core agent patterns
│   ├── logfireExample.py         # Observability patterns
│   ├── openrouterexample.py      # Multi-model provider
│   └── PydanticEvalDoc/          # Complete evaluation examples
└── tests/                        # Comprehensive test suite
```

## Development

### Running Tests

```bash
# Run all mocked tests (CI-safe, no API keys needed)
uv run pytest tests/ -v --ignore=tests/integration/

# Run integration tests (requires API keys)
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/ --cov=skills --cov=exampleCode
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy skills/ exampleCode/
```

## Core Principles

This project follows:

- **TDD** - Test-Driven Development
- **KISS** - Keep It Simple, Stupid
- **SOLID** - Clean, maintainable architecture
- **EDD** - Evaluation-Driven Development

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Resources

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic Evals Documentation](https://ai.pydantic.dev/evals/)
- [Logfire Documentation](https://logfire.pydantic.dev/)
- [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills)
