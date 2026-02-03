# Contributing to OpenSymbolicAI Core

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/OpenSymbolicAI/core-py.git
cd core-py

# Install dependencies
uv sync

# Set up pre-commit hooks
uv run pre-commit install

# Copy environment template
cp .env.example .env
# Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
```

## Code Style

We use **ruff** for linting and **mypy** for type checking.

### Linting

```bash
uv run ruff check .          # Check for issues
uv run ruff check --fix .    # Auto-fix issues
```

### Type Checking

```bash
uv run mypy src
```

### Type Annotation Guidelines

Use modern Python 3.12+ syntax:

```python
# Good
def process(items: list[str], config: dict[str, Any] | None = None) -> str | None:
    ...

# Avoid
from typing import List, Dict, Optional
def process(items: List[str], config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    ...
```

### Data Structures

Use **Pydantic models** instead of dataclasses or plain dicts:

```python
from pydantic import BaseModel, Field

class TaskConfig(BaseModel):
    name: str = Field(..., description="Task name")
    retries: int = Field(default=3, ge=0)
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_core.py

# Run with verbose output
uv run pytest -v

# Run integration tests (requires LLM API keys)
uv run pytest integration_tests/
```

### Test Organization

- `tests/` — Unit tests (no external dependencies)
- `integration_tests/` — Tests that require LLM providers
- `benchmarks/` — Performance benchmarks

## Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Write code**: Follow the style guidelines above
3. **Add tests**: Include unit tests in `tests/`, integration tests in `integration_tests/` if applicable
4. **Run checks**: Ensure `ruff check .`, `mypy src`, and `pytest` all pass
5. **Commit**: Use conventional commit messages (see below)
6. **Open PR**: Describe your changes and link any related issues

### Commit Messages

We use **Semantic Release** with conventional commits:

```
<type>: <description>
```

| Type | Description | Version Impact |
|------|-------------|----------------|
| `fix:` | Bug fixes | Patch (0.1.0 → 0.1.1) |
| `feat:` | New features | Minor (0.1.0 → 0.2.0) |
| `feat!:` | Breaking changes | Major (0.1.0 → 1.0.0) |
| `docs:` | Documentation only | No release |
| `chore:` | Maintenance tasks | No release |
| `refactor:` | Code refactoring | No release |
| `test:` | Adding/updating tests | No release |

Examples:

```bash
git commit -m "fix: resolve null pointer in calculator execution"
git commit -m "feat: add support for nested function calls"
git commit -m "feat!: change LLM interface to async-only"
```

## Architecture Overview

```
src/opensymbolicai/
├── core.py              # @primitive, @decomposition decorators
├── blueprints/
│   ├── plan_execute.py  # PlanExecute base class
│   └── planner.py       # LLM planning logic
├── llm.py               # Multi-provider LLM abstraction
├── checkpoint.py        # Pause/resume for distributed execution
└── models.py            # Pydantic models
```

### Key Concepts

- **Primitives** (`@primitive`): Atomic operations the agent can execute
- **Decompositions** (`@decomposition`): Examples showing how to break intents into primitives
- **PlanExecute**: Base class that uses LLM to plan, then executes deterministically

### Example

```python
from opensymbolicai import PlanExecute, primitive, decomposition

class Calculator(PlanExecute):
    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @decomposition(
        intent="What is 2 plus 3?",
        expanded_intent="Add the two numbers",
    )
    def _example_add(self) -> float:
        return self.add(a=2, b=3)
```

## Good First Issues

Looking for a place to start? Check out issues labeled [`good first issue`](https://github.com/OpenSymbolicAI/core-py/labels/good%20first%20issue).

## Questions?

Open an issue or start a discussion. We're happy to help!
