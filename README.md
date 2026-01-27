# OpenSymbolicAI Core (Python)

Prompt engineering is often treated like black magic: vibes, lore, and "try this phrasing".

**OpenSymbolicAI turns prompts into code** — so LLM behavior becomes **predictable, testable, reviewable, and composable** using the engineering workflows we already trust: types, modules, unit tests, CI, and benchmarks.

> **Thesis:** Stop *prompting*. Start *programming*.

---

## What This Repo Is

`core-py` is the **Python runtime for OpenSymbolicAI**: the core primitives and execution model for building LLM-powered systems as *software*, not as a pile of strings.

**Core concepts:**
- **Primitives** — Atomic operations your agent can directly execute
- **Decompositions** — Examples showing how to break complex intents into primitive sequences
- **PlanExecute** — Blueprint that uses LLM to plan, then executes deterministically

---

## Why "Prompt → Code" Matters

| Prompts as strings | Prompts as code |
|-------------------|-----------------|
| Hard to reproduce | **Version** behavior, not just text |
| Hard to review | **Diff** and code review changes |
| Brittle or no tests | **Test** expectations (unit + integration) |
| "Model mood" mysteries | **Debug** with execution traces |
| Copy-paste reuse | **Compose** as reusable modules |

---

## Quickstart

### 1. Install

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
```

### 3. Run an example

```bash
cd examples/calculator
uv run python run_calculator.py
```

---

## Example: Scientific Calculator Agent

```python
from opensymbolicai import PlanExecute, primitive, decomposition

class ScientificCalculator(PlanExecute):

    @primitive(read_only=True)
    def add_numbers(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @primitive(read_only=True)
    def convert_degrees_to_radians(self, angle: float) -> float:
        """Convert degrees to radians."""
        return angle * 3.14159 / 180

    @decomposition(
        intent="What is sine of 90 degrees?",
        expanded_intent="Convert to radians, then calculate sine",
    )
    def _example_sine(self) -> float:
        rad = self.convert_degrees_to_radians(angle=90)
        return self.sine(angle_in_radians=rad)
```

The LLM learns from decomposition examples to plan new queries using your primitives.

---

## Supported Providers

- **Ollama** — Local models
- **OpenAI** — GPT-4, etc.
- **Anthropic** — Claude
- **Fireworks** — Fast inference
- **Groq** — Ultra-fast inference

---

## Benchmarks

Run the calculator benchmark to evaluate model performance:

```bash
cd benchmarks/calculator
uv run python benchmark.py
```

### Model Recommendations (Ollama)

| Model | Accuracy | Notes |
|-------|----------|-------|
| `gpt-oss:20b` | 100% | Best accuracy, larger model |
| `qwen3:1.7b` | 100% | Best balance of accuracy & size |
| `qwen3:8b` | 100% | Perfect accuracy |
| `gemma3:4b` | 94% | Tested on 120 intents |
| `phi4:14b` | 80% | Strong, larger model |

**Recommendations:**
- **Primary choice:** `qwen3:1.7b` — fast, accurate, small footprint
- **Higher accuracy:** `gemma3:4b` — proven on larger test set
- **Best accuracy:** `gpt-oss:20b` or `qwen3:8b` — 100% on all tests

---

## Development

### Pre-commit hooks

```bash
uv run pre-commit install          # one-time
uv run pre-commit run --all-files  # run manually
```

### Commands

```bash
uv run ruff check .        # lint
uv run ruff check --fix .  # lint + autofix
uv run mypy src            # type-check
uv run pytest              # run tests
```

---

## Repository Structure

```
src/opensymbolicai/     # Core package
  ├── core.py           # @primitive, @decomposition decorators
  ├── blueprints/       # PlanExecute and Planner
  ├── llm.py            # Multi-provider LLM abstraction
  ├── checkpoint.py     # Distributed execution support
  └── models.py         # Pydantic models
examples/calculator/    # Working example agent
tests/                  # Unit tests
integration_tests/      # Integration tests (requires LLM)
benchmarks/             # Performance benchmarks
```

---

## Contributing

PRs welcome. Please include:
- Unit test in `tests/`
- Integration test in `integration_tests/` (when relevant)
- Benchmark if it impacts runtime-critical paths

---

## License

MIT
