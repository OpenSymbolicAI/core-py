# OpenSymbolicAI Core (Python)

<p align="center">
  <img src="assets/demo.gif" alt="OpenSymbolicAI Demo" width="800">
</p>

**Make AI a software engineering discipline.**

## Why This Architecture?

**LLMs are untrusted.** They're stochastic, may be trained on poisoned data, and change under the hood without notice. The more tokens they produce, the further they drift. More instructions often make things *worse*.

**Current orchestration is risky.** Most agent frameworks dump instructions and data together in the context window, then let the LLM loop freely:

```
Instructions + Data + Tools → LLM → Tool call → Output → LLM → Tool call → ...
```

This creates injection risks: data can masquerade as instructions, like SQL injection attacks. And since LLMs are autoregressive, the more context you add, the less reliable they become.

**OpenSymbolicAI separates concerns:**

| Problem | How We Solve It |
|---------|-----------------|
| Data influences planning unpredictably | **Planning is isolated.** LLM sees only the query and primitive signatures—not your data |
| LLM can make unplanned tool calls | **Execution is deterministic.** LLM is a leaf node—it plans, then execution happens without LLM in the loop |
| Prompt injection and data exfiltration | **Symbolic Firewall.** LLM operates on variable names, not raw content. Data stays in application memory, never tokenized. [Learn more](https://www.opensymbolic.ai/blog/security-by-design) |
| Side effects are hidden | **Mutations are explicit.** `read_only=False` primitives trigger approval hooks before execution |
| Outputs are unpredictable JSON/markdown | **Outputs are typed.** Pydantic models guarantee structured, validated results |
| Long contexts cause drift | **Context is minimal.** Only what's needed goes to the LLM—faster, cheaper, more reliable |
| Model changes break prompts | **Model-agnostic.** Constrained inputs/outputs minimize variability across models |
| Failures lose progress | **Checkpoint system.** Pause/resume execution across distributed workers with full state serialization |
| Hard to debug what happened | **Full tracing.** Before/after namespace snapshots, argument expressions, resolved values, timing—every step recorded |

> **Thesis:** Stop *prompting*. Start *programming*.

---

## What This Repo Is

`core-py` is the **Python runtime for OpenSymbolicAI**: the core primitives and execution model for building LLM-powered systems as *software*, not as a pile of strings.

**Core concepts:**
- **Primitives** - Atomic operations your agent can directly execute
- **Decompositions** - Examples showing how to break complex intents into primitive sequences
- **PlanExecute** - Blueprint that uses LLM to plan, then executes deterministically

**Related:** [opensymbolicai-cli](https://github.com/OpenSymbolicAI/cli-py) — Interactive TUI for discovering and running agents

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
uv run python run_calculator.py              # uses gpt-oss:20b by default
uv run python run_calculator.py qwen3:1.7b   # specify a model
uv run python run_calculator.py qwen3:1.7b -v # verbose mode (shows plans)
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

Ollama, OpenAI, Anthropic, Fireworks, Groq, or add your own.

---

## Benchmarks

Run the calculator benchmark to evaluate model performance:

```bash
uv run python benchmarks/calculator/benchmark.py                  # all models
uv run python benchmarks/calculator/benchmark.py --models qwen3:1.7b  # specific model
uv run python benchmarks/calculator/benchmark.py --limit 20 -v    # quick test, verbose
```

See [benchmarks/calculator/README.md](benchmarks/calculator/README.md) for full options (parallel execution, categories, JSON export).

### Model Recommendations (Ollama)

| Model | Accuracy | Notes |
|-------|----------|-------|
| `gpt-oss:20b` | 100% | Best accuracy, larger model |
| `qwen3:1.7b` | 100% | Best balance of accuracy & size |
| `qwen3:8b` | 100% | Perfect accuracy |
| `gemma3:4b` | 94% | Tested on 120 intents |
| `phi4:14b` | 80% | Strong, larger model |

**Recommendations:**
- **Primary choice:** `qwen3:1.7b` - fast, accurate, small footprint
- **Higher accuracy:** `gemma3:4b` - proven on larger test set
- **Best accuracy:** `gpt-oss:20b` or `qwen3:8b` - 100% on all tests

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
