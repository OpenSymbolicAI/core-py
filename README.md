<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OpenSymbolicAI/.github/main/profile/opensymbolicai-horizontal-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/OpenSymbolicAI/.github/main/profile/opensymbolicai-horizontal.svg">
    <img alt="OpenSymbolicAI" src="https://raw.githubusercontent.com/OpenSymbolicAI/.github/main/profile/opensymbolicai-horizontal.svg" height="48">
  </picture>
</p>

# OpenSymbolicAI Core (Python)

[![PyPI version](https://img.shields.io/pypi/v/opensymbolicai-core)](https://pypi.org/project/opensymbolicai-core/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/opensymbolicai-core)](https://pypi.org/project/opensymbolicai-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/opensymbolicai-core)](https://pypi.org/project/opensymbolicai-core/)

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

## Performance

Tested on the [TravelPlanner benchmark](https://osu-nlp-group.github.io/TravelPlanner/) (ICML 2024 Spotlight) — 1,225 real-world planning tasks where GPT-4 alone achieves **0.6%**.

### TravelPlanner Results (1,000 test tasks)

| Metric | OpenSymbolicAI |
|--------|----------------|
| **Pass rate** | 97.9% |
| Hard constraints | 100% |
| Commonsense checks | 97.9% |
| Avg. latency | 52.4s |

### Framework Comparison (45 train tasks, same model)

| Metric | OpenSymbolicAI | LangChain | CrewAI |
|--------|----------------|-----------|--------|
| **Pass rate** | 100% | 77.8% | 73.3% |
| Tokens/task | 13,936 | 43,801 | 81,331 |
| LLM calls/task | 2.3 | 13.5 | 39.6 |
| Cost/task | $0.013 | $0.051 | $0.100 |
| Latency | 47s | 73s | 124s |

**Key takeaways:** 3.1x fewer tokens than LangChain, 5.8x fewer than CrewAI. A [$0.006/task open-source model](https://www.opensymbolic.ai/blog/travelplanner-benchmark) (Llama 3.3 70B on Groq) outperforms standalone GPT-4.

> Read more: [TravelPlanner Benchmark Deep Dive](https://www.opensymbolic.ai/blog/travelplanner-benchmark) · [Token Economics](https://www.opensymbolic.ai/blog/illustration-token-economics) · [Cost & Reliability](https://www.opensymbolic.ai/blog/illustration-cost-reliability)

---

## What This Repo Is

`core-py` is the **Python runtime for OpenSymbolicAI**: the core primitives and execution model for building LLM-powered systems as *software*, not as a pile of strings.

**Core concepts:**
- **Primitives** (`@primitive`) - Atomic operations your agent can execute
- **Decompositions** (`@decomposition`) - Examples showing how to break complex intents into primitive sequences
- **Evaluators** (`@evaluator`) - Goal evaluation methods for iterative agents

**Blueprints** (pick the one that fits your problem):

| Blueprint | When to Use |
|-----------|-------------|
| **PlanExecute** | Single-turn tasks with a fixed sequence of primitives |
| **DesignExecute** | Tasks needing loops and conditionals (dynamic-length data) |
| **GoalSeeking** | Iterative problems where progress is evaluated each step |

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
pip install opensymbolicai-core   # from PyPI
# or for development:
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

## Example: Shopping Cart Agent (DesignExecute)

When tasks involve dynamic-length data, you need loops and conditionals. `DesignExecute` extends `PlanExecute` with control flow support and loop guards to prevent runaway execution.

```python
from opensymbolicai import DesignExecute, primitive, decomposition

class ShoppingCart(DesignExecute):

    @primitive(read_only=True)
    def lookup_price(self, item: str) -> float:
        """Look up the unit price of an item from the catalog."""
        return CATALOG[item.lower()]

    @primitive(read_only=True)
    def apply_discount(self, price: float, percent: float) -> float:
        """Apply a percentage discount to a price."""
        return round(price * (1 - percent / 100), 2)

    @decomposition(
        intent="I need 5 apples and 1 laptop shipped to California",
        expanded_intent="Loop over items, apply bulk discounts, add state tax",
    )
    def _example_cart(self) -> float:
        cart = [("apples", 5), ("laptop", 1)]
        subtotal = 0.0
        for raw_name, qty in cart:
            price = self.lookup_price(item=raw_name)
            line = self.multiply(price=price, quantity=qty)
            if qty >= 3:
                line = self.apply_discount(price=line, percent=10.0)
            subtotal = self.add(a=subtotal, b=line)
        tax_rate = self.lookup_tax_rate(state="CA")
        return self.add_tax(subtotal=subtotal, rate=tax_rate)
```

The LLM generates plans with `for` loops and `if` statements. Loop guards automatically prevent infinite loops.

---

## Example: Function Optimizer (GoalSeeking)

For iterative problems where you can't solve it in one shot, `GoalSeeking` runs a plan-execute-evaluate loop until the goal is achieved.

```python
from opensymbolicai import GoalSeeking, primitive, evaluator, decomposition
from opensymbolicai import GoalContext, GoalEvaluation

class FunctionOptimizer(GoalSeeking):

    @primitive(read_only=True)
    def evaluate(self, x: float) -> float:
        """Evaluate the mystery function at point x."""
        return round(target_function(x), 6)

    @evaluator
    def check_converged(self, goal: str, context: GoalContext) -> GoalEvaluation:
        """Goal is achieved when we find a value close to the true maximum."""
        return GoalEvaluation(goal_achieved=context.converged)

    @decomposition(
        intent="Explore the function across the range",
        expanded_intent="Sample spread-out points to understand the function shape",
    )
    def _example_explore(self) -> float:
        v1 = self.evaluate(x=3.0)
        v2 = self.evaluate(x=8.0)
        v3 = self.evaluate(x=14.0)
        return v3
```

Each iteration: **plan** (pick sample points) → **execute** (call primitives) → **introspect** (extract knowledge into context) → **evaluate** (check goal). The LLM never sees raw execution results—only structured `GoalContext`.

---

## Auto-Documented Type Definitions

When primitives use Pydantic models as parameters or return types, the LLM prompt automatically includes a **Type Definitions** section listing each model's fields and types. This eliminates guesswork — the LLM knows exactly which attributes to use.

```python
from pydantic import BaseModel
from opensymbolicai import DesignExecute, primitive

class Flight(BaseModel):
    flight_number: str
    price: float
    origin: str
    destination: str

class TravelAgent(DesignExecute):

    @primitive(read_only=True)
    def search_flights(self, origin: str, destination: str) -> list[Flight]:
        """Search for available flights."""
        ...
```

The generated prompt will include:

```
## Type Definitions

Flight(flight_number: str, price: float, origin: str, destination: str)
```

This works across all blueprints (PlanExecute, DesignExecute, GoalSeeking) and handles generic types — `list[Flight]`, `Flight | None`, `Optional[Flight]`, `Union[Flight, Hotel]` are all unwrapped to discover the underlying models. Models are deduplicated and sorted alphabetically.

---

## Structured Exceptions

Primitives can raise typed exceptions that are captured in the execution trace:

```python
from opensymbolicai import ValidationError, PreconditionError, RetryableError

@primitive(read_only=True)
def divide(self, a: float, b: float) -> float:
    if b == 0:
        raise PreconditionError("Cannot divide by zero", code="DIVISION_BY_ZERO")
    return a / b
```

| Exception | Use Case |
|-----------|----------|
| `ValidationError` | Invalid inputs, out-of-range values |
| `PreconditionError` | Missing prerequisites (division by zero, empty collection) |
| `ResourceError` | Unavailable external resources (DB, API, file) |
| `OperationError` | Runtime failures during execution |
| `RetryableError` | Transient errors (rate limits, timeouts) — does not halt execution |

All exceptions serialize to dict for trace persistence and carry optional `code` and `details` fields.

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

## Anonymous Telemetry

OpenSymbolicAI collects anonymous usage data to help us understand adoption and
prioritize development. **No prompts, responses, API keys, or personal data are
ever collected.**

What we collect: framework version, Python version, OS, blueprint type used,
LLM provider name, and whether the run succeeded.

### Opting Out

Set either environment variable:

```bash
export OPENSYMBOLICAI_TELEMETRY_DISABLED=1
# or the universal convention:
export DO_NOT_TRACK=1
```

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
src/opensymbolicai/
  ├── core.py              # @primitive, @decomposition, @evaluator decorators
  ├── models.py            # Pydantic models (configs, traces, results)
  ├── llm.py               # Multi-provider LLM abstraction
  ├── checkpoint.py        # Distributed execution & state serialization
  ├── exceptions.py        # Structured exception hierarchy
  └── blueprints/
      ├── plan_execute.py    # PlanExecute — single-turn plan & execute
      ├── design_execute.py  # DesignExecute — adds loops & conditionals
      └── goal_seeking.py    # GoalSeeking — iterative plan-execute-evaluate
examples/
  ├── calculator/          # Scientific calculator (PlanExecute)
  ├── shopping_cart/       # Shopping cart with tax (DesignExecute)
  └── function_optimizer/  # Black-box optimization (GoalSeeking)
tests/                     # Unit tests
integration_tests/         # Integration tests (requires LLM)
benchmarks/                # Performance benchmarks
docs/                      # MkDocs documentation
```

---

## Contributing

This project is developed by a small core team and we are not currently accepting outside code contributions. Bug reports, feature ideas, and questions are very welcome via the [issue tracker](https://github.com/OpenSymbolicAI/core-py/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for the full policy.

---

## License

MIT
