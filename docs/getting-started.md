# Getting Started

## Installation

```bash
pip install opensymbolicai-core
```

Or with `uv`:

```bash
uv add opensymbolicai-core
```

## Core Concepts

An OpenSymbolicAI agent is a subclass of a **blueprint** (e.g. `PlanExecute`) that exposes **primitives** (atomic operations the agent may call) and **decompositions** (worked examples that teach the LLM how to compose primitives into plans).

- **`@primitive`** — Mark a method as a primitive the planner can call. Set `read_only=False` for primitives that mutate state.
- **`@decomposition`** — Mark a method as a worked example. The `intent` is the user-style query; `expanded_intent` is the natural-language strategy. The method body shows the primitive sequence to use.

## A Minimal Agent

```python
from opensymbolicai import PlanExecute, primitive, decomposition
from opensymbolicai.llm import LLMConfig, Provider


class Calculator(PlanExecute):
    """A tiny calculator agent."""

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=True)
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    @decomposition(
        intent="What is (2 + 3) times 4?",
        expanded_intent="Add 2 and 3 first, then multiply the result by 4",
    )
    def _example_add_then_multiply(self) -> float:
        partial = self.add(a=2, b=3)
        return self.multiply(a=partial, b=4)


llm = LLMConfig(provider=Provider.OLLAMA, model="gpt-oss:20b")
calc = Calculator(llm=llm)

result = calc.run("What is (10 + 20) times 5?")
if result.success:
    print(result.result)
else:
    print(f"Error: {result.error}")
```

## Configuration

`PlanExecute` accepts an optional `PlanExecuteConfig` for behavior tuning. A few useful fields:

- `max_plan_retries` — Retry plan generation on validation failures (default `0`).
- `multi_turn` — Persist variables and conversation history across `.run()` calls.
- `on_mutation` — Hook invoked before any `read_only=False` primitive runs; return a string to block.
- `observability` — Attach an `ObservabilityConfig` to emit trace events for planning and execution.

See the API Reference for the full list.

## Next Steps

- [API Reference](api.md) — auto-generated from docstrings.
- [Examples in the repository](https://github.com/OpenSymbolicAI/core-py/tree/main/examples) — calculator, shopping cart (with control flow), function optimizer (goal-seeking).
