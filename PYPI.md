# OpenSymbolicAI Core

**Build LLM-powered systems as typed, testable Python software — not prompt strings.**

The LLM plans once over your primitive signatures; execution is deterministic Python with no model in the loop. Inputs and outputs are typed variables.

## Install

```bash
pip install opensymbolicai-core
```

## Example

```python
from opensymbolicai import PlanExecute, primitive, decomposition

class Calculator(PlanExecute):
    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @decomposition(intent="What is 2 + 3?")
    def _example(self) -> float:
        return self.add(a=2, b=3)
```

## Documentation

Full docs, examples, benchmarks, and architecture: **https://github.com/OpenSymbolicAI/core-py**

## License

MIT
