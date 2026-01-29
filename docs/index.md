# OpenSymbolicAI Core

Welcome to the OpenSymbolicAI Core documentation.

OpenSymbolicAI Core is a Python framework for symbolic AI agents with LLM-powered planning and execution capabilities.

## Installation

```bash
uv add opensymbolicai-core
```

## Quick Start

Here's a scientific calculator that uses primitives for basic operations and decompositions for complex calculations:

```python
from opensymbolicai import PlanExecute, primitive, decomposition

class ScientificCalculator(PlanExecute):
    """A calculator with LLM-powered planning."""

    @primitive(read_only=True)
    def add_numbers(self, a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    @primitive(read_only=True)
    def multiply_numbers(self, a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    @primitive(read_only=True)
    def square_root_of(self, number: float) -> float:
        """Calculate the square root of a number."""
        return number ** 0.5

    @primitive(read_only=True)
    def raise_to_power(self, base: float, exponent: float) -> float:
        """Raise base to the given exponent."""
        return base ** exponent

    @decomposition(
        intent="Calculate the hypotenuse of a right triangle with sides 3 and 4",
        expanded_intent="Square both sides, add them, then take the square root",
    )
    def calculate_hypotenuse(self) -> float:
        """Calculate hypotenuse using the Pythagorean theorem: c = sqrt(a^2 + b^2)"""
        a_squared = self.raise_to_power(base=3, exponent=2)
        b_squared = self.raise_to_power(base=4, exponent=2)
        sum_of_squares = self.add_numbers(a=a_squared, b=b_squared)
        return self.square_root_of(number=sum_of_squares)
```

## Features

- **Symbolic AI Framework**: Define operations as primitives and decompositions
- **LLM-Powered Planning**: Automatic plan generation using language models
- **Checkpoint Support**: Save and resume execution state
- **Type Safety**: Full Pydantic model support with Python 3.12+ type hints
