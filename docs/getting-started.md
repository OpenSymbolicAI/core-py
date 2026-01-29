# Getting Started

## Installation

```bash
pip install opensymbolicai-core
```

Or with uv:

```bash
uv add opensymbolicai-core
```

## Basic Concepts

### Primitives

Primitives are the basic building blocks - atomic operations that cannot be decomposed further.

```python
from opensymbolicai import primitive

@primitive
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

### Decompositions

Decompositions are compound operations that can be broken down into primitive operations.

```python
from opensymbolicai import decomposition

@decomposition
def sum_three(a: int, b: int, c: int) -> int:
    """Sum three numbers using two additions."""
    partial = add(a, b)
    return add(partial, c)
```

### PlanExecute

The `PlanExecute` blueprint orchestrates planning and execution of symbolic operations.

```python
from opensymbolicai import PlanExecute, PlanExecuteConfig

config = PlanExecuteConfig(
    max_iterations=10,
    enable_checkpoints=True,
)

executor = PlanExecute(config=config)
```

## Next Steps

- Check out the [API Reference](api.md) for detailed documentation
- Explore the examples in the repository
