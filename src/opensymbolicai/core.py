"""Core decorators and types for OpenSymbolicAI."""

from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class MethodType(Enum):
    """Classification of agent methods."""

    PRIMITIVE = "primitive"
    DECOMPOSITION = "decomposition"


def primitive(read_only: bool = False) -> Callable[[F], F]:
    """Mark a method as a primitive operation.

    Primitives are atomic operations that the agent can directly execute.
    They serve as the building blocks for more complex behaviors.

    Args:
        read_only: If True, indicates this primitive does not modify state.

    Returns:
        A decorator that marks the function as a primitive.

    Example:
        @primitive(read_only=True)
        def add_numbers(self, a: float, b: float) -> float:
            return a + b
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper.__method_type__ = MethodType.PRIMITIVE  # type: ignore[attr-defined]
        wrapper.__primitive_read_only__ = read_only  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorator


def decomposition(intent: str, expanded_intent: str = "") -> Callable[[F], F]:
    """Mark a method as a decomposition example.

    Decompositions demonstrate how to break down high-level intents into
    sequences of primitive operations. They serve as examples for the LLM
    to learn the patterns of composition.

    Args:
        intent: A high-level description of what this decomposition achieves.
            This is the natural language query that this example answers.
        expanded_intent: An optional step-by-step breakdown of the approach.
            Provides additional context about the reasoning or methodology.

    Returns:
        A decorator that marks the function as a decomposition example.

    Example:
        @decomposition(
            intent="What is sine of 90 degrees?",
            expanded_intent="First convert 90 degrees to radians, then calculate sine",
        )
        def _example_sine_90(self) -> float:
            angle_rad: float = self.convert_degrees_to_radians(angle_in_degrees=90)
            sin_90: float = self.sine(angle_in_radians=angle_rad)
            return sin_90
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper.__method_type__ = MethodType.DECOMPOSITION  # type: ignore[attr-defined]
        wrapper.__decomposition_intent__ = intent  # type: ignore[attr-defined]
        wrapper.__decomposition_expanded_intent__ = expanded_intent  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorator
