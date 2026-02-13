"""Number guessing agent — a simple GoalSeeking example.

The agent plays a number guessing game: it tries to find a secret number
between 1 and 100 using binary search. Each iteration, the LLM generates
a plan that calls `guess(n)`, and the agent narrows the search range based
on the "too_high" / "too_low" / "correct" feedback.

This demonstrates:
- Custom GoalContext subclass (GuessingContext with search range)
- Introspection boundary (update_context extracts structured feedback)
- Static @evaluator (checks context.found)
- @decomposition (shows the LLM how to compose a guess)
- Iterative convergence toward a goal
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from opensymbolicai.blueprints import GoalSeeking
from opensymbolicai.core import decomposition, evaluator, primitive
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
)

if TYPE_CHECKING:
    from opensymbolicai.llm import LLM, LLMConfig


class GuessingContext(GoalContext):
    """Tracks the search range and guess history."""

    low: int = Field(default=1, description="Lower bound of search range (inclusive)")
    high: int = Field(default=100, description="Upper bound of search range (inclusive)")
    last_feedback: str = Field(default="", description="Feedback from the last guess")
    found: bool = Field(default=False, description="Whether the number was found")


class NumberGuesser(GoalSeeking):
    """Agent that finds a secret number through iterative guessing.

    Uses binary search strategy: the LLM sees the current [low, high] range
    and generates a guess. The environment responds with "too_low", "too_high",
    or "correct", and update_context narrows the range accordingly.
    """

    def __init__(
        self,
        llm: LLM | LLMConfig,
        secret: int,
        max_guesses: int = 10,
    ) -> None:
        """Initialize the number guesser.

        Args:
            llm: LLM for plan generation.
            secret: The secret number to find (1-100).
            max_guesses: Maximum number of guesses allowed.
        """
        super().__init__(
            llm=llm,
            name="NumberGuesser",
            description="Finds a secret number between 1 and 100 using binary search",
            config=GoalSeekingConfig(max_iterations=max_guesses),
        )
        self._secret = secret

    # ---- Primitives ----

    @primitive(read_only=True)
    def guess(self, n: int) -> str:
        """Guess a number and get feedback.

        Args:
            n: The number to guess.

        Returns:
            "too_low" if n < secret, "too_high" if n > secret, "correct" if n == secret.
        """
        if n < self._secret:
            return "too_low"
        if n > self._secret:
            return "too_high"
        return "correct"

    # ---- Decomposition ----

    @decomposition(
        intent="Guess the midpoint of the current range",
        expanded_intent="Calculate the middle of [low, high] and guess it",
    )
    def _example_guess(self) -> str:
        """Example: guess the midpoint."""
        result: str = self.guess(n=50)
        return result

    # ---- Context & Introspection ----

    def create_context(self, goal: str) -> GuessingContext:
        return GuessingContext(goal=goal)

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        """Extract feedback from the guess and narrow the search range."""
        assert isinstance(context, GuessingContext)
        last_step = execution_result.trace.last_step
        if last_step is None:
            return

        feedback = last_step.result_value
        context.last_feedback = str(feedback)

        if feedback == "correct":
            context.found = True
            return

        # Extract the guessed number from the step args
        guessed = last_step.args.get("n") or last_step.args.get("arg0")
        if guessed is None:
            return
        n = int(guessed.resolved_value)

        if feedback == "too_low":
            context.low = max(context.low, n + 1)
        elif feedback == "too_high":
            context.high = min(context.high, n - 1)

    # ---- Evaluator ----

    @evaluator
    def check_found(self, goal: str, context: GoalContext) -> GoalEvaluation:
        """Goal is achieved when the number is found."""
        assert isinstance(context, GuessingContext)
        return GoalEvaluation(goal_achieved=context.found)

    # ---- Custom Goal Prompt ----

    def build_goal_prompt(self, goal: str, context: GoalContext) -> str:
        """Override to include the current search range in the prompt."""
        assert isinstance(context, GuessingContext)

        primitives = self._get_primitive_methods()
        primitive_docs = [
            self._format_primitive_signature(name, method)
            for name, method in primitives
        ]

        decompositions = self._get_decomposition_methods()
        examples = []
        for _name, method, intent, expanded in decompositions:
            source = self._get_decomposition_source(method)
            if source:
                example = f"Intent: {intent}"
                if expanded:
                    example += f"\nApproach: {expanded}"
                example += f"\nPython:\n{source}"
                examples.append(example)

        range_info = f"The number is between {context.low} and {context.high} (inclusive)."
        if context.last_feedback:
            range_info += f"\nLast feedback: {context.last_feedback}"

        midpoint = (context.low + context.high) // 2

        return f"""You are {self.name}, an AI agent that guesses a secret number.

## Goal

{goal}

## Current State

{range_info}
Suggested next guess (midpoint): {midpoint}

## Available Methods

```python
{chr(10).join(primitive_docs)}
```

## Examples

{chr(10).join(f"### Example {i + 1}{chr(10)}{ex}" for i, ex in enumerate(examples)) if examples else "No examples."}

## Task

Generate ONE Python assignment statement that guesses the midpoint of the current range.

## Rules

1. Output ONLY one assignment statement
2. Call self.guess(n=...) with the midpoint value
3. Do NOT use imports, loops, or conditionals

## Output

```python
"""
