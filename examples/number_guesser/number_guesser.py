"""Number guessing agent — a simple GoalSeeking example.

The agent plays a number guessing game: it tries to find a secret number
between 1 and 100. Each iteration, the LLM provides a search range
[low, high], and a deterministic method calculates the midpoint and
guesses it. The agent feeds back "too_low" / "too_high" / "correct"
so the LLM can narrow the range on the next iteration.

This demonstrates:
- Custom GoalContext subclass (GuessingContext with search range)
- Introspection boundary (update_context extracts structured feedback)
- Static @evaluator (checks context.found)
- @decomposition (shows the LLM how to compose a search call)
- LLM as range-provider, method as guesser
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

    low: int = Field(default=1_000_000, description="Lower bound of search range (inclusive)")
    high: int = Field(default=2_000_000, description="Upper bound of search range (inclusive)")
    last_feedback: str = Field(default="", description="Feedback from the last guess")
    found: bool = Field(default=False, description="Whether the number was found")


class NumberGuesser(GoalSeeking):
    """Agent that finds a secret number through iterative guessing.

    The LLM provides a search range [low, high] each iteration, and a
    deterministic method calculates the midpoint and guesses it. The
    environment responds with "too_low", "too_high", or "correct", so
    the LLM can narrow the range on the next iteration.
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
            description="Finds a secret number using binary search",
            config=GoalSeekingConfig(max_iterations=max_guesses),
        )
        self._secret = secret

    # ---- Primitives ----

    @primitive(read_only=True)
    def search(self, low: int, high: int) -> str:
        """Provide a search range; the midpoint is guessed automatically.

        Args:
            low: Lower bound of the search range (inclusive).
            high: Upper bound of the search range (inclusive).

        Returns:
            "too_low" if midpoint < secret, "too_high" if midpoint > secret,
            "correct" if midpoint == secret.
        """
        midpoint = (low + high) // 2
        if midpoint < self._secret:
            return "too_low"
        if midpoint > self._secret:
            return "too_high"
        return "correct"

    # ---- Decomposition ----

    @decomposition(
        intent="Narrow the search range based on feedback",
        expanded_intent="Provide a [low, high] range; the method guesses the midpoint",
    )
    def _example_search(self) -> str:
        """Example: search the full range."""
        result: str = self.search(low=1_000_000, high=2_000_000)
        return result

    # ---- Context & Introspection ----

    def create_context(self, goal: str) -> GuessingContext:
        return GuessingContext(goal=goal)

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        """Extract feedback and record the range the LLM provided."""
        assert isinstance(context, GuessingContext)
        last_step = execution_result.trace.last_step
        if last_step is None:
            return

        feedback = last_step.result_value
        context.last_feedback = str(feedback)

        if feedback == "correct":
            context.found = True
            return

        # Record the range the LLM chose so it appears in the next prompt
        low_arg = last_step.args.get("low") or last_step.args.get("arg0")
        high_arg = last_step.args.get("high") or last_step.args.get("arg1")
        if low_arg is not None:
            context.low = int(low_arg.resolved_value)
        if high_arg is not None:
            context.high = int(high_arg.resolved_value)

    # ---- Evaluator ----

    @evaluator
    def check_found(self, goal: str, context: GoalContext) -> GoalEvaluation:
        """Goal is achieved when the number is found."""
        assert isinstance(context, GuessingContext)
        return GoalEvaluation(goal_achieved=context.found)

