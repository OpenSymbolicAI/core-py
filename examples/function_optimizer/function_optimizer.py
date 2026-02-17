"""Black-box function optimizer — a GoalSeeking example.

The agent tries to find the global maximum of a multi-modal function
by strategically sampling points. Unlike binary search, the function
has multiple peaks and valleys, so the agent must reason about the
landscape from observed samples and balance exploration vs exploitation.

This demonstrates:
- Non-monotonic optimization where binary search fails
- LLM reasoning about function landscape from sparse samples
- Strategic exploration vs exploitation decisions
- Custom GoalContext tracking all sampled points and best values
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

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


class Sample(BaseModel):
    """A single function evaluation."""

    x: float
    value: float

    def __repr__(self) -> str:
        return f"f({self.x:.4f}) = {self.value:.6f}"


def target_function(x: float) -> float:
    """Multi-modal function with non-obvious global maximum.

    f(x) = sin(3x) * exp(-0.1*(x-5)^2) + 0.5 * cos(7x) * exp(-0.05*(x-12)^2)

    Has multiple peaks near x~5 and x~12, with the global maximum near x~4.7.
    """
    return (
        math.sin(3 * x) * math.exp(-0.1 * (x - 5) ** 2)
        + 0.5 * math.cos(7 * x) * math.exp(-0.05 * (x - 12) ** 2)
    )


class OptimizationContext(GoalContext):
    """Tracks sampled points and the best value found."""

    samples: list[Sample] = Field(
        default_factory=list, description="All evaluated points so far"
    )
    best_x: float | None = Field(default=None, description="x with the highest f(x)")
    best_value: float | None = Field(
        default=None, description="Highest f(x) found so far"
    )
    converged: bool = Field(
        default=False, description="Whether the optimum was found"
    )


class FunctionOptimizer(GoalSeeking):
    """Agent that finds the global maximum of a mystery function.

    The function is non-monotonic with multiple peaks and valleys.
    The agent can only observe f(x) at chosen points and must reason
    about where to sample next based on accumulated observations.
    """

    def __init__(
        self,
        llm: LLM | LLMConfig,
        tolerance: float = 0.01,
        max_iterations: int = 25,
    ) -> None:
        """Initialize the function optimizer.

        Args:
            llm: LLM for plan generation.
            tolerance: How close to the true max to consider converged.
            max_iterations: Maximum number of plan-execute iterations.
        """
        super().__init__(
            llm=llm,
            name="FunctionOptimizer",
            description="Finds the global maximum of a mystery function f(x) on [0, 20]",
            config=GoalSeekingConfig(max_iterations=max_iterations),
        )
        self._tolerance = tolerance
        self._true_max_x, self._true_max_value = self._compute_true_max()

    @staticmethod
    def _compute_true_max() -> tuple[float, float]:
        """Brute-force the true global maximum for evaluation."""
        best_x = 0.0
        best_val = target_function(0.0)
        for i in range(200_001):
            x = i * 20.0 / 200_000
            val = target_function(x)
            if val > best_val:
                best_val = val
                best_x = x
        return best_x, best_val

    # ---- Primitives ----

    @primitive(read_only=True)
    def evaluate(self, x: float) -> float:
        """Evaluate the mystery function at point x.

        Args:
            x: The point to evaluate, must be in [0, 20].

        Returns:
            The function value f(x).
        """
        x = max(0.0, min(20.0, float(x)))
        return round(target_function(x), 6)

    # ---- Decomposition ----

    @decomposition(
        intent="Explore the function across the range",
        expanded_intent="Sample multiple spread-out points to understand the function shape",
    )
    def _example_explore(self) -> float:
        """Example: broad exploration with several samples."""
        v1: float = self.evaluate(x=3.0)  # noqa: F841
        v2: float = self.evaluate(x=8.0)  # noqa: F841
        v3: float = self.evaluate(x=14.0)
        result: float = v3
        return result

    @decomposition(
        intent="Refine around a promising region",
        expanded_intent="Sample nearby points around the current best to find the peak precisely",
    )
    def _example_refine(self) -> float:
        """Example: refine around x=5."""
        v1: float = self.evaluate(x=4.5)  # noqa: F841
        v2: float = self.evaluate(x=5.0)  # noqa: F841
        v3: float = self.evaluate(x=5.5)
        result: float = v3
        return result

    # ---- Context & Introspection ----

    def create_context(self, goal: str) -> OptimizationContext:
        return OptimizationContext(goal=goal)

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        """Extract all evaluate() calls and update best known value."""
        assert isinstance(context, OptimizationContext)

        for step in execution_result.trace.steps:
            if step.primitive_called != "evaluate" or not step.success:
                continue

            x_arg = step.args.get("x") or step.args.get("arg0")
            if x_arg is None:
                continue

            x_val = float(x_arg.resolved_value)
            f_val = float(step.result_value)
            context.samples.append(Sample(x=x_val, value=f_val))

            if context.best_value is None or f_val > context.best_value:
                context.best_x = x_val
                context.best_value = f_val

        # Check if we're close enough to the true maximum
        if (
            context.best_value is not None
            and abs(context.best_value - self._true_max_value) < self._tolerance
        ):
            context.converged = True

    # ---- Evaluator ----

    @evaluator
    def check_converged(self, goal: str, context: GoalContext) -> GoalEvaluation:
        """Goal is achieved when we find a value close to the true maximum."""
        assert isinstance(context, OptimizationContext)
        return GoalEvaluation(goal_achieved=context.converged)
