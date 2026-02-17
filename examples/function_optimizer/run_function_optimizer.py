#!/usr/bin/env python3
"""Console runner for the Function Optimizer agent.

Usage:
    uv run python examples/function_optimizer/run_function_optimizer.py [model]
    uv run python examples/function_optimizer/run_function_optimizer.py -v        # verbose
    uv run python examples/function_optimizer/run_function_optimizer.py gpt-oss:20b
"""

from __future__ import annotations

import sys

from function_optimizer import FunctionOptimizer, OptimizationContext

from opensymbolicai.llm import LLMConfig, Provider
from opensymbolicai.models import GoalContext, GoalStatus, Iteration


class LiveFunctionOptimizer(FunctionOptimizer):
    """FunctionOptimizer that prints each iteration as it happens."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.final_context: OptimizationContext | None = None

    def on_iteration_complete(self, iteration: Iteration, context: GoalContext) -> None:
        assert isinstance(context, OptimizationContext)
        self.final_context = context

        # Count new samples from this iteration
        new_eval_count = sum(
            1
            for s in iteration.execution_result.trace.steps
            if s.primitive_called == "evaluate" and s.success
        )
        new_samples = context.samples[-new_eval_count:] if new_eval_count else []

        sample_strs = [f"f({s.x:.2f})={s.value:.4f}" for s in new_samples]
        best_str = (
            f"best: f({context.best_x:.4f})={context.best_value:.6f}"
            if context.best_x is not None
            else ""
        )
        print(
            f"  #{iteration.iteration_number}: "
            f"{', '.join(sample_strs) or '(no evals)'}  "
            f"[{best_str}]"
        )


def main() -> None:
    """Run the function optimizer."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model = args[0] if args else "gpt-oss:20b"

    print(f"Function Optimizer Agent (Ollama - {model})")
    print("=" * 60)
    print("Goal: Find the global maximum of f(x) on [0, 20]")
    print("f(x) = sin(3x)*exp(-0.1*(x-5)^2) + 0.5*cos(7x)*exp(-0.05*(x-12)^2)")
    print()

    config = LLMConfig(provider=Provider.OLLAMA, model=model)
    agent = LiveFunctionOptimizer(llm=config, tolerance=0.01, max_iterations=25)

    if verbose:
        print(f"True maximum: f({agent._true_max_x:.4f}) = {agent._true_max_value:.6f}")
        print()

    result = agent.seek("Find the x in [0, 20] that maximizes the function f(x)")

    print(f"\n{'=' * 60}")
    if result.status == GoalStatus.ACHIEVED:
        print("Converged! Found maximum near the true optimum.")
    else:
        print("Ran out of iterations without converging.")

    # Show final results from tracked context
    ctx = agent.final_context
    if ctx:
        total_evals = len(ctx.samples)
        print(f"Total evaluations: {total_evals}")
        print(f"Best found:   f({ctx.best_x:.4f}) = {ctx.best_value:.6f}")
        print(f"True maximum: f({agent._true_max_x:.4f}) = {agent._true_max_value:.6f}")
        if ctx.best_value is not None:
            gap = abs(agent._true_max_value - ctx.best_value)
            print(f"Gap: {gap:.6f}")
    print(f"Iterations used: {result.iteration_count}")


if __name__ == "__main__":
    main()
