#!/usr/bin/env python3
"""Console runner for the Number Guesser agent.

Usage:
    uv run python examples/number_guesser/run_number_guesser.py [model]
    uv run python examples/number_guesser/run_number_guesser.py -v        # verbose
    uv run python examples/number_guesser/run_number_guesser.py gpt-oss:20b
"""

from __future__ import annotations

import random
import sys

from number_guesser import GuessingContext, NumberGuesser

from opensymbolicai.llm import LLMConfig, Provider
from opensymbolicai.models import GoalContext, GoalStatus, Iteration


class LiveNumberGuesser(NumberGuesser):
    """NumberGuesser that prints each guess as it happens."""

    def on_iteration_complete(self, iteration: Iteration, context: GoalContext) -> None:
        assert isinstance(context, GuessingContext)
        last_step = iteration.execution_result.trace.last_step
        feedback = last_step.result_value if last_step else "?"
        plan = iteration.plan_result.plan.strip()
        range_str = f"[{context.low}, {context.high}]"
        print(f"  #{iteration.iteration_number}: {plan} → {feedback}  {range_str}")


def main() -> None:
    """Run the number guesser game."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model = args[0] if args else "gpt-oss:20b"

    low, high = 1_000_000, 2_000_000
    secret = random.randint(low, high)

    print(f"Number Guesser Agent (Ollama - {model})")
    print("=" * 50)
    print(f"Secret number chosen ({low:,}-{high:,}). Can the agent find it in 25 guesses?")
    if verbose:
        print(f"(Secret: {secret})")
    print()

    config = LLMConfig(provider=Provider.OLLAMA, model=model)
    agent = LiveNumberGuesser(llm=config, secret=secret, max_guesses=25)

    result = agent.seek(f"Find the secret number between {low} and {high}")

    print(f"\n{'=' * 50}")
    if result.status == GoalStatus.ACHIEVED:
        print(f"Found it! The number was {secret}.")
    else:
        print(f"Ran out of guesses. The number was {secret}.")

    print(f"Guesses used: {result.iteration_count}")


if __name__ == "__main__":
    main()
