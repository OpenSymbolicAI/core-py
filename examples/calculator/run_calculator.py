#!/usr/bin/env python3
"""Console runner for the Scientific Calculator agent with Ollama."""

from __future__ import annotations

import sys

from calculator import ScientificCalculator

from opensymbolicai.llm import LLMConfig, Provider


def main() -> None:
    """Run the calculator in interactive console mode."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    # Get model from first non-flag argument
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model = args[0] if args else "gpt-oss:20b"

    print(f"Scientific Calculator Agent (Ollama - {model})")
    print("=" * 50)
    print("Type your math questions or 'quit' to exit.")
    if verbose:
        print("(Verbose mode enabled)")
    print()

    config = LLMConfig(
        provider=Provider.OLLAMA,
        model=model,
    )

    calc = ScientificCalculator(llm=config)

    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            result = calc.run(query)
            if verbose and result.plan_attempts:
                for i, attempt in enumerate(result.plan_attempts):
                    print(f"--- Plan Attempt {i + 1} ---")
                    print(attempt.plan_generation.extracted_code)
                    print()
            if result.success:
                print(f"= {result.result}\n")
            else:
                print(f"Error: {result.error}")
                if result.plan:
                    print(f"Final Plan: {repr(result.plan)}\n")
                else:
                    print()
        except Exception as e:
            print(f"Exception: {e}\n")


if __name__ == "__main__":
    main()
