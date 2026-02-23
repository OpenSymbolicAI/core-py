#!/usr/bin/env python3
"""Console runner for the RecipeNutrition agent — demonstrates observability.

Shows how to:
- Send traces to the local observability backend (http://localhost:8100/events)
- Inspect the trace event hierarchy (spans, parent spans)
- View execution steps with complex Pydantic return types

Prerequisites:
    Start the observability stack first:
        cd /path/to/OpenSymbolicAI/observability && docker compose up

    Then view traces in the dashboard at http://localhost:8101

Usage:
    uv run python examples/recipe_book/run_recipe_book.py [model]
    uv run python examples/recipe_book/run_recipe_book.py -v          # verbose
    uv run python examples/recipe_book/run_recipe_book.py gpt-oss:20b
"""

from __future__ import annotations

import sys

from recipe_book import NUTRITION_PER_100G, MealSummary, NutritionInfo, RecipeNutrition

from opensymbolicai.llm import LLMConfig, Provider
from opensymbolicai.models import DesignExecuteConfig
from opensymbolicai.observability import ObservabilityConfig

COLLECTOR_URL = "http://localhost:8100/events"


def print_nutrition_db() -> None:
    """Print available ingredients."""
    print("Available ingredients (per 100g):")
    for name, info in sorted(NUTRITION_PER_100G.items()):
        print(
            f"  {name:<16} {info.calories:>6.0f} cal  "
            f"{info.protein_g:>5.1f}g protein  "
            f"{info.carbs_g:>5.1f}g carbs  "
            f"{info.fat_g:>5.1f}g fat"
        )
    print()


def format_nutrition(info: NutritionInfo) -> str:
    """Format NutritionInfo for display."""
    return (
        f"{info.calories:.0f} cal | "
        f"{info.protein_g:.1f}g protein | "
        f"{info.carbs_g:.1f}g carbs | "
        f"{info.fat_g:.1f}g fat | "
        f"{info.fiber_g:.1f}g fiber"
    )


def main() -> None:
    """Run the recipe nutrition demo with observability."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model = args[0] if args else "gpt-oss:20b"

    print(f"Recipe Nutrition Agent (Ollama - {model})")
    print("=" * 55)
    print(f"Sending traces to: {COLLECTOR_URL}")
    print("View dashboard at: http://localhost:8101\n")
    print_nutrition_db()

    # --- Observability: send traces to local collector ---
    obs_config = ObservabilityConfig(
        enabled=True,
        collector_url=COLLECTOR_URL,
        capture_llm_prompts=True,
        capture_llm_responses=True,
        capture_execution_steps=True,
        capture_plan_source=True,
        tags={"example": "recipe_book", "env": "demo"},
    )

    agent = RecipeNutrition(
        llm=LLMConfig(provider=Provider.OLLAMA, model=model),
        config=DesignExecuteConfig(
            max_plan_retries=2,
            observability=obs_config,
        ),
    )

    # --- Demo queries ---
    queries = [
        # Multi-ingredient meal with servings — exercises loops + all primitives
        (
            "What is the nutrition for a meal with 200g chicken breast, "
            "150g rice, and 100g broccoli? It serves 2."
        ),
        # Single ingredient lookup — simple case
        "How many calories and protein in 250g of salmon?",
        # Another multi-ingredient recipe
        (
            "Calculate nutrition for pasta with cheese: 200g pasta, 50g cheese, "
            "15g olive oil, 100g tomato. Serves 3."
        ),
        # Error case: unknown ingredient
        "What's the nutrition in 100g of dragon fruit?",
    ]

    for task in queries:
        print(f"Task: {task}")
        result = agent.run(task)

        if verbose and result.plan:
            print(f"Plan:\n{result.plan}")

        if verbose and result.trace:
            print(f"Primitive calls: {len(result.trace.steps)}")
            for step in result.trace.steps:
                args_str = ", ".join(
                    f"{k}={v.resolved_value}" for k, v in step.args.items()
                )
                print(f"  {step.primitive_called}({args_str}) -> {step.result_type}")
                if step.result_value is not None and hasattr(
                    step.result_value, "model_dump"
                ):
                    print(f"    = {step.result_value}")

        if result.success:
            value = result.result
            if isinstance(value, MealSummary):
                print(f"Meal: {value.meal_name}")
                print(
                    f"  Ingredients: {', '.join(i.name for i in value.ingredients)}"
                )
                print(f"  Total:       {format_nutrition(value.total_nutrition)}")
                print(
                    f"  Per serving: {format_nutrition(value.per_serving)} "
                    f"({value.servings} servings)"
                )
            elif isinstance(value, NutritionInfo):
                print(f"Nutrition: {format_nutrition(value)}")
            else:
                print(f"Result: {value}")
        else:
            print(f"Error: {result.error}")

        print()


if __name__ == "__main__":
    main()
