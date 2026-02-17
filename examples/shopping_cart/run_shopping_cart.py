#!/usr/bin/env python3
"""Console runner for the Shopping Cart agent.

Demonstrates DesignExecute: the LLM generates plans with loops and
conditionals to process a dynamic shopping cart with state-specific tax.

Usage:
    uv run python examples/shopping_cart/run_shopping_cart.py [model]
    uv run python examples/shopping_cart/run_shopping_cart.py -v        # verbose
    uv run python examples/shopping_cart/run_shopping_cart.py gpt-oss:20b
"""

from __future__ import annotations

import sys

from shopping_cart import (
    BULK_DISCOUNT_PERCENT,
    BULK_THRESHOLD,
    CATALOG,
    STATE_TAX_RATES,
    ShoppingCart,
)

from opensymbolicai.llm import LLMConfig, Provider
from opensymbolicai.models import DesignExecuteConfig


def print_catalog() -> None:
    """Print the available items and tax info."""
    print("Available items:")
    for item, price in sorted(CATALOG.items()):
        print(f"  {item:<15} ${price:.2f}")
    print(
        f"\nBulk discount: {BULK_DISCOUNT_PERCENT:.0f}% off "
        f"when buying {BULK_THRESHOLD}+ of the same item"
    )
    tax_free = [s for s, r in STATE_TAX_RATES.items() if r == 0.0]
    print(f"Tax-free states: {', '.join(sorted(tax_free))}")


def main() -> None:
    """Run the shopping cart demo."""
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    model = args[0] if args else "gpt-oss:20b"

    print(f"Shopping Cart Agent (Ollama - {model})")
    print("=" * 55)
    print_catalog()
    print()

    config = LLMConfig(provider=Provider.OLLAMA, model=model)
    agent = ShoppingCart(
        llm=config,
        config=DesignExecuteConfig(max_plan_retries=2),
    )

    # Example carts — different combinations to demo loops + conditionals + state tax
    carts = [
        # California: high tax state (7.25%), with bulk discount
        "What's the total for 5 apples and 1 laptop, shipping to CA?",
        # Oregon: tax-free state — tests the if tax_rate > 0 conditional
        "I want 2 headphones, 3 books, and 10 pens. Shipping to OR.",
        # Texas (6.25%), no bulk discounts (all qty=1)
        "Calculate the total for 1 coffee, 1 mouse, and 1 notebook for TX.",
        # New York (4.0%), large mixed cart
        "I need 4 oranges, 3 granola bars, 2 water bottles, 6 pencils, "
        "1 monitor, and 3 usb cables. Shipping to NY.",
        # Montana: another tax-free state, all bulk-eligible
        "Give me 5 sticky notes, 3 folders, 4 juice, and 3 magazines. Ship to MT.",
        # Minnesota: highest tax rate (6.875%), single expensive item
        "Just 1 laptop to MN please.",
        # Delaware: tax-free, big office supply haul with all bulk discounts
        "I'm stocking up: 10 pens, 10 pencils, 5 notebooks, 3 staplers, "
        "and 8 folders. Shipping to DE.",
        # --- Error cases ---
        # Unknown item: "dragon fruit" is not in the catalog
        "I'd like 2 dragon fruits shipped to CA.",
        # Unknown state: "XX" is not a valid US state code
        "1 apple and 1 pen, shipping to XX.",
        # Missing state: no shipping destination at all
        "Just give me 3 bananas.",
    ]

    for task in carts:
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
                print(f"  {step.primitive_called}({args_str}) → {step.result_value}")

        if result.success:
            print(f"Total: ${result.result:.2f}\n")
        else:
            print(f"Error: {result.error}\n")


if __name__ == "__main__":
    main()
