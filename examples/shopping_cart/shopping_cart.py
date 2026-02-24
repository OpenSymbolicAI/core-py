"""Shopping cart agent — demonstrates DesignExecute with loops and conditionals.

This example shows how DesignExecute enables LLM-generated plans that use
control flow to process a shopping cart: looping over items, conditionally
applying bulk discounts, looking up state-specific tax rates, and computing
a final total.

This is something PlanExecute cannot do — it would require one hardcoded
line per cart item, making it impossible to handle dynamic-length carts.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from opensymbolicai.blueprints.design_execute import DesignExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.models import (
    DesignExecuteConfig,
    ExecutionResult,
    ExecutionStep,
    ExecutionTrace,
)

if TYPE_CHECKING:
    from opensymbolicai.llm import LLM, LLMConfig

# Simulated product catalog
CATALOG: dict[str, float] = {
    # Fruits & snacks
    "apple": 1.50,
    "banana": 0.75,
    "orange": 1.25,
    "grapes": 3.99,
    "granola bar": 2.49,
    "chips": 4.29,
    # Drinks
    "coffee": 8.99,
    "tea": 5.49,
    "water bottle": 1.99,
    "juice": 3.49,
    # Office supplies
    "pen": 2.49,
    "pencil": 1.29,
    "notebook": 4.99,
    "stapler": 7.99,
    "sticky notes": 3.49,
    "folder": 2.99,
    # Electronics
    "laptop": 999.99,
    "mouse": 29.99,
    "keyboard": 49.99,
    "headphones": 79.99,
    "usb cable": 9.99,
    "charger": 24.99,
    "monitor": 349.99,
    "webcam": 59.99,
    # Books & media
    "book": 12.99,
    "magazine": 6.99,
    "ebook": 9.99,
}

# US state sales tax rates (%)
# Source: approximate 2024 general state sales tax rates
# 0.0 = no state sales tax
STATE_TAX_RATES: dict[str, float] = {
    "AL": 4.0,
    "AK": 0.0,
    "AZ": 5.6,
    "AR": 6.5,
    "CA": 7.25,
    "CO": 2.9,
    "CT": 6.35,
    "DE": 0.0,
    "FL": 6.0,
    "GA": 4.0,
    "HI": 4.0,
    "ID": 6.0,
    "IL": 6.25,
    "IN": 7.0,
    "IA": 6.0,
    "KS": 6.5,
    "KY": 6.0,
    "LA": 4.45,
    "ME": 5.5,
    "MD": 6.0,
    "MA": 6.25,
    "MI": 6.0,
    "MN": 6.875,
    "MS": 7.0,
    "MO": 4.225,
    "MT": 0.0,
    "NE": 5.5,
    "NV": 6.85,
    "NH": 0.0,
    "NJ": 6.625,
    "NM": 5.0,
    "NY": 4.0,
    "NC": 4.75,
    "ND": 5.0,
    "OH": 5.75,
    "OK": 4.5,
    "OR": 0.0,
    "PA": 6.0,
    "RI": 7.0,
    "SC": 6.0,
    "SD": 4.2,
    "TN": 7.0,
    "TX": 6.25,
    "UT": 6.1,
    "VT": 6.0,
    "VA": 5.3,
    "WA": 6.5,
    "WV": 6.0,
    "WI": 5.0,
    "WY": 4.0,
}

# Bulk discount: 10% off if you buy 3+ of the same item
BULK_THRESHOLD = 3
BULK_DISCOUNT_PERCENT = 10.0


class ShoppingCart(DesignExecute):
    """A shopping cart agent that computes totals with discounts and state tax.

    Demonstrates DesignExecute's control flow capabilities:
    - Loops over cart items to compute line totals
    - LLM-powered item name resolution (e.g. "books" -> "book")
    - Conditionally applies bulk discounts (10% off for 3+ of same item)
    - Looks up state-specific tax rates (5 states have 0% tax)
    - Conditionally skips tax for tax-free states

    The LLM generates plans like::

        cart = [("apples", 5), ("laptop", 1)]
        subtotal = 0.0
        for raw_name, qty in cart:
            item = resolve_item(name=raw_name)
            price = lookup_price(item=item)
            line = multiply(price=price, quantity=qty)
            if qty >= 3:
                line = apply_discount(price=line, percent=10.0)
            subtotal = add(a=subtotal, b=line)
        tax_rate = lookup_tax_rate(state="CA")
        if tax_rate > 0:
            result = add_tax(subtotal=subtotal, rate=tax_rate)
        else:
            result = subtotal
    """

    def __init__(
        self,
        llm: LLM | LLMConfig,
        config: DesignExecuteConfig | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            name="ShoppingCart",
            description=(
                "A shopping cart that computes totals with state-specific tax. "
                "Items with quantity >= 3 get a 10% bulk discount. "
                "A US state MUST be specified — call lookup_tax_rate(state) exactly once. "
                "5 states have 0%% tax: AK, DE, MT, NH, OR."
            ),
            config=config,
        )

    # ---- Primitives ----

    @primitive(read_only=True, deterministic=False)
    def resolve_item(self, name: str) -> str:
        """Resolve a possibly-plural or informal item name to its catalog key.

        Uses the LLM to semantically match the input to the closest catalog
        entry (e.g. "books" -> "book", "USB Cables" -> "usb cable").

        Args:
            name: The raw item name from the user (may be plural or informal).

        Returns:
            The exact catalog key as a string.
        """
        # Fast path: exact match
        canonical = name.lower().strip()
        if canonical in CATALOG:
            return canonical

        catalog_items = ", ".join(sorted(CATALOG))
        prompt = (
            f"The catalog contains these items: {catalog_items}\n\n"
            f'Which catalog item best matches "{name}"?\n'
            "Reply with ONLY the exact catalog item name, nothing else."
        )
        response = self._llm.generate(prompt)
        resolved = response.text.strip().lower().strip('"\'')
        if resolved not in CATALOG:
            msg = (
                f"Could not resolve '{name}' to a catalog item. "
                f"LLM suggested '{resolved}'. Available: {catalog_items}"
            )
            raise ValueError(msg)

        # Guard against hallucinated matches: verify the resolved name is
        # close enough to the input using edit distance (SequenceMatcher).
        similarity = SequenceMatcher(None, canonical, resolved).ratio()
        if similarity < 0.5:
            msg = (
                f"Could not resolve '{name}' to a catalog item. "
                f"LLM suggested '{resolved}' but it is too dissimilar "
                f"(similarity={similarity:.2f}). Available: {catalog_items}"
            )
            raise ValueError(msg)

        return resolved

    @primitive(read_only=True)
    def lookup_price(self, item: str) -> float:
        """Look up the unit price of an item from the catalog.

        Args:
            item: The item name (e.g., "apple", "laptop").

        Returns:
            The unit price as a float.
        """
        name = item.lower().strip()
        if name not in CATALOG:
            msg = f"Item '{item}' not found in catalog. Available: {', '.join(sorted(CATALOG))}"
            raise ValueError(msg)
        return CATALOG[name]

    @primitive(read_only=True)
    def lookup_tax_rate(self, state: str) -> float:
        """Look up the sales tax rate for a US state.

        Args:
            state: Two-letter state code (e.g., "CA", "NY", "OR").

        Returns:
            The state sales tax rate as a percentage (e.g., 7.25 for California).
            Returns 0.0 for tax-free states (AK, DE, MT, NH, OR).
        """
        code = state.upper().strip()
        if code not in STATE_TAX_RATES:
            msg = f"Unknown state '{state}'. Use a two-letter state code (e.g., CA, NY, TX)."
            raise ValueError(msg)
        return STATE_TAX_RATES[code]

    @primitive(read_only=True)
    def multiply(self, price: float, quantity: int) -> float:
        """Multiply a unit price by a quantity to get a line total.

        Args:
            price: The unit price.
            quantity: The number of items.

        Returns:
            The line total (price * quantity).
        """
        return round(price * quantity, 2)

    @primitive(read_only=True)
    def apply_discount(self, price: float, percent: float) -> float:
        """Apply a percentage discount to a price.

        Args:
            price: The original price.
            percent: The discount percentage (e.g., 10.0 for 10% off).

        Returns:
            The discounted price.
        """
        return round(price * (1 - percent / 100), 2)

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.

        Returns:
            The sum a + b.
        """
        return round(a + b, 2)

    @primitive(read_only=True)
    def add_tax(self, subtotal: float, rate: float) -> float:
        """Add sales tax to a subtotal.

        Args:
            subtotal: The pre-tax subtotal.
            rate: The tax rate as a percentage (e.g., 7.25 for 7.25%).

        Returns:
            The total including tax.
        """
        return round(subtotal * (1 + rate / 100), 2)

    # ---- Execution validation ----

    def execute(self, plan: str) -> ExecutionResult:
        """Execute the plan, then verify exactly one state was looked up."""
        result = super().execute(plan)

        # Count how many times lookup_tax_rate was called in the trace
        tax_calls = sum(
            1 for step in result.trace.steps if step.primitive_called == "lookup_tax_rate"
        )

        if tax_calls != 1:
            if tax_calls == 0:
                error_msg = "A US state must be specified — lookup_tax_rate was never called."
            else:
                error_msg = (
                    f"Exactly one state must be specified, but lookup_tax_rate "
                    f"was called {tax_calls} times."
                )
            # Append a failed step so run() sees the failure via trace.all_succeeded
            failed_steps = list(result.trace.steps)
            failed_steps.append(
                ExecutionStep(
                    step_number=len(failed_steps) + 1,
                    statement="<state validation>",
                    success=False,
                    error=error_msg,
                )
            )
            return ExecutionResult(
                value_type="NoneType",
                value_name="",
                value_json="null",
                trace=ExecutionTrace(
                    steps=failed_steps,
                    total_time_seconds=result.trace.total_time_seconds,
                ),
            )

        return result

    # ---- Decompositions (teach the LLM the patterns) ----

    @decomposition(
        intent="I need 5 apples and 1 laptop shipped to California",
        expanded_intent=(
            "Loop over each (item, quantity) pair. Resolve the item name to "
            "its catalog key, look up the unit price, compute the line total, "
            "apply 10% discount if quantity >= 3, accumulate into a subtotal, "
            "then look up the state tax rate and add tax."
        ),
    )
    def _example_cart_with_state_tax(self) -> float:
        cart = [("apples", 5), ("laptop", 1)]
        subtotal: float = 0.0
        for raw_name, qty in cart:
            item: str = self.resolve_item(name=raw_name)
            price: float = self.lookup_price(item=item)
            line: float = self.multiply(price=price, quantity=qty)
            if qty >= 3:
                line = self.apply_discount(price=line, percent=10.0)
            subtotal = self.add(a=subtotal, b=line)
        tax_rate: float = self.lookup_tax_rate(state="CA")
        if tax_rate > 0:
            result: float = self.add_tax(subtotal=subtotal, rate=tax_rate)
        else:
            result = subtotal
        return result

    @decomposition(
        intent="I'd like to buy 3 bananas",
        expanded_intent=(
            "The user did NOT provide a US state for shipping. Do NOT invent "
            "or assume a state. Raise a ValueError to signal that the shipping "
            "state is required."
        ),
    )
    def _example_missing_state(self) -> float:
        raise ValueError("No shipping state specified. Please provide a US state code (e.g., CA, NY, TX).")

