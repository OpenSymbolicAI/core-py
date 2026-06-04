"""Unit tests for the ShoppingCart DesignExecute example."""

from typing import Any

import pytest

from examples.shopping_cart.shopping_cart import (
    ShoppingCart,
)
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage


class MockLLM(LLM):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        config = LLMConfig(provider="mock", model="mock-model")
        super().__init__(config, cache=None)
        self.responses = responses or []
        self.call_count = 0
        self.prompts: list[str] = []

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.prompts.append(prompt)
        response_text = (
            self.responses[self.call_count]
            if self.call_count < len(self.responses)
            else "result = 0"
        )
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            provider="mock",
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


# =============================================================================
# Primitive Unit Tests
# =============================================================================


class TestResolveItem:
    """Test the resolve_item primitive."""

    def test_exact_match(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.resolve_item("apple") == "apple"

    def test_case_insensitive(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.resolve_item("Apple") == "apple"
        assert cart.resolve_item("LAPTOP") == "laptop"

    def test_whitespace_stripped(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.resolve_item("  apple  ") == "apple"

    def test_llm_fallback_resolves(self):
        """LLM resolves a non-exact name to a catalog item."""
        mock_llm = MockLLM(responses=["book"])
        cart = ShoppingCart(llm=mock_llm)
        assert cart.resolve_item("books") == "book"
        assert mock_llm.call_count == 1

    def test_llm_fallback_fails(self):
        """LLM suggests a name not in the catalog."""
        mock_llm = MockLLM(responses=["flying carpet"])
        cart = ShoppingCart(llm=mock_llm)
        with pytest.raises(ValueError, match="Could not resolve"):
            cart.resolve_item("magic item")


class TestLookupPrice:
    """Test the lookup_price primitive."""

    def test_known_item(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.lookup_price("apple") == 1.50
        assert cart.lookup_price("laptop") == 999.99

    def test_case_insensitive(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.lookup_price("Apple") == 1.50

    def test_unknown_item(self):
        cart = ShoppingCart(llm=MockLLM())
        with pytest.raises(ValueError, match="not found in catalog"):
            cart.lookup_price("dragon fruit")


class TestLookupTaxRate:
    """Test the lookup_tax_rate primitive."""

    def test_known_state(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.lookup_tax_rate("CA") == 7.25
        assert cart.lookup_tax_rate("TX") == 6.25

    def test_case_insensitive(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.lookup_tax_rate("ca") == 7.25

    def test_tax_free_states(self):
        cart = ShoppingCart(llm=MockLLM())
        for state in ("AK", "DE", "MT", "NH", "OR"):
            assert cart.lookup_tax_rate(state) == 0.0

    def test_unknown_state(self):
        cart = ShoppingCart(llm=MockLLM())
        with pytest.raises(ValueError, match="Unknown state"):
            cart.lookup_tax_rate("XX")


class TestMultiply:
    """Test the multiply primitive."""

    def test_basic(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.multiply(1.50, 5) == 7.50

    def test_rounding(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.multiply(1.333, 3) == 4.0  # 3.999 rounded to 2dp


class TestApplyDiscount:
    """Test the apply_discount primitive."""

    def test_ten_percent(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.apply_discount(100.0, 10.0) == 90.0

    def test_zero_discount(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.apply_discount(50.0, 0.0) == 50.0

    def test_rounding(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.apply_discount(7.50, 10.0) == 6.75


class TestAddTax:
    """Test the add_tax primitive."""

    def test_ca_tax(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.add_tax(1006.74, 7.25) == 1079.73

    def test_zero_tax(self):
        cart = ShoppingCart(llm=MockLLM())
        assert cart.add_tax(100.0, 0.0) == 100.0


# =============================================================================
# Execute Validation Tests (lookup_tax_rate call count)
# =============================================================================


class TestExecuteTaxValidation:
    """Test that execute() enforces exactly one lookup_tax_rate call."""

    def test_zero_tax_lookups_fails(self):
        """Plan that never calls lookup_tax_rate should fail."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = "result = lookup_price(item='apple')\nreturn result"
        result = cart.execute(plan)
        assert not result.trace.all_succeeded
        assert any("never called" in (s.error or "") for s in result.trace.steps)

    def test_multiple_tax_lookups_fails(self):
        """Plan that calls lookup_tax_rate twice should fail."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "r1 = lookup_tax_rate(state='CA')\n"
            "r2 = lookup_tax_rate(state='TX')\n"
            "result = add(a=r1, b=r2)\n"
            "return result"
        )
        result = cart.execute(plan)
        assert not result.trace.all_succeeded
        assert any("called 2 times" in (s.error or "") for s in result.trace.steps)

    def test_exactly_one_tax_lookup_succeeds(self):
        """Plan with exactly one lookup_tax_rate call should pass validation."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "price = lookup_price(item='apple')\n"
            "line = multiply(price=price, quantity=1)\n"
            "tax = lookup_tax_rate(state='CA')\n"
            "result = add_tax(subtotal=line, rate=tax)\n"
            "return result"
        )
        result = cart.execute(plan)
        assert result.trace.all_succeeded


# =============================================================================
# End-to-End Math Tests
# =============================================================================


class TestEndToEndMath:
    """Test full plans and verify the math."""

    def test_5_apples_1_laptop_to_ca(self):
        """5 apples + 1 laptop, shipping to CA = $1079.73."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "cart = [('apple', 5), ('laptop', 1)]\n"
            "subtotal = 0.0\n"
            "for raw_name, qty in cart:\n"
            "    item = resolve_item(name=raw_name)\n"
            "    price = lookup_price(item=item)\n"
            "    line = multiply(price=price, quantity=qty)\n"
            "    if qty >= 3:\n"
            "        line = apply_discount(price=line, percent=10.0)\n"
            "    subtotal = add(a=subtotal, b=line)\n"
            "tax_rate = lookup_tax_rate(state='CA')\n"
            "if tax_rate > 0:\n"
            "    result = add_tax(subtotal=subtotal, rate=tax_rate)\n"
            "else:\n"
            "    result = subtotal\n"
            "return result"
        )
        # apples: 5 * 1.50 = 7.50, 10% off = 6.75
        # laptop: 1 * 999.99 = 999.99, no discount
        # subtotal: 6.75 + 999.99 = 1006.74
        # CA tax 7.25%: 1006.74 * 1.0725 = 1079.7261 → round = 1079.73
        result = cart.execute(plan)
        assert result.trace.all_succeeded
        assert result.get_value() == 1079.73

    def test_tax_free_state_skips_tax(self):
        """Items shipped to OR (0% tax) should not have tax added."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "price = lookup_price(item='laptop')\n"
            "subtotal = multiply(price=price, quantity=1)\n"
            "tax_rate = lookup_tax_rate(state='OR')\n"
            "if tax_rate > 0:\n"
            "    result = add_tax(subtotal=subtotal, rate=tax_rate)\n"
            "else:\n"
            "    result = subtotal\n"
            "return result"
        )
        result = cart.execute(plan)
        assert result.trace.all_succeeded
        assert result.get_value() == 999.99

    def test_bulk_discount_at_threshold(self):
        """Exactly 3 items (the threshold) should get 10% off."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "price = lookup_price(item='pen')\n"
            "line = multiply(price=price, quantity=3)\n"
            "line = apply_discount(price=line, percent=10.0)\n"
            "tax_rate = lookup_tax_rate(state='OR')\n"
            "if tax_rate > 0:\n"
            "    result = add_tax(subtotal=line, rate=tax_rate)\n"
            "else:\n"
            "    result = line\n"
            "return result"
        )
        # 3 * 2.49 = 7.47, 10% off = 6.72 (7.47 * 0.9 = 6.723 → round = 6.72)
        result = cart.execute(plan)
        assert result.trace.all_succeeded
        assert result.get_value() == 6.72

    def test_no_bulk_discount_below_threshold(self):
        """2 items (below threshold) should not get a discount."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "price = lookup_price(item='pen')\n"
            "line = multiply(price=price, quantity=2)\n"
            "tax_rate = lookup_tax_rate(state='OR')\n"
            "if tax_rate > 0:\n"
            "    result = add_tax(subtotal=line, rate=tax_rate)\n"
            "else:\n"
            "    result = line\n"
            "return result"
        )
        # 2 * 2.49 = 4.98, no discount, OR = 0% tax
        result = cart.execute(plan)
        assert result.trace.all_succeeded
        assert result.get_value() == 4.98

    def test_mixed_cart_to_ny(self):
        """Mixed cart with some bulk-eligible items, shipped to NY (4.0%)."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "cart = [('book', 3), ('mouse', 1)]\n"
            "subtotal = 0.0\n"
            "for item_name, qty in cart:\n"
            "    price = lookup_price(item=item_name)\n"
            "    line = multiply(price=price, quantity=qty)\n"
            "    if qty >= 3:\n"
            "        line = apply_discount(price=line, percent=10.0)\n"
            "    subtotal = add(a=subtotal, b=line)\n"
            "tax_rate = lookup_tax_rate(state='NY')\n"
            "if tax_rate > 0:\n"
            "    result = add_tax(subtotal=subtotal, rate=tax_rate)\n"
            "else:\n"
            "    result = subtotal\n"
            "return result"
        )
        # book: 3 * 12.99 = 38.97, 10% off = 35.07 (38.97 * 0.9 = 35.073 → 35.07)
        # mouse: 1 * 29.99 = 29.99, no discount
        # subtotal: 35.07 + 29.99 = 65.06
        # NY tax 4.0%: 65.06 * 1.04 = 67.6624 → 67.66
        result = cart.execute(plan)
        assert result.trace.all_succeeded
        assert result.get_value() == 67.66


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestErrorPropagation:
    """Test that primitive errors propagate correctly through plans."""

    def test_unknown_item_in_loop(self):
        """A bad item name mid-loop should produce a failed trace."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = "result = lookup_price(item='dragon fruit')\nreturn result"
        result = cart.execute(plan)
        assert not result.trace.all_succeeded
        assert "not found in catalog" in result.trace.failed_steps[0].error

    def test_unknown_state_in_plan(self):
        """An invalid state code should produce a failed trace."""
        mock_llm = MockLLM()
        cart = ShoppingCart(llm=mock_llm)

        plan = (
            "price = lookup_price(item='apple')\n"
            "line = multiply(price=price, quantity=1)\n"
            "tax_rate = lookup_tax_rate(state='ZZ')\n"
            "result = add_tax(subtotal=line, rate=tax_rate)\n"
            "return result"
        )
        result = cart.execute(plan)
        assert not result.trace.all_succeeded
        assert any("Unknown state" in (s.error or "") for s in result.trace.steps)
