"""Unit tests for the RecipeNutrition DesignExecute example with observability."""

from typing import Any

import pytest

from examples.recipe_book.recipe_book import (
    Ingredient,
    MealSummary,
    NutritionInfo,
    RecipeNutrition,
)
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import DesignExecuteConfig
from opensymbolicai.observability import (
    EventType,
    InMemoryTransport,
    ObservabilityConfig,
)


class MockLLM(LLM):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        config = LLMConfig(provider="mock", model="mock-model")
        super().__init__(config, cache=None)
        self.responses = responses or []
        self.call_count = 0

    def _generate_impl(self, prompt: str, **kwargs: Any) -> LLMResponse:
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


class TestGetNutrition:
    """Test the get_nutrition primitive."""

    def test_known_ingredient(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = agent.get_nutrition("chicken breast", 100)
        assert info.calories == 165
        assert info.protein_g == 31.0
        assert info.fat_g == 3.6

    def test_scaled_quantity(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = agent.get_nutrition("rice", 200)
        assert info.calories == 260.0  # 130 * 2
        assert info.carbs_g == 56.4  # 28.2 * 2

    def test_unknown_ingredient(self):
        agent = RecipeNutrition(llm=MockLLM())
        with pytest.raises(ValueError, match="Unknown ingredient"):
            agent.get_nutrition("dragon fruit", 100)

    def test_case_insensitive(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = agent.get_nutrition("Chicken Breast", 100)
        assert info.calories == 165

    def test_returns_nutrition_info_model(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = agent.get_nutrition("salmon", 100)
        assert isinstance(info, NutritionInfo)
        assert info.fiber_g == 0.0


class TestZeroNutrition:
    """Test the zero_nutrition primitive."""

    def test_all_zeros(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = agent.zero_nutrition()
        assert info.calories == 0
        assert info.protein_g == 0
        assert info.carbs_g == 0
        assert info.fat_g == 0
        assert info.fiber_g == 0


class TestAddNutrition:
    """Test the add_nutrition primitive."""

    def test_add_two_infos(self):
        agent = RecipeNutrition(llm=MockLLM())
        a = NutritionInfo(calories=100, protein_g=10, carbs_g=20, fat_g=5, fiber_g=3)
        b = NutritionInfo(calories=200, protein_g=15, carbs_g=30, fat_g=8, fiber_g=2)
        result = agent.add_nutrition(a, b)
        assert result.calories == 300
        assert result.protein_g == 25
        assert result.carbs_g == 50
        assert result.fat_g == 13
        assert result.fiber_g == 5


class TestDivideNutrition:
    """Test the divide_nutrition primitive."""

    def test_divide_by_servings(self):
        agent = RecipeNutrition(llm=MockLLM())
        total = NutritionInfo(calories=600, protein_g=30, carbs_g=60, fat_g=20, fiber_g=10)
        per_serving = agent.divide_nutrition(total, 3)
        assert per_serving.calories == 200
        assert per_serving.protein_g == 10

    def test_zero_servings_raises(self):
        agent = RecipeNutrition(llm=MockLLM())
        total = NutritionInfo(calories=100, protein_g=10, carbs_g=20, fat_g=5, fiber_g=3)
        with pytest.raises(ValueError, match="Servings must be positive"):
            agent.divide_nutrition(total, 0)


class TestMakeIngredient:
    """Test the make_ingredient primitive."""

    def test_returns_ingredient_model(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = NutritionInfo(calories=165, protein_g=31, carbs_g=0, fat_g=3.6, fiber_g=0)
        ing = agent.make_ingredient("chicken breast", 200, info)
        assert isinstance(ing, Ingredient)
        assert ing.name == "chicken breast"
        assert ing.grams == 200
        assert ing.nutrition.calories == 165


class TestBuildMealSummary:
    """Test the build_meal_summary primitive."""

    def test_returns_meal_summary(self):
        agent = RecipeNutrition(llm=MockLLM())
        info = NutritionInfo(calories=300, protein_g=30, carbs_g=40, fat_g=10, fiber_g=5)
        per = NutritionInfo(calories=150, protein_g=15, carbs_g=20, fat_g=5, fiber_g=2.5)
        ing = Ingredient(name="chicken", grams=200, nutrition=info)
        summary = agent.build_meal_summary(
            meal_name="Test Meal",
            ingredients=[ing],
            total_nutrition=info,
            servings=2,
            per_serving=per,
        )
        assert isinstance(summary, MealSummary)
        assert summary.meal_name == "Test Meal"
        assert len(summary.ingredients) == 1
        assert summary.servings == 2


# =============================================================================
# End-to-End Execution Tests (with mock LLM plan)
# =============================================================================


class TestExecutePlan:
    """Test direct plan execution with complex Pydantic types."""

    def test_single_ingredient_plan(self):
        """Execute a plan that looks up a single ingredient's nutrition."""
        agent = RecipeNutrition(llm=MockLLM())
        plan = 'result = get_nutrition(ingredient="salmon", grams=100)'
        result = agent.execute(plan)
        assert result.trace.all_succeeded
        assert result.value_type == "NutritionInfo"

    def test_multi_ingredient_loop_plan(self):
        """Execute a plan with loops over ingredients and aggregation."""
        agent = RecipeNutrition(llm=MockLLM())
        plan = """\
items = [("chicken breast", 200), ("rice", 150), ("broccoli", 100)]
total = zero_nutrition()
ingredient_list = []
for name, grams in items:
    info = get_nutrition(ingredient=name, grams=grams)
    ing = make_ingredient(name=name, grams=grams, nutrition=info)
    ingredient_list.append(ing)
    total = add_nutrition(a=total, b=info)
per_serving = divide_nutrition(nutrition=total, servings=2)
result = build_meal_summary(
    meal_name="Chicken Rice Bowl",
    ingredients=ingredient_list,
    total_nutrition=total,
    servings=2,
    per_serving=per_serving,
)
"""
        result = agent.execute(plan)
        assert result.trace.all_succeeded
        assert result.value_type == "MealSummary"

    def test_unknown_ingredient_in_plan(self):
        """Plan with unknown ingredient should produce a failed step."""
        agent = RecipeNutrition(llm=MockLLM())
        plan = 'result = get_nutrition(ingredient="dragon fruit", grams=100)'
        result = agent.execute(plan)
        assert not result.trace.all_succeeded


# =============================================================================
# Observability Tests
# =============================================================================


class TestObservability:
    """Test that observability captures trace events for complex types."""

    def _make_traced_agent(self) -> tuple[RecipeNutrition, InMemoryTransport]:
        transport = InMemoryTransport()
        obs = ObservabilityConfig(enabled=True, transport=transport)
        config = DesignExecuteConfig(observability=obs)
        agent = RecipeNutrition(llm=MockLLM(), config=config)
        return agent, transport

    def test_single_lookup_emits_events(self):
        """A single-ingredient lookup should emit lifecycle + execution events."""
        agent, transport = self._make_traced_agent()
        plan = 'result = get_nutrition(ingredient="salmon", grams=100)'
        # Use run() with a mock LLM that returns this plan
        mock_llm = MockLLM(responses=[plan])
        agent._llm = mock_llm

        result = agent.run("How much protein in 100g salmon?")
        assert result.success

        event_types = [e.event_type for e in transport.events]
        assert EventType.RUN_START in event_types
        assert EventType.EXECUTION_STEP in event_types
        assert EventType.RUN_COMPLETE in event_types

    def test_execution_steps_capture_pydantic_types(self):
        """Execution step events should include Pydantic model type info."""
        agent, transport = self._make_traced_agent()
        plan = 'result = get_nutrition(ingredient="rice", grams=150)'
        mock_llm = MockLLM(responses=[plan])
        agent._llm = mock_llm

        agent.run("Nutrition for 150g rice")

        step_events = [
            e for e in transport.events if e.event_type == EventType.EXECUTION_STEP
        ]
        assert len(step_events) >= 1

        step_payload = step_events[0].payload
        assert step_payload["primitive_called"] == "get_nutrition"
        assert step_payload["result_type"] == "NutritionInfo"
        assert step_payload["success"] is True

    def test_loop_plan_emits_multiple_steps(self):
        """A loop plan should emit one EXECUTION_STEP per primitive call."""
        agent, transport = self._make_traced_agent()
        plan = """\
items = [("chicken breast", 200), ("rice", 150)]
total = zero_nutrition()
for name, grams in items:
    info = get_nutrition(ingredient=name, grams=grams)
    total = add_nutrition(a=total, b=info)
result = total
"""
        mock_llm = MockLLM(responses=[plan])
        agent._llm = mock_llm

        result = agent.run("Nutrition for chicken and rice")
        assert result.success

        step_events = [
            e for e in transport.events if e.event_type == EventType.EXECUTION_STEP
        ]
        # 1 zero_nutrition + 2 get_nutrition + 2 add_nutrition = 5
        assert len(step_events) == 5

    def test_all_events_have_agent_class(self):
        """All events should carry the agent class name."""
        agent, transport = self._make_traced_agent()
        plan = 'result = get_nutrition(ingredient="egg", grams=50)'
        mock_llm = MockLLM(responses=[plan])
        agent._llm = mock_llm

        agent.run("Egg nutrition")

        for event in transport.events:
            assert event.agent_class == "RecipeNutrition"

    def test_tags_propagated_to_events(self):
        """User-supplied tags should appear on all events."""
        transport = InMemoryTransport()
        obs = ObservabilityConfig(
            enabled=True,
            transport=transport,
            tags={"env": "test", "example": "recipe"},
        )
        config = DesignExecuteConfig(observability=obs)
        plan = 'result = get_nutrition(ingredient="oats", grams=100)'
        agent = RecipeNutrition(llm=MockLLM(responses=[plan]), config=config)

        agent.run("Oats nutrition")

        for event in transport.events:
            assert event.tags["env"] == "test"
            assert event.tags["example"] == "recipe"
