"""Recipe nutrition agent — demonstrates DesignExecute with complex Pydantic types.

This example shows how DesignExecute handles rich structured data:
- Primitives that accept and return Pydantic models (not just ints/strings)
- Loops over ingredient lists with per-item nutrition lookups
- Aggregation of nested model fields across iterations
- Per-serving calculations on structured nutrition data

The framework automatically:
- Generates type definitions in the LLM prompt from Pydantic annotations
- Allows attribute access on returned models in LLM-generated plans
- Serializes complex results in execution traces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from opensymbolicai.blueprints.design_execute import DesignExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.models import DesignExecuteConfig

if TYPE_CHECKING:
    from opensymbolicai.llm import LLM, LLMConfig

# ---------------------------------------------------------------------------
# Domain models — complex types returned by primitives
# ---------------------------------------------------------------------------


class NutritionInfo(BaseModel):
    """Nutritional breakdown for a food item or meal."""

    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class Ingredient(BaseModel):
    """A single ingredient with its quantity and nutrition."""

    name: str
    grams: float
    nutrition: NutritionInfo


class MealSummary(BaseModel):
    """Final summary of a meal's nutritional content."""

    meal_name: str
    ingredients: list[Ingredient]
    total_nutrition: NutritionInfo
    servings: int
    per_serving: NutritionInfo


# ---------------------------------------------------------------------------
# Nutrition database (per 100g)
# ---------------------------------------------------------------------------

NUTRITION_PER_100G: dict[str, NutritionInfo] = {
    "chicken breast": NutritionInfo(
        calories=165, protein_g=31.0, carbs_g=0.0, fat_g=3.6, fiber_g=0.0
    ),
    "rice": NutritionInfo(
        calories=130, protein_g=2.7, carbs_g=28.2, fat_g=0.3, fiber_g=0.4
    ),
    "broccoli": NutritionInfo(
        calories=34, protein_g=2.8, carbs_g=7.0, fat_g=0.4, fiber_g=2.6
    ),
    "salmon": NutritionInfo(
        calories=208, protein_g=20.4, carbs_g=0.0, fat_g=13.4, fiber_g=0.0
    ),
    "egg": NutritionInfo(
        calories=155, protein_g=13.0, carbs_g=1.1, fat_g=11.0, fiber_g=0.0
    ),
    "pasta": NutritionInfo(
        calories=131, protein_g=5.0, carbs_g=25.0, fat_g=1.1, fiber_g=1.8
    ),
    "spinach": NutritionInfo(
        calories=23, protein_g=2.9, carbs_g=3.6, fat_g=0.4, fiber_g=2.2
    ),
    "olive oil": NutritionInfo(
        calories=884, protein_g=0.0, carbs_g=0.0, fat_g=100.0, fiber_g=0.0
    ),
    "tomato": NutritionInfo(
        calories=18, protein_g=0.9, carbs_g=3.9, fat_g=0.2, fiber_g=1.2
    ),
    "cheese": NutritionInfo(
        calories=402, protein_g=25.0, carbs_g=1.3, fat_g=33.1, fiber_g=0.0
    ),
    "potato": NutritionInfo(
        calories=77, protein_g=2.0, carbs_g=17.5, fat_g=0.1, fiber_g=2.2
    ),
    "banana": NutritionInfo(
        calories=89, protein_g=1.1, carbs_g=22.8, fat_g=0.3, fiber_g=2.6
    ),
    "oats": NutritionInfo(
        calories=389, protein_g=16.9, carbs_g=66.3, fat_g=6.9, fiber_g=10.6
    ),
    "avocado": NutritionInfo(
        calories=160, protein_g=2.0, carbs_g=8.5, fat_g=14.7, fiber_g=6.7
    ),
    "lentils": NutritionInfo(
        calories=116, protein_g=9.0, carbs_g=20.1, fat_g=0.4, fiber_g=7.9
    ),
}


class RecipeNutrition(DesignExecute):
    """An agent that calculates nutritional content for recipes.

    Demonstrates complex Pydantic return types:
    - ``get_nutrition`` returns a ``NutritionInfo`` model
    - ``make_ingredient`` returns an ``Ingredient`` model with nested ``NutritionInfo``
    - ``build_meal_summary`` returns a ``MealSummary`` model

    The LLM generates plans that loop over ingredients, access model attributes
    (e.g. ``info.calories``, ``ingredient.nutrition``), and aggregate structured data.
    """

    def __init__(
        self,
        llm: LLM | LLMConfig,
        config: DesignExecuteConfig | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            name="RecipeNutrition",
            description=(
                "A nutrition calculator for recipes. Given a list of ingredients "
                "with quantities in grams, calculates total and per-serving "
                "nutritional content (calories, protein, carbs, fat, fiber)."
            ),
            config=config,
        )

    # ---- Primitives ----

    @primitive(read_only=True)
    def get_nutrition(self, ingredient: str, grams: float) -> NutritionInfo:
        """Look up nutrition info for an ingredient, scaled to the given weight.

        Args:
            ingredient: Name of the ingredient (e.g. "chicken breast", "rice").
            grams: Weight in grams.

        Returns:
            NutritionInfo with calories, protein_g, carbs_g, fat_g, fiber_g
            scaled to the specified weight.
        """
        key = ingredient.lower().strip()
        if key not in NUTRITION_PER_100G:
            available = ", ".join(sorted(NUTRITION_PER_100G))
            msg = f"Unknown ingredient '{ingredient}'. Available: {available}"
            raise ValueError(msg)

        base = NUTRITION_PER_100G[key]
        factor = grams / 100.0
        return NutritionInfo(
            calories=round(base.calories * factor, 1),
            protein_g=round(base.protein_g * factor, 1),
            carbs_g=round(base.carbs_g * factor, 1),
            fat_g=round(base.fat_g * factor, 1),
            fiber_g=round(base.fiber_g * factor, 1),
        )

    @primitive(read_only=True)
    def make_ingredient(self, name: str, grams: float, nutrition: NutritionInfo) -> Ingredient:
        """Bundle an ingredient name, quantity, and nutrition into an Ingredient object.

        Args:
            name: Ingredient name.
            grams: Weight in grams.
            nutrition: The NutritionInfo for this ingredient at the given weight.

        Returns:
            An Ingredient with name, grams, and nutrition fields.
        """
        return Ingredient(name=name, grams=grams, nutrition=nutrition)

    @primitive(read_only=True)
    def zero_nutrition(self) -> NutritionInfo:
        """Return a zeroed-out NutritionInfo, useful as a starting accumulator.

        Returns:
            NutritionInfo with all values set to 0.
        """
        return NutritionInfo(
            calories=0, protein_g=0, carbs_g=0, fat_g=0, fiber_g=0
        )

    @primitive(read_only=True)
    def add_nutrition(self, a: NutritionInfo, b: NutritionInfo) -> NutritionInfo:
        """Add two NutritionInfo values together (component-wise sum).

        Args:
            a: First nutrition info.
            b: Second nutrition info.

        Returns:
            Combined NutritionInfo with summed values.
        """
        return NutritionInfo(
            calories=round(a.calories + b.calories, 1),
            protein_g=round(a.protein_g + b.protein_g, 1),
            carbs_g=round(a.carbs_g + b.carbs_g, 1),
            fat_g=round(a.fat_g + b.fat_g, 1),
            fiber_g=round(a.fiber_g + b.fiber_g, 1),
        )

    @primitive(read_only=True)
    def divide_nutrition(self, nutrition: NutritionInfo, servings: int) -> NutritionInfo:
        """Divide nutrition values by the number of servings.

        Args:
            nutrition: Total nutrition to divide.
            servings: Number of servings.

        Returns:
            NutritionInfo per serving.
        """
        if servings <= 0:
            raise ValueError("Servings must be positive")
        return NutritionInfo(
            calories=round(nutrition.calories / servings, 1),
            protein_g=round(nutrition.protein_g / servings, 1),
            carbs_g=round(nutrition.carbs_g / servings, 1),
            fat_g=round(nutrition.fat_g / servings, 1),
            fiber_g=round(nutrition.fiber_g / servings, 1),
        )

    @primitive(read_only=True)
    def build_meal_summary(
        self,
        meal_name: str,
        ingredients: list[Ingredient],
        total_nutrition: NutritionInfo,
        servings: int,
        per_serving: NutritionInfo,
    ) -> MealSummary:
        """Build a complete meal summary with all nutritional details.

        Args:
            meal_name: Name of the meal.
            ingredients: List of Ingredient objects used.
            total_nutrition: Total nutrition for the entire recipe.
            servings: Number of servings the recipe makes.
            per_serving: Nutrition per single serving.

        Returns:
            A MealSummary with all recipe and nutrition details.
        """
        return MealSummary(
            meal_name=meal_name,
            ingredients=ingredients,
            total_nutrition=total_nutrition,
            servings=servings,
            per_serving=per_serving,
        )

    # ---- Decompositions ----

    @decomposition(
        intent="What is the nutrition for a chicken and rice bowl with 200g chicken breast, 150g rice, and 100g broccoli? Serves 2.",
        expanded_intent=(
            "Loop over each (ingredient, grams) pair. Look up nutrition for each, "
            "build Ingredient objects, accumulate total nutrition, then divide by "
            "servings and build a MealSummary."
        ),
    )
    def _example_chicken_rice(self) -> MealSummary:
        items = [("chicken breast", 200), ("rice", 150), ("broccoli", 100)]
        total: NutritionInfo = self.zero_nutrition()
        ingredient_list: list[Ingredient] = []

        for name, grams in items:
            info: NutritionInfo = self.get_nutrition(ingredient=name, grams=grams)
            ing: Ingredient = self.make_ingredient(name=name, grams=grams, nutrition=info)
            ingredient_list.append(ing)
            total = self.add_nutrition(a=total, b=info)

        per_serving: NutritionInfo = self.divide_nutrition(nutrition=total, servings=2)
        result: MealSummary = self.build_meal_summary(
            meal_name="Chicken and Rice Bowl",
            ingredients=ingredient_list,
            total_nutrition=total,
            servings=2,
            per_serving=per_serving,
        )
        return result

    @decomposition(
        intent="How many calories in 100g of salmon?",
        expanded_intent=(
            "Look up nutrition for a single ingredient and return the NutritionInfo."
        ),
    )
    def _example_single_lookup(self) -> NutritionInfo:
        result: NutritionInfo = self.get_nutrition(ingredient="salmon", grams=100)
        return result
