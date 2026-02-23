# Recipe Nutrition Example

A nutrition calculator agent that demonstrates **DesignExecute with complex Pydantic types** and **observability**.

## What It Demonstrates

### Complex Pydantic Types

Unlike the calculator (floats) and shopping cart (floats + strings), this example uses nested Pydantic models as primitive inputs and outputs:

```
NutritionInfo
├── calories: float
├── protein_g: float
├── carbs_g: float
├── fat_g: float
└── fiber_g: float

Ingredient
├── name: str
├── grams: float
└── nutrition: NutritionInfo    ← nested model

MealSummary
├── meal_name: str
├── ingredients: list[Ingredient]  ← list of nested models
├── total_nutrition: NutritionInfo
├── servings: int
└── per_serving: NutritionInfo
```

The framework automatically generates type definitions in the LLM prompt and allows attribute access on returned models in plans (e.g. `info.calories`, `ingredient.nutrition`).

### Observability

The runner sends traces to the local observability backend. Each `run()` produces a separate trace with the full span hierarchy:

```
run.start
├── plan.start
│   ├── plan.llm_request
│   ├── plan.llm_response
│   └── plan.complete
├── execution.start
│   ├── execution.step  (zero_nutrition)
│   ├── execution.step  (get_nutrition → NutritionInfo)
│   ├── execution.step  (make_ingredient → Ingredient)
│   ├── execution.step  (add_nutrition → NutritionInfo)
│   ├── ...
│   └── execution.step  (build_meal_summary → MealSummary)
├── execution.complete
└── run.complete
```

### Primitives

| Primitive | Returns | Description |
|-----------|---------|-------------|
| `get_nutrition(ingredient, grams)` | `NutritionInfo` | Look up nutrition scaled to weight |
| `zero_nutrition()` | `NutritionInfo` | Zeroed accumulator for loops |
| `make_ingredient(name, grams, nutrition)` | `Ingredient` | Bundle into an Ingredient |
| `add_nutrition(a, b)` | `NutritionInfo` | Component-wise sum |
| `divide_nutrition(nutrition, servings)` | `NutritionInfo` | Per-serving calculation |
| `build_meal_summary(...)` | `MealSummary` | Final structured result |

15 built-in ingredients with real nutritional data (per 100g): chicken breast, rice, salmon, pasta, egg, broccoli, spinach, cheese, olive oil, tomato, potato, banana, oats, avocado, lentils.

## Running

Start the observability stack:

```bash
cd /path/to/OpenSymbolicAI/observability
docker compose up
```

Run the example:

```bash
uv run python examples/recipe_book/run_recipe_book.py
```

Options:

```bash
# Verbose mode — shows generated plans and primitive call traces
uv run python examples/recipe_book/run_recipe_book.py -v

# Specify a different Ollama model
uv run python examples/recipe_book/run_recipe_book.py qwen2:7b
```

View traces in the dashboard at http://localhost:8101.

## Example LLM-Generated Plan

For the query "200g chicken breast, 150g rice, 100g broccoli, serves 2", the LLM generates:

```python
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
    meal_name="Chicken and Rice Bowl",
    ingredients=ingredient_list,
    total_nutrition=total,
    servings=2,
    per_serving=per_serving,
)
```

## Tests

```bash
uv run pytest tests/test_recipe_book.py -v
```

19 tests covering primitives, plan execution with loops, and observability (event emission, type capture in traces, tag propagation).
