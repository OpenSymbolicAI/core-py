# Calculator Benchmark

Benchmarks LLM models on their ability to use the ScientificCalculator agent to solve mathematical problems.

## Overview

This benchmark evaluates how well different LLM models can:
- Understand natural language math queries
- Generate correct plans using calculator primitives
- Produce accurate numerical results

## Intents

The benchmark includes 120 intents across 6 categories:

| Category | Count | Description |
|----------|-------|-------------|
| `basic_arithmetic` | 22 | Addition, subtraction, multiplication, division |
| `powers_roots` | 15 | Squares, cubes, powers, square roots |
| `trigonometry` | 18 | Sine, cosine, tangent with degree conversion |
| `logarithms` | 14 | Natural log, log base 10, exponentials |
| `constants` | 4 | Pi, Euler's number |
| `complex` | 47 | Multi-step: circle area, Pythagorean theorem, percentages, averages |

## Prerequisites

1. **Ollama** must be running locally:
   ```bash
   ollama serve
   ```

2. **At least one model** must be available:
   ```bash
   ollama pull llama3.2
   ollama pull qwen2.5
   ollama pull mistral
   ```

## Usage

### Basic Usage

Run against all available Ollama models:
```bash
uv run python benchmarks/calculator/benchmark.py
```

### Specify Models

Test specific models:
```bash
uv run python benchmarks/calculator/benchmark.py --models llama3.2 qwen2.5:7b mistral
```

### Limit Intents

Run a quick test with fewer intents:
```bash
uv run python benchmarks/calculator/benchmark.py --limit 20
```

### Filter by Category

Run only specific categories:
```bash
uv run python benchmarks/calculator/benchmark.py --category trigonometry
uv run python benchmarks/calculator/benchmark.py --category complex
```

### Verbose Output

See each intent result as it runs:
```bash
uv run python benchmarks/calculator/benchmark.py -v
```

The benchmark shows a live progress bar with accuracy and error count when running in a terminal. When piped or redirected, it shows a simpler progress indicator.

### Parallel Execution

Run multiple intents concurrently for faster benchmarking:
```bash
uv run python benchmarks/calculator/benchmark.py --parallel 4
# or
uv run python benchmarks/calculator/benchmark.py -p 4
```

Each parallel worker creates its own calculator instance to avoid contention. Results are sorted by intent ID to maintain consistent ordering in reports.

### Save JSON Report

Export detailed results to JSON:
```bash
uv run python benchmarks/calculator/benchmark.py --output results.json
```

### Combined Example

```bash
uv run python benchmarks/calculator/benchmark.py \
  --models llama3.2 qwen2.5 \
  --limit 50 \
  --parallel 4 \
  --output benchmark_results.json \
  -v
```

## Output

### Text Report

The benchmark produces a summary table:

```
--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
Model                            Accuracy   Passed   Failed   Errors   Avg Time
--------------------------------------------------------------------------------
llama3.2                            85.0%      102        8       10      1.23s
qwen2.5                             78.3%       94       16       10      0.98s
--------------------------------------------------------------------------------
```

And detailed per-model statistics:
- Results by category
- Token usage (input/output)
- List of failed intents with expected vs actual values

### JSON Report

Results are automatically saved to `benchmarks/calculator/results/` with timestamped filenames.

Use `--output` to additionally save to a custom path.

JSON structure:
```json
{
  "metadata": {
    "timestamp": "2026-01-25T21:41:20",
    "intents_file": "...",
    "total_runtime_seconds": 8.27,
    "models_tested": ["llama3.2"]
  },
  "summary": [
    {
      "model": "llama3.2",
      "accuracy_percent": 85.0,
      "passed": 102,
      "failed": 8,
      "errors": 10
    }
  ],
  "details": {
    "llama3.2": {
      "results_by_category": {...},
      "intent_results": [
        {
          "id": 1,
          "intent": "What is 5 plus 3?",
          "expected": 8.0,
          "actual": 8.0,
          "passed": true,
          "error": null,
          "plan": "result = self.add_numbers(5, 3)",
          "llm_response": "```python\nresult = ...\n```",
          "input_tokens": 1024,
          "output_tokens": 32
        }
      ]
    }
  }
}
```

Key fields for debugging:
- `plan`: Extracted Python code that was attempted
- `llm_response`: Raw LLM output before extraction
- `error`: Error message if execution failed

## Interpreting Results

- **Passed**: Result within tolerance of expected value
- **Failed**: Result computed but incorrect
- **Errors**: Plan generation or execution failed

Common error causes:
- Invalid Python syntax in generated plan
- Calling non-existent primitives
- Incorrect argument names
- Division by zero or invalid math operations

## Adding New Intents

Edit `intents.json` to add new test cases:

```json
{
  "id": 121,
  "category": "basic_arithmetic",
  "intent": "What is 7 plus 8?",
  "expected": 15.0,
  "tolerance": 0.0001
}
```

Fields:
- `id`: Unique identifier
- `category`: One of the category names
- `intent`: Natural language query
- `expected`: Expected numerical result
- `tolerance`: Acceptable error margin (use larger values for floating-point results)
