"""Calculator benchmark runner for evaluating LLM models.

Runs calculator intents against different Ollama models and produces
accuracy and performance reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from opensymbolicai.llm import LLMConfig, Provider

# Import calculator from examples
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples" / "calculator"))
from calculator import ScientificCalculator


@dataclass
class IntentResult:
    """Result of running a single intent."""

    id: int
    category: str
    intent: str
    expected: float
    actual: float | None
    tolerance: float
    passed: bool
    error: str | None
    plan: str | None
    llm_response: str | None
    execution_time_seconds: float
    plan_time_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass
class ModelResult:
    """Aggregated results for a single model."""

    model: str
    total_intents: int
    passed: int
    failed: int
    errors: int
    accuracy: float
    total_time_seconds: float
    avg_time_per_intent: float
    total_input_tokens: int
    total_output_tokens: int
    results_by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    intent_results: list[IntentResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    timestamp: str
    intents_file: str
    models: list[str]
    model_results: list[ModelResult]
    total_runtime_seconds: float


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running."""
    try:
        request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def get_available_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Get list of available Ollama models."""
    try:
        request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [model["name"] for model in data.get("models", [])]
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return []


def load_intents(intents_file: Path) -> dict[str, Any]:
    """Load intents from JSON file."""
    with open(intents_file) as f:
        return json.load(f)


def run_intent(
    calculator: ScientificCalculator,
    intent: dict[str, Any],
) -> IntentResult:
    """Run a single intent and return the result."""
    intent_id = intent["id"]
    category = intent["category"]
    intent_text = intent["intent"]
    expected = intent["expected"]
    tolerance = intent["tolerance"]

    actual: float | None = None
    error: str | None = None
    plan: str | None = None
    llm_response: str | None = None
    passed = False
    execution_time = 0.0
    plan_time = 0.0
    input_tokens = 0
    output_tokens = 0

    try:
        result = calculator.run(intent_text)

        # Always capture plan and metrics for debugging
        plan = result.plan
        # Capture raw LLM response if available
        if result.plan_attempts:
            llm_response = result.plan_attempts[
                -1
            ].plan_generation.llm_interaction.response
        if result.metrics:
            execution_time = result.metrics.execute_time_seconds
            plan_time = result.metrics.plan_time_seconds
            input_tokens = result.metrics.plan_tokens.input_tokens
            output_tokens = result.metrics.plan_tokens.output_tokens

        if result.success:
            actual = result.result

            # Check if result is within tolerance
            if actual is not None and isinstance(actual, (int, float)):
                passed = abs(actual - expected) <= tolerance
            else:
                passed = False
                error = f"Invalid result type: {type(actual)}"
        else:
            error = result.error or "Unknown error"
    except Exception as e:
        error = str(e)

    return IntentResult(
        id=intent_id,
        category=category,
        intent=intent_text,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
        passed=passed,
        error=error,
        plan=plan,
        llm_response=llm_response,
        execution_time_seconds=execution_time,
        plan_time_seconds=plan_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def run_intent_worker(
    model: str,
    intent: dict[str, Any],
) -> IntentResult:
    """Worker function to run a single intent with its own calculator instance."""
    config = LLMConfig(provider=Provider.OLLAMA, model=model)
    calculator = ScientificCalculator(llm=config)
    return run_intent(calculator, intent)


def run_benchmark_for_model(
    model: str,
    intents: list[dict[str, Any]],
    verbose: bool = False,
    parallel: int = 1,
) -> ModelResult:
    """Run benchmark for a single model.

    Args:
        model: The Ollama model name
        intents: List of intents to run
        verbose: Whether to print verbose output
        parallel: Number of parallel workers (1 = sequential)
    """
    intent_results: list[IntentResult] = []
    results_by_category: dict[str, dict[str, int]] = {}
    passed_count = 0
    error_count = 0

    start_time = time.time()

    # Create progress bar (disable if not a TTY for cleaner piped output)
    is_tty = sys.stderr.isatty()

    if parallel > 1:
        # Parallel execution
        pbar = tqdm(
            total=len(intents),
            desc=f"{model:<25}",
            unit="intent",
            leave=True,
            dynamic_ncols=True,
            disable=not is_tty,
        )

        if not is_tty:
            print(f"Running {model} with {parallel} workers...", end=" ", flush=True)

        # Use a lock for thread-safe updates
        lock = Lock()

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_intent_worker, model, intent): intent
                for intent in intents
            }

            for future in as_completed(futures):
                intent = futures[future]
                result = future.result()

                with lock:
                    intent_results.append(result)

                    # Track by category
                    cat = result.category
                    if cat not in results_by_category:
                        results_by_category[cat] = {
                            "passed": 0,
                            "failed": 0,
                            "errors": 0,
                        }

                    if result.error:
                        results_by_category[cat]["errors"] += 1
                        error_count += 1
                    elif result.passed:
                        results_by_category[cat]["passed"] += 1
                        passed_count += 1
                    else:
                        results_by_category[cat]["failed"] += 1

                    # Update progress bar
                    accuracy = passed_count / len(intent_results) * 100
                    pbar.set_postfix(
                        acc=f"{accuracy:.0f}%", err=error_count, refresh=True
                    )
                    pbar.update(1)

                    if verbose:
                        if result.passed:
                            tqdm.write(f"  PASS: {intent['intent'][:60]}")
                        elif result.error:
                            tqdm.write(
                                f"  ERR:  {intent['intent'][:45]}... {result.error[:25]}"
                            )
                        else:
                            tqdm.write(
                                f"  FAIL: {intent['intent'][:45]}... "
                                f"(exp={result.expected}, got={result.actual})"
                            )

        pbar.close()

        # Sort results by intent ID to maintain consistent ordering
        intent_results.sort(key=lambda r: r.id)
    else:
        # Sequential execution (original behavior)
        config = LLMConfig(provider=Provider.OLLAMA, model=model)
        calculator = ScientificCalculator(llm=config)

        pbar = tqdm(
            intents,
            desc=f"{model:<25}",
            unit="intent",
            leave=True,
            dynamic_ncols=True,
            disable=not is_tty,
        )

        # Print simple progress for non-TTY
        if not is_tty:
            print(f"Running {model}...", end=" ", flush=True)

        for intent in pbar:
            result = run_intent(calculator, intent)
            intent_results.append(result)

            # Track by category
            cat = result.category
            if cat not in results_by_category:
                results_by_category[cat] = {"passed": 0, "failed": 0, "errors": 0}

            if result.error:
                results_by_category[cat]["errors"] += 1
                error_count += 1
            elif result.passed:
                results_by_category[cat]["passed"] += 1
                passed_count += 1
            else:
                results_by_category[cat]["failed"] += 1

            # Update progress bar with current stats
            accuracy = passed_count / len(intent_results) * 100
            pbar.set_postfix(acc=f"{accuracy:.0f}%", err=error_count, refresh=True)

            if verbose:
                if result.passed:
                    tqdm.write(f"  PASS: {intent['intent'][:60]}")
                elif result.error:
                    tqdm.write(
                        f"  ERR:  {intent['intent'][:45]}... {result.error[:25]}"
                    )
                else:
                    tqdm.write(
                        f"  FAIL: {intent['intent'][:45]}... "
                        f"(exp={result.expected}, got={result.actual})"
                    )

        pbar.close()

    # Print summary for non-TTY mode
    if not is_tty:
        accuracy = passed_count / len(intents) * 100 if intents else 0
        print(f"done. acc={accuracy:.1f}% ({passed_count}/{len(intents)})")

    total_time = time.time() - start_time

    passed = sum(1 for r in intent_results if r.passed)
    failed = sum(1 for r in intent_results if not r.passed and not r.error)
    errors = sum(1 for r in intent_results if r.error)

    return ModelResult(
        model=model,
        total_intents=len(intents),
        passed=passed,
        failed=failed,
        errors=errors,
        accuracy=passed / len(intents) * 100 if intents else 0,
        total_time_seconds=total_time,
        avg_time_per_intent=total_time / len(intents) if intents else 0,
        total_input_tokens=sum(r.input_tokens for r in intent_results),
        total_output_tokens=sum(r.output_tokens for r in intent_results),
        results_by_category=results_by_category,
        intent_results=intent_results,
    )


def generate_report(report: BenchmarkReport) -> str:
    """Generate a text report from benchmark results."""
    lines = [
        "=" * 80,
        "CALCULATOR BENCHMARK REPORT",
        "=" * 80,
        f"Timestamp: {report.timestamp}",
        f"Intents file: {report.intents_file}",
        f"Total runtime: {report.total_runtime_seconds:.2f}s",
        f"Models tested: {len(report.models)}",
        "",
    ]

    # Summary table
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(
        f"{'Model':<30} {'Accuracy':>10} {'Passed':>8} {'Failed':>8} {'Errors':>8} {'Avg Time':>10}"
    )
    lines.append("-" * 80)

    for result in report.model_results:
        lines.append(
            f"{result.model:<30} {result.accuracy:>9.1f}% {result.passed:>8} "
            f"{result.failed:>8} {result.errors:>8} {result.avg_time_per_intent:>9.2f}s"
        )

    lines.append("-" * 80)
    lines.append("")

    # Per-model details
    for result in report.model_results:
        lines.append("=" * 80)
        lines.append(f"MODEL: {result.model}")
        lines.append("=" * 80)
        lines.append(f"Total intents: {result.total_intents}")
        lines.append(f"Passed: {result.passed} ({result.accuracy:.1f}%)")
        lines.append(f"Failed: {result.failed}")
        lines.append(f"Errors: {result.errors}")
        lines.append(f"Total time: {result.total_time_seconds:.2f}s")
        lines.append(
            f"Total tokens: {result.total_input_tokens + result.total_output_tokens} "
            f"(in: {result.total_input_tokens}, out: {result.total_output_tokens})"
        )
        lines.append("")

        # Category breakdown
        lines.append("Results by category:")
        lines.append(
            f"  {'Category':<20} {'Passed':>8} {'Failed':>8} {'Errors':>8} {'Accuracy':>10}"
        )
        lines.append("  " + "-" * 56)
        for cat, stats in sorted(result.results_by_category.items()):
            total = stats["passed"] + stats["failed"] + stats["errors"]
            acc = stats["passed"] / total * 100 if total > 0 else 0
            lines.append(
                f"  {cat:<20} {stats['passed']:>8} {stats['failed']:>8} "
                f"{stats['errors']:>8} {acc:>9.1f}%"
            )
        lines.append("")

        # Failed intents
        failed_intents = [r for r in result.intent_results if not r.passed]
        if failed_intents:
            lines.append("Failed/Error intents:")
            for r in failed_intents[:20]:  # Limit to first 20
                if r.error:
                    lines.append(f"  [{r.id}] ERROR: {r.intent[:40]}...")
                    lines.append(f"       {r.error[:60]}")
                else:
                    lines.append(
                        f"  [{r.id}] FAIL: {r.intent[:40]}... "
                        f"(expected={r.expected}, actual={r.actual})"
                    )
            if len(failed_intents) > 20:
                lines.append(f"  ... and {len(failed_intents) - 20} more")
            lines.append("")

    return "\n".join(lines)


def generate_json_report(report: BenchmarkReport) -> dict[str, Any]:
    """Generate a JSON-serializable report."""
    return {
        "metadata": {
            "timestamp": report.timestamp,
            "intents_file": report.intents_file,
            "total_runtime_seconds": report.total_runtime_seconds,
            "models_tested": report.models,
        },
        "summary": [
            {
                "model": r.model,
                "accuracy_percent": round(r.accuracy, 2),
                "passed": r.passed,
                "failed": r.failed,
                "errors": r.errors,
                "total_intents": r.total_intents,
                "avg_time_seconds": round(r.avg_time_per_intent, 3),
                "total_tokens": r.total_input_tokens + r.total_output_tokens,
            }
            for r in report.model_results
        ],
        "details": {
            r.model: {
                "results_by_category": r.results_by_category,
                "intent_results": [
                    {
                        "id": ir.id,
                        "category": ir.category,
                        "intent": ir.intent,
                        "expected": ir.expected,
                        "actual": ir.actual,
                        "passed": ir.passed,
                        "error": ir.error,
                        "plan": ir.plan,
                        "llm_response": ir.llm_response,
                        "execution_time_seconds": round(ir.execution_time_seconds, 3),
                        "plan_time_seconds": round(ir.plan_time_seconds, 3),
                        "input_tokens": ir.input_tokens,
                        "output_tokens": ir.output_tokens,
                    }
                    for ir in r.intent_results
                ],
            }
            for r in report.model_results
        },
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run calculator benchmark against Ollama models"
    )
    parser.add_argument(
        "--intents",
        type=Path,
        default=Path(__file__).parent / "intents.json",
        help="Path to intents JSON file",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Ollama models to test (default: all available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of intents to run",
    )
    parser.add_argument(
        "--category",
        help="Only run intents from this category",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for running intents (default: 1 = sequential)",
    )
    args = parser.parse_args()

    # Check Ollama availability
    if not is_ollama_available():
        print("ERROR: Ollama is not running. Please start Ollama first.")
        return 1

    # Get models to test
    if args.models:
        models = args.models
    else:
        models = get_available_ollama_models()
        if not models:
            print("ERROR: No Ollama models available. Pull a model first:")
            print("  ollama pull llama3.2")
            return 1

    print(f"Models to test: {', '.join(models)}")

    # Load intents
    intents_data = load_intents(args.intents)
    intents = intents_data["intents"]

    # Filter by category if specified
    if args.category:
        intents = [i for i in intents if i["category"] == args.category]
        print(f"Filtered to {len(intents)} intents in category '{args.category}'")

    # Limit if specified
    if args.limit:
        intents = intents[: args.limit]
        print(f"Limited to {len(intents)} intents")

    parallel_msg = (
        f" with {args.parallel} parallel workers" if args.parallel > 1 else ""
    )
    print(
        f"Running {len(intents)} intents against {len(models)} model(s){parallel_msg}..."
    )
    print()

    # Run benchmark
    start_time = time.time()
    model_results: list[ModelResult] = []

    for model in models:
        result = run_benchmark_for_model(
            model, intents, verbose=args.verbose, parallel=args.parallel
        )
        model_results.append(result)
        print()

    total_time = time.time() - start_time

    # Generate report
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        intents_file=str(args.intents),
        models=models,
        model_results=model_results,
        total_runtime_seconds=total_time,
    )

    # Print text report
    print(generate_report(report))

    # Always save JSON report
    json_report = generate_json_report(report)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_names = "_".join(m.replace(":", "-").replace("/", "-") for m in models[:3])
    if len(models) > 3:
        model_names += f"_+{len(models) - 3}"
    output_file = results_dir / f"benchmark_{timestamp}_{model_names}.json"

    with open(output_file, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Also save to custom path if specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(json_report, f, indent=2)
        print(f"Also saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
