"""Tests for the FunctionOptimizer example agent."""

import math
import sys
from pathlib import Path
from typing import Any

from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import (
    ArgumentValue,
    ExecutionResult,
    ExecutionStep,
    ExecutionTrace,
    GoalStatus,
)

# Add examples directory to path so we can import the agent
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "function_optimizer"))

from function_optimizer import (
    FunctionOptimizer,
    OptimizationContext,
    Sample,
    target_function,
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
            else "result = evaluate(x=10.0)"
        )
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            provider="mock",
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


# =============================================================================
# Target Function Tests
# =============================================================================


class TestTargetFunction:
    def test_at_zero(self):
        val = target_function(0.0)
        expected = math.sin(0) * math.exp(-0.1 * 25) + 0.5 * math.cos(0) * math.exp(-0.05 * 144)
        assert abs(val - expected) < 1e-10

    def test_at_five(self):
        """Near x=5, the first Gaussian peaks."""
        val = target_function(5.0)
        # exp(-0.1*(5-5)^2) = 1.0, so first term = sin(15)
        # exp(-0.05*(5-12)^2) = exp(-2.45), so second term is small
        expected = math.sin(15) * 1.0 + 0.5 * math.cos(35) * math.exp(-2.45)
        assert abs(val - expected) < 1e-10

    def test_symmetry_not_present(self):
        """Function is NOT symmetric — confirms non-trivial landscape."""
        assert target_function(3.0) != target_function(17.0)

    def test_multiple_peaks(self):
        """There are multiple local maxima across the range."""
        # Sample at many points and count sign changes in the derivative
        values = [target_function(i * 0.1) for i in range(201)]
        peaks = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks += 1
        assert peaks >= 3, f"Expected multiple peaks, found {peaks}"


# =============================================================================
# OptimizationContext Tests
# =============================================================================


class TestOptimizationContext:
    def test_initial_state(self):
        ctx = OptimizationContext(goal="test")
        assert ctx.samples == []
        assert ctx.best_x is None
        assert ctx.best_value is None
        assert ctx.converged is False

    def test_sample_tracking(self):
        ctx = OptimizationContext(goal="test")
        ctx.samples.append(Sample(x=5.0, value=0.6))
        ctx.samples.append(Sample(x=10.0, value=0.2))
        assert len(ctx.samples) == 2
        assert ctx.samples[0].x == 5.0
        assert ctx.samples[1].value == 0.2


# =============================================================================
# Evaluate Primitive Tests
# =============================================================================


class TestEvaluatePrimitive:
    def test_returns_function_value(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        result = agent.evaluate(5.0)
        assert abs(result - round(target_function(5.0), 6)) < 1e-7

    def test_clamps_below_zero(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        result = agent.evaluate(-5.0)
        assert result == round(target_function(0.0), 6)

    def test_clamps_above_twenty(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        result = agent.evaluate(25.0)
        assert result == round(target_function(20.0), 6)

    def test_rounds_to_six_decimals(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        result = agent.evaluate(4.7)
        assert result == round(target_function(4.7), 6)


# =============================================================================
# Update Context Tests
# =============================================================================


def _make_eval_step(x: float, result: float, success: bool = True) -> ExecutionStep:
    """Create a mock ExecutionStep for an evaluate() call."""
    return ExecutionStep(
        step_number=1,
        statement=f"result = evaluate(x={x})",
        variable_name="result",
        primitive_called="evaluate",
        args={"x": ArgumentValue(expression=str(x), resolved_value=x)},
        result_type="float",
        result_value=result,
        success=success,
    )


class TestUpdateContext:
    def test_extracts_single_sample(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        ctx = OptimizationContext(goal="test")

        step = _make_eval_step(5.0, 0.611)
        exec_result = ExecutionResult(
            value_type="float",
            trace=ExecutionTrace(steps=[step]),
        )
        agent.update_context(ctx, exec_result)

        assert len(ctx.samples) == 1
        assert ctx.samples[0].x == 5.0
        assert ctx.samples[0].value == 0.611
        assert ctx.best_x == 5.0
        assert ctx.best_value == 0.611

    def test_extracts_multiple_samples(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        ctx = OptimizationContext(goal="test")

        steps = [
            _make_eval_step(3.0, 0.2),
            _make_eval_step(8.0, -0.1),
            _make_eval_step(14.0, 0.3),
        ]
        exec_result = ExecutionResult(
            value_type="float",
            trace=ExecutionTrace(steps=steps),
        )
        agent.update_context(ctx, exec_result)

        assert len(ctx.samples) == 3
        assert ctx.best_x == 14.0
        assert ctx.best_value == 0.3

    def test_skips_failed_steps(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        ctx = OptimizationContext(goal="test")

        steps = [
            _make_eval_step(5.0, 0.6, success=True),
            _make_eval_step(10.0, 0.0, success=False),
        ]
        exec_result = ExecutionResult(
            value_type="float",
            trace=ExecutionTrace(steps=steps),
        )
        agent.update_context(ctx, exec_result)

        assert len(ctx.samples) == 1
        assert ctx.best_x == 5.0

    def test_skips_non_evaluate_steps(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        ctx = OptimizationContext(goal="test")

        step = ExecutionStep(
            step_number=1,
            statement="x = 5",
            variable_name="x",
            primitive_called=None,
            result_value=5,
            success=True,
        )
        exec_result = ExecutionResult(
            value_type="int",
            trace=ExecutionTrace(steps=[step]),
        )
        agent.update_context(ctx, exec_result)

        assert len(ctx.samples) == 0

    def test_convergence_when_close_to_true_max(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm, tolerance=0.01)
        ctx = OptimizationContext(goal="test")

        # Evaluate at a point very close to the true maximum
        true_val = agent._true_max_value
        step = _make_eval_step(agent._true_max_x, true_val)
        exec_result = ExecutionResult(
            value_type="float",
            trace=ExecutionTrace(steps=[step]),
        )
        agent.update_context(ctx, exec_result)

        assert ctx.converged is True

    def test_no_convergence_when_far_from_max(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm, tolerance=0.01)
        ctx = OptimizationContext(goal="test")

        step = _make_eval_step(0.0, target_function(0.0))
        exec_result = ExecutionResult(
            value_type="float",
            trace=ExecutionTrace(steps=[step]),
        )
        agent.update_context(ctx, exec_result)

        assert ctx.converged is False


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestCheckConverged:
    def test_not_converged(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        ctx = OptimizationContext(goal="test", converged=False)
        result = agent.check_converged("test", ctx)
        assert result.goal_achieved is False

    def test_converged(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        ctx = OptimizationContext(goal="test", converged=True)
        result = agent.check_converged("test", ctx)
        assert result.goal_achieved is True


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    def test_seek_converges_near_true_max(self):
        """Agent converges when LLM samples near the true maximum."""
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm, tolerance=0.01)
        true_x = agent._true_max_x

        # LLM generates code that samples near the true max
        mock_llm.responses = [
            f"result = evaluate(x={true_x:.4f})",
        ]

        result = agent.seek("Find the maximum of f(x) on [0, 20]")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 1

    def test_seek_hits_max_iterations(self):
        """Agent stops at max_iterations when never converging."""
        mock_llm = MockLLM(
            responses=[
                "result = evaluate(x=0.0)",
                "result = evaluate(x=0.0)",
                "result = evaluate(x=0.0)",
            ]
        )
        agent = FunctionOptimizer(llm=mock_llm, tolerance=0.001, max_iterations=3)
        result = agent.seek("Find the maximum")

        assert result.status == GoalStatus.MAX_ITERATIONS
        assert result.iteration_count == 3

    def test_seek_with_multiple_evals_per_iteration(self):
        """Agent can make multiple evaluate calls in a single plan."""
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm, tolerance=0.01)
        true_x = agent._true_max_x

        mock_llm.responses = [
            # First iteration: explore broadly
            "v1 = evaluate(x=3.0)\nv2 = evaluate(x=10.0)\nv3 = evaluate(x=17.0)",
            # Second iteration: refine near the true max
            f"result = evaluate(x={true_x:.4f})",
        ]

        result = agent.seek("Find the maximum of f(x)")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 2


# =============================================================================
# Compute True Max Tests
# =============================================================================


class TestComputeTrueMax:
    def test_true_max_is_near_4_7(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        assert 4.5 < agent._true_max_x < 5.0

    def test_true_max_value_is_positive(self):
        mock_llm = MockLLM()
        agent = FunctionOptimizer(llm=mock_llm)
        assert agent._true_max_value > 0.9
