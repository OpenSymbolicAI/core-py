"""Unit tests for DesignExecute blueprint."""

from typing import Any

import pytest

from opensymbolicai.blueprints.design_execute import DesignExecute
from opensymbolicai.core import primitive
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import DesignExecuteConfig, MutationHookContext


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


class SimpleCalculator(DesignExecute):
    """Simple calculator for testing."""

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=True)
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


class CalculatorWithMemory(DesignExecute):
    """Calculator with mutable memory for testing mutation hooks."""

    def __init__(
        self,
        llm: LLM | LLMConfig,
        config: DesignExecuteConfig | None = None,
    ):
        super().__init__(llm=llm, config=config)
        self._memory: float = 0.0

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=False)
    def memory_store(self, value: float) -> float:
        """Store a value in memory."""
        self._memory = value
        return self._memory

    @primitive(read_only=False)
    def memory_add(self, value: float) -> float:
        """Add to memory."""
        self._memory += value
        return self._memory

    @primitive(read_only=True)
    def memory_recall(self) -> float:
        """Get memory value."""
        return self._memory


# =============================================================================
# Config Tests
# =============================================================================


class TestDesignExecuteConfig:
    """Test DesignExecuteConfig defaults and settings."""

    def test_defaults(self):
        config = DesignExecuteConfig()
        assert config.max_loop_iterations == 100
        assert config.max_total_primitive_calls == 1000
        assert config.allow_break_continue is True

    def test_custom_values(self):
        config = DesignExecuteConfig(
            max_loop_iterations=50,
            max_total_primitive_calls=500,
            allow_break_continue=False,
        )
        assert config.max_loop_iterations == 50
        assert config.max_total_primitive_calls == 500
        assert config.allow_break_continue is False

    def test_inherits_plan_execute_config_fields(self):
        config = DesignExecuteConfig(multi_turn=True, max_plan_retries=3)
        assert config.multi_turn is True
        assert config.max_plan_retries == 3


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidationAllowsControlFlow:
    """Test that DesignExecute validation allows control flow constructs."""

    def test_allows_for_loops(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "results = []\nfor i in range(3):\n    x = add(float(i), 1.0)\n    results.append(x)\nresult = results"
        # Should not raise
        calc.validate_plan(plan)

    def test_allows_while_loops(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = 0.0\nwhile x < 10:\n    x = add(x, 1.0)\nresult = x"
        calc.validate_plan(plan)

    def test_allows_if_elif_else(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 2.0)\nif x > 5:\n    result = multiply(x, 2.0)\nelse:\n    result = multiply(x, 3.0)"
        calc.validate_plan(plan)

    def test_allows_nested_loops(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "total = 0.0\nfor i in range(3):\n    for j in range(3):\n        total = add(total, 1.0)\nresult = total"
        calc.validate_plan(plan)

    def test_allows_augmented_assignment(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = 0.0\nx += 1.0\nresult = x"
        calc.validate_plan(plan)

    def test_allows_bare_expressions(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "results = []\nresults.append(add(1.0, 2.0))\nresult = results"
        calc.validate_plan(plan)

    def test_allows_break_by_default(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = 0.0\nwhile True:\n    x = add(x, 1.0)\n    if x > 5:\n        break\nresult = x"
        calc.validate_plan(plan)

    def test_allows_continue_by_default(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "total = 0.0\nfor i in range(10):\n    if i < 5:\n        continue\n    total = add(total, float(i))\nresult = total"
        calc.validate_plan(plan)


class TestValidationBlocksDangerous:
    """Test that DesignExecute still blocks dangerous operations."""

    def test_blocks_function_definitions(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="FunctionDef"):
            calc.validate_plan("def foo():\n    pass")

    def test_blocks_class_definitions(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="ClassDef"):
            calc.validate_plan("class Foo:\n    pass")

    def test_blocks_imports(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="Import"):
            calc.validate_plan("import os")

    def test_blocks_import_from(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="ImportFrom"):
            calc.validate_plan("from os import path")

    def test_allows_try_except(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        # try/except is allowed in DesignExecute
        calc.validate_plan("try:\n    x = add(1.0, 2.0)\nexcept ValueError:\n    x = 0.0")

    def test_blocks_with_statement(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="With"):
            calc.validate_plan("with open('f') as f:\n    pass")

    def test_blocks_exec(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="exec"):
            calc.validate_plan("exec('x = 1')")

    def test_blocks_eval(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="eval"):
            calc.validate_plan("x = eval('1+1')")

    def test_blocks_open(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="open"):
            calc.validate_plan("x = open('file.txt')")

    def test_blocks_private_attributes(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="private"):
            calc.validate_plan("x = self._secret")

    def test_blocks_unknown_functions(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="not allowed"):
            calc.validate_plan("x = unknown_func()")

    def test_blocks_non_primitive_self_methods(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="Do not use 'self.' prefix"):
            calc.validate_plan("x = self.not_a_primitive()")

    def test_blocks_break_when_disabled(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(allow_break_continue=False),
        )

        with pytest.raises(ValueError, match="Break"):
            calc.validate_plan(
                "for i in range(10):\n    if i > 5:\n        break\n    x = add(float(i), 0.0)"
            )

    def test_blocks_continue_when_disabled(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(allow_break_continue=False),
        )

        with pytest.raises(ValueError, match="Continue"):
            calc.validate_plan(
                "for i in range(10):\n    if i < 5:\n        continue\n    x = add(float(i), 0.0)"
            )

    def test_allows_raise(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        # raise is allowed in DesignExecute
        calc.validate_plan("if True:\n    raise ValueError('test')")

    def test_blocks_assert(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="Assert"):
            calc.validate_plan("assert True")


# =============================================================================
# Execution Tests
# =============================================================================


class TestForLoopExecution:
    """Test for loop execution."""

    def test_simple_for_loop(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "total = 0.0\nfor i in range(5):\n    total = add(total, 1.0)\nresult = total"
        result = calc.execute(plan)
        assert result.value_name == "result"
        assert result.get_value() == 5.0

    def test_for_loop_traces_each_call(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "total = 0.0\nfor i in range(3):\n    total = add(total, 1.0)\nresult = total"
        result = calc.execute(plan)
        # Should have 3 traced primitive calls (one per iteration)
        assert result.trace.step_count == 3
        assert all(s.primitive_called == "add" for s in result.trace.steps)
        assert all(s.success for s in result.trace.steps)

    def test_for_loop_with_accumulator_list(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = (
            "results = []\n"
            "for i in range(3):\n"
            "    x = add(float(i), 10.0)\n"
            "    results.append(x)\n"
            "result = results"
        )
        result = calc.execute(plan)
        assert result.get_value() == [10.0, 11.0, 12.0]


class TestWhileLoopExecution:
    """Test while loop execution."""

    def test_simple_while_loop(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = 0.0\nwhile x < 5:\n    x = add(x, 1.0)\nresult = x"
        result = calc.execute(plan)
        assert result.get_value() == 5.0

    def test_while_with_break(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = 0.0\nwhile True:\n    x = add(x, 1.0)\n    if x >= 3:\n        break\nresult = x"
        result = calc.execute(plan)
        assert result.get_value() == 3.0


class TestConditionalExecution:
    """Test if/elif/else execution."""

    def test_if_true_branch(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(5.0, 5.0)\nif x > 5:\n    result = multiply(x, 2.0)\nelse:\n    result = multiply(x, 3.0)"
        result = calc.execute(plan)
        assert result.get_value() == 20.0  # 10 * 2

    def test_else_branch(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 1.0)\nif x > 5:\n    result = multiply(x, 2.0)\nelse:\n    result = multiply(x, 3.0)"
        result = calc.execute(plan)
        assert result.get_value() == 6.0  # 2 * 3

    def test_no_trace_for_skipped_branch(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 1.0)\nif x > 100:\n    y = multiply(x, 2.0)\nresult = x"
        result = calc.execute(plan)
        # Only the add call should be traced, not the multiply
        assert result.trace.step_count == 1
        assert result.trace.steps[0].primitive_called == "add"


class TestNestedControlFlow:
    """Test nested loops and conditionals."""

    def test_nested_for_loops(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = (
            "total = 0.0\n"
            "for i in range(3):\n"
            "    for j in range(2):\n"
            "        total = add(total, 1.0)\n"
            "result = total"
        )
        result = calc.execute(plan)
        assert result.get_value() == 6.0  # 3 * 2 iterations
        assert result.trace.step_count == 6

    def test_loop_with_conditional(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = (
            "total = 0.0\n"
            "for i in range(6):\n"
            "    if i % 2 == 0:\n"
            "        total = add(total, float(i))\n"
            "result = total"
        )
        result = calc.execute(plan)
        assert result.get_value() == 6.0  # 0 + 2 + 4
        assert result.trace.step_count == 3  # Only even iterations call add


# =============================================================================
# Result Variable Tests
# =============================================================================


class TestResultVariable:
    """Test result variable determination."""

    def test_explicit_result_variable(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 2.0)\nresult = multiply(x, 3.0)"
        result = calc.execute(plan)
        assert result.value_name == "result"
        assert result.get_value() == 9.0

    def test_fallback_to_last_variable(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 2.0)\ny = multiply(x, 3.0)"
        result = calc.execute(plan)
        # No 'result' variable, should use last variable in namespace
        assert result.get_value() == 9.0

    def test_result_from_loop(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "total = 0.0\nfor i in range(3):\n    total = add(total, 1.0)\nresult = total"
        result = calc.execute(plan)
        assert result.value_name == "result"
        assert result.get_value() == 3.0


# =============================================================================
# Safety Tests
# =============================================================================


class TestLoopSafety:
    """Test loop iteration limits and safety."""

    def test_infinite_while_loop_stopped(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_loop_iterations=10),
        )

        plan = "x = 0.0\nwhile True:\n    x = add(x, 1.0)"
        result = calc.execute(plan)
        # Should have a failed step for loop limit
        assert not result.trace.all_succeeded
        last_step = result.trace.last_step
        assert last_step is not None
        assert not last_step.success
        assert "Loop exceeded maximum iterations" in last_step.error

    def test_custom_loop_limit(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_loop_iterations=5),
        )

        plan = "x = 0.0\nwhile True:\n    x = add(x, 1.0)"
        result = calc.execute(plan)
        # Should stop after ~5 iterations (5 primitive calls before guard triggers)
        successful = result.trace.successful_steps
        assert len(successful) == 5

    def test_nested_loop_each_has_own_counter(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_loop_iterations=10),
        )

        # Inner loop runs 3 times per outer iteration, outer runs 4 times = 12 total
        # Each individual loop is under the limit of 10
        plan = (
            "total = 0.0\n"
            "for i in range(4):\n"
            "    for j in range(3):\n"
            "        total = add(total, 1.0)\n"
            "result = total"
        )
        result = calc.execute(plan)
        assert result.get_value() == 12.0
        assert result.trace.step_count == 12

    def test_max_total_primitive_calls(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(
                max_loop_iterations=1000,
                max_total_primitive_calls=10,
            ),
        )

        plan = "x = 0.0\nfor i in range(100):\n    x = add(x, 1.0)\nresult = x"
        result = calc.execute(plan)
        # Should stop after 10 primitive calls
        assert not result.trace.all_succeeded
        successful = result.trace.successful_steps
        assert len(successful) == 10

    def test_for_loop_within_limit_succeeds(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_loop_iterations=10),
        )

        plan = "total = 0.0\nfor i in range(5):\n    total = add(total, 1.0)\nresult = total"
        result = calc.execute(plan)
        assert result.trace.all_succeeded
        assert result.get_value() == 5.0


# =============================================================================
# Tracing Tests
# =============================================================================


class TestTracing:
    """Test execution tracing."""

    def test_traces_all_primitive_calls(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 2.0)\ny = multiply(3.0, 4.0)\nresult = add(x, y)"
        result = calc.execute(plan)
        assert result.trace.step_count == 3
        assert result.trace.steps[0].primitive_called == "add"
        assert result.trace.steps[1].primitive_called == "multiply"
        assert result.trace.steps[2].primitive_called == "add"

    def test_trace_includes_args(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "result = add(1.0, 2.0)"
        result = calc.execute(plan)
        assert result.trace.step_count == 1
        step = result.trace.steps[0]
        assert "a" in step.args
        assert step.args["a"].resolved_value == 1.0
        assert "b" in step.args
        assert step.args["b"].resolved_value == 2.0

    def test_trace_includes_result_value(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "result = add(1.0, 2.0)"
        result = calc.execute(plan)
        step = result.trace.steps[0]
        assert step.result_value == 3.0
        assert step.result_type == "float"

    def test_trace_has_timing(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "result = add(1.0, 2.0)"
        result = calc.execute(plan)
        assert result.trace.total_time_seconds > 0
        assert result.trace.steps[0].time_seconds >= 0

    def test_trace_step_numbers_sequential(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "total = 0.0\nfor i in range(3):\n    total = add(total, 1.0)\nresult = total"
        result = calc.execute(plan)
        for i, step in enumerate(result.trace.steps):
            assert step.step_number == i + 1


# =============================================================================
# Mutation Hook Tests
# =============================================================================


class TestMutationHooksInLoops:
    """Test mutation hooks with control flow."""

    def test_hook_called_per_iteration(self):
        calls: list[MutationHookContext] = []

        def track_hook(ctx: MutationHookContext) -> str | None:
            calls.append(ctx)
            return None

        mock_llm = MockLLM()
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=DesignExecuteConfig(on_mutation=track_hook),
        )

        plan = "for i in range(3):\n    memory_store(value=float(i))"
        calc.execute(plan)
        assert len(calls) == 3
        assert all(c.method_name == "memory_store" for c in calls)

    def test_hook_not_called_for_read_only(self):
        calls: list[MutationHookContext] = []

        def track_hook(ctx: MutationHookContext) -> str | None:
            calls.append(ctx)
            return None

        mock_llm = MockLLM()
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=DesignExecuteConfig(on_mutation=track_hook),
        )

        plan = "total = 0.0\nfor i in range(5):\n    total = add(total, 1.0)\nresult = total"
        calc.execute(plan)
        assert len(calls) == 0

    def test_rejection_stops_loop_execution(self):
        call_count = 0

        def reject_after_two(ctx: MutationHookContext) -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                return "Too many mutations"
            return None

        mock_llm = MockLLM()
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=DesignExecuteConfig(on_mutation=reject_after_two),
        )

        plan = "for i in range(10):\n    memory_store(value=float(i))"
        result = calc.execute(plan)
        assert not result.trace.all_succeeded
        # 2 successful + 1 rejected
        assert len(result.trace.successful_steps) == 2
        assert len(result.trace.failed_steps) == 1
        assert "Mutation rejected" in result.trace.failed_steps[0].error

    def test_rejection_prevents_mutation(self):
        def reject_all(ctx: MutationHookContext) -> str | None:
            return "Rejected"

        mock_llm = MockLLM()
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=DesignExecuteConfig(on_mutation=reject_all),
        )

        plan = "memory_store(value=999.0)"
        calc.execute(plan)
        assert calc._memory == 0.0  # Memory unchanged


# =============================================================================
# Multi-Turn Tests
# =============================================================================


class TestMultiTurnWithControlFlow:
    """Test multi-turn mode with control flow plans."""

    def test_variables_persist_across_turns(self):
        mock_llm = MockLLM(
            responses=[
                "result = add(2.0, 3.0)",
                "total = 0.0\nfor i in range(3):\n    total = add(total, result)\nresult = total",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(multi_turn=True),
        )

        r1 = calc.run("Add 2 and 3")
        assert r1.success is True
        assert r1.result == 5.0

        r2 = calc.run("Sum result three times")
        assert r2.success is True
        assert r2.result == 15.0  # 5 + 5 + 5

    def test_loop_result_persists(self):
        mock_llm = MockLLM(
            responses=[
                "items = []\nfor i in range(3):\n    items.append(add(float(i), 1.0))\nresult = items",
                "total = 0.0\nfor x in items:\n    total = add(total, x)\nresult = total",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(multi_turn=True),
        )

        r1 = calc.run("Create items")
        assert r1.success is True
        assert r1.result == [1.0, 2.0, 3.0]

        r2 = calc.run("Sum all items")
        assert r2.success is True
        assert r2.result == 6.0


# =============================================================================
# Prompt Tests
# =============================================================================


class TestPrompt:
    """Test that the prompt allows control flow."""

    def test_prompt_mentions_loops_and_conditionals(self):
        mock_llm = MockLLM(responses=["result = add(1.0, 2.0)"])
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("Do something")
        prompt = mock_llm.prompts[0]

        assert "for loops" in prompt
        assert "while loops" in prompt
        assert "if/elif/else" in prompt

    def test_prompt_does_not_forbid_loops(self):
        mock_llm = MockLLM(responses=["result = add(1.0, 2.0)"])
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("Do something")
        prompt = mock_llm.prompts[0]

        assert "Do NOT use imports, loops, conditionals" not in prompt

    def test_prompt_mentions_result_variable(self):
        mock_llm = MockLLM(responses=["result = add(1.0, 2.0)"])
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("Do something")
        prompt = mock_llm.prompts[0]

        assert "result" in prompt

    def test_prompt_mentions_loop_limit(self):
        mock_llm = MockLLM(responses=["result = add(1.0, 2.0)"])
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_loop_iterations=50),
        )

        calc.run("Do something")
        prompt = mock_llm.prompts[0]

        assert "50" in prompt


# =============================================================================
# Integration Tests (full run() cycle)
# =============================================================================


class TestRunIntegration:
    """Test complete run() cycle with control flow."""

    def test_run_with_for_loop(self):
        mock_llm = MockLLM(
            responses=[
                "total = 0.0\nfor i in range(5):\n    total = add(total, 1.0)\nresult = total"
            ]
        )
        calc = SimpleCalculator(llm=mock_llm)

        result = calc.run("Count to 5")
        assert result.success is True
        assert result.result == 5.0

    def test_run_with_while_loop(self):
        mock_llm = MockLLM(
            responses=[
                "x = 1.0\nwhile x < 100:\n    x = multiply(x, 2.0)\nresult = x"
            ]
        )
        calc = SimpleCalculator(llm=mock_llm)

        result = calc.run("Double until >= 100")
        assert result.success is True
        assert result.result == 128.0

    def test_run_with_conditional(self):
        mock_llm = MockLLM(
            responses=[
                "x = add(10.0, 20.0)\nif x > 25:\n    result = multiply(x, 2.0)\nelse:\n    result = x"
            ]
        )
        calc = SimpleCalculator(llm=mock_llm)

        result = calc.run("Double if sum > 25")
        assert result.success is True
        assert result.result == 60.0

    def test_run_plan_retry_with_control_flow(self):
        mock_llm = MockLLM(
            responses=[
                "import os",  # Invalid
                "total = 0.0\nfor i in range(3):\n    total = add(total, 1.0)\nresult = total",  # Valid
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_plan_retries=1),
        )

        result = calc.run("Count to 3")
        assert result.success is True
        assert result.result == 3.0
        assert mock_llm.call_count == 2

    def test_run_loop_limit_reports_failure(self):
        mock_llm = MockLLM(
            responses=["x = 0.0\nwhile True:\n    x = add(x, 1.0)"]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=DesignExecuteConfig(max_loop_iterations=5),
        )

        result = calc.run("Infinite loop")
        assert result.success is False
        assert result.error is not None


# =============================================================================
# Stepwise Execution Tests
# =============================================================================


class TestStepwiseNotSupported:
    """Test that stepwise execution raises NotImplementedError."""

    def test_execute_stepwise_raises(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(NotImplementedError, match="stepwise"):
            list(calc.execute_stepwise("test task"))


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_loop(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "for i in range(0):\n    x = add(float(i), 1.0)\nresult = 0.0"
        result = calc.execute(plan)
        assert result.trace.step_count == 0  # No primitive calls
        assert result.get_value() == 0.0

    def test_flat_plan_still_works(self):
        """DesignExecute should still handle flat plans like PlanExecute."""
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = add(1.0, 2.0)\ny = multiply(x, 3.0)\nresult = y"
        result = calc.execute(plan)
        assert result.get_value() == 9.0
        assert result.trace.step_count == 2

    def test_plan_with_no_primitive_calls(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        plan = "x = 5\ny = x + 10\nresult = y"
        result = calc.execute(plan)
        assert result.trace.step_count == 0  # No primitive calls traced
        assert result.get_value() == 15

    def test_syntax_error_in_plan(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        with pytest.raises(ValueError, match="syntax"):
            calc.execute("for i in ")
