"""Unit tests for PlanExecute multi-turn functionality."""

from typing import Any

from opensymbolicai.blueprints.plan_execute import PlanExecute
from opensymbolicai.core import primitive
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import MutationHookContext, PlanExecuteConfig


class MockLLM(LLM):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        # Create a dummy config for the base class
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


class SimpleCalculator(PlanExecute):
    """Simple calculator for testing."""

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=True)
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


class CalculatorWithMemory(PlanExecute):
    """Calculator with mutable memory for testing mutation hooks."""

    def __init__(self, llm: LLM | LLMConfig, config: PlanExecuteConfig | None = None):
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

    @primitive(read_only=False)
    def memory_recall(self) -> float:
        """Get memory value."""
        return self._memory


class TestMultiTurnConfig:
    """Test multi_turn configuration option."""

    def test_multi_turn_defaults_to_false(self):
        config = PlanExecuteConfig()
        assert config.multi_turn is False

    def test_multi_turn_can_be_enabled(self):
        config = PlanExecuteConfig(multi_turn=True)
        assert config.multi_turn is True


class TestSingleTurnMode:
    """Test default single-turn behavior."""

    def test_variables_do_not_persist_between_runs(self):
        mock_llm = MockLLM(
            responses=[
                "result = add(2, 3)",
                "final = multiply(result, 10)",  # 'result' won't exist
            ]
        )
        calc = SimpleCalculator(llm=mock_llm)

        r1 = calc.run("Add 2 and 3")
        assert r1.success is True
        assert r1.result == 5

        r2 = calc.run("Multiply result by 10")
        assert r2.success is False  # 'result' is not defined
        assert "result" in r2.error.lower() or "name" in r2.error.lower()

    def test_history_is_empty_in_single_turn_mode(self):
        mock_llm = MockLLM(responses=["result = add(1, 2)"])
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("Add 1 and 2")
        assert len(calc.history) == 0

    def test_persisted_variables_empty_in_single_turn_mode(self):
        mock_llm = MockLLM(responses=["result = add(1, 2)"])
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("Add 1 and 2")
        assert calc.persisted_variables == {}


class TestMultiTurnMode:
    """Test multi-turn conversation mode."""

    def test_variables_persist_between_runs(self):
        mock_llm = MockLLM(
            responses=[
                "result = add(2, 3)",
                "final = multiply(result, 10)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        r1 = calc.run("Add 2 and 3")
        assert r1.success is True
        assert r1.result == 5

        r2 = calc.run("Multiply result by 10")
        assert r2.success is True
        assert r2.result == 50

    def test_history_tracks_successful_turns(self):
        mock_llm = MockLLM(
            responses=[
                "x = add(1, 2)",
                "y = multiply(x, 3)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        calc.run("Add 1 and 2")
        calc.run("Multiply by 3")

        assert len(calc.history) == 2

        assert calc.history[0].task == "Add 1 and 2"
        assert calc.history[0].plan == "x = add(1, 2)"
        assert calc.history[0].result == 3
        assert calc.history[0].success is True

        assert calc.history[1].task == "Multiply by 3"
        assert calc.history[1].plan == "y = multiply(x, 3)"
        assert calc.history[1].result == 9
        assert calc.history[1].success is True

    def test_history_tracks_failed_turns(self):
        mock_llm = MockLLM(
            responses=[
                "result = undefined_function(1, 2)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        r = calc.run("Do something invalid")
        assert r.success is False

        assert len(calc.history) == 1
        assert calc.history[0].success is False
        assert calc.history[0].error is not None

    def test_persisted_variables_are_accessible(self):
        mock_llm = MockLLM(
            responses=[
                "x = add(5, 5)",
                "y = multiply(x, 2)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        calc.run("First task")
        assert "x" in calc.persisted_variables
        assert calc.persisted_variables["x"] == 10

        calc.run("Second task")
        assert "x" in calc.persisted_variables
        assert "y" in calc.persisted_variables
        assert calc.persisted_variables["y"] == 20

    def test_persisted_variables_returns_copy(self):
        """Ensure modifications to returned dict don't affect internal state."""
        mock_llm = MockLLM(responses=["x = add(1, 1)"])
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        calc.run("Task")
        vars_copy = calc.persisted_variables
        vars_copy["x"] = 999

        assert calc.persisted_variables["x"] == 2  # Original unchanged


class TestMultiTurnPrompt:
    """Test that history is included in prompts for multi-turn mode."""

    def test_prompt_includes_history_section(self):
        mock_llm = MockLLM(
            responses=[
                "result = add(1, 2)",
                "final = multiply(result, 3)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        calc.run("First task")
        calc.run("Second task")

        # Check second prompt includes history
        second_prompt = mock_llm.prompts[1]
        assert "Conversation History" in second_prompt
        assert "Turn 1" in second_prompt
        assert "First task" in second_prompt
        assert "result = add(1, 2)" in second_prompt

    def test_prompt_does_not_include_history_in_single_turn_mode(self):
        mock_llm = MockLLM(
            responses=[
                "result = add(1, 2)",
                "final = add(3, 4)",
            ]
        )
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("First task")
        calc.run("Second task")

        second_prompt = mock_llm.prompts[1]
        assert "Conversation History" not in second_prompt

    def test_prompt_shows_error_for_failed_turns(self):
        mock_llm = MockLLM(
            responses=[
                "result = undefined_func()",
                "x = add(1, 2)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        calc.run("Failing task")
        calc.run("Next task")

        second_prompt = mock_llm.prompts[1]
        assert "Error:" in second_prompt


class TestMultiTurnChainedOperations:
    """Test complex multi-turn scenarios."""

    def test_three_turn_conversation(self):
        mock_llm = MockLLM(
            responses=[
                "a = add(10, 5)",
                "b = multiply(a, 2)",
                "c = add(a, b)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        r1 = calc.run("Start with 10 + 5")
        assert r1.result == 15

        r2 = calc.run("Double it")
        assert r2.result == 30

        r3 = calc.run("Add original to doubled")
        assert r3.result == 45  # 15 + 30

        assert len(calc.history) == 3

    def test_can_reuse_variables_from_any_previous_turn(self):
        mock_llm = MockLLM(
            responses=[
                "first = add(1, 1)",
                "second = add(2, 2)",
                "third = add(3, 3)",
                "total = add(add(first, second), third)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        calc.run("Turn 1")
        calc.run("Turn 2")
        calc.run("Turn 3")
        r4 = calc.run("Sum all")

        assert r4.success is True
        assert r4.result == 12  # 2 + 4 + 6


class TestMultiTurnWithExecuteDirectly:
    """Test that execute() also respects multi-turn settings."""

    def test_execute_uses_persisted_namespace(self):
        mock_llm = MockLLM(responses=["x = add(5, 5)"])
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(multi_turn=True),
        )

        # First run to establish variable
        calc.run("Create x")

        # Direct execute should have access to x
        result = calc.execute("y = multiply(x, 3)")
        assert result.trace.all_succeeded
        assert calc.persisted_variables["y"] == 30

    def test_execute_without_multi_turn_has_fresh_namespace(self):
        mock_llm = MockLLM(responses=["x = add(5, 5)"])
        calc = SimpleCalculator(llm=mock_llm)

        calc.run("Create x")

        # Direct execute should NOT have access to x
        result = calc.execute("y = multiply(x, 3)")
        assert not result.trace.all_succeeded


class TestMutationHookConfig:
    """Test mutation hook configuration."""

    def test_on_mutation_defaults_to_none(self):
        config = PlanExecuteConfig()
        assert config.on_mutation is None

    def test_on_mutation_can_be_set(self):
        def my_hook(ctx: MutationHookContext) -> str | None:
            return None

        config = PlanExecuteConfig(on_mutation=my_hook)
        assert config.on_mutation is my_hook


class TestMutationHookCalled:
    """Test that mutation hooks are called correctly."""

    def test_hook_called_for_non_read_only_primitives(self):
        calls: list[MutationHookContext] = []

        def track_hook(ctx: MutationHookContext) -> str | None:
            calls.append(ctx)
            return None

        mock_llm = MockLLM(responses=["stored = memory_store(value=42.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=track_hook),
        )

        result = calc.run("Store 42 in memory")
        assert result.success is True
        assert len(calls) == 1
        assert calls[0].method_name == "memory_store"
        assert calls[0].args == {"value": 42.0}

    def test_hook_not_called_for_read_only_primitives(self):
        calls: list[MutationHookContext] = []

        def track_hook(ctx: MutationHookContext) -> str | None:
            calls.append(ctx)
            return None

        mock_llm = MockLLM(responses=["result = add(a=1.0, b=2.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=track_hook),
        )

        result = calc.run("Add 1 and 2")
        assert result.success is True
        assert result.result == 3.0
        assert len(calls) == 0  # Hook should not be called for read-only

    def test_hook_called_multiple_times_for_multiple_mutations(self):
        calls: list[MutationHookContext] = []

        def track_hook(ctx: MutationHookContext) -> str | None:
            calls.append(ctx)
            return None

        mock_llm = MockLLM(
            responses=["a = memory_store(value=10.0)\nb = memory_add(value=5.0)"]
        )
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=track_hook),
        )

        result = calc.run("Store 10 then add 5")
        assert result.success is True
        assert len(calls) == 2
        assert calls[0].method_name == "memory_store"
        assert calls[1].method_name == "memory_add"


class TestMutationHookRejection:
    """Test that mutation hooks can reject mutations."""

    def test_hook_can_reject_mutation(self):
        def reject_all(ctx: MutationHookContext) -> str | None:
            return "All mutations are rejected"

        mock_llm = MockLLM(responses=["stored = memory_store(value=42.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=reject_all),
        )

        result = calc.run("Store 42 in memory")
        assert result.success is False
        assert "Mutation rejected" in result.error
        assert "All mutations are rejected" in result.error

    def test_rejection_stops_execution(self):
        calls: list[str] = []

        def reject_memory_add(ctx: MutationHookContext) -> str | None:
            calls.append(ctx.method_name)
            if ctx.method_name == "memory_add":
                return "Cannot add to memory"
            return None

        mock_llm = MockLLM(
            responses=[
                "a = memory_store(value=10.0)\nb = memory_add(value=5.0)\nc = memory_recall()"
            ]
        )
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=reject_memory_add),
        )

        result = calc.run("Store, add, recall")
        assert result.success is False
        # First mutation (memory_store) should succeed, second (memory_add) rejected
        assert len(calls) == 2
        assert calls[0] == "memory_store"
        assert calls[1] == "memory_add"
        # memory_recall should not have been called due to early termination
        assert "memory_recall" not in calls

    def test_rejection_prevents_mutation_from_executing(self):
        def reject_all(ctx: MutationHookContext) -> str | None:
            return "Rejected"

        mock_llm = MockLLM(responses=["stored = memory_store(value=999.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=reject_all),
        )

        # Memory should still be 0 since mutation was rejected
        initial_memory = calc._memory
        calc.run("Store 999")
        assert calc._memory == initial_memory  # Memory unchanged

    def test_hook_returning_none_allows_mutation(self):
        def allow_all(ctx: MutationHookContext) -> str | None:
            return None  # Explicitly return None to allow

        mock_llm = MockLLM(responses=["stored = memory_store(value=42.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=allow_all),
        )

        result = calc.run("Store 42")
        assert result.success is True
        assert calc._memory == 42.0

    def test_conditional_rejection_based_on_args(self):
        def reject_negative(ctx: MutationHookContext) -> str | None:
            value = ctx.args.get("value", 0)
            if isinstance(value, (int, float)) and value < 0:
                return f"Cannot store negative value: {value}"
            return None

        mock_llm = MockLLM()
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=reject_negative),
        )

        # Positive value should work
        result1 = calc.execute("a = memory_store(value=10.0)")
        assert result1.trace.all_succeeded
        assert calc._memory == 10.0

        # Negative value should be rejected
        result2 = calc.execute("b = memory_store(value=-5.0)")
        assert not result2.trace.all_succeeded
        assert "Cannot store negative value" in result2.trace.steps[-1].error


class TestMutationHookContext:
    """Test the MutationHookContext data."""

    def test_context_contains_method_name(self):
        captured_ctx: MutationHookContext | None = None

        def capture_hook(ctx: MutationHookContext) -> str | None:
            nonlocal captured_ctx
            captured_ctx = ctx
            return None

        mock_llm = MockLLM(responses=["x = memory_store(value=5.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=capture_hook),
        )

        calc.run("Store 5")
        assert captured_ctx is not None
        assert captured_ctx.method_name == "memory_store"

    def test_context_contains_args(self):
        captured_ctx: MutationHookContext | None = None

        def capture_hook(ctx: MutationHookContext) -> str | None:
            nonlocal captured_ctx
            captured_ctx = ctx
            return None

        mock_llm = MockLLM(responses=["x = memory_add(value=7.5)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=capture_hook),
        )

        calc.run("Add 7.5 to memory")
        assert captured_ctx is not None
        assert captured_ctx.args == {"value": 7.5}

    def test_context_result_is_none_before_execution(self):
        """Hook is called before execution, so result should be None."""
        captured_ctx: MutationHookContext | None = None

        def capture_hook(ctx: MutationHookContext) -> str | None:
            nonlocal captured_ctx
            captured_ctx = ctx
            return None

        mock_llm = MockLLM(responses=["x = memory_store(value=100.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=capture_hook),
        )

        calc.run("Store 100")
        assert captured_ctx is not None
        assert captured_ctx.result is None  # Called before execution

    def test_context_step_is_none_before_execution(self):
        """Hook is called before execution, so step should be None."""
        captured_ctx: MutationHookContext | None = None

        def capture_hook(ctx: MutationHookContext) -> str | None:
            nonlocal captured_ctx
            captured_ctx = ctx
            return None

        mock_llm = MockLLM(responses=["x = memory_store(value=100.0)"])
        calc = CalculatorWithMemory(
            llm=mock_llm,
            config=PlanExecuteConfig(on_mutation=capture_hook),
        )

        calc.run("Store 100")
        assert captured_ctx is not None
        assert captured_ctx.step is None  # Called before execution


class TestExtractCodeBlock:
    """Test _extract_code_block handles various LLM response formats."""

    def test_extracts_code_from_python_code_block(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        response = "```python\nresult = add(1, 2)\n```"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"

    def test_extracts_code_from_generic_code_block(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        response = "```\nresult = add(1, 2)\n```"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"

    def test_strips_trailing_backticks_without_opening(self):
        """Fix for LLMs that add closing ``` without proper opening."""
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        # This is the problematic case from the benchmark
        response = "result = add(1, 2)\n```"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"
        assert "```" not in code

    def test_strips_trailing_backticks_with_extra_whitespace(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        response = "result = add(1, 2)  \n  ```"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"

    def test_handles_plain_code_without_backticks(self):
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        response = "result = add(1, 2)"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"

    def test_handles_annotated_assignment_with_trailing_backticks(self):
        """Reproduces the exact benchmark failure case."""
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        response = (
            "result1: float = self.add_numbers(first_number=5, second_number=3)\n```"
        )
        code = calc._extract_code_block(response)
        assert (
            code == "result1: float = self.add_numbers(first_number=5, second_number=3)"
        )
        assert "```" not in code

    def test_dedents_code_from_code_block(self):
        """Code blocks sometimes have extra indentation that should be removed."""
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        # Code inside a markdown block with extra indentation
        response = "```python\n    result = add(1, 2)\n```"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"

    def test_handles_code_in_python_block_with_trailing_backticks(self):
        """Properly extracts code even if there are extra backticks after the block."""
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)

        # LLM returns properly formatted block - should extract cleanly
        response = "```python\nresult = add(1, 2)\n```\n```"
        code = calc._extract_code_block(response)
        assert code == "result = add(1, 2)"


class TestPlanRetryConfig:
    """Test max_plan_retries configuration option."""

    def test_max_plan_retries_defaults_to_zero(self):
        config = PlanExecuteConfig()
        assert config.max_plan_retries == 0

    def test_max_plan_retries_can_be_set(self):
        config = PlanExecuteConfig(max_plan_retries=3)
        assert config.max_plan_retries == 3


class TestPlanRetryBehavior:
    """Test plan retry on validation failure."""

    def test_no_retry_by_default(self):
        """With default max_plan_retries=0, invalid plan fails immediately."""
        # Generate an invalid plan (uses a loop which is disallowed)
        mock_llm = MockLLM(
            responses=[
                "for i in range(5): x = add(i, 1)",  # Invalid - loop not allowed
            ]
        )
        calc = SimpleCalculator(llm=mock_llm)

        result = calc.run("Count to 5")
        assert result.success is False
        assert "For" in result.error or "not allowed" in result.error
        assert mock_llm.call_count == 1  # No retry

    def test_retry_on_validation_failure(self):
        """With max_plan_retries > 0, retries on validation failure."""
        mock_llm = MockLLM(
            responses=[
                "for i in range(5): x = add(i, 1)",  # Invalid - loop
                "result = add(1, 2)",  # Valid on retry
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(max_plan_retries=1),
        )

        result = calc.run("Add numbers")
        assert result.success is True
        assert result.result == 3
        assert mock_llm.call_count == 2  # Initial + 1 retry

    def test_feedback_included_in_retry_prompt(self):
        """Error feedback is passed to the LLM on retry."""
        mock_llm = MockLLM(
            responses=[
                "import os",  # Invalid - import not allowed
                "result = add(5, 5)",  # Valid
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(max_plan_retries=1),
        )

        calc.run("Add numbers")

        # Check that second prompt contains feedback about the error
        assert len(mock_llm.prompts) == 2
        second_prompt = mock_llm.prompts[1]
        assert "Previous Attempt Failed" in second_prompt
        assert "Import" in second_prompt or "not allowed" in second_prompt

    def test_max_retries_exhausted(self):
        """Fails after exhausting all retries."""
        mock_llm = MockLLM(
            responses=[
                "if True: x = 1",  # Invalid
                "while True: pass",  # Invalid
                "for i in []: pass",  # Invalid
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(max_plan_retries=2),
        )

        result = calc.run("Do something")
        assert result.success is False
        assert mock_llm.call_count == 3  # Initial + 2 retries

    def test_execution_errors_not_retried(self):
        """Errors during execution (not validation) don't trigger retry."""
        # This plan passes validation but fails at runtime due to undefined variable
        mock_llm = MockLLM(
            responses=[
                "result = add(undefined_var, 5)",  # Valid syntax, passes validation, fails at runtime
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(max_plan_retries=2),
        )

        result = calc.run("Add numbers")
        assert result.success is False
        assert "undefined_var" in result.error or "name" in result.error.lower()
        assert mock_llm.call_count == 1  # No retry for runtime errors

    def test_successful_first_attempt_no_retry_needed(self):
        """Valid plan on first attempt succeeds without retry."""
        mock_llm = MockLLM(
            responses=[
                "result = add(10, 20)",
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(max_plan_retries=3),
        )

        result = calc.run("Add 10 and 20")
        assert result.success is True
        assert result.result == 30
        assert mock_llm.call_count == 1  # No retry needed

    def test_retry_with_different_validation_errors(self):
        """Each retry can fail with different validation errors."""
        mock_llm = MockLLM(
            responses=[
                "exec('x = 1')",  # Invalid - dangerous builtin
                "x = open('file.txt')",  # Invalid - dangerous builtin
                "result = multiply(2, 3)",  # Valid
            ]
        )
        calc = SimpleCalculator(
            llm=mock_llm,
            config=PlanExecuteConfig(max_plan_retries=2),
        )

        result = calc.run("Calculate")
        assert result.success is True
        assert result.result == 6
        assert mock_llm.call_count == 3

        # Verify each retry prompt contained feedback from previous error
        assert "Previous Attempt Failed" in mock_llm.prompts[1]
        assert "exec" in mock_llm.prompts[1]
        assert "Previous Attempt Failed" in mock_llm.prompts[2]
        assert "open" in mock_llm.prompts[2]


# =========================================================================
# Type Definitions Tests
# =========================================================================


class TestTypeDefinitions:
    """Tests for _format_type_definitions() and its inclusion in prompts."""

    def test_no_type_definitions_for_primitive_types(self):
        """No Type Definitions section when primitives use only primitive types."""
        mock_llm = MockLLM()
        calc = SimpleCalculator(llm=mock_llm)
        prompt = calc.build_plan_prompt("Do something")
        assert "## Type Definitions" not in prompt

    def test_type_definitions_for_basemodel_param(self):
        """Type Definitions section appears when a primitive param is a BaseModel."""
        from pydantic import BaseModel

        class Flight(BaseModel):
            flight_number: str
            price: float
            origin: str

        class ModelAgent(PlanExecute):
            @primitive(read_only=True)
            def book_flight(self, flight: Flight) -> str:
                """Book a flight."""
                return flight.flight_number

        agent = ModelAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Book a flight")
        assert "## Type Definitions" in prompt
        assert "Flight(flight_number: str, price: float, origin: str)" in prompt

    def test_type_definitions_for_basemodel_return(self):
        """Type Definitions section for BaseModel in return type."""
        from pydantic import BaseModel

        class Restaurant(BaseModel):
            name: str
            rating: float

        class FoodAgent(PlanExecute):
            @primitive(read_only=True)
            def find_restaurant(self, city: str) -> Restaurant:
                """Find a restaurant."""
                return Restaurant(name="Test", rating=4.5)

        agent = FoodAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Find food")
        assert "## Type Definitions" in prompt
        assert "Restaurant(name: str, rating: float)" in prompt

    def test_type_definitions_unwraps_list(self):
        """Extracts BaseModel from list[Model] annotations."""
        from pydantic import BaseModel

        class Accommodation(BaseModel):
            name: str
            price: float

        class TravelAgent(PlanExecute):
            @primitive(read_only=True)
            def search_hotels(self, city: str) -> list[Accommodation]:
                """Search hotels."""
                return []

        agent = TravelAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Search")
        assert "## Type Definitions" in prompt
        assert "Accommodation(name: str, price: float)" in prompt

    def test_type_definitions_unwraps_optional(self):
        """Extracts BaseModel from Optional[Model] / Model | None annotations."""
        from pydantic import BaseModel

        class Hotel(BaseModel):
            name: str
            stars: int

        class HotelAgent(PlanExecute):
            @primitive(read_only=True)
            def find_hotel(self, city: str) -> Hotel | None:
                """Find a hotel."""
                return None

        agent = HotelAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Find hotel")
        assert "## Type Definitions" in prompt
        assert "Hotel(name: str, stars: int)" in prompt

    def test_type_definitions_deduplicates(self):
        """Models referenced by multiple primitives appear only once."""
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            cost: float

        class ShopAgent(PlanExecute):
            @primitive(read_only=True)
            def buy_item(self, item: Item) -> str:
                """Buy an item."""
                return item.name

            @primitive(read_only=True)
            def sell_item(self, item: Item) -> float:
                """Sell an item."""
                return item.cost

        agent = ShopAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Trade")
        assert prompt.count("Item(name: str, cost: float)") == 1

    def test_type_definitions_sorted_alphabetically(self):
        """Models are sorted alphabetically by class name."""
        from pydantic import BaseModel

        class Zebra(BaseModel):
            stripes: int

        class Alpha(BaseModel):
            value: str

        class Middle(BaseModel):
            count: int

        class ZooAgent(PlanExecute):
            @primitive(read_only=True)
            def get_zebra(self) -> Zebra:
                return Zebra(stripes=10)

            @primitive(read_only=True)
            def get_alpha(self) -> Alpha:
                return Alpha(value="a")

            @primitive(read_only=True)
            def get_middle(self) -> Middle:
                return Middle(count=5)

        agent = ZooAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Zoo")
        # Find positions of each model in the prompt
        alpha_pos = prompt.index("Alpha(")
        middle_pos = prompt.index("Middle(")
        zebra_pos = prompt.index("Zebra(")
        assert alpha_pos < middle_pos < zebra_pos

    def test_type_definitions_multiple_models_across_primitives(self):
        """Collects models from params and returns across all primitives."""
        from pydantic import BaseModel

        class Flight(BaseModel):
            number: str
            price: float

        class Hotel(BaseModel):
            name: str
            stars: int

        class TripAgent(PlanExecute):
            @primitive(read_only=True)
            def search_flights(self, origin: str) -> list[Flight]:
                return []

            @primitive(read_only=True)
            def book_hotel(self, hotel: Hotel) -> str:
                return hotel.name

        agent = TripAgent(llm=MockLLM())
        prompt = agent.build_plan_prompt("Plan trip")
        assert "Flight(number: str, price: float)" in prompt
        assert "Hotel(name: str, stars: int)" in prompt
