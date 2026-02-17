"""Tests for the NumberGuesser example agent."""

import sys
from pathlib import Path
from typing import Any

from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import (
    ExecutionResult,
    ExecutionTrace,
    GoalEvaluation,
    GoalStatus,
    Iteration,
    PlanResult,
)

# Add examples directory to path so we can import the agent
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "number_guesser"))

from number_guesser import GuessingContext, NumberGuesser


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
            else "result = self.search(low=1, high=99)"
        )
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            provider="mock",
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


def _dummy_iteration() -> Iteration:
    """Create a minimal Iteration for tests that need iteration_count > 0."""
    return Iteration(
        iteration_number=1,
        plan_result=PlanResult(plan="result = self.search(low=1, high=99)"),
        execution_result=ExecutionResult(value_type="str"),
        evaluation=GoalEvaluation(goal_achieved=False),
    )


class TestNumberGuesserBinarySearch:
    """Test that NumberGuesser converges via binary search."""

    def test_finds_number_50_in_one_guess(self):
        """Secret=50, search(1, 99) → midpoint=50, found immediately."""
        mock_llm = MockLLM(responses=["result = self.search(low=1, high=99)"])
        agent = NumberGuesser(llm=mock_llm, secret=50)
        result = agent.seek("Find the secret number between 1 and 100")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 1

    def test_binary_search_finds_73(self):
        """Simulate binary search for secret=73."""
        # Binary search steps for 73:
        # search(1, 100)  → mid=50 → too_low
        # search(51, 100) → mid=75 → too_high
        # search(51, 74)  → mid=62 → too_low
        # search(63, 74)  → mid=68 → too_low
        # search(69, 74)  → mid=71 → too_low
        # search(72, 74)  → mid=73 → correct!
        mock_llm = MockLLM(
            responses=[
                "result = self.search(low=1, high=100)",
                "result = self.search(low=51, high=100)",
                "result = self.search(low=51, high=74)",
                "result = self.search(low=63, high=74)",
                "result = self.search(low=69, high=74)",
                "result = self.search(low=72, high=74)",
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=73)
        result = agent.seek("Find the secret number between 1 and 100")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 6
        assert result.final_answer == "correct"

    def test_finds_number_1_at_boundary(self):
        """Secret=1, edge case at lower bound."""
        mock_llm = MockLLM(
            responses=[
                "result = self.search(low=1, high=100)",  # mid=50 → too_high
                "result = self.search(low=1, high=49)",   # mid=25 → too_high
                "result = self.search(low=1, high=24)",   # mid=12 → too_high
                "result = self.search(low=1, high=11)",   # mid=6  → too_high
                "result = self.search(low=1, high=5)",    # mid=3  → too_high
                "result = self.search(low=1, high=2)",    # mid=1  → correct!
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=1)
        result = agent.seek("Find the secret number")

        assert result.status == GoalStatus.ACHIEVED
        assert result.final_answer == "correct"

    def test_finds_number_100_at_boundary(self):
        """Secret=100, edge case at upper bound."""
        mock_llm = MockLLM(
            responses=[
                "result = self.search(low=1, high=100)",    # mid=50  → too_low
                "result = self.search(low=51, high=100)",   # mid=75  → too_low
                "result = self.search(low=76, high=100)",   # mid=88  → too_low
                "result = self.search(low=89, high=100)",   # mid=94  → too_low
                "result = self.search(low=95, high=100)",   # mid=97  → too_low
                "result = self.search(low=98, high=100)",   # mid=99  → too_low
                "result = self.search(low=100, high=100)",  # mid=100 → correct!
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=100)
        result = agent.seek("Find the secret number")

        assert result.status == GoalStatus.ACHIEVED


class TestNumberGuesserMaxIterations:
    def test_stops_at_max_guesses(self):
        """If LLM keeps guessing wrong, stops at max_guesses."""
        # Always search a range whose midpoint is 2, but secret is 99
        mock_llm = MockLLM(
            responses=["result = self.search(low=1, high=3)"] * 3
        )
        agent = NumberGuesser(
            llm=mock_llm, secret=99, max_guesses=3
        )
        result = agent.seek("Find the secret number")

        assert result.status == GoalStatus.MAX_ITERATIONS
        assert result.iteration_count == 3


class TestGuessingContext:
    def test_context_range_updates_on_too_low(self):
        """After a too_low search, the context records the range used."""
        mock_llm = MockLLM(
            responses=[
                "result = self.search(low=1, high=59)",   # mid=30 → too_low
                "result = self.search(low=72, high=74)",  # mid=73 → correct
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=73)
        result = agent.seek("Find the number")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 2

    def test_context_range_updates_on_too_high(self):
        """After a too_high search, the context records the range used."""
        mock_llm = MockLLM(
            responses=[
                "result = self.search(low=79, high=81)",  # mid=80 → too_high
                "result = self.search(low=72, high=74)",  # mid=73 → correct
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=73)
        result = agent.seek("Find the number")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 2

    def test_initial_context(self):
        mock_llm = MockLLM()
        agent = NumberGuesser(llm=mock_llm, secret=42)
        ctx = agent.create_context("test")
        assert isinstance(ctx, GuessingContext)
        assert ctx.low == 1_000_000
        assert ctx.high == 2_000_000
        assert ctx.found is False
        assert ctx.last_feedback == ""


class TestGoalPrompt:
    def test_prompt_includes_range(self):
        mock_llm = MockLLM()
        agent = NumberGuesser(llm=mock_llm, secret=42)
        ctx = GuessingContext(
            goal="Find the number",
            low=20,
            high=60,
            iterations=[_dummy_iteration()],
        )
        prompt = agent.build_goal_prompt("Find the number", ctx)

        assert "20" in prompt
        assert "60" in prompt

    def test_prompt_includes_feedback(self):
        mock_llm = MockLLM()
        agent = NumberGuesser(llm=mock_llm, secret=42)
        ctx = GuessingContext(
            goal="Find the number",
            low=20,
            high=60,
            last_feedback="too_low",
            iterations=[_dummy_iteration()],
        )
        prompt = agent.build_goal_prompt("Find the number", ctx)

        assert "too_low" in prompt
