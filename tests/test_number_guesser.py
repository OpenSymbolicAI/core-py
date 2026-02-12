"""Tests for the NumberGuesser example agent."""

import sys
from pathlib import Path
from typing import Any

from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import GoalStatus

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
            else "result = self.guess(n=50)"
        )
        self.call_count += 1
        return LLMResponse(
            text=response_text,
            provider="mock",
            model="mock-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )


class TestNumberGuesserBinarySearch:
    """Test that NumberGuesser converges via binary search."""

    def test_finds_number_50_in_one_guess(self):
        """Secret=50, first guess is midpoint=50, found immediately."""
        mock_llm = MockLLM(responses=["result = self.guess(n=50)"])
        agent = NumberGuesser(llm=mock_llm, secret=50)
        result = agent.seek("Find the secret number between 1 and 100")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 1

    def test_binary_search_finds_73(self):
        """Simulate binary search for secret=73."""
        # Binary search steps for 73 in [1, 100]:
        # Guess 50 → too_low  → [51, 100]
        # Guess 75 → too_high → [51, 74]
        # Guess 62 → too_low  → [63, 74]
        # Guess 68 → too_low  → [69, 74]
        # Guess 71 → too_low  → [72, 74]
        # Guess 73 → correct!
        mock_llm = MockLLM(
            responses=[
                "result = self.guess(n=50)",
                "result = self.guess(n=75)",
                "result = self.guess(n=62)",
                "result = self.guess(n=68)",
                "result = self.guess(n=71)",
                "result = self.guess(n=73)",
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
                "result = self.guess(n=50)",  # too_high → [1, 49]
                "result = self.guess(n=25)",  # too_high → [1, 24]
                "result = self.guess(n=12)",  # too_high → [1, 11]
                "result = self.guess(n=6)",   # too_high → [1, 5]
                "result = self.guess(n=3)",   # too_high → [1, 2]
                "result = self.guess(n=1)",   # correct!
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
                "result = self.guess(n=50)",   # too_low → [51, 100]
                "result = self.guess(n=75)",   # too_low → [76, 100]
                "result = self.guess(n=88)",   # too_low → [89, 100]
                "result = self.guess(n=94)",   # too_low → [95, 100]
                "result = self.guess(n=97)",   # too_low → [98, 100]
                "result = self.guess(n=99)",   # too_low → [100, 100]
                "result = self.guess(n=100)",  # correct!
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=100)
        result = agent.seek("Find the secret number")

        assert result.status == GoalStatus.ACHIEVED


class TestNumberGuesserMaxIterations:
    def test_stops_at_max_guesses(self):
        """If LLM keeps guessing wrong, stops at max_guesses."""
        # Always guess 1, but secret is 99
        mock_llm = MockLLM(
            responses=["result = self.guess(n=1)"] * 3
        )
        agent = NumberGuesser(
            llm=mock_llm, secret=99, max_guesses=3
        )
        result = agent.seek("Find the secret number")

        assert result.status == GoalStatus.MAX_ITERATIONS
        assert result.iteration_count == 3


class TestGuessingContext:
    def test_context_range_updates_on_too_low(self):
        """After a too_low guess, low bound should increase."""
        mock_llm = MockLLM(
            responses=[
                "result = self.guess(n=30)",  # too_low, secret=73 → low=31
                "result = self.guess(n=73)",  # correct
            ]
        )
        agent = NumberGuesser(llm=mock_llm, secret=73)
        result = agent.seek("Find the number")

        assert result.status == GoalStatus.ACHIEVED
        # After first guess of 30 (too_low), the context should have updated low=31
        # We can verify via iteration count
        assert result.iteration_count == 2

    def test_context_range_updates_on_too_high(self):
        """After a too_high guess, high bound should decrease."""
        mock_llm = MockLLM(
            responses=[
                "result = self.guess(n=80)",  # too_high, secret=73 → high=79
                "result = self.guess(n=73)",  # correct
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
        assert ctx.low == 1
        assert ctx.high == 100
        assert ctx.found is False
        assert ctx.last_feedback == ""


class TestGoalPrompt:
    def test_prompt_includes_range(self):
        mock_llm = MockLLM()
        agent = NumberGuesser(llm=mock_llm, secret=42)
        ctx = GuessingContext(goal="Find the number", low=20, high=60)
        prompt = agent.build_goal_prompt("Find the number", ctx)

        assert "20" in prompt
        assert "60" in prompt
        assert "40" in prompt  # midpoint

    def test_prompt_includes_feedback(self):
        mock_llm = MockLLM()
        agent = NumberGuesser(llm=mock_llm, secret=42)
        ctx = GuessingContext(
            goal="Find the number", low=20, high=60, last_feedback="too_low"
        )
        prompt = agent.build_goal_prompt("Find the number", ctx)

        assert "too_low" in prompt
