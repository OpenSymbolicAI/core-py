"""Tests for GoalSeeking blueprint."""

from typing import Any

from opensymbolicai.blueprints.goal_seeking import GoalSeeking
from opensymbolicai.core import evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
    GoalSeekingResult,
    GoalStatus,
    Iteration,
)

# =============================================================================
# Test Helpers
# =============================================================================


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


class ResearchContext(GoalContext):
    """Custom context for testing."""

    findings: list[str] = []


class CountingAgent(GoalSeeking):
    """Agent that counts up. Uses @evaluator for static evaluation."""

    def __init__(
        self,
        llm: LLM,
        target: int = 3,
        config: GoalSeekingConfig | None = None,
    ):
        super().__init__(
            llm=llm,
            name="CountingAgent",
            description="Counts to a target number",
            config=config,
        )
        self.target = target
        self.count = 0

    @primitive(read_only=True)
    def increment(self) -> int:
        """Increment the counter by 1."""
        self.count += 1
        return self.count

    @primitive(read_only=True)
    def get_count(self) -> int:
        """Get the current count."""
        return self.count

    @evaluator
    def check_count(self, goal: str, context: GoalContext) -> GoalEvaluation:
        return GoalEvaluation(goal_achieved=self.count >= self.target)


class ResearchAgent(GoalSeeking):
    """Agent with custom context and update_context. Uses @evaluator."""

    def __init__(self, llm: LLM, min_findings: int = 2):
        super().__init__(
            llm=llm,
            name="ResearchAgent",
            description="Collects findings",
            config=GoalSeekingConfig(max_iterations=5),
        )
        self.min_findings = min_findings

    @primitive(read_only=True)
    def search(self, query: str) -> list[str]:
        """Search for information."""
        return [f"fact about {query}"]

    def create_context(self, goal: str) -> ResearchContext:
        return ResearchContext(goal=goal)

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        assert isinstance(context, ResearchContext)
        last_step = execution_result.trace.last_step
        if last_step and isinstance(last_step.result_value, list):
            context.findings.extend(last_step.result_value)

    @evaluator
    def check_research(self, goal: str, context: GoalContext) -> GoalEvaluation:
        assert isinstance(context, ResearchContext)
        return GoalEvaluation(goal_achieved=len(context.findings) >= self.min_findings)


class DynamicAgent(GoalSeeking):
    """Agent without @evaluator — uses dynamic evaluator generation."""

    def __init__(self, llm: LLM, config: GoalSeekingConfig | None = None):
        super().__init__(
            llm=llm,
            name="DynamicAgent",
            description="Uses dynamic evaluator",
            config=config,
        )
        self.value = 0

    @primitive(read_only=True)
    def compute(self, x: int) -> int:
        """Compute and store a value."""
        self.value = x
        return x


# =============================================================================
# Model Tests
# =============================================================================


class TestGoalStatus:
    def test_enum_values(self):
        assert GoalStatus.PURSUING == "pursuing"
        assert GoalStatus.ACHIEVED == "achieved"
        assert GoalStatus.FAILED == "failed"
        assert GoalStatus.MAX_ITERATIONS == "max_iterations"


class TestGoalEvaluation:
    def test_achieved(self):
        ev = GoalEvaluation(goal_achieved=True)
        assert ev.goal_achieved is True

    def test_not_achieved(self):
        ev = GoalEvaluation(goal_achieved=False)
        assert ev.goal_achieved is False


class TestGoalContext:
    def test_empty_context(self):
        ctx = GoalContext(goal="test goal")
        assert ctx.goal == "test goal"
        assert ctx.iterations == []
        assert ctx.iteration_count == 0
        assert ctx.last_evaluation is None

    def test_with_iterations(self):
        from opensymbolicai.models import (
            ExecutionResult,
            ExecutionTrace,
            PlanResult,
        )

        iteration = Iteration(
            iteration_number=1,
            plan_result=PlanResult(plan="x = 1"),
            execution_result=ExecutionResult(
                value_type="int", trace=ExecutionTrace()
            ),
            evaluation=GoalEvaluation(goal_achieved=False),
        )
        ctx = GoalContext(goal="test", iterations=[iteration])
        assert ctx.iteration_count == 1
        assert ctx.last_evaluation is not None
        assert ctx.last_evaluation.goal_achieved is False


class TestGoalSeekingResult:
    def test_succeeded(self):
        result = GoalSeekingResult(
            goal="test", status=GoalStatus.ACHIEVED, final_answer=42
        )
        assert result.succeeded is True
        assert result.iteration_count == 0

    def test_not_succeeded(self):
        result = GoalSeekingResult(
            goal="test", status=GoalStatus.MAX_ITERATIONS
        )
        assert result.succeeded is False


# =============================================================================
# GoalSeeking Class Tests
# =============================================================================


class TestEvaluatorIntrospection:
    def test_finds_static_evaluator(self):
        llm = MockLLM()
        agent = CountingAgent(llm=llm)
        method = agent._get_evaluator_method()
        assert method is not None

    def test_no_evaluator(self):
        llm = MockLLM()
        agent = DynamicAgent(llm=llm)
        method = agent._get_evaluator_method()
        assert method is None


class TestStaticEvaluatorSeek:
    def test_seek_with_static_evaluator(self):
        """CountingAgent increments until count >= target, then evaluator returns achieved."""
        # Each iteration: LLM generates "result = self.increment()"
        # plan_iteration calls self.plan() which calls self._llm.generate()
        # The prompt is the goal prompt, but plan() wraps it further
        # Actually, plan_iteration calls self.plan(prompt) which adds its own prompt wrapper.
        # Let's just provide the raw code responses.
        mock_llm = MockLLM(
            responses=[
                # Iteration 1: plan
                "result = self.increment()",
                # Iteration 2: plan
                "result = self.increment()",
                # Iteration 3: plan (count reaches target=3)
                "result = self.increment()",
            ]
        )
        agent = CountingAgent(
            llm=mock_llm,
            target=3,
            config=GoalSeekingConfig(max_iterations=10),
        )
        result = agent.seek("Count to 3")

        assert result.status == GoalStatus.ACHIEVED
        assert result.succeeded is True
        assert result.iteration_count == 3
        assert result.goal == "Count to 3"
        assert agent.count == 3

    def test_seek_achieves_in_one_iteration(self):
        mock_llm = MockLLM(responses=["result = self.increment()"])
        agent = CountingAgent(
            llm=mock_llm,
            target=1,
            config=GoalSeekingConfig(max_iterations=10),
        )
        result = agent.seek("Count to 1")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 1
        assert agent.count == 1


class TestMaxIterations:
    def test_stops_at_max_iterations(self):
        """If evaluator never returns achieved, stops at max_iterations."""
        mock_llm = MockLLM(
            responses=[
                "result = self.get_count()",
                "result = self.get_count()",
                "result = self.get_count()",
            ]
        )
        agent = CountingAgent(
            llm=mock_llm,
            target=100,  # Will never reach this
            config=GoalSeekingConfig(max_iterations=3),
        )
        result = agent.seek("Count to 100")

        assert result.status == GoalStatus.MAX_ITERATIONS
        assert result.succeeded is False
        assert result.iteration_count == 3


class TestShouldContinue:
    def test_goal_achieved_stops(self):
        llm = MockLLM()
        agent = CountingAgent(llm=llm)
        ctx = GoalContext(goal="test")
        evaluation = GoalEvaluation(goal_achieved=True)

        should_cont, status = agent.should_continue(ctx, evaluation)
        assert should_cont is False
        assert status == GoalStatus.ACHIEVED

    def test_max_iterations_stops(self):
        llm = MockLLM()
        agent = CountingAgent(llm=llm, config=GoalSeekingConfig(max_iterations=2))
        ctx = GoalContext(goal="test")
        # Simulate 2 completed iterations
        from opensymbolicai.models import ExecutionResult, ExecutionTrace, PlanResult

        for i in range(2):
            ctx.iterations.append(
                Iteration(
                    iteration_number=i + 1,
                    plan_result=PlanResult(plan="x = 1"),
                    execution_result=ExecutionResult(
                        value_type="int", trace=ExecutionTrace()
                    ),
                    evaluation=GoalEvaluation(goal_achieved=False),
                )
            )
        evaluation = GoalEvaluation(goal_achieved=False)

        should_cont, status = agent.should_continue(ctx, evaluation)
        assert should_cont is False
        assert status == GoalStatus.MAX_ITERATIONS

    def test_pursuing_continues(self):
        llm = MockLLM()
        agent = CountingAgent(llm=llm, config=GoalSeekingConfig(max_iterations=10))
        ctx = GoalContext(goal="test")
        evaluation = GoalEvaluation(goal_achieved=False)

        should_cont, status = agent.should_continue(ctx, evaluation)
        assert should_cont is True
        assert status == GoalStatus.PURSUING


class TestCustomContext:
    def test_create_context_returns_subclass(self):
        llm = MockLLM()
        agent = ResearchAgent(llm=llm)
        ctx = agent.create_context("find info")
        assert isinstance(ctx, ResearchContext)
        assert ctx.goal == "find info"
        assert ctx.findings == []

    def test_update_context_populates_findings(self):
        mock_llm = MockLLM(
            responses=[
                'result = self.search(query="topic1")',
                'result = self.search(query="topic2")',
            ]
        )
        agent = ResearchAgent(llm=mock_llm, min_findings=2)
        result = agent.seek("Find 2 facts")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 2


class TestDynamicEvaluator:
    def test_dynamic_evaluator_generation(self):
        """Agent without @evaluator generates evaluator code from LLM."""
        mock_llm = MockLLM(
            responses=[
                # First call: plan_evaluator generates evaluator code
                # Note: evaluator runs BEFORE iteration is appended to context,
                # so iteration_count >= 1 becomes true during the 2nd iteration
                # (when 1 iteration is already recorded).
                'result = GoalEvaluation(goal_achieved=context.iteration_count >= 1)',
                # Iteration 1: plan
                "result = self.compute(x=1)",
                # Iteration 2: plan
                "result = self.compute(x=2)",
            ]
        )
        agent = DynamicAgent(
            llm=mock_llm,
            config=GoalSeekingConfig(max_iterations=5),
        )
        result = agent.seek("Compute twice")

        assert result.status == GoalStatus.ACHIEVED
        assert result.iteration_count == 2
        # Evaluator was generated once (first LLM call), then two plan calls
        assert mock_llm.call_count == 3

    def test_dynamic_evaluator_max_iterations(self):
        """Dynamic evaluator that never succeeds respects max_iterations."""
        mock_llm = MockLLM(
            responses=[
                # Evaluator code: always returns False
                "result = GoalEvaluation(goal_achieved=False)",
                # Iteration plans
                "result = self.compute(x=1)",
                "result = self.compute(x=2)",
            ]
        )
        agent = DynamicAgent(
            llm=mock_llm,
            config=GoalSeekingConfig(max_iterations=2),
        )
        result = agent.seek("Never finish")

        assert result.status == GoalStatus.MAX_ITERATIONS
        assert result.succeeded is False


class TestLifecycleHooks:
    def test_on_iteration_start_called(self):
        starts: list[int] = []

        class HookedAgent(CountingAgent):
            def on_iteration_start(
                self, iteration_number: int, context: GoalContext
            ) -> None:
                starts.append(iteration_number)

        mock_llm = MockLLM(
            responses=[
                "result = self.increment()",
                "result = self.increment()",
            ]
        )
        agent = HookedAgent(llm=mock_llm, target=2)
        agent.seek("Count to 2")

        assert starts == [1, 2]

    def test_on_iteration_complete_called(self):
        completions: list[int] = []

        class HookedAgent(CountingAgent):
            def on_iteration_complete(
                self, iteration: Iteration, context: GoalContext
            ) -> None:
                completions.append(iteration.iteration_number)

        mock_llm = MockLLM(
            responses=[
                "result = self.increment()",
                "result = self.increment()",
            ]
        )
        agent = HookedAgent(llm=mock_llm, target=2)
        agent.seek("Count to 2")

        assert completions == [1, 2]

    def test_on_goal_achieved_called(self):
        achieved_results: list[GoalSeekingResult] = []

        class HookedAgent(CountingAgent):
            def on_goal_achieved(self, result: GoalSeekingResult) -> None:
                achieved_results.append(result)

        mock_llm = MockLLM(responses=["result = self.increment()"])
        agent = HookedAgent(llm=mock_llm, target=1)
        agent.seek("Count to 1")

        assert len(achieved_results) == 1
        assert achieved_results[0].status == GoalStatus.ACHIEVED

    def test_on_goal_achieved_not_called_on_max_iterations(self):
        achieved_results: list[GoalSeekingResult] = []

        class HookedAgent(CountingAgent):
            def on_goal_achieved(self, result: GoalSeekingResult) -> None:
                achieved_results.append(result)

        mock_llm = MockLLM(
            responses=["result = self.get_count()", "result = self.get_count()"]
        )
        agent = HookedAgent(
            llm=mock_llm,
            target=100,
            config=GoalSeekingConfig(max_iterations=2),
        )
        agent.seek("Count to 100")

        assert len(achieved_results) == 0


class TestFinalAnswer:
    def test_extract_final_answer(self):
        mock_llm = MockLLM(responses=["result = self.increment()"])
        agent = CountingAgent(llm=mock_llm, target=1)
        result = agent.seek("Count to 1")

        assert result.final_answer == 1

    def test_extract_final_answer_none_on_empty(self):
        llm = MockLLM()
        agent = CountingAgent(llm=llm)
        ctx = GoalContext(goal="test")
        assert agent._extract_final_answer(ctx) is None


class TestGoalPrompt:
    def test_build_goal_prompt_includes_goal(self):
        llm = MockLLM()
        agent = CountingAgent(llm=llm)
        ctx = GoalContext(goal="Count to 5")
        prompt = agent.build_goal_prompt("Count to 5", ctx)

        assert "Count to 5" in prompt
        assert "increment" in prompt
        assert "get_count" in prompt

    def test_build_goal_prompt_includes_context_info(self):
        from opensymbolicai.models import ExecutionResult, ExecutionTrace, PlanResult

        llm = MockLLM()
        agent = CountingAgent(llm=llm)
        ctx = GoalContext(goal="test")
        ctx.iterations.append(
            Iteration(
                iteration_number=1,
                plan_result=PlanResult(plan="x = 1"),
                execution_result=ExecutionResult(
                    value_type="int", trace=ExecutionTrace()
                ),
                evaluation=GoalEvaluation(goal_achieved=False),
            )
        )
        prompt = agent.build_goal_prompt("test", ctx)
        assert "1 iteration(s) completed" in prompt


class TestGoalSeekingConfig:
    def test_default_max_iterations(self):
        config = GoalSeekingConfig()
        assert config.max_iterations == 10

    def test_custom_max_iterations(self):
        config = GoalSeekingConfig(max_iterations=5)
        assert config.max_iterations == 5
