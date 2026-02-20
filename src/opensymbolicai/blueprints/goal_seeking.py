"""GoalSeeking blueprint: iterative goal-directed agent with plan-execute-evaluate loops."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from opensymbolicai.blueprints.design_execute import DesignExecute
from opensymbolicai.core import MethodType
from opensymbolicai.llm import LLM, LLMConfig
from opensymbolicai.models import (
    ExecutionResult,
    ExecutionStep,
    ExecutionTrace,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
    GoalSeekingResult,
    GoalStatus,
    Iteration,
    LLMInteraction,
    PlanAttempt,
    PlanGeneration,
    PlanResult,
)
from opensymbolicai.models import TokenUsage as ModelTokenUsage


class GoalSeeking(DesignExecute):
    """Agent that iteratively pursues a goal through plan-execute-evaluate cycles.

    GoalSeeking extends DesignExecute with an iterative loop that:
    1. Plans the next step toward a goal
    2. Executes the plan
    3. Introspects results into structured context (the introspection boundary)
    4. Evaluates progress toward the goal
    5. Repeats until the goal is achieved or termination conditions are met

    Evaluation uses a two-tier approach:
    - Tier 1 (static): An @evaluator-decorated method on the agent
    - Tier 2 (dynamic): LLM-generated evaluator code from the goal

    Subclasses should:
    1. Define @primitive methods for available operations
    2. Optionally define an @evaluator method for static evaluation
    3. Override update_context() to extract insights from execution results
    4. Override create_context() to return a custom GoalContext subclass
    """

    def __init__(
        self,
        llm: LLM | LLMConfig,
        name: str = "",
        description: str = "",
        config: GoalSeekingConfig | None = None,
    ) -> None:
        """Initialize the GoalSeeking agent.

        Args:
            llm: LLM instance or config for plan generation.
            name: Agent name for prompts.
            description: Agent description for prompts.
            config: GoalSeeking-specific configuration.
        """
        cfg = config or GoalSeekingConfig()
        super().__init__(llm=llm, name=name, description=description, config=cfg)
        self.goal_config = cfg

    # -------------------------------------------------------------------------
    # Evaluator Introspection
    # -------------------------------------------------------------------------

    def _get_evaluator_method(
        self,
    ) -> Callable[[str, GoalContext], GoalEvaluation] | None:
        """Find the @evaluator-decorated method on this agent.

        Returns:
            The bound evaluator method, or None if no @evaluator is defined.
        """
        for name in dir(self):
            if name.startswith("__"):
                continue
            method = getattr(self, name, None)
            if (
                callable(method)
                and hasattr(method, "__method_type__")
                and method.__method_type__ == MethodType.EVALUATOR
            ):
                return method  # type: ignore[no-any-return]
        return None

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def build_goal_prompt(
        self, goal: str, context: GoalContext, feedback: str | None = None
    ) -> str:
        """Build the prompt for planning the next iteration.

        Includes the original goal, available primitives, decomposition examples,
        and structured insights from context (NOT raw execution results).

        Args:
            goal: The goal being pursued.
            context: Accumulated context with structured insights.
            feedback: Error feedback from a failed plan attempt (for retry).

        Returns:
            Complete prompt for plan generation.
        """
        primitives = self._get_primitive_methods()
        decompositions = self._get_decomposition_methods()

        primitive_docs = [
            self._format_primitive_signature(name, method)
            for name, method in primitives
        ]

        examples = []
        for _name, method, intent, expanded in decompositions:
            source = self._get_decomposition_source(method)
            if source:
                example = f"Intent: {intent}"
                if expanded:
                    example += f"\nApproach: {expanded}"
                example += f"\nPython:\n{source}"
                examples.append(example)

        # Build type definitions section for Pydantic models
        type_defs_section = self._format_type_definitions(primitives)

        # Build context summary from structured insights (not raw results)
        context_section = ""
        if context.iteration_count > 0:
            context_section = f"""
## Previous Iterations

{context.iteration_count} iteration(s) completed so far.
"""
            # Include any custom fields from context subclasses
            custom_fields = _get_custom_context_fields(context)
            if custom_fields:
                context_section += "\n## Accumulated Knowledge\n\n"
                for field_name, field_value in custom_fields.items():
                    context_section += f"- {field_name}: {field_value!r}\n"

        feedback_section = ""
        if feedback:
            feedback_section = f"""
## Previous Attempt Failed

Your previous plan was invalid. Please fix the following error and regenerate:

{feedback}

"""

        prompt = f"""You are {self.name}, an AI agent that generates Python code plans to achieve a goal.

{self.description}

## Goal

{goal}

## Available Primitive Methods

You can ONLY call these methods:

```python
{chr(10).join(primitive_docs)}
```
{type_defs_section}
## Example Decompositions

{chr(10).join(f"### Example {i + 1}{chr(10)}{ex}" for i, ex in enumerate(examples)) if examples else "No examples available."}
{context_section}{feedback_section}## Task

Generate Python code for the NEXT step toward achieving the goal: {goal}

## Rules

1. You can use: assignments, for/while loops, if/elif/else, try/except
2. You can ONLY call the primitive methods listed above
3. Do NOT use imports, function definitions, class definitions, or with statements
4. Do NOT use any dangerous operations (exec, eval, open, etc.)
5. Call primitives directly (e.g. `search(query="...")`), do NOT use `self.`
6. Use if/else to handle missing data or check results before proceeding
7. Use try/except around search() or read_page() to handle errors gracefully
8. The last assigned variable will be the final result

## Output

```python
"""
        return prompt

    def build_evaluator_prompt(self, goal: str) -> str:
        """Build the prompt for evaluator code generation.

        Args:
            goal: The goal to generate evaluator code for.

        Returns:
            Prompt for the LLM to generate evaluator code.
        """
        primitives = self._get_primitive_methods()
        primitive_docs = [
            self._format_primitive_signature(name, method)
            for name, method in primitives
        ]

        return f"""You are generating Python evaluation code for a goal-seeking agent.

## Goal

{goal}

## Available Primitives

```python
{chr(10).join(primitive_docs)}
```

## Task

Generate Python code that evaluates whether the goal has been achieved.

The code has access to these variables:
- `goal` (str): The goal being pursued
- `context` (GoalContext): Accumulated context with structured insights
- `self`: The agent instance (can call primitives)
- `GoalEvaluation`: The evaluation result class

The code MUST assign `result = GoalEvaluation(goal_achieved=...)`.

## Rules

1. Output ONLY Python assignment statements
2. Check context fields (insights), NOT raw execution results
3. The final statement must be: `result = GoalEvaluation(goal_achieved=...)`
4. Do NOT use imports, loops, conditionals, or function definitions

## Output

```python
"""

    # -------------------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------------------

    def plan_iteration(
        self, goal: str, context: GoalContext, feedback: str | None = None
    ) -> PlanResult:
        """Generate a plan for the next iteration.

        Uses build_goal_prompt() directly (not self.plan()) to avoid
        double-wrapping the prompt through build_plan_prompt().

        Args:
            goal: The goal being pursued.
            context: Accumulated context from previous iterations.
            feedback: Error feedback from a failed plan attempt (for retry).

        Returns:
            PlanResult with the generated plan.
        """
        prompt = self.build_goal_prompt(goal, context, feedback=feedback)
        start_time = time.perf_counter()
        response = self._llm.generate(prompt)
        elapsed = time.perf_counter() - start_time

        raw_response = response.text
        extracted_code = self._extract_code_block(raw_response)
        plan_text = self.on_code_extracted(raw_response, extracted_code)

        llm_interaction = LLMInteraction(
            prompt=prompt,
            response=raw_response,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            time_seconds=elapsed,
            provider=response.provider,
            model=response.model,
        )
        plan_generation = PlanGeneration(
            llm_interaction=llm_interaction,
            extracted_code=extracted_code,
        )

        return PlanResult(
            plan=plan_text,
            usage=ModelTokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            time_seconds=elapsed,
            provider=response.provider,
            model=response.model,
            plan_generation=plan_generation,
        )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def plan_evaluator(self, goal: str) -> str:
        """Generate evaluator code from the goal using the LLM.

        Called once before the loop starts. The generated code must assign
        `result = GoalEvaluation(goal_achieved=...)`.

        Args:
            goal: The goal to generate evaluator code for.

        Returns:
            Python code string for evaluation.
        """
        prompt = self.build_evaluator_prompt(goal)
        response = self._llm.generate(prompt)
        code = self._extract_code_block(response.text)
        return code

    def run_evaluator(
        self,
        evaluator_code: str,
        goal: str,
        context: GoalContext,
    ) -> GoalEvaluation:
        """Execute generated evaluator code in a sandboxed namespace.

        Args:
            evaluator_code: The Python evaluator code to execute.
            goal: The goal being pursued.
            context: Accumulated context.

        Returns:
            GoalEvaluation from the `result` variable in the executed code.
        """
        namespace: dict[str, Any] = {
            "self": self,
            "goal": goal,
            "context": context,
            "GoalEvaluation": GoalEvaluation,
        }
        # Add primitives to namespace
        for name, method in self._get_primitive_methods():
            namespace[name] = method
        namespace.update(self.allowed_builtins)

        exec(  # noqa: S102
            compile(evaluator_code, "<evaluator>", "exec"),
            {"__builtins__": {}},
            namespace,
        )

        result = namespace.get("result")
        if not isinstance(result, GoalEvaluation):
            return GoalEvaluation(goal_achieved=False)
        return result

    # -------------------------------------------------------------------------
    # Termination Logic
    # -------------------------------------------------------------------------

    def should_continue(
        self,
        context: GoalContext,
        evaluation: GoalEvaluation,
    ) -> tuple[bool, GoalStatus]:
        """Determine if the loop should continue.

        Args:
            context: The current goal context.
            evaluation: The latest evaluation result.

        Returns:
            Tuple of (should_continue, status_if_stopping).
        """
        if evaluation.goal_achieved:
            return False, GoalStatus.ACHIEVED

        if context.iteration_count >= self.goal_config.max_iterations:
            return False, GoalStatus.MAX_ITERATIONS

        return True, GoalStatus.PURSUING

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def on_iteration_start(self, iteration_number: int, context: GoalContext) -> None:
        """Hook called at the start of each iteration.

        Override for logging, metrics, or custom setup.
        """

    def on_iteration_complete(
        self, iteration: Iteration, context: GoalContext
    ) -> None:
        """Hook called after each iteration completes.

        Override for logging, metrics, or triggering side effects.
        """

    def on_goal_achieved(self, result: GoalSeekingResult) -> None:
        """Hook called when goal is achieved.

        Override for notifications, cleanup, or celebration.
        """

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        """THE INTROSPECTION BOUNDARY. Called after each execution.

        This is where raw ExecutionResult is introspected into structured
        insights on the context. The planner and evaluator only see what
        this method writes into context — never the raw result.

        Override to extract domain-specific insights from the execution result.
        """

    def create_context(self, goal: str) -> GoalContext:
        """Factory hook to create the initial context.

        Override to return a custom context subclass.
        """
        return GoalContext(goal=goal)

    # -------------------------------------------------------------------------
    # Answer Extraction
    # -------------------------------------------------------------------------

    def _extract_final_answer(self, context: GoalContext) -> Any:
        """Extract the final answer from context.

        Default: returns last execution result value.
        Override for custom answer extraction logic.
        """
        if context.iterations:
            last_step = context.iterations[-1].execution_result.trace.last_step
            if last_step is not None:
                return last_step.result_value
        return None

    # -------------------------------------------------------------------------
    # Main Orchestration
    # -------------------------------------------------------------------------

    def seek(self, goal: str) -> GoalSeekingResult:
        """Pursue a goal through iterative plan-execute-evaluate cycles.

        Args:
            goal: The goal to achieve (natural language).

        Returns:
            GoalSeekingResult with final answer and iteration history.
        """
        context = self.create_context(goal)

        # Resolve evaluator: static (@evaluator) or dynamic (LLM-generated)
        static_evaluator = self._get_evaluator_method()
        evaluator_code: str | None = None
        if not static_evaluator:
            evaluator_code = self.plan_evaluator(goal)

        max_plan_attempts = 1 + self.goal_config.max_plan_retries

        while True:
            iteration_number = context.iteration_count + 1

            # Hook: iteration start
            self.on_iteration_start(iteration_number, context)

            # 1. Plan + Execute with retry on validation errors
            plan_result: PlanResult | None = None
            exec_result: ExecutionResult | None = None
            plan_attempts: list[PlanAttempt] = []
            feedback: str | None = None

            for attempt in range(max_plan_attempts):
                plan_result = self.plan_iteration(goal, context, feedback=feedback)

                plan_attempt = PlanAttempt(
                    attempt_number=attempt + 1,
                    plan_generation=plan_result.plan_generation
                    or PlanGeneration(
                        llm_interaction=LLMInteraction(
                            prompt="",
                            response="",
                            input_tokens=plan_result.usage.input_tokens,
                            output_tokens=plan_result.usage.output_tokens,
                            time_seconds=plan_result.time_seconds,
                            provider=plan_result.provider,
                            model=plan_result.model,
                        ),
                        extracted_code=plan_result.plan,
                    ),
                    feedback=feedback,
                )

                try:
                    exec_result = self.execute(plan_result.plan)
                    plan_attempt.success = True
                    plan_attempts.append(plan_attempt)
                    break
                except ValueError as e:
                    # Validation error — retry with feedback if attempts remain
                    error_msg = str(e)
                    plan_attempt.validation_error = error_msg
                    plan_attempt.success = False
                    plan_attempts.append(plan_attempt)

                    if attempt < max_plan_attempts - 1:
                        feedback = error_msg
                        continue

                    # All retries exhausted — create a synthetic failed result
                    exec_result = ExecutionResult(
                        value_type="error",
                        trace=ExecutionTrace(
                            steps=[
                                ExecutionStep(
                                    step_number=1,
                                    statement="<plan validation failed>",
                                    success=False,
                                    error=f"Plan validation failed after "
                                    f"{max_plan_attempts} attempts: {error_msg}",
                                )
                            ]
                        ),
                    )

            assert plan_result is not None
            assert exec_result is not None

            # 2. Introspect: derive structured insights from raw result
            self.update_context(context, exec_result)

            # 3. Evaluate progress (checks context insights, not raw result)
            if static_evaluator:
                evaluation = static_evaluator(goal, context)
            else:
                assert evaluator_code is not None
                evaluation = self.run_evaluator(evaluator_code, goal, context)

            # 4. Record iteration (raw result preserved for traceability)
            iteration = Iteration(
                iteration_number=iteration_number,
                plan_result=plan_result,
                execution_result=exec_result,
                evaluation=evaluation,
                plan_attempts=plan_attempts,
            )
            context.iterations.append(iteration)

            # Hook: iteration complete
            self.on_iteration_complete(iteration, context)

            # 5. Check termination
            should_cont, status = self.should_continue(context, evaluation)

            if not should_cont:
                result = GoalSeekingResult(
                    goal=goal,
                    status=status,
                    final_answer=self._extract_final_answer(context),
                    iterations=context.iterations,
                )

                if status == GoalStatus.ACHIEVED:
                    self.on_goal_achieved(result)

                return result


def _get_custom_context_fields(context: GoalContext) -> dict[str, Any]:
    """Extract custom fields from a GoalContext subclass (non-base fields)."""
    base_fields = set(GoalContext.model_fields.keys())
    custom: dict[str, Any] = {}
    for field_name in type(context).model_fields:
        if field_name not in base_fields:
            custom[field_name] = getattr(context, field_name)
    return custom
