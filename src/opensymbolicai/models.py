"""Pydantic models for plan-and-execute orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from opensymbolicai.observability.config import ObservabilityConfig

# ── Shared sentinel / constant strings ────────────────────────────────────────
NULL_JSON: str = "null"
"""Default JSON representation for absent/null values."""

NONE_TYPE_NAME: str = "NoneType"
"""String name used when a result value is None."""

PLAN_COMPILE_SOURCE: str = "<plan>"
"""Source name passed to compile() for plan execution."""

EVALUATOR_COMPILE_SOURCE: str = "<evaluator>"
"""Source name passed to compile() for evaluator execution."""

def empty_builtins() -> dict[str, Any]:
    """Return a fresh restricted builtins dict for exec()/eval() sandboxing.

    A new dict is returned each call to prevent cross-contamination between
    executions (CPython's exec() may mutate the globals dict it receives).
    """
    return {"__builtins__": {}}

MUTATION_REJECTED_PREFIX: str = "Mutation rejected"
"""Error message prefix when a mutation hook rejects an operation."""


class MutationDetection(BaseModel):
    """Result of checking whether an AST statement is a mutating primitive call."""

    is_mutation: bool = False
    method_name: str | None = None
    variable_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class ArgumentValue(BaseModel):
    """Captured argument value with both reference and resolved value."""

    expression: str = Field(
        ..., description="The source expression (e.g., 'x + 1' or '42')"
    )
    resolved_value: Any = Field(default=None, description="The actual runtime value")
    variable_reference: str | None = Field(
        default=None,
        description="The variable name if it was a simple variable reference",
    )

    model_config = {"arbitrary_types_allowed": True}


class LLMInteraction(BaseModel):
    """Generic record of an LLM call."""

    prompt: str = Field(..., description="The prompt sent to the LLM")
    response: str = Field(..., description="The text response from the LLM")
    input_tokens: int = Field(default=0, description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    time_seconds: float = Field(default=0.0, description="Time taken for LLM call")
    provider: str = Field(default="", description="LLM provider used")
    model: str = Field(default="", description="Model used")


class PlanGeneration(BaseModel):
    """Record of a plan generation LLM call with code extraction."""

    llm_interaction: LLMInteraction = Field(
        ..., description="The underlying LLM interaction"
    )
    extracted_code: str = Field(
        ..., description="The code extracted from the LLM response"
    )


class PlanAttempt(BaseModel):
    """A single plan generation attempt, including retries."""

    attempt_number: int = Field(..., description="1-based attempt number")
    plan_generation: PlanGeneration = Field(
        ..., description="The plan generation details for this attempt"
    )
    feedback: str | None = Field(
        default=None, description="Error feedback from previous attempt (if retry)"
    )
    validation_error: str | None = Field(
        default=None, description="Validation error if plan was invalid"
    )
    success: bool = Field(
        default=True, description="Whether this attempt produced a valid plan"
    )


class MutationHookContext(BaseModel):
    """Context passed to mutation hooks when a non-read-only primitive is called."""

    method_name: str = Field(..., description="Name of the primitive method called")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the method"
    )
    result: Any = Field(default=None, description="Result returned by the method")
    step: ExecutionStep | None = Field(
        default=None, description="The execution step containing full details"
    )

    model_config = {"arbitrary_types_allowed": True}


# Type alias for mutation hook callbacks
# Returns None to allow the mutation, or a string rejection reason to block it
MutationHook = Callable[[MutationHookContext], str | None]


class PlanExecuteConfig(BaseModel):
    """Extended configuration for PlanExecute agents."""

    allowed_builtins: dict[str, Any] | None = Field(
        default=None,
        description="Dict of builtins allowed in execution. None uses defaults.",
    )
    skip_result_serialization: bool = Field(
        default=False,
        description="Skip JSON serialization of results. Useful for large or non-serializable outputs.",
    )
    multi_turn: bool = Field(
        default=False,
        description="Enable multi-turn mode. Persists variables and tracks conversation history.",
    )
    on_mutation: MutationHook | None = Field(
        default=None,
        description="Hook called when a primitive with read_only=False is executed. "
        "Return None to allow, or a string rejection reason to block the mutation.",
    )
    max_plan_retries: int = Field(
        default=0,
        description="Maximum number of times to retry plan generation if validation fails. "
        "The LLM will receive the error message as feedback for regeneration.",
    )

    # Checkpoint/distributed execution settings
    require_mutation_approval: bool = Field(
        default=True,
        description="If True, execution pauses before mutations and yields a checkpoint "
        "for external approval. Use with execute_stepwise() for distributed execution.",
    )
    worker_id: str | None = Field(
        default=None,
        description="Identifier for this worker instance. Used in checkpoints to track "
        "which worker created/updated the checkpoint.",
    )
    observability: ObservabilityConfig | None = Field(
        default=None,
        description="Observability configuration. When set and enabled, trace events "
        "are emitted for planning, execution, and LLM interactions.",
    )

    model_config = {"arbitrary_types_allowed": True}


class DesignExecuteConfig(PlanExecuteConfig):
    """Configuration for DesignExecute agents with control flow support."""

    max_loop_iterations: int = Field(
        default=100,
        description="Maximum iterations per loop (for/while). Prevents infinite loops.",
    )
    max_total_primitive_calls: int = Field(
        default=1000,
        description="Maximum total primitive calls across the entire plan execution.",
    )
    allow_break_continue: bool = Field(
        default=True,
        description="Whether to allow break/continue in loops.",
    )


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    task: str = Field(..., description="The user's task/query")
    plan: str = Field(..., description="The generated Python plan")
    result: Any = Field(default=None, description="The final result value")
    success: bool = Field(default=True, description="Whether the turn succeeded")
    error: str | None = Field(default=None, description="Error message if failed")


class TokenUsage(BaseModel):
    """Token usage statistics from LLM generation."""

    input_tokens: int = Field(default=0, description="Number of input/prompt tokens")
    output_tokens: int = Field(
        default=0, description="Number of output/completion tokens"
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class PlanResult(BaseModel):
    """Result from the planning phase."""

    plan: str = Field(..., description="Generated Python statements")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage")
    time_seconds: float = Field(default=0.0, description="Time taken to generate plan")
    provider: str = Field(default="", description="LLM provider used")
    model: str = Field(default="", description="Model used")
    plan_generation: PlanGeneration | None = Field(
        default=None, description="Full LLM interaction details for plan generation"
    )


class PrimitiveCall(BaseModel):
    """Information about a primitive method call in a plan."""

    method_name: str = Field(..., description="Name of the primitive method")
    read_only: bool = Field(
        default=False, description="Whether the method is read-only"
    )
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed")


class PlanAnalysis(BaseModel):
    """Analysis of a plan's structure."""

    calls: list[PrimitiveCall] = Field(
        default_factory=list, description="All primitive calls in the plan"
    )

    @property
    def has_mutations(self) -> bool:
        """Check if any calls are not read-only."""
        return any(not call.read_only for call in self.calls)

    @property
    def method_names(self) -> list[str]:
        """Get list of all method names called."""
        return [call.method_name for call in self.calls]


class ExecutionStep(BaseModel):
    """A single step in plan execution."""

    step_number: int = Field(..., description="1-based step number")
    statement: str = Field(..., description="The Python statement executed")
    variable_name: str = Field(default="", description="Variable being assigned")
    primitive_called: str | None = Field(
        default=None, description="Name of primitive method called"
    )
    args: dict[str, ArgumentValue] = Field(
        default_factory=dict,
        description="Arguments passed to the primitive with both expression and resolved value",
    )
    namespace_before: dict[str, Any] = Field(
        default_factory=dict, description="Namespace snapshot before execution"
    )
    namespace_after: dict[str, Any] = Field(
        default_factory=dict, description="Namespace snapshot after execution"
    )
    result_type: str = Field(default="", description="Type of the result")
    result_value: Any = Field(default=None, description="The computed value")
    result_json: str = Field(default=NULL_JSON, description="JSON-serialized result")
    time_seconds: float = Field(default=0.0, description="Time taken for this step")
    success: bool = Field(default=True, description="Whether the step succeeded")
    error: str | None = Field(default=None, description="Error message if failed")

    model_config = {"arbitrary_types_allowed": True}


class ExecutionTrace(BaseModel):
    """Complete trace of plan execution."""

    steps: list[ExecutionStep] = Field(
        default_factory=list, description="All executed steps"
    )
    total_time_seconds: float = Field(default=0.0, description="Total execution time")

    @property
    def step_count(self) -> int:
        """Number of steps executed."""
        return len(self.steps)

    @property
    def successful_steps(self) -> list[ExecutionStep]:
        """Get all successful steps."""
        return [s for s in self.steps if s.success]

    @property
    def failed_steps(self) -> list[ExecutionStep]:
        """Get all failed steps."""
        return [s for s in self.steps if not s.success]

    @property
    def all_succeeded(self) -> bool:
        """Check if all steps succeeded."""
        return all(s.success for s in self.steps)

    @property
    def last_step(self) -> ExecutionStep | None:
        """Get the last executed step."""
        return self.steps[-1] if self.steps else None

    @property
    def primitives_called(self) -> list[str]:
        """Get list of all primitives called."""
        return [s.primitive_called for s in self.steps if s.primitive_called]


class ExecutionResult(BaseModel):
    """Result from executing a plan."""

    value_type: str = Field(..., description="Python type name of the result")
    value_name: str = Field(default="", description="Variable name of the result")
    value_json: str = Field(default=NULL_JSON, description="JSON-serialized value")
    trace: ExecutionTrace = Field(
        default_factory=ExecutionTrace, description="Step-by-step execution trace"
    )

    def get_value(self) -> Any:
        """Deserialize and return the actual value."""
        return json.loads(self.value_json)


class ExecutionMetrics(BaseModel):
    """Metrics from a complete orchestration run."""

    plan_tokens: TokenUsage = Field(
        default_factory=TokenUsage, description="Tokens used for planning"
    )
    plan_time_seconds: float = Field(default=0.0, description="Time for planning")
    execute_time_seconds: float = Field(default=0.0, description="Time for execution")
    steps_executed: int = Field(default=0, description="Number of steps executed")
    provider: str = Field(default="", description="LLM provider used")
    model: str = Field(default="", description="Model used")

    @property
    def total_time_seconds(self) -> float:
        """Total time for planning and execution."""
        return self.plan_time_seconds + self.execute_time_seconds


class OrchestrationResult(BaseModel):
    """Result from a complete plan-and-execute run."""

    success: bool = Field(..., description="Whether the orchestration succeeded")
    result: Any = Field(default=None, description="The computed result")
    error: str | None = Field(default=None, description="Error message if failed")
    metrics: ExecutionMetrics | None = Field(
        default=None, description="Execution metrics"
    )
    plan: str | None = Field(default=None, description="The generated plan")
    trace: ExecutionTrace | None = Field(
        default=None, description="Step-by-step execution trace"
    )
    plan_attempts: list[PlanAttempt] = Field(
        default_factory=list,
        description="All plan generation attempts including retries",
    )
    task: str = Field(default="", description="The original task description")


# =============================================================================
# Goal-Seeking Models
# =============================================================================


class GoalStatus(str, Enum):
    """Status of goal pursuit."""

    PURSUING = "pursuing"
    ACHIEVED = "achieved"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


class GoalEvaluation(BaseModel):
    """Result of evaluating progress toward a goal.

    Subclass to add domain-specific fields (findings, confidence, etc.)
    """

    goal_achieved: bool
    """Whether the goal has been achieved."""


class Iteration(BaseModel):
    """A single iteration of the goal-seeking loop."""

    iteration_number: int = Field(..., description="1-based iteration number")

    plan_result: PlanResult = Field(
        ..., description="The plan generated for this iteration"
    )

    execution_result: ExecutionResult = Field(
        ..., description="Result of executing the plan"
    )

    evaluation: GoalEvaluation = Field(
        ..., description="Evaluation of progress toward goal"
    )

    plan_attempts: list[PlanAttempt] = Field(
        default_factory=list,
        description="Plan generation attempts including retries (populated when max_plan_retries > 0)",
    )


class GoalContext(BaseModel):
    """Accumulated structured knowledge across iterations.

    This is the introspection boundary. Subclass to add domain-specific
    insight fields that update_context() populates from raw execution results.
    The planner and evaluator only see these fields — never raw results.
    """

    goal: str = Field(..., description="The original goal")

    iterations: list[Iteration] = Field(
        default_factory=list, description="All completed iterations"
    )

    @property
    def iteration_count(self) -> int:
        """Number of completed iterations."""
        return len(self.iterations)

    @property
    def last_evaluation(self) -> GoalEvaluation | None:
        """Get the evaluation from the last iteration."""
        if self.iterations:
            return self.iterations[-1].evaluation
        return None


class GoalSeekingConfig(DesignExecuteConfig):
    """Configuration for GoalSeeking agents."""

    max_iterations: int = Field(
        default=10, description="Maximum iterations before stopping"
    )
    max_plan_retries: int = Field(
        default=2,
        description="Maximum number of times to retry plan generation if validation fails. "
        "When a plan fails validation, the error is fed back to the LLM for regeneration.",
    )


class GoalSeekingResult(BaseModel):
    """Result from a complete goal-seeking run.

    Subclass to add domain-specific result fields.
    """

    goal: str = Field(..., description="The original goal")

    status: GoalStatus = Field(..., description="Final status")

    final_answer: Any = Field(default=None, description="The final result/answer")

    iterations: list[Iteration] = Field(
        default_factory=list, description="All iterations performed"
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def iteration_count(self) -> int:
        """Number of iterations performed."""
        return len(self.iterations)

    @property
    def succeeded(self) -> bool:
        """Whether the goal was achieved."""
        return self.status == GoalStatus.ACHIEVED
