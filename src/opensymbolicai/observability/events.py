"""Trace event models for observability."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of trace events emitted during agent execution."""

    # Lifecycle
    RUN_START = "run.start"
    RUN_COMPLETE = "run.complete"
    RUN_ERROR = "run.error"

    # Planning
    PLAN_START = "plan.start"
    PLAN_LLM_REQUEST = "plan.llm_request"
    PLAN_LLM_RESPONSE = "plan.llm_response"
    PLAN_VALIDATION_ERROR = "plan.validation_error"
    PLAN_COMPLETE = "plan.complete"

    # Execution
    EXECUTION_START = "execution.start"
    EXECUTION_STEP = "execution.step"
    EXECUTION_COMPLETE = "execution.complete"
    MUTATION_BLOCKED = "execution.mutation_blocked"

    # Goal seeking
    GOAL_SEEK_START = "goal.seek_start"
    GOAL_ITERATION_START = "goal.iteration_start"
    GOAL_EVALUATION = "goal.evaluation"
    GOAL_ITERATION_COMPLETE = "goal.iteration_complete"
    GOAL_SEEK_COMPLETE = "goal.seek_complete"


class ExecutionSummary(BaseModel):
    """Payload for ``EXECUTION_COMPLETE`` trace events."""

    value_type: str
    value_name: str
    step_count: int
    all_succeeded: bool
    total_time_seconds: float


class RunCompleteSummary(BaseModel):
    """Payload for ``RUN_COMPLETE`` trace events."""

    success: bool = True
    result_type: str
    steps_executed: int


class RunErrorSummary(BaseModel):
    """Payload for ``RUN_ERROR`` trace events."""

    success: bool = False
    error: str


class GoalSeekStartPayload(BaseModel):
    """Payload for ``GOAL_SEEK_START`` trace events."""

    goal: str
    max_iterations: int


class GoalIterationSummary(BaseModel):
    """Payload for ``GOAL_ITERATION_COMPLETE`` trace events."""

    iteration: int
    goal_achieved: bool


class GoalSeekSummary(BaseModel):
    """Payload for ``GOAL_SEEK_COMPLETE`` trace events."""

    status: str
    iteration_count: int


class TraceEvent(BaseModel):
    """A single trace event emitted during agent execution.

    Events are grouped by trace_id (one per run/seek call) and form
    a tree via span_id / parent_span_id.
    """

    event_id: str = Field(..., description="Unique event identifier (UUID)")
    trace_id: str = Field(
        ..., description="Groups all events from one run()/seek() call"
    )
    span_id: str = Field(..., description="Identifies this logical unit")
    parent_span_id: str | None = Field(
        default=None, description="Parent span for nesting"
    )
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event occurred",
    )
    agent_class: str = Field(..., description="Agent class name")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific data"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="User-supplied metadata"
    )
