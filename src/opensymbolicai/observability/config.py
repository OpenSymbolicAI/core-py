"""Configuration for the observability layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from opensymbolicai.observability.transports.protocol import TraceTransport


class ObservabilityConfig(BaseModel):
    """Controls what gets captured and where it goes.

    When ``enabled`` is False (the default), no tracing overhead is incurred.
    """

    enabled: bool = Field(default=False, description="Master switch for observability")

    # Granular capture filters
    capture_llm_prompts: bool = Field(
        default=True, description="Capture LLM prompts (can be large)"
    )
    capture_llm_responses: bool = Field(
        default=True, description="Capture LLM responses"
    )
    capture_namespace_snapshots: bool = Field(
        default=False, description="Capture namespace before/after each step (large)"
    )
    capture_execution_steps: bool = Field(
        default=True, description="Capture individual execution steps"
    )
    capture_plan_source: bool = Field(
        default=True, description="Capture generated plan source code"
    )

    # Metadata attached to all events
    session_id: str | None = Field(
        default=None, description="Group multiple runs into a session"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom metadata attached to all events"
    )

    # Transport config
    collector_url: str | None = Field(
        default=None, description="URL for HttpTransport (e.g. http://localhost:8100/events)"
    )
    output_path: str | None = Field(
        default=None, description="File path for FileTransport (JSONL)"
    )

    # Custom transport takes precedence over collector_url / output_path
    transport: TraceTransport | None = Field(
        default=None,
        description="Custom TraceTransport instance. Takes precedence over collector_url/output_path.",
        exclude=True,
    )

    model_config = {"arbitrary_types_allowed": True}
