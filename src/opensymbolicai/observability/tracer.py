"""Internal tracer that emits events through a configured transport."""

from __future__ import annotations

import uuid
from typing import Any

from opensymbolicai.observability.config import ObservabilityConfig
from opensymbolicai.observability.events import EventType, TraceEvent
from opensymbolicai.observability.transports.file import FileTransport
from opensymbolicai.observability.transports.http import HttpTransport
from opensymbolicai.observability.transports.protocol import TraceTransport


def _create_transport(config: ObservabilityConfig) -> TraceTransport:
    """Resolve a transport from the config."""
    if config.transport is not None:
        return config.transport  # type: ignore[no-any-return]
    if config.collector_url is not None:
        return HttpTransport(url=config.collector_url)
    if config.output_path is not None:
        return FileTransport(path=config.output_path)
    from opensymbolicai.observability.transports.memory import InMemoryTransport

    return InMemoryTransport()


def _deep_remove(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Remove top-level keys from a dict, returning the same dict."""
    for key in keys:
        data.pop(key, None)
    return data


def _nested_dict(data: dict[str, Any], *path: str) -> dict[str, Any] | None:
    """Traverse nested dicts by key path. Returns None if any key is missing."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


class PayloadFilter:
    """Applies capture filters to event payloads based on ObservabilityConfig."""

    def __init__(self, config: ObservabilityConfig) -> None:
        self._config = config

    def plan_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Filter a PlanResult payload."""
        if not self._config.capture_plan_source:
            _deep_remove(data, "plan")
            generation = _nested_dict(data, "plan_generation")
            if generation:
                _deep_remove(generation, "extracted_code")

        interaction = _nested_dict(data, "plan_generation", "llm_interaction")
        if interaction:
            if not self._config.capture_llm_prompts:
                _deep_remove(interaction, "prompt")
            if not self._config.capture_llm_responses:
                _deep_remove(interaction, "response")

        return data

    def execution_step(self, data: dict[str, Any]) -> dict[str, Any]:
        """Filter an ExecutionStep payload."""
        if not self._config.capture_namespace_snapshots:
            _deep_remove(data, "namespace_before", "namespace_after")
        return data

    def llm_interaction(self, data: dict[str, Any]) -> dict[str, Any]:
        """Filter an LLMInteraction payload."""
        if not self._config.capture_llm_prompts:
            _deep_remove(data, "prompt")
        if not self._config.capture_llm_responses:
            _deep_remove(data, "response")
        return data


class Tracer:
    """Internal event emitter used by blueprint classes.

    Not part of the public API — users configure observability via
    ``ObservabilityConfig`` and never touch the ``Tracer`` directly.

    The tracer manages trace/span IDs, applies capture filters, and
    routes events to the configured transport.
    """

    def __init__(self, config: ObservabilityConfig, agent_class: str) -> None:
        self._config = config
        self._agent_class = agent_class
        self._transport = _create_transport(config)
        self._trace_id: str = ""
        self._span_stack: list[str] = []
        self.filter = PayloadFilter(config)

    @property
    def config(self) -> ObservabilityConfig:
        """The observability configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Trace / span management
    # ------------------------------------------------------------------

    def new_trace(self) -> str:
        """Start a new trace. Returns the trace_id."""
        self._trace_id = uuid.uuid4().hex
        self._span_stack = []
        return self._trace_id

    def _new_span_id(self) -> str:
        return uuid.uuid4().hex

    @property
    def current_parent_span(self) -> str | None:
        """The span at the top of the stack (current parent)."""
        return self._span_stack[-1] if self._span_stack else None

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def start_span(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> str:
        """Emit a start event and push a new span onto the stack.

        Returns the span_id so callers can reference it as a parent.
        """
        span_id = self._new_span_id()
        parent = parent_span_id if parent_span_id is not None else self.current_parent_span
        self._emit(event_type, payload or {}, span_id=span_id, parent_span_id=parent)
        self._span_stack.append(span_id)
        return span_id

    def end_span(
        self,
        span_id: str,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit an end event and pop the span from the stack."""
        parent = None
        if self._span_stack and self._span_stack[-1] == span_id:
            self._span_stack.pop()
            parent = self.current_parent_span
        self._emit(event_type, payload or {}, span_id=span_id, parent_span_id=parent)

    def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        """Emit a standalone event (not a span start/end)."""
        span_id = self._new_span_id()
        parent = parent_span_id if parent_span_id is not None else self.current_parent_span
        self._emit(event_type, payload or {}, span_id=span_id, parent_span_id=parent)

    def _emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        span_id: str,
        parent_span_id: str | None,
    ) -> None:
        """Create and send a TraceEvent."""
        event = TraceEvent(
            event_id=uuid.uuid4().hex,
            trace_id=self._trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            event_type=event_type,
            agent_class=self._agent_class,
            payload=payload,
            tags=dict(self._config.tags),
        )
        self._transport.send([event])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush buffered events without closing the transport.

        Safe to call between ``run()`` invocations on the same agent.
        """
        if hasattr(self._transport, "flush"):
            self._transport.flush()

    def close(self) -> None:
        """Flush and close the transport."""
        self._transport.close()
