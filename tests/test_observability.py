"""Unit tests for the observability framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from opensymbolicai.blueprints.design_execute import DesignExecute
from opensymbolicai.blueprints.plan_execute import PlanExecute
from opensymbolicai.core import evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import (
    MUTATION_REJECTED_PREFIX,
    DesignExecuteConfig,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
    GoalStatus,
    PlanExecuteConfig,
)
from opensymbolicai.observability.config import ObservabilityConfig
from opensymbolicai.observability.events import EventType, TraceEvent
from opensymbolicai.observability.tracer import PayloadFilter, Tracer, _create_transport
from opensymbolicai.observability.transports.file import FileTransport
from opensymbolicai.observability.transports.http import HttpTransport
from opensymbolicai.observability.transports.memory import InMemoryTransport

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


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
    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=True)
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


class LoopCalculator(DesignExecute):
    @primitive(read_only=True)
    def double(self, x: float) -> float:
        """Double a number."""
        return x * 2


def _make_obs_config(**kwargs: Any) -> ObservabilityConfig:
    """Create an enabled ObservabilityConfig with InMemoryTransport."""
    transport = InMemoryTransport()
    return ObservabilityConfig(
        enabled=True,
        transport=transport,
        **kwargs,
    )


def _get_transport(agent: PlanExecute) -> InMemoryTransport:
    """Extract the InMemoryTransport from an agent's tracer."""
    assert agent._tracer is not None
    transport = agent._tracer._transport
    assert isinstance(transport, InMemoryTransport)
    return transport


def _event_types(transport: InMemoryTransport) -> list[EventType]:
    """Get the list of event types from a transport."""
    return [e.event_type for e in transport.events]


# ===========================================================================
# TraceEvent model tests
# ===========================================================================


class TestTraceEvent:
    def test_serialization_roundtrip(self) -> None:
        event = TraceEvent(
            event_id="abc123",
            trace_id="trace-1",
            session_id="session-1",
            span_id="span-1",
            parent_span_id=None,
            event_type=EventType.RUN_START,
            agent_class="TestAgent",
            payload={"task": "do something"},
            tags={"env": "test"},
        )
        dumped = event.model_dump_json()
        loaded = TraceEvent.model_validate_json(dumped)
        assert loaded.event_id == "abc123"
        assert loaded.event_type == EventType.RUN_START
        assert loaded.payload == {"task": "do something"}
        assert loaded.tags == {"env": "test"}

    def test_default_timestamp(self) -> None:
        event = TraceEvent(
            event_id="x",
            trace_id="t",
            session_id="s",
            span_id="s",
            event_type=EventType.RUN_START,
            agent_class="A",
        )
        assert event.timestamp is not None


# ===========================================================================
# ObservabilityConfig tests
# ===========================================================================


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        config = ObservabilityConfig()
        assert config.enabled is False
        assert config.capture_llm_prompts is True
        assert config.capture_namespace_snapshots is False
        assert config.collector_url is None
        assert config.tags == {}

    def test_with_tags(self) -> None:
        config = ObservabilityConfig(enabled=True, tags={"env": "prod"})
        assert config.tags == {"env": "prod"}


# ===========================================================================
# Transport tests
# ===========================================================================


class TestInMemoryTransport:
    def test_send_and_close(self) -> None:
        transport = InMemoryTransport()
        event = TraceEvent(
            event_id="1",
            trace_id="t",
            session_id="test",
            span_id="s",
            event_type=EventType.RUN_START,
            agent_class="A",
        )
        transport.send([event])
        assert len(transport.events) == 1
        transport.close()  # no-op, should not raise

    def test_multiple_sends(self) -> None:
        transport = InMemoryTransport()
        for i in range(5):
            transport.send(
                [
                    TraceEvent(
                        event_id=str(i),
                        trace_id="t",
                        session_id="test",
                        span_id=str(i),
                        event_type=EventType.EXECUTION_STEP,
                        agent_class="A",
                    )
                ]
            )
        assert len(transport.events) == 5


class TestFileTransport:
    def test_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "trace.jsonl")
            transport = FileTransport(path)

            events = [
                TraceEvent(
                    event_id=str(i),
                    trace_id="t",
                    session_id="test",
                    span_id=str(i),
                    event_type=EventType.RUN_START,
                    agent_class="A",
                    payload={"step": i},
                )
                for i in range(3)
            ]
            transport.send(events)
            transport.close()

            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 3

            parsed = json.loads(lines[0])
            assert parsed["event_type"] == "run.start"
            assert parsed["payload"]["step"] == 0

    def test_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "deep" / "nested" / "trace.jsonl")
            transport = FileTransport(path)
            transport.send(
                [
                    TraceEvent(
                        event_id="1",
                        trace_id="t",
                        session_id="test",
                        span_id="s",
                        event_type=EventType.RUN_START,
                        agent_class="A",
                    )
                ]
            )
            transport.close()
            assert Path(path).exists()

    def test_append_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "trace.jsonl")
            # Write, close, write again
            t1 = FileTransport(path)
            t1.send(
                [
                    TraceEvent(
                        event_id="1",
                        trace_id="t",
                        session_id="test",
                        span_id="s",
                        event_type=EventType.RUN_START,
                        agent_class="A",
                    )
                ]
            )
            t1.close()

            t2 = FileTransport(path)
            t2.send(
                [
                    TraceEvent(
                        event_id="2",
                        trace_id="t",
                        session_id="test",
                        span_id="s2",
                        event_type=EventType.RUN_COMPLETE,
                        agent_class="A",
                    )
                ]
            )
            t2.close()

            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 2


class TestHttpTransport:
    def test_send_queues_and_close_flushes(self) -> None:
        transport = HttpTransport(
            url="http://localhost:9999/events",
            batch_size=100,
            flush_interval_seconds=10,
        )

        event = TraceEvent(
            event_id="1",
            trace_id="t",
            session_id="test",
            span_id="s",
            event_type=EventType.RUN_START,
            agent_class="A",
        )
        transport.send([event])

        # Patch _send_http to capture what gets sent
        sent_batches: list[list[TraceEvent]] = []
        transport._send_http = lambda events: sent_batches.append(events)  # type: ignore[assignment]

        transport.close()
        assert len(sent_batches) == 1
        assert len(sent_batches[0]) == 1

    def test_no_send_after_close(self) -> None:
        transport = HttpTransport(
            url="http://localhost:9999/events",
            batch_size=100,
            flush_interval_seconds=10,
        )
        transport.close()
        # Sending after close should not raise
        transport.send(
            [
                TraceEvent(
                    event_id="1",
                    trace_id="t",
                    session_id="test",
                    span_id="s",
                    event_type=EventType.RUN_START,
                    agent_class="A",
                )
            ]
        )


# ===========================================================================
# _create_transport tests
# ===========================================================================


class TestCreateTransport:
    def test_custom_transport_takes_precedence(self) -> None:
        custom = InMemoryTransport()
        config = ObservabilityConfig(
            enabled=True,
            transport=custom,
            collector_url="http://example.com",
        )
        result = _create_transport(config)
        assert result is custom

    def test_collector_url_creates_http(self) -> None:
        config = ObservabilityConfig(
            enabled=True, collector_url="http://localhost:8100/events"
        )
        result = _create_transport(config)
        assert isinstance(result, HttpTransport)
        result.close()

    def test_output_path_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "out.jsonl")
            config = ObservabilityConfig(enabled=True, output_path=path)
            result = _create_transport(config)
            assert isinstance(result, FileTransport)
            result.close()

    def test_fallback_creates_inmemory(self) -> None:
        config = ObservabilityConfig(enabled=True)
        result = _create_transport(config)
        assert isinstance(result, InMemoryTransport)


# ===========================================================================
# PayloadFilter tests
# ===========================================================================


class TestPayloadFilter:
    def test_plan_result_strips_prompt_when_disabled(self) -> None:
        f = PayloadFilter(ObservabilityConfig(capture_llm_prompts=False))
        data = {
            "plan": "x = add(1, 2)",
            "plan_generation": {
                "llm_interaction": {
                    "prompt": "Generate...",
                    "response": "x = add(1, 2)",
                },
                "extracted_code": "x = add(1, 2)",
            },
        }
        filtered = f.plan_result(data)
        interaction = filtered["plan_generation"]["llm_interaction"]
        assert "prompt" not in interaction
        assert "response" in interaction

    def test_plan_result_strips_response_when_disabled(self) -> None:
        f = PayloadFilter(ObservabilityConfig(capture_llm_responses=False))
        data = {
            "plan": "x = add(1, 2)",
            "plan_generation": {
                "llm_interaction": {
                    "prompt": "Generate...",
                    "response": "x = add(1, 2)",
                },
                "extracted_code": "x = add(1, 2)",
            },
        }
        filtered = f.plan_result(data)
        interaction = filtered["plan_generation"]["llm_interaction"]
        assert "prompt" in interaction
        assert "response" not in interaction

    def test_plan_result_strips_plan_source_when_disabled(self) -> None:
        f = PayloadFilter(ObservabilityConfig(capture_plan_source=False))
        data = {
            "plan": "x = add(1, 2)",
            "plan_generation": {
                "llm_interaction": {"prompt": "p", "response": "r"},
                "extracted_code": "x = add(1, 2)",
            },
        }
        filtered = f.plan_result(data)
        assert "plan" not in filtered
        assert "extracted_code" not in filtered["plan_generation"]

    def test_execution_step_strips_namespaces_when_disabled(self) -> None:
        f = PayloadFilter(ObservabilityConfig(capture_namespace_snapshots=False))
        data = {
            "step_number": 1,
            "namespace_before": {"x": 1},
            "namespace_after": {"x": 1, "y": 2},
        }
        filtered = f.execution_step(data)
        assert "namespace_before" not in filtered
        assert "namespace_after" not in filtered
        assert filtered["step_number"] == 1

    def test_execution_step_keeps_namespaces_when_enabled(self) -> None:
        f = PayloadFilter(ObservabilityConfig(capture_namespace_snapshots=True))
        data = {
            "step_number": 1,
            "namespace_before": {"x": 1},
            "namespace_after": {"x": 1, "y": 2},
        }
        filtered = f.execution_step(data)
        assert "namespace_before" in filtered
        assert "namespace_after" in filtered

    def test_llm_interaction_filter(self) -> None:
        f = PayloadFilter(
            ObservabilityConfig(
                capture_llm_prompts=False, capture_llm_responses=False
            )
        )
        data = {"prompt": "Generate...", "response": "result = 42", "time_seconds": 0.5}
        filtered = f.llm_interaction(data)
        assert "prompt" not in filtered
        assert "response" not in filtered
        assert filtered["time_seconds"] == 0.5


# ===========================================================================
# Tracer tests
# ===========================================================================


class TestTracer:
    def test_new_trace_generates_unique_ids(self) -> None:
        config = ObservabilityConfig(enabled=True)
        tracer = Tracer(config, "TestAgent")
        id1 = tracer.new_trace()
        id2 = tracer.new_trace()
        assert id1 != id2

    def test_span_stack(self) -> None:
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        assert tracer.current_parent_span is None

        span1 = tracer.start_span(EventType.RUN_START, {"task": "x"})
        assert tracer.current_parent_span == span1

        span2 = tracer.start_span(EventType.PLAN_START, {"attempt": 1})
        assert tracer.current_parent_span == span2

        tracer.end_span(span2, EventType.PLAN_COMPLETE, {})
        assert tracer.current_parent_span == span1

        tracer.end_span(span1, EventType.RUN_COMPLETE, {})
        assert tracer.current_parent_span is None

    def test_emit_uses_current_parent(self) -> None:
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        span = tracer.start_span(EventType.RUN_START)
        tracer.emit(EventType.EXECUTION_STEP, {"step": 1})

        # The EXECUTION_STEP event should have RUN_START's span as parent
        step_event = transport.events[-1]
        assert step_event.parent_span_id == span

    def test_tags_propagated(self) -> None:
        transport = InMemoryTransport()
        config = ObservabilityConfig(
            enabled=True, transport=transport, tags={"env": "test", "run": "42"}
        )
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()
        tracer.emit(EventType.RUN_START, {})
        assert transport.events[0].tags == {"env": "test", "run": "42"}

    def test_agent_class_on_all_events(self) -> None:
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "MyCustomAgent")
        tracer.new_trace()
        tracer.emit(EventType.RUN_START)
        tracer.emit(EventType.PLAN_START)
        for event in transport.events:
            assert event.agent_class == "MyCustomAgent"

    def test_close_delegates_to_transport(self) -> None:
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "A")
        tracer.close()  # should not raise

    def test_deferred_span_sends_both_events_together(self) -> None:
        """start_span(defer=True) holds the event until end_span sends both."""
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        span = tracer.start_span(EventType.PLAN_LLM_REQUEST, {"prompt": "hi"}, defer=True)

        # Nothing sent yet — the start event is deferred
        assert len(transport.events) == 0

        tracer.end_span(span, EventType.PLAN_LLM_RESPONSE, {"response": "hello"})

        # Both events sent together
        assert len(transport.events) == 2
        assert transport.events[0].event_type == EventType.PLAN_LLM_REQUEST
        assert transport.events[1].event_type == EventType.PLAN_LLM_RESPONSE
        assert transport.events[0].span_id == transport.events[1].span_id

    def test_deferred_span_preserves_start_timestamp(self) -> None:
        """The deferred start event keeps its original timestamp."""
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        span = tracer.start_span(EventType.PLAN_LLM_REQUEST, defer=True)
        tracer.end_span(span, EventType.PLAN_LLM_RESPONSE)

        start_ts = transport.events[0].timestamp
        end_ts = transport.events[1].timestamp
        assert start_ts <= end_ts

    def test_non_deferred_span_sends_immediately(self) -> None:
        """start_span without defer sends the event right away."""
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        tracer.start_span(EventType.RUN_START, {"task": "x"})

        # Sent immediately
        assert len(transport.events) == 1
        assert transport.events[0].event_type == EventType.RUN_START

    def test_deferred_span_flushed_on_close(self) -> None:
        """Orphaned deferred events are flushed when the tracer is closed."""
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        tracer.start_span(EventType.PLAN_LLM_REQUEST, defer=True)
        assert len(transport.events) == 0

        tracer.close()
        assert len(transport.events) == 1
        assert transport.events[0].event_type == EventType.PLAN_LLM_REQUEST

    def test_deferred_span_flushed_on_new_trace(self) -> None:
        """Orphaned deferred events are flushed when a new trace starts."""
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        tracer.start_span(EventType.PLAN_LLM_REQUEST, defer=True)
        assert len(transport.events) == 0

        tracer.new_trace()
        assert len(transport.events) == 1
        assert transport.events[0].event_type == EventType.PLAN_LLM_REQUEST

    def test_deferred_span_parent_is_correct(self) -> None:
        """Deferred span events have the correct parent span IDs."""
        transport = InMemoryTransport()
        config = ObservabilityConfig(enabled=True, transport=transport)
        tracer = Tracer(config, "TestAgent")
        tracer.new_trace()

        plan_span = tracer.start_span(EventType.PLAN_START)
        llm_span = tracer.start_span(
            EventType.PLAN_LLM_REQUEST, {"prompt": "hi"}, defer=True
        )
        tracer.end_span(llm_span, EventType.PLAN_LLM_RESPONSE, {"response": "ok"})
        tracer.end_span(plan_span, EventType.PLAN_COMPLETE)

        # Find the deferred pair
        llm_events = [e for e in transport.events if e.span_id == llm_span]
        assert len(llm_events) == 2
        # Both should have plan_span as parent
        assert llm_events[0].parent_span_id == plan_span
        assert llm_events[1].parent_span_id == plan_span


# ===========================================================================
# Integration: PlanExecute with observability
# ===========================================================================


class TestPlanExecuteObservability:
    def test_disabled_by_default(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        agent = SimpleCalculator(llm=llm)
        assert agent._tracer is None

    def test_no_tracer_when_config_is_none(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        agent = SimpleCalculator(llm=llm, config=PlanExecuteConfig())
        assert agent._tracer is None

    def test_no_tracer_when_disabled(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        config = PlanExecuteConfig(
            observability=ObservabilityConfig(enabled=False)
        )
        agent = SimpleCalculator(llm=llm, config=config)
        assert agent._tracer is None

    def test_run_emits_full_lifecycle(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)\nreturn result"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.run("What is 1 + 2?")
        assert result.success
        assert result.result == 3.0

        types = _event_types(transport)

        # Verify lifecycle ordering
        assert types[0] == EventType.RUN_START
        assert EventType.PLAN_START in types
        assert EventType.PLAN_LLM_REQUEST in types
        assert EventType.PLAN_LLM_RESPONSE in types
        assert EventType.PLAN_COMPLETE in types
        assert EventType.EXECUTION_START in types
        assert EventType.EXECUTION_STEP in types
        assert EventType.EXECUTION_COMPLETE in types
        assert types[-1] == EventType.RUN_COMPLETE

    def test_run_emits_error_on_failure(self) -> None:
        # Return code that will fail execution
        llm = MockLLM(["result = nonexistent_func()"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.run("Do something impossible")
        assert not result.success

        types = _event_types(transport)
        # Should have RUN_ERROR at the end (validation failure path)
        assert EventType.RUN_ERROR in types

    def test_trace_id_consistent_across_events(self) -> None:
        llm = MockLLM(["result = add(a=3, b=4)"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("3 + 4")

        trace_ids = {e.trace_id for e in transport.events}
        assert len(trace_ids) == 1, "All events should share one trace_id"

    def test_span_hierarchy(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)\nreturn result"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("1+2")

        events = transport.events
        run_start = next(e for e in events if e.event_type == EventType.RUN_START)
        plan_start = next(e for e in events if e.event_type == EventType.PLAN_START)
        exec_start = next(e for e in events if e.event_type == EventType.EXECUTION_START)

        # Plan and execution spans should be children of the run span
        assert plan_start.parent_span_id == run_start.span_id
        assert exec_start.parent_span_id == run_start.span_id

    def test_payload_contains_task(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("Calculate 1 + 2")

        run_start = next(
            e for e in transport.events if e.event_type == EventType.RUN_START
        )
        assert run_start.payload["task"] == "Calculate 1 + 2"

    def test_execution_step_events(self) -> None:
        llm = MockLLM(["x = add(a=1, b=2)\nresult = multiply(a=x, b=3)\nreturn result"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("(1+2)*3")

        step_events = [
            e for e in transport.events if e.event_type == EventType.EXECUTION_STEP
        ]
        # 2 assignment steps + 1 required final return step
        assert len(step_events) == 3
        assert step_events[0].payload["primitive_called"] == "add"
        assert step_events[1].payload["primitive_called"] == "multiply"


# ===========================================================================
# Integration: Capture filtering
# ===========================================================================


class TestCaptureFiltering:
    def test_no_llm_prompts_when_disabled(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        obs = _make_obs_config(capture_llm_prompts=False)
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("1+2")

        # The LLM span still exists (for duration tracking) but prompt is stripped
        request_events = [
            e for e in transport.events
            if e.event_type == EventType.PLAN_LLM_REQUEST
        ]
        assert len(request_events) == 1
        assert "prompt" not in request_events[0].payload

    def test_no_llm_responses_when_disabled(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        obs = _make_obs_config(capture_llm_responses=False)
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("1+2")

        # The LLM span still exists (for duration tracking) but response is stripped
        response_events = [
            e for e in transport.events
            if e.event_type == EventType.PLAN_LLM_RESPONSE
        ]
        assert len(response_events) == 1
        assert "response" not in response_events[0].payload

    def test_no_execution_steps_when_disabled(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)\nreturn result"])
        obs = _make_obs_config(capture_execution_steps=False)
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("1+2")

        types = _event_types(transport)
        assert EventType.EXECUTION_STEP not in types
        # But EXECUTION_START and COMPLETE should still be there
        assert EventType.EXECUTION_START in types
        assert EventType.EXECUTION_COMPLETE in types

    def test_plan_validation_error_event(self) -> None:
        # LLM returns invalid code (uses imports), retry returns valid code
        llm = MockLLM(
            ["import os\nresult = os.getcwd()", "result = add(a=1, b=2)\nreturn result"]
        )
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs, max_plan_retries=1)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.run("1+2")
        assert result.success

        types = _event_types(transport)
        assert EventType.PLAN_VALIDATION_ERROR in types


# ===========================================================================
# Integration: DesignExecute with observability
# ===========================================================================


class TestDesignExecuteObservability:
    def test_traced_primitives_emit_steps(self) -> None:
        llm = MockLLM(
            [
                "results = []\nfor i in range(3):\n    results.append(double(x=float(i)))\nresult = results\nreturn result"
            ]
        )
        obs = _make_obs_config()
        config = DesignExecuteConfig(observability=obs)
        agent = LoopCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.run("Double 0, 1, 2")
        assert result.success

        _event_types(transport)
        step_events = [
            e for e in transport.events if e.event_type == EventType.EXECUTION_STEP
        ]
        # 3 calls to double
        assert len(step_events) == 3
        for step_e in step_events:
            assert step_e.payload["primitive_called"] == "double"

    def test_design_execute_full_lifecycle(self) -> None:
        llm = MockLLM(["result = double(x=5.0)\nreturn result"])
        obs = _make_obs_config()
        config = DesignExecuteConfig(observability=obs)
        agent = LoopCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.run("Double 5")
        assert result.success
        assert result.result == 10.0

        types = _event_types(transport)
        assert types[0] == EventType.RUN_START
        assert EventType.EXECUTION_START in types
        assert EventType.EXECUTION_COMPLETE in types
        assert types[-1] == EventType.RUN_COMPLETE


# ===========================================================================
# Integration: GoalSeeking with observability
# ===========================================================================


class TestGoalSeekingObservability:
    def test_seek_emits_goal_events(self) -> None:
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class SimpleGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def check(self, x: int) -> bool:
                """Check if x > 0."""
                return x > 0

            @evaluator
            def evaluate(self, goal: str, context: GoalContext) -> GoalEvaluation:
                return GoalEvaluation(goal_achieved=context.iteration_count >= 1)

        llm = MockLLM(["result = check(x=1)"])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=3)
        agent = SimpleGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.seek("Check a number")
        assert result.succeeded

        types = _event_types(transport)
        assert EventType.GOAL_SEEK_START in types
        assert EventType.GOAL_ITERATION_START in types
        assert EventType.GOAL_EVALUATION in types
        assert EventType.GOAL_ITERATION_COMPLETE in types
        assert EventType.GOAL_SEEK_COMPLETE in types

    def test_seek_span_hierarchy(self) -> None:
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class SimpleGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def noop(self) -> int:
                """No-op."""
                return 0

            @evaluator
            def evaluate(self, goal: str, context: GoalContext) -> GoalEvaluation:
                return GoalEvaluation(goal_achieved=context.iteration_count >= 1)

        llm = MockLLM(["result = noop()"])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=3)
        agent = SimpleGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Do nothing")

        events = transport.events
        seek_start = next(
            e for e in events if e.event_type == EventType.GOAL_SEEK_START
        )
        iter_start = next(
            e for e in events if e.event_type == EventType.GOAL_ITERATION_START
        )
        # Iteration should be child of seek
        assert iter_start.parent_span_id == seek_start.span_id

    def test_seek_payload_contents(self) -> None:
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class SimpleGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def noop(self) -> int:
                """No-op."""
                return 0

            @evaluator
            def evaluate(self, goal: str, context: GoalContext) -> GoalEvaluation:
                return GoalEvaluation(goal_achieved=True)

        llm = MockLLM(["result = noop()"])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=5)
        agent = SimpleGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Achieve the goal")

        seek_start = next(
            e for e in transport.events if e.event_type == EventType.GOAL_SEEK_START
        )
        assert seek_start.payload["goal"] == "Achieve the goal"
        assert seek_start.payload["max_iterations"] == 5

        seek_complete = next(
            e for e in transport.events if e.event_type == EventType.GOAL_SEEK_COMPLETE
        )
        assert seek_complete.payload["status"] == "achieved"

    def test_plan_iteration_emits_plan_span(self) -> None:
        """plan_iteration() should wrap LLM events in PLAN_START/PLAN_COMPLETE span."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class SimpleGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def check(self, x: int) -> bool:
                """Check if x > 0."""
                return x > 0

            @evaluator
            def evaluate(self, goal: str, context: GoalContext) -> GoalEvaluation:
                return GoalEvaluation(goal_achieved=context.iteration_count >= 1)

        llm = MockLLM(["result = check(x=1)"])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=3)
        agent = SimpleGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Check a number")

        types = _event_types(transport)
        assert EventType.PLAN_START in types
        assert EventType.PLAN_LLM_REQUEST in types
        assert EventType.PLAN_LLM_RESPONSE in types
        assert EventType.PLAN_COMPLETE in types

        # PLAN_START should appear before PLAN_COMPLETE
        plan_start_idx = types.index(EventType.PLAN_START)
        plan_complete_idx = types.index(EventType.PLAN_COMPLETE)
        assert plan_start_idx < plan_complete_idx

    def test_plan_iteration_plan_span_hierarchy(self) -> None:
        """PLAN_LLM_REQUEST/RESPONSE should be children of the PLAN_START span."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class SimpleGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def noop(self) -> int:
                """No-op."""
                return 0

            @evaluator
            def evaluate(self, goal: str, context: GoalContext) -> GoalEvaluation:
                return GoalEvaluation(goal_achieved=context.iteration_count >= 1)

        llm = MockLLM(["result = noop()"])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=3)
        agent = SimpleGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Do nothing")

        events = transport.events
        plan_start = next(
            e for e in events if e.event_type == EventType.PLAN_START
        )
        llm_request = next(
            e for e in events if e.event_type == EventType.PLAN_LLM_REQUEST
        )
        llm_response = next(
            e for e in events if e.event_type == EventType.PLAN_LLM_RESPONSE
        )

        # LLM events should be children of the plan span
        assert llm_request.parent_span_id == plan_start.span_id
        assert llm_response.parent_span_id == plan_start.span_id

    def test_plan_evaluator_traces_llm_call(self) -> None:
        """plan_evaluator() should emit GOAL_EVALUATOR_LLM_REQUEST/RESPONSE."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x

        llm = MockLLM([
            # First call: plan_evaluator generates evaluator code
            "result = GoalEvaluation(goal_achieved=context.iteration_count >= 1)",
            # Iteration 1: plan
            "result = compute(x=1)",
            # Iteration 2: plan (evaluator succeeds after iteration 1 recorded)
            "result = compute(x=2)",
        ])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=5)
        agent = DynamicGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.seek("Compute twice")
        assert result.succeeded

        types = _event_types(transport)
        assert EventType.GOAL_EVALUATOR_LLM_REQUEST in types
        assert EventType.GOAL_EVALUATOR_LLM_RESPONSE in types

    def test_plan_evaluator_respects_capture_flags(self) -> None:
        """Evaluator LLM events should respect capture_llm_prompts/responses."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x

        llm = MockLLM([
            "result = GoalEvaluation(goal_achieved=True)",
            "result = compute(x=1)",
        ])
        obs = _make_obs_config(
            capture_llm_prompts=False,
            capture_llm_responses=False,
        )
        config = GoalSeekingConfig(observability=obs, max_iterations=5)
        agent = DynamicGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Do it")

        # LLM spans still exist (for duration tracking) but payloads are stripped
        request_events = [
            e for e in transport.events
            if e.event_type == EventType.GOAL_EVALUATOR_LLM_REQUEST
        ]
        assert len(request_events) == 1
        assert "prompt" not in request_events[0].payload

        response_events = [
            e for e in transport.events
            if e.event_type == EventType.GOAL_EVALUATOR_LLM_RESPONSE
        ]
        assert len(response_events) == 1
        assert "response" not in response_events[0].payload

    def test_plan_evaluator_llm_response_contains_interaction_data(self) -> None:
        """GOAL_EVALUATOR_LLM_RESPONSE should contain LLM interaction details."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x

        llm = MockLLM([
            "result = GoalEvaluation(goal_achieved=True)",
            "result = compute(x=1)",
        ])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=5)
        agent = DynamicGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Do it")

        response_event = next(
            e for e in transport.events
            if e.event_type == EventType.GOAL_EVALUATOR_LLM_RESPONSE
        )
        payload = response_event.payload
        assert "response" in payload
        assert "input_tokens" in payload
        assert "output_tokens" in payload
        assert "time_seconds" in payload
        assert payload["provider"] == "mock"
        assert payload["model"] == "mock-model"

    def test_evaluator_primitive_calls_traced(self) -> None:
        """Primitives called from evaluator code should emit GOAL_EVALUATOR_STEP."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x * 2

            @primitive(read_only=True)
            def count_items(self) -> int:
                """Count items."""
                return 5

        llm = MockLLM([
            # Evaluator code: calls count_items() primitive
            "n = count_items()\nresult = GoalEvaluation(goal_achieved=n >= 5)",
            # Iteration 1: plan
            "result = compute(x=3)",
        ])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=5)
        agent = DynamicGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.seek("Compute something")
        assert result.succeeded

        types = _event_types(transport)
        assert EventType.GOAL_EVALUATOR_STEP in types

        eval_steps = [
            e for e in transport.events
            if e.event_type == EventType.GOAL_EVALUATOR_STEP
        ]
        assert len(eval_steps) >= 1
        assert eval_steps[0].payload["primitive_called"] == "count_items"
        assert eval_steps[0].payload["success"] is True

    def test_evaluator_primitive_calls_not_traced_when_disabled(self) -> None:
        """With capture_execution_steps=False, evaluator primitives are not traced."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x

            @primitive(read_only=True)
            def count_items(self) -> int:
                """Count items."""
                return 5

        llm = MockLLM([
            "n = count_items()\nresult = GoalEvaluation(goal_achieved=n >= 5)",
            "result = compute(x=3)",
        ])
        obs = _make_obs_config(capture_execution_steps=False)
        config = GoalSeekingConfig(observability=obs, max_iterations=5)
        agent = DynamicGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.seek("Compute something")

        types = _event_types(transport)
        assert EventType.GOAL_EVALUATOR_STEP not in types

    def test_plan_validation_error_emitted_in_seek(self) -> None:
        """Validation errors during seek should emit PLAN_VALIDATION_ERROR."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class SimpleGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def increment(self) -> int:
                """Increment."""
                return 1

            @evaluator
            def evaluate(self, goal: str, context: GoalContext) -> GoalEvaluation:
                return GoalEvaluation(goal_achieved=context.iteration_count >= 1)

        llm = MockLLM([
            # Bad plan (import), then valid plan on retry
            "import os\nresult = increment()",
            "result = increment()",
        ])
        obs = _make_obs_config()
        config = GoalSeekingConfig(
            observability=obs, max_iterations=3, max_plan_retries=1
        )
        agent = SimpleGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.seek("Do it")
        assert result.succeeded

        types = _event_types(transport)
        assert EventType.PLAN_VALIDATION_ERROR in types

        error_event = next(
            e for e in transport.events
            if e.event_type == EventType.PLAN_VALIDATION_ERROR
        )
        assert "error" in error_event.payload
        assert error_event.payload["attempt"] == 1

    def test_seek_without_observability_still_works(self) -> None:
        """GoalSeeking should work fine with observability disabled."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x

        llm = MockLLM([
            "result = GoalEvaluation(goal_achieved=True)",
            "result = compute(x=1)",
        ])
        config = GoalSeekingConfig(max_iterations=5)
        agent = DynamicGoalAgent(llm=llm, config=config)
        assert agent._tracer is None

        result = agent.seek("Do it")
        assert result.succeeded

    def test_evaluator_primitive_error_traced(self) -> None:
        """Failed primitive calls in evaluator should emit GOAL_EVALUATOR_STEP with error."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class DynamicGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def compute(self, x: int) -> int:
                """Compute a value."""
                return x

            @primitive(read_only=True)
            def failing_check(self) -> bool:
                """Always fails."""
                raise ValueError("check failed")

        llm = MockLLM([
            # Evaluator code catches the error itself — the traced wrapper
            # records the failed step (with re-raise), and the evaluator's
            # try/except catches it so run_evaluator still returns a result.
            "try:\n    failing_check()\n    result = GoalEvaluation(goal_achieved=True)\nexcept ValueError:\n    result = GoalEvaluation(goal_achieved=False)",
            # Iteration 1: plan
            "result = compute(x=1)",
            # Iteration 2: evaluator runs again with same code, fails again
            "result = compute(x=2)",
        ])
        obs = _make_obs_config()
        config = GoalSeekingConfig(observability=obs, max_iterations=2)
        agent = DynamicGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.seek("Test errors")
        assert result.status == GoalStatus.MAX_ITERATIONS

        eval_steps = [
            e for e in transport.events
            if e.event_type == EventType.GOAL_EVALUATOR_STEP
        ]
        # At least one failing_check call should have been traced
        assert len(eval_steps) >= 1
        failed_steps = [e for e in eval_steps if not e.payload["success"]]
        assert len(failed_steps) >= 1
        assert "check failed" in failed_steps[0].payload["error"]

    def test_evaluator_mutation_hook_rejection_traced(self) -> None:
        """Mutation rejection in evaluator should emit GOAL_EVALUATOR_STEP with error."""
        from opensymbolicai.blueprints.goal_seeking import GoalSeeking

        class MutatingGoalAgent(GoalSeeking):
            @primitive(read_only=True)
            def read_value(self) -> int:
                """Read a value."""
                return 42

            @primitive(read_only=False)
            def write_value(self, x: int) -> None:
                """Write a value (mutating)."""

        def reject_all(ctx: Any) -> str:
            return "writes not allowed in evaluator"

        llm = MockLLM([
            # Evaluator code: tries to call the mutating primitive
            "try:\n    write_value(x=1)\n    result = GoalEvaluation(goal_achieved=True)\n"
            "except RuntimeError:\n    result = GoalEvaluation(goal_achieved=False)",
            # Iteration 1: plan (read-only, works fine)
            "result = read_value()",
            # Iteration 2: plan
            "result = read_value()",
        ])
        obs = _make_obs_config()
        config = GoalSeekingConfig(
            observability=obs, max_iterations=2, on_mutation=reject_all
        )
        agent = MutatingGoalAgent(llm=llm, config=config)
        transport = _get_transport(agent)

        result = agent.seek("Test mutation rejection")
        assert result.status == GoalStatus.MAX_ITERATIONS

        eval_steps = [
            e for e in transport.events
            if e.event_type == EventType.GOAL_EVALUATOR_STEP
        ]
        assert len(eval_steps) >= 1
        rejected_steps = [
            e for e in eval_steps
            if MUTATION_REJECTED_PREFIX in (e.payload.get("error") or "")
        ]
        assert len(rejected_steps) >= 1
        assert rejected_steps[0].payload["primitive_called"] == "write_value"
        assert rejected_steps[0].payload["success"] is False


# ===========================================================================
# Integration: File transport end-to-end
# ===========================================================================


class TestFileTransportEndToEnd:
    def test_run_writes_events_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "trace.jsonl")
            obs = ObservabilityConfig(
                enabled=True,
                output_path=path,
            )
            config = PlanExecuteConfig(observability=obs)
            llm = MockLLM(["result = add(a=1, b=2)\nreturn result"])
            agent = SimpleCalculator(llm=llm, config=config)

            result = agent.run("1+2")
            assert result.success

            # Close the tracer to flush
            assert agent._tracer is not None
            agent._tracer.close()

            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) > 0

            # Each line should be valid JSON
            for line in lines:
                parsed = json.loads(line)
                assert "event_type" in parsed
                assert "trace_id" in parsed


# ===========================================================================
# Zero-cost when disabled: verify no overhead
# ===========================================================================


class TestZeroCostWhenDisabled:
    def test_run_works_without_observability(self) -> None:
        llm = MockLLM(["result = add(a=10, b=20)\nreturn result"])
        agent = SimpleCalculator(llm=llm)

        result = agent.run("10 + 20")
        assert result.success
        assert result.result == 30.0
        assert agent._tracer is None

    def test_run_works_with_disabled_observability(self) -> None:
        llm = MockLLM(["result = add(a=10, b=20)\nreturn result"])
        config = PlanExecuteConfig(
            observability=ObservabilityConfig(enabled=False)
        )
        agent = SimpleCalculator(llm=llm, config=config)

        result = agent.run("10 + 20")
        assert result.success
        assert result.result == 30.0
        assert agent._tracer is None
