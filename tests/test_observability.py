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
    DesignExecuteConfig,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
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
        llm = MockLLM(["result = add(a=1, b=2)"])
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
        llm = MockLLM(["result = add(a=1, b=2)"])
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
        llm = MockLLM(["x = add(a=1, b=2)\nresult = multiply(a=x, b=3)"])
        obs = _make_obs_config()
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("(1+2)*3")

        step_events = [
            e for e in transport.events if e.event_type == EventType.EXECUTION_STEP
        ]
        assert len(step_events) == 2
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

        types = _event_types(transport)
        assert EventType.PLAN_LLM_REQUEST not in types

    def test_no_llm_responses_when_disabled(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
        obs = _make_obs_config(capture_llm_responses=False)
        config = PlanExecuteConfig(observability=obs)
        agent = SimpleCalculator(llm=llm, config=config)
        transport = _get_transport(agent)

        agent.run("1+2")

        types = _event_types(transport)
        assert EventType.PLAN_LLM_RESPONSE not in types

    def test_no_execution_steps_when_disabled(self) -> None:
        llm = MockLLM(["result = add(a=1, b=2)"])
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
        llm = MockLLM(["import os\nresult = os.getcwd()", "result = add(a=1, b=2)"])
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
                "results = []\nfor i in range(3):\n    results.append(double(x=float(i)))\nresult = results"
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
        llm = MockLLM(["result = double(x=5.0)"])
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
            llm = MockLLM(["result = add(a=1, b=2)"])
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
        llm = MockLLM(["result = add(a=10, b=20)"])
        agent = SimpleCalculator(llm=llm)

        result = agent.run("10 + 20")
        assert result.success
        assert result.result == 30.0
        assert agent._tracer is None

    def test_run_works_with_disabled_observability(self) -> None:
        llm = MockLLM(["result = add(a=10, b=20)"])
        config = PlanExecuteConfig(
            observability=ObservabilityConfig(enabled=False)
        )
        agent = SimpleCalculator(llm=llm, config=config)

        result = agent.run("10 + 20")
        assert result.success
        assert result.result == 30.0
        assert agent._tracer is None
