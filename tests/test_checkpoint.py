"""Unit tests for checkpoint system (distributed/interruptible execution)."""

import tempfile
from datetime import datetime
from typing import Any

import pytest

from opensymbolicai.blueprints.plan_execute import PlanExecute
from opensymbolicai.checkpoint import (
    CheckpointStatus,
    ExecutionCheckpoint,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    PendingMutation,
    PlanContext,
    SerializedValue,
    SerializerRegistry,
)
from opensymbolicai.core import primitive
from opensymbolicai.llm import LLM, LLMConfig, LLMResponse, TokenUsage
from opensymbolicai.models import PlanExecuteConfig


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


# =============================================================================
# Test Agents
# =============================================================================


class ReadOnlyCalculator(PlanExecute):
    """Calculator with only read-only operations (no mutations)."""

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=True)
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


class MutatingCalculator(PlanExecute):
    """Calculator with mutable operations for testing mutation approval."""

    def __init__(self, llm: LLM | LLMConfig, config: PlanExecuteConfig | None = None):
        super().__init__(llm=llm, config=config)
        self._memory: float = 0.0

    @primitive(read_only=True)
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @primitive(read_only=False)
    def store(self, value: float) -> float:
        """Store a value (mutation)."""
        self._memory = value
        return self._memory

    @primitive(read_only=False)
    def increment(self, amount: float) -> float:
        """Increment memory (mutation)."""
        self._memory += amount
        return self._memory

    @property
    def memory(self) -> float:
        return self._memory


# =============================================================================
# SerializerRegistry Tests
# =============================================================================


class TestSerializerRegistryPrimitives:
    """Test serialization of primitive types."""

    def test_serialize_none(self):
        registry = SerializerRegistry()
        result = registry.serialize(None)
        assert result.type_name == "NoneType"
        assert result.data is None
        assert result.is_primitive is True

    def test_serialize_int(self):
        registry = SerializerRegistry()
        result = registry.serialize(42)
        assert result.type_name == "builtins.int"
        assert result.data == 42
        assert result.is_primitive is True

    def test_serialize_float(self):
        registry = SerializerRegistry()
        result = registry.serialize(3.14)
        assert result.type_name == "builtins.float"
        assert result.data == 3.14
        assert result.is_primitive is True

    def test_serialize_str(self):
        registry = SerializerRegistry()
        result = registry.serialize("hello")
        assert result.type_name == "builtins.str"
        assert result.data == "hello"
        assert result.is_primitive is True

    def test_serialize_bool(self):
        registry = SerializerRegistry()
        result = registry.serialize(True)
        assert result.type_name == "builtins.bool"
        assert result.data is True
        assert result.is_primitive is True


class TestSerializerRegistryCollections:
    """Test serialization of collection types."""

    def test_serialize_list(self):
        registry = SerializerRegistry()
        result = registry.serialize([1, 2, 3])
        assert result.type_name == "builtins.list"
        assert result.is_primitive is False
        # Deserialize and verify
        deserialized = registry.deserialize(result)
        assert deserialized == [1, 2, 3]

    def test_serialize_nested_list(self):
        registry = SerializerRegistry()
        result = registry.serialize([[1, 2], [3, 4]])
        deserialized = registry.deserialize(result)
        assert deserialized == [[1, 2], [3, 4]]

    def test_serialize_dict(self):
        registry = SerializerRegistry()
        result = registry.serialize({"a": 1, "b": 2})
        assert result.type_name == "builtins.dict"
        deserialized = registry.deserialize(result)
        assert deserialized == {"a": 1, "b": 2}

    def test_serialize_nested_dict(self):
        registry = SerializerRegistry()
        data = {"outer": {"inner": 42}}
        result = registry.serialize(data)
        deserialized = registry.deserialize(result)
        assert deserialized == data


class TestSerializerRegistryBuiltinTypes:
    """Test serialization of built-in types with default serializers."""

    def test_serialize_set(self):
        registry = SerializerRegistry()
        result = registry.serialize({1, 2, 3})
        deserialized = registry.deserialize(result)
        assert deserialized == {1, 2, 3}

    def test_serialize_frozenset(self):
        registry = SerializerRegistry()
        result = registry.serialize(frozenset([1, 2, 3]))
        deserialized = registry.deserialize(result)
        assert deserialized == frozenset([1, 2, 3])

    def test_serialize_tuple(self):
        registry = SerializerRegistry()
        result = registry.serialize((1, 2, 3))
        deserialized = registry.deserialize(result)
        assert deserialized == (1, 2, 3)

    def test_serialize_bytes(self):
        registry = SerializerRegistry()
        result = registry.serialize(b"hello")
        deserialized = registry.deserialize(result)
        assert deserialized == b"hello"

    def test_serialize_datetime(self):
        registry = SerializerRegistry()
        now = datetime.now()
        result = registry.serialize(now)
        deserialized = registry.deserialize(result)
        assert deserialized == now


class TestSerializerRegistryCustomTypes:
    """Test custom type serialization."""

    def test_register_custom_serializer(self):
        class Point:
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y

        registry = SerializerRegistry()
        registry.register(
            Point,
            serializer=lambda p: {"x": p.x, "y": p.y},
            deserializer=lambda d: Point(d["x"], d["y"]),
        )

        point = Point(3.0, 4.0)
        result = registry.serialize(point)
        deserialized = registry.deserialize(result)

        assert deserialized.x == 3.0
        assert deserialized.y == 4.0

    def test_register_serializer_decorator(self):
        class Circle:
            def __init__(self, radius: float):
                self.radius = radius

        registry = SerializerRegistry()

        @registry.register_serializer(Circle)
        def serialize_circle(c: Circle) -> dict:
            return {"radius": c.radius}

        @registry.register_deserializer("tests.test_checkpoint.Circle")
        def deserialize_circle(d: dict) -> Circle:
            return Circle(d["radius"])

        circle = Circle(5.0)
        result = registry.serialize(circle)
        assert result.data == {"radius": 5.0}

    def test_unserializable_type_has_repr(self):
        registry = SerializerRegistry()

        class Unserializable:
            pass

        obj = Unserializable()
        result = registry.serialize(obj)

        assert "__repr__" in result.data
        assert "__unserializable__" in result.data

    def test_cannot_deserialize_unserializable(self):
        registry = SerializerRegistry()

        serialized = SerializedValue(
            type_name="some.UnknownType",
            data={"__repr__": "<object>", "__unserializable__": True},
            is_primitive=False,
        )

        with pytest.raises(ValueError, match="Cannot deserialize"):
            registry.deserialize(serialized)


class TestSerializerRegistryNamespace:
    """Test namespace serialization helpers."""

    def test_serialize_namespace(self):
        registry = SerializerRegistry()
        namespace = {"x": 10, "y": 20, "name": "test"}
        result = registry.serialize_namespace(namespace)

        assert "x" in result
        assert "y" in result
        assert "name" in result
        assert result["x"].data == 10

    def test_serialize_namespace_with_exclude(self):
        registry = SerializerRegistry()
        namespace = {"x": 10, "y": 20, "self": object()}
        result = registry.serialize_namespace(namespace, exclude={"self"})

        assert "x" in result
        assert "y" in result
        assert "self" not in result

    def test_deserialize_namespace(self):
        registry = SerializerRegistry()
        serialized = {
            "x": SerializedValue(type_name="builtins.int", data=10, is_primitive=True),
            "y": SerializedValue(type_name="builtins.int", data=20, is_primitive=True),
        }
        result = registry.deserialize_namespace(serialized)

        assert result == {"x": 10, "y": 20}


# =============================================================================
# CheckpointStore Tests
# =============================================================================


class TestInMemoryCheckpointStore:
    """Test in-memory checkpoint store."""

    def test_save_and_load(self):
        store = InMemoryCheckpointStore()
        checkpoint = ExecutionCheckpoint(
            task="test task", plan="x = add(1, 2)", total_steps=1
        )

        store.save(checkpoint)
        loaded = store.load(checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.task == "test task"
        assert loaded.plan == "x = add(1, 2)"

    def test_load_nonexistent_returns_none(self):
        store = InMemoryCheckpointStore()
        result = store.load("nonexistent-id")
        assert result is None

    def test_delete(self):
        store = InMemoryCheckpointStore()
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)

        store.save(checkpoint)
        store.delete(checkpoint.checkpoint_id)

        assert store.load(checkpoint.checkpoint_id) is None

    def test_list_by_status(self):
        store = InMemoryCheckpointStore()

        cp1 = ExecutionCheckpoint(
            task="task1", plan="x = 1", total_steps=1, status=CheckpointStatus.RUNNING
        )
        cp2 = ExecutionCheckpoint(
            task="task2", plan="y = 2", total_steps=1, status=CheckpointStatus.COMPLETED
        )
        cp3 = ExecutionCheckpoint(
            task="task3", plan="z = 3", total_steps=1, status=CheckpointStatus.RUNNING
        )

        store.save(cp1)
        store.save(cp2)
        store.save(cp3)

        running = store.list_by_status(CheckpointStatus.RUNNING)
        assert len(running) == 2
        assert cp1.checkpoint_id in running
        assert cp3.checkpoint_id in running

    def test_list_all(self):
        store = InMemoryCheckpointStore()

        cp1 = ExecutionCheckpoint(task="task1", plan="x = 1", total_steps=1)
        cp2 = ExecutionCheckpoint(task="task2", plan="y = 2", total_steps=1)

        store.save(cp1)
        store.save(cp2)

        all_ids = store.list_all()
        assert len(all_ids) == 2

    def test_clear(self):
        store = InMemoryCheckpointStore()
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)

        store.save(checkpoint)
        store.clear()

        assert len(store.list_all()) == 0

    def test_stored_copy_is_independent(self):
        store = InMemoryCheckpointStore()
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)

        store.save(checkpoint)
        checkpoint.task = "modified"

        loaded = store.load(checkpoint.checkpoint_id)
        assert loaded is not None
        assert loaded.task == "test"  # Original value


class TestFileCheckpointStore:
    """Test file-based checkpoint store."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            checkpoint = ExecutionCheckpoint(
                task="test task", plan="x = add(1, 2)", total_steps=1
            )

            store.save(checkpoint)
            loaded = store.load(checkpoint.checkpoint_id)

            assert loaded is not None
            assert loaded.task == "test task"

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            result = store.load("nonexistent-id")
            assert result is None

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)

            store.save(checkpoint)
            store.delete(checkpoint.checkpoint_id)

            assert store.load(checkpoint.checkpoint_id) is None

    def test_list_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)

            cp1 = ExecutionCheckpoint(task="task1", plan="x = 1", total_steps=1)
            cp2 = ExecutionCheckpoint(task="task2", plan="y = 2", total_steps=1)

            store.save(cp1)
            store.save(cp2)

            all_ids = store.list_all()
            assert len(all_ids) == 2


# =============================================================================
# ExecutionCheckpoint Model Tests
# =============================================================================


class TestExecutionCheckpointModel:
    """Test ExecutionCheckpoint model."""

    def test_default_checkpoint_id_is_uuid(self):
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)
        assert checkpoint.checkpoint_id is not None
        assert len(checkpoint.checkpoint_id) == 36  # UUID format

    def test_is_resumable_for_running(self):
        checkpoint = ExecutionCheckpoint(
            task="test", plan="x = 1", total_steps=1, status=CheckpointStatus.RUNNING
        )
        assert checkpoint.is_resumable is True

    def test_is_resumable_for_awaiting_approval(self):
        checkpoint = ExecutionCheckpoint(
            task="test",
            plan="x = 1",
            total_steps=1,
            status=CheckpointStatus.AWAITING_APPROVAL,
        )
        assert checkpoint.is_resumable is True

    def test_is_terminal_for_completed(self):
        checkpoint = ExecutionCheckpoint(
            task="test", plan="x = 1", total_steps=1, status=CheckpointStatus.COMPLETED
        )
        assert checkpoint.is_terminal is True
        assert checkpoint.is_resumable is False

    def test_is_terminal_for_failed(self):
        checkpoint = ExecutionCheckpoint(
            task="test", plan="x = 1", total_steps=1, status=CheckpointStatus.FAILED
        )
        assert checkpoint.is_terminal is True

    def test_progress_fraction(self):
        checkpoint = ExecutionCheckpoint(
            task="test", plan="x = 1", total_steps=4, current_step=2
        )
        assert checkpoint.progress_fraction == 0.5

    def test_progress_fraction_zero_steps(self):
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=0)
        assert checkpoint.progress_fraction == 0.0

    def test_touch_updates_timestamp(self):
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)
        original_time = checkpoint.updated_at

        checkpoint.touch()

        assert checkpoint.updated_at >= original_time

    def test_touch_updates_worker_id(self):
        checkpoint = ExecutionCheckpoint(task="test", plan="x = 1", total_steps=1)

        checkpoint.touch(worker_id="worker-123")

        assert checkpoint.worker_id == "worker-123"


# =============================================================================
# execute_stepwise Tests
# =============================================================================


class TestExecuteStepwiseReadOnly:
    """Test execute_stepwise with read-only operations (no mutations)."""

    def test_yields_checkpoint_for_each_step(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\ny = add(x, 3)\nreturn y"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("add numbers"))

        # Yields checkpoint after each step (2 steps) + final completed checkpoint
        # Actually: yields after step 1, after step 2 (which is RUNNING), then sets COMPLETED
        # The last yield is the final state after all steps
        assert len(checkpoints) >= 2
        assert checkpoints[-1].status == CheckpointStatus.COMPLETED

    def test_final_checkpoint_has_result(self):
        mock_llm = MockLLM(responses=["result = add(2, 3)\nreturn result"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("add 2 and 3"))
        final = checkpoints[-1]

        assert final.status == CheckpointStatus.COMPLETED
        # The final step is the return, which binds no variable.
        assert final.result_variable == ""
        assert final.result_value is not None

    def test_checkpoint_tracks_completed_steps(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\ny = multiply(x, 3)\nreturn y"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("compute"))

        # Final checkpoint should have all steps (2 assignments + final return)
        assert len(checkpoints[-1].completed_steps) == 3
        # Verify step details
        assert checkpoints[-1].completed_steps[0].primitive_called == "add"
        assert checkpoints[-1].completed_steps[1].primitive_called == "multiply"

    def test_checkpoint_tracks_namespace(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\nreturn x"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("add"))
        final = checkpoints[-1]

        # x should be in namespace
        assert "x" in final.namespace_snapshot
        registry = SerializerRegistry()
        x_value = registry.deserialize(final.namespace_snapshot["x"])
        assert x_value == 3.0


class TestExecuteStepwiseMutations:
    """Test execute_stepwise with mutations requiring approval."""

    def test_pauses_before_mutation(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\ny = store(x)\nreturn y"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("store result"))

        # Should pause at mutation (step 2), after executing read-only step 1
        # Last checkpoint should be awaiting approval
        assert checkpoints[-1].status == CheckpointStatus.AWAITING_APPROVAL
        # Should have completed the add step
        assert len(checkpoints[-1].completed_steps) == 1
        assert checkpoints[-1].completed_steps[0].primitive_called == "add"

    def test_pending_mutation_info(self):
        mock_llm = MockLLM(responses=["result = store(42.0)\nreturn result"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("store 42"))

        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        assert checkpoint.status == CheckpointStatus.AWAITING_APPROVAL
        assert checkpoint.pending_mutation is not None
        assert checkpoint.pending_mutation.method_name == "store"
        assert checkpoint.pending_mutation.step_number == 1

    def test_mutation_not_executed_when_paused(self):
        mock_llm = MockLLM(responses=["x = store(100.0)\nreturn x"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        list(agent.execute_stepwise("store 100"))

        # Memory should still be 0 since mutation was not approved
        assert agent.memory == 0.0

    def test_no_pause_when_approval_not_required(self):
        mock_llm = MockLLM(responses=["x = store(42.0)\nreturn x"])
        config = PlanExecuteConfig(require_mutation_approval=False)
        agent = MutatingCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("store"))

        # Should complete without pausing
        assert checkpoints[-1].status == CheckpointStatus.COMPLETED
        assert agent.memory == 42.0


class TestExecuteStepwiseErrors:
    """Test execute_stepwise error handling."""

    def test_invalid_plan_yields_failed_checkpoint(self):
        mock_llm = MockLLM(responses=["for x in range(10): pass"])  # Invalid
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("invalid"))

        assert len(checkpoints) == 1
        assert checkpoints[0].status == CheckpointStatus.FAILED
        assert checkpoints[0].error is not None

    def test_execution_error_yields_failed_checkpoint(self):
        mock_llm = MockLLM(responses=["x = undefined_function()"])  # Will fail
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("bad call"))

        # Should fail validation since undefined_function is not a primitive
        assert checkpoints[-1].status == CheckpointStatus.FAILED
        assert checkpoints[-1].error is not None


# =============================================================================
# resume_from_checkpoint Tests
# =============================================================================


class TestResumeFromCheckpoint:
    """Test resume_from_checkpoint functionality."""

    def test_resume_after_mutation_approval(self):
        mock_llm = MockLLM(
            responses=["x = add(1, 2)\ny = store(x)\nz = add(y, 1)\nreturn z"]
        )
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        # Execute until pause
        checkpoints = list(agent.execute_stepwise("compute and store"))
        paused = checkpoints[-1]

        assert paused.status == CheckpointStatus.AWAITING_APPROVAL
        assert agent.memory == 0.0  # Not executed yet

        # Resume with approval
        resumed_checkpoints = list(
            agent.resume_from_checkpoint(paused, approve_mutation=True)
        )

        # Should have executed mutation and continued
        assert agent.memory == 3.0  # store(3.0) was executed
        assert resumed_checkpoints[-1].status == CheckpointStatus.COMPLETED

    def test_resume_continues_from_correct_step(self):
        mock_llm = MockLLM(
            responses=["a = add(1, 1)\nb = store(a)\nc = add(b, 1)\nreturn c"]
        )
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        # Execute until pause at store
        checkpoints = list(agent.execute_stepwise("multi step"))
        paused = checkpoints[-1]

        # Should have completed step 1 (add)
        assert len(paused.completed_steps) == 1

        # Resume and complete
        resumed = list(agent.resume_from_checkpoint(paused, approve_mutation=True))

        # Final checkpoint should have all steps (3 assignments + final return)
        assert len(resumed[-1].completed_steps) == 4

    def test_cannot_resume_completed_checkpoint(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\nreturn x"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("add"))
        completed = checkpoints[-1]

        with pytest.raises(ValueError, match="terminal checkpoint"):
            list(agent.resume_from_checkpoint(completed))

    def test_cannot_resume_failed_checkpoint(self):
        checkpoint = ExecutionCheckpoint(
            task="test",
            plan="x = 1",
            total_steps=1,
            status=CheckpointStatus.FAILED,
            error="some error",
        )

        mock_llm = MockLLM()
        agent = ReadOnlyCalculator(mock_llm)

        with pytest.raises(ValueError, match="terminal checkpoint"):
            list(agent.resume_from_checkpoint(checkpoint))

    def test_must_approve_if_awaiting(self):
        checkpoint = ExecutionCheckpoint(
            task="test",
            plan="x = store(1)",
            total_steps=1,
            status=CheckpointStatus.AWAITING_APPROVAL,
            pending_mutation=PendingMutation(
                method_name="store", args={}, statement="x = store(1)", step_number=1
            ),
        )

        mock_llm = MockLLM()
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        with pytest.raises(ValueError, match="awaiting mutation approval"):
            list(agent.resume_from_checkpoint(checkpoint, approve_mutation=False))

    def test_multiple_mutations_require_multiple_approvals(self):
        mock_llm = MockLLM(
            responses=["a = store(1)\nb = increment(1)\nc = increment(1)\nreturn c"]
        )
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        # First pause at store
        checkpoints = list(agent.execute_stepwise("multi mutate"))
        assert checkpoints[-1].status == CheckpointStatus.AWAITING_APPROVAL
        assert checkpoints[-1].pending_mutation.method_name == "store"

        # Resume - should pause at first increment
        cp1 = list(agent.resume_from_checkpoint(checkpoints[-1], approve_mutation=True))
        assert cp1[-1].status == CheckpointStatus.AWAITING_APPROVAL
        assert cp1[-1].pending_mutation.method_name == "increment"

        # Resume again - should pause at second increment
        cp2 = list(agent.resume_from_checkpoint(cp1[-1], approve_mutation=True))
        assert cp2[-1].status == CheckpointStatus.AWAITING_APPROVAL

        # Final resume - should complete
        cp3 = list(agent.resume_from_checkpoint(cp2[-1], approve_mutation=True))
        assert cp3[-1].status == CheckpointStatus.COMPLETED
        assert agent.memory == 3.0  # 1 + 1 + 1


class TestResumeNamespaceRestoration:
    """Test that namespace is properly restored on resume."""

    def test_variables_restored_from_checkpoint(self):
        mock_llm = MockLLM(responses=["x = add(10, 20)\ny = store(x)\nreturn y"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)

        # Execute and pause
        checkpoints = list(agent.execute_stepwise("compute"))
        paused = checkpoints[-1]

        # Verify x is in namespace snapshot
        assert "x" in paused.namespace_snapshot

        # Create new agent instance (simulating distributed worker)
        new_agent = MutatingCalculator(mock_llm, config=config)

        # Resume should restore x and use it
        list(new_agent.resume_from_checkpoint(paused, approve_mutation=True))

        # store(x) should have stored 30
        assert new_agent.memory == 30.0


# =============================================================================
# run_with_checkpoints Tests
# =============================================================================


class TestRunWithCheckpoints:
    """Test run_with_checkpoints convenience method."""

    def test_saves_all_checkpoints(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\ny = add(x, 3)\nreturn y"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = ReadOnlyCalculator(mock_llm, config=config)
        store = InMemoryCheckpointStore()

        result = agent.run_with_checkpoints("compute", store)

        assert result.status == CheckpointStatus.COMPLETED
        assert len(store.list_all()) >= 1

    def test_auto_approve_completes_mutations(self):
        mock_llm = MockLLM(responses=["x = store(42.0)\nreturn x"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)
        store = InMemoryCheckpointStore()

        result = agent.run_with_checkpoints("store", store, auto_approve=True)

        assert result.status == CheckpointStatus.COMPLETED
        assert agent.memory == 42.0

    def test_returns_pending_without_auto_approve(self):
        mock_llm = MockLLM(responses=["x = store(42.0)\nreturn x"])
        config = PlanExecuteConfig(require_mutation_approval=True)
        agent = MutatingCalculator(mock_llm, config=config)
        store = InMemoryCheckpointStore()

        result = agent.run_with_checkpoints("store", store, auto_approve=False)

        assert result.status == CheckpointStatus.AWAITING_APPROVAL
        assert agent.memory == 0.0  # Not executed


# =============================================================================
# PlanContext Tests
# =============================================================================


class TestPlanContext:
    """Test PlanContext model."""

    def test_attempt_count(self):
        context = PlanContext(plan_attempts=[], final_plan="x = 1")
        assert context.attempt_count == 0

    def test_had_retries(self):
        from opensymbolicai.models import LLMInteraction, PlanAttempt, PlanGeneration

        attempts = [
            PlanAttempt(
                attempt_number=1,
                plan_generation=PlanGeneration(
                    llm_interaction=LLMInteraction(prompt="p", response="r"),
                    extracted_code="bad",
                ),
                success=False,
            ),
            PlanAttempt(
                attempt_number=2,
                plan_generation=PlanGeneration(
                    llm_interaction=LLMInteraction(prompt="p2", response="r2"),
                    extracted_code="x = 1",
                ),
                success=True,
            ),
        ]

        context = PlanContext(plan_attempts=attempts, final_plan="x = 1")
        assert context.had_retries is True

    def test_all_llm_interactions(self):
        from opensymbolicai.models import LLMInteraction, PlanAttempt, PlanGeneration

        attempts = [
            PlanAttempt(
                attempt_number=1,
                plan_generation=PlanGeneration(
                    llm_interaction=LLMInteraction(prompt="prompt1", response="resp1"),
                    extracted_code="code1",
                ),
                success=True,
            ),
        ]

        context = PlanContext(plan_attempts=attempts, final_plan="code1")
        interactions = context.all_llm_interactions

        assert len(interactions) == 1
        assert interactions[0].prompt == "prompt1"


# =============================================================================
# Config Tests
# =============================================================================


class TestCheckpointConfig:
    """Test PlanExecuteConfig checkpoint settings."""

    def test_require_mutation_approval_defaults_to_true(self):
        config = PlanExecuteConfig()
        assert config.require_mutation_approval is True

    def test_worker_id_defaults_to_none(self):
        config = PlanExecuteConfig()
        assert config.worker_id is None

    def test_worker_id_can_be_set(self):
        config = PlanExecuteConfig(worker_id="worker-abc")
        assert config.worker_id == "worker-abc"

    def test_worker_id_appears_in_checkpoint(self):
        mock_llm = MockLLM(responses=["x = add(1, 2)\nreturn x"])
        config = PlanExecuteConfig(
            require_mutation_approval=True, worker_id="my-worker"
        )
        agent = ReadOnlyCalculator(mock_llm, config=config)

        checkpoints = list(agent.execute_stepwise("add"))

        assert checkpoints[-1].worker_id == "my-worker"
