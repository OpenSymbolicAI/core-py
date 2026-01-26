"""Checkpoint system for distributed, interruptible execution.

This module provides:
- SerializerRegistry: Custom type serialization for JSON persistence
- CheckpointStore: Protocol for pluggable storage backends (DB, Redis, etc.)
- ExecutionCheckpoint: Serializable execution state for pause/resume
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from opensymbolicai.models import ExecutionStep, LLMInteraction, PlanAttempt


class CheckpointStatus(str, Enum):
    """Status of an execution checkpoint."""

    PENDING = "pending"  # Not yet started
    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Manually paused
    AWAITING_APPROVAL = "awaiting_approval"  # Waiting for mutation approval
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Execution failed


class SerializedValue(BaseModel):
    """A serialized value with type information for deserialization."""

    type_name: str = Field(..., description="Fully qualified type name")
    data: Any = Field(..., description="Serialized data (JSON-compatible)")
    is_primitive: bool = Field(
        default=True,
        description="Whether this is a JSON primitive (no custom deserializer needed)",
    )


class PendingMutation(BaseModel):
    """Information about a mutation awaiting approval."""

    method_name: str = Field(..., description="Name of the mutating primitive")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to the mutation"
    )
    statement: str = Field(..., description="The full statement to execute")
    step_number: int = Field(..., description="Which step this mutation is")
    variable_name: str = Field(
        default="", description="Variable being assigned the result"
    )


class PlanContext(BaseModel):
    """Context about how the plan was generated."""

    plan_attempts: list[PlanAttempt] = Field(
        default_factory=list,
        description="All plan generation attempts including retries",
    )
    final_plan: str = Field(default="", description="The final validated plan")
    generation_time_seconds: float = Field(
        default=0.0, description="Total time spent generating the plan"
    )

    @property
    def attempt_count(self) -> int:
        """Number of plan generation attempts."""
        return len(self.plan_attempts)

    @property
    def had_retries(self) -> bool:
        """Whether the plan required retries."""
        return len(self.plan_attempts) > 1

    @property
    def all_llm_interactions(self) -> list[LLMInteraction]:
        """Get all LLM interactions from all attempts."""
        return [
            attempt.plan_generation.llm_interaction for attempt in self.plan_attempts
        ]


class ExecutionCheckpoint(BaseModel):
    """Serializable execution state for pause/resume across distributed workers."""

    # Identity
    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique checkpoint identifier",
    )
    task: str = Field(..., description="The original task description")
    plan: str = Field(..., description="The Python plan being executed")

    # Plan generation context (LLM history)
    plan_context: PlanContext | None = Field(
        default=None,
        description="Full context of how the plan was generated, including LLM interactions",
    )

    # Execution progress
    current_step: int = Field(default=0, description="Next step to execute (0-indexed)")
    total_steps: int = Field(..., description="Total number of steps in the plan")
    status: CheckpointStatus = Field(
        default=CheckpointStatus.PENDING, description="Current execution status"
    )

    # State
    namespace_snapshot: dict[str, SerializedValue] = Field(
        default_factory=dict,
        description="Serialized namespace variables",
    )
    completed_steps: list[ExecutionStep] = Field(
        default_factory=list, description="Steps that have been executed"
    )

    # Mutation approval
    pending_mutation: PendingMutation | None = Field(
        default=None,
        description="Mutation awaiting approval (if status is awaiting_approval)",
    )

    # Error tracking
    error: str | None = Field(
        default=None, description="Error message if status is failed"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the checkpoint was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the checkpoint was last updated",
    )
    worker_id: str | None = Field(
        default=None, description="ID of the worker that created/last updated this"
    )

    # Result (when completed)
    result_value: SerializedValue | None = Field(
        default=None, description="The final result value (when status is completed)"
    )
    result_variable: str = Field(
        default="", description="Name of the variable containing the result"
    )

    def touch(self, worker_id: str | None = None) -> None:
        """Update the updated_at timestamp and optionally worker_id."""
        self.updated_at = datetime.now(UTC)
        if worker_id is not None:
            self.worker_id = worker_id

    @property
    def is_resumable(self) -> bool:
        """Check if this checkpoint can be resumed."""
        return self.status in (
            CheckpointStatus.PENDING,
            CheckpointStatus.RUNNING,
            CheckpointStatus.PAUSED,
            CheckpointStatus.AWAITING_APPROVAL,
        )

    @property
    def is_terminal(self) -> bool:
        """Check if this checkpoint is in a terminal state."""
        return self.status in (CheckpointStatus.COMPLETED, CheckpointStatus.FAILED)

    @property
    def progress_fraction(self) -> float:
        """Get execution progress as a fraction (0.0 to 1.0)."""
        if self.total_steps == 0:
            return 0.0
        return self.current_step / self.total_steps

    model_config = {"arbitrary_types_allowed": True}


# Type aliases for serializer functions
Serializer = Callable[[Any], Any]  # value -> JSON-compatible data
Deserializer = Callable[[Any], Any]  # JSON-compatible data -> value


class SerializerRegistry:
    """Registry for custom type serializers.

    Allows registering serializers/deserializers for types that aren't
    natively JSON-serializable.

    Example:
        registry = SerializerRegistry()

        # Register a custom class
        registry.register(
            MyClass,
            serializer=lambda obj: {"field": obj.field},
            deserializer=lambda data: MyClass(field=data["field"])
        )

        # Or use decorators
        @registry.register_serializer(MyClass)
        def serialize_myclass(obj: MyClass) -> dict:
            return {"field": obj.field}

        @registry.register_deserializer("mymodule.MyClass")
        def deserialize_myclass(data: dict) -> MyClass:
            return MyClass(field=data["field"])
    """

    def __init__(self) -> None:
        self._serializers: dict[type, Serializer] = {}
        self._deserializers: dict[str, Deserializer] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register serializers for common types."""
        import base64

        # datetime
        self._serializers[datetime] = lambda dt: dt.isoformat()
        self._deserializers["datetime.datetime"] = lambda s: datetime.fromisoformat(s)

        # bytes
        self._serializers[bytes] = lambda b: base64.b64encode(b).decode("ascii")
        self._deserializers["builtins.bytes"] = lambda s: base64.b64decode(
            s.encode("ascii")
        )

        # set and frozenset
        self._serializers[set] = lambda s: list(s)
        self._deserializers["builtins.set"] = lambda lst: set(lst)
        self._serializers[frozenset] = lambda s: list(s)
        self._deserializers["builtins.frozenset"] = lambda lst: frozenset(lst)

        # tuple (preserve as tuple, not list)
        self._serializers[tuple] = lambda t: list(t)
        self._deserializers["builtins.tuple"] = lambda lst: tuple(lst)

    def register(
        self, type_: type, serializer: Serializer, deserializer: Deserializer
    ) -> None:
        """Register a serializer/deserializer pair for a type.

        Args:
            type_: The Python type to register.
            serializer: Function that converts instance to JSON-compatible data.
            deserializer: Function that converts JSON data back to instance.
        """
        type_name = f"{type_.__module__}.{type_.__qualname__}"
        self._serializers[type_] = serializer
        self._deserializers[type_name] = deserializer

    def register_serializer(self, type_: type) -> Callable[[Serializer], Serializer]:
        """Decorator to register a serializer for a type.

        Example:
            @registry.register_serializer(MyClass)
            def serialize(obj: MyClass) -> dict:
                return {"field": obj.field}
        """

        def decorator(fn: Serializer) -> Serializer:
            self._serializers[type_] = fn
            return fn

        return decorator

    def register_deserializer(
        self, type_name: str
    ) -> Callable[[Deserializer], Deserializer]:
        """Decorator to register a deserializer for a type name.

        Example:
            @registry.register_deserializer("mymodule.MyClass")
            def deserialize(data: dict) -> MyClass:
                return MyClass(field=data["field"])
        """

        def decorator(fn: Deserializer) -> Deserializer:
            self._deserializers[type_name] = fn
            return fn

        return decorator

    def _get_type_name(self, value_type: type) -> str:
        """Get the fully qualified type name."""
        return f"{value_type.__module__}.{value_type.__qualname__}"

    def serialize(self, value: Any) -> SerializedValue:
        """Serialize a value to a SerializedValue.

        Args:
            value: The value to serialize.

        Returns:
            SerializedValue with type info and serialized data.
        """
        # Handle None
        if value is None:
            return SerializedValue(type_name="NoneType", data=None, is_primitive=True)

        value_type = type(value)
        type_name = self._get_type_name(value_type)

        # Check for JSON primitives
        if isinstance(value, bool):  # Must check bool before int (bool is subclass)
            return SerializedValue(
                type_name="builtins.bool", data=value, is_primitive=True
            )
        if isinstance(value, int):
            return SerializedValue(
                type_name="builtins.int", data=value, is_primitive=True
            )
        if isinstance(value, float):
            return SerializedValue(
                type_name="builtins.float", data=value, is_primitive=True
            )
        if isinstance(value, str):
            return SerializedValue(
                type_name="builtins.str", data=value, is_primitive=True
            )

        # Check for registered serializer (before generic handling)
        if value_type in self._serializers:
            serializer = self._serializers[value_type]
            return SerializedValue(
                type_name=type_name, data=serializer(value), is_primitive=False
            )

        # Check for lists (recursively serialize)
        if isinstance(value, list):
            serialized_items = [self.serialize(item).model_dump() for item in value]
            return SerializedValue(
                type_name="builtins.list",
                data=serialized_items,
                is_primitive=False,
            )

        # Check for dicts (recursively serialize)
        if isinstance(value, dict):
            serialized_dict = {
                k: self.serialize(v).model_dump() for k, v in value.items()
            }
            return SerializedValue(
                type_name="builtins.dict",
                data=serialized_dict,
                is_primitive=False,
            )

        # Check if it's a Pydantic model
        if hasattr(value, "model_dump"):
            return SerializedValue(
                type_name=type_name,
                data={"__pydantic__": True, "data": value.model_dump()},
                is_primitive=False,
            )

        # Fallback: try to convert to string representation
        return SerializedValue(
            type_name=type_name,
            data={"__repr__": repr(value), "__unserializable__": True},
            is_primitive=False,
        )

    def deserialize(self, serialized: SerializedValue) -> Any:
        """Deserialize a SerializedValue back to its original value.

        Args:
            serialized: The SerializedValue to deserialize.

        Returns:
            The deserialized value.

        Raises:
            ValueError: If no deserializer is registered for the type.
        """
        # Handle None
        if serialized.type_name == "NoneType":
            return None

        # Handle primitives
        if serialized.is_primitive:
            return serialized.data

        # Handle lists
        if serialized.type_name == "builtins.list":
            return [
                self.deserialize(SerializedValue(**item)) for item in serialized.data
            ]

        # Handle dicts
        if serialized.type_name == "builtins.dict":
            return {
                k: self.deserialize(SerializedValue(**v))
                for k, v in serialized.data.items()
            }

        # Check for registered deserializer
        if serialized.type_name in self._deserializers:
            deserializer = self._deserializers[serialized.type_name]
            return deserializer(serialized.data)

        # Check for Pydantic model marker
        if isinstance(serialized.data, dict) and serialized.data.get("__pydantic__"):
            raise ValueError(
                f"Cannot deserialize Pydantic model '{serialized.type_name}': "
                f"register a deserializer that can reconstruct the model from: "
                f"{serialized.data['data']}"
            )

        # Check for __repr__ fallback (non-deserializable)
        if isinstance(serialized.data, dict) and serialized.data.get(
            "__unserializable__"
        ):
            raise ValueError(
                f"Cannot deserialize type '{serialized.type_name}': "
                f"no deserializer registered. Original repr: {serialized.data.get('__repr__', 'unknown')}"
            )

        raise ValueError(
            f"No deserializer registered for type '{serialized.type_name}'"
        )

    def can_deserialize(self, type_name: str) -> bool:
        """Check if a type can be deserialized."""
        if type_name in ("NoneType", "builtins.list", "builtins.dict"):
            return True
        # Primitives
        if type_name in (
            "builtins.bool",
            "builtins.int",
            "builtins.float",
            "builtins.str",
        ):
            return True
        return type_name in self._deserializers

    def serialize_namespace(
        self, namespace: dict[str, Any], exclude: set[str] | None = None
    ) -> dict[str, SerializedValue]:
        """Serialize an entire namespace dict.

        Args:
            namespace: The namespace to serialize.
            exclude: Keys to exclude from serialization.

        Returns:
            Dict mapping variable names to SerializedValues.
        """
        exclude = exclude or set()
        result: dict[str, SerializedValue] = {}
        for key, value in namespace.items():
            if key in exclude:
                continue
            result[key] = self.serialize(value)
        return result

    def deserialize_namespace(
        self, serialized: dict[str, SerializedValue]
    ) -> dict[str, Any]:
        """Deserialize a namespace back to regular values.

        Args:
            serialized: Dict of serialized values.

        Returns:
            Dict mapping variable names to deserialized values.
        """
        result: dict[str, Any] = {}
        for key, value in serialized.items():
            result[key] = self.deserialize(value)
        return result


# Default global registry instance
default_serializer_registry = SerializerRegistry()


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends.

    Implement this protocol to store checkpoints in your preferred backend
    (PostgreSQL, Redis, MongoDB, filesystem, etc.).

    Example:
        class RedisCheckpointStore:
            def __init__(self, redis_client):
                self.redis = redis_client

            def save(self, checkpoint: ExecutionCheckpoint) -> None:
                self.redis.set(
                    f"checkpoint:{checkpoint.checkpoint_id}",
                    checkpoint.model_dump_json()
                )

            def load(self, checkpoint_id: str) -> ExecutionCheckpoint | None:
                data = self.redis.get(f"checkpoint:{checkpoint_id}")
                if data:
                    return ExecutionCheckpoint.model_validate_json(data)
                return None

            def delete(self, checkpoint_id: str) -> None:
                self.redis.delete(f"checkpoint:{checkpoint_id}")

            def list_by_status(self, status: CheckpointStatus) -> list[str]:
                # Implementation depends on your indexing strategy
                ...
    """

    def save(self, checkpoint: ExecutionCheckpoint) -> None:
        """Save a checkpoint to storage."""
        ...

    def load(self, checkpoint_id: str) -> ExecutionCheckpoint | None:
        """Load a checkpoint from storage. Returns None if not found."""
        ...

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint from storage."""
        ...

    def list_by_status(self, status: CheckpointStatus) -> list[str]:
        """List checkpoint IDs with the given status."""
        ...


class InMemoryCheckpointStore:
    """In-memory checkpoint store for testing and development."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, ExecutionCheckpoint] = {}

    def save(self, checkpoint: ExecutionCheckpoint) -> None:
        """Save a checkpoint to memory."""
        checkpoint.touch()
        # Store a copy to prevent mutation issues
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint.model_copy(deep=True)

    def load(self, checkpoint_id: str) -> ExecutionCheckpoint | None:
        """Load a checkpoint from memory."""
        cp = self._checkpoints.get(checkpoint_id)
        if cp:
            return cp.model_copy(deep=True)
        return None

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint from memory."""
        self._checkpoints.pop(checkpoint_id, None)

    def list_by_status(self, status: CheckpointStatus) -> list[str]:
        """List checkpoint IDs with the given status."""
        return [
            cp.checkpoint_id for cp in self._checkpoints.values() if cp.status == status
        ]

    def list_all(self) -> list[str]:
        """List all checkpoint IDs."""
        return list(self._checkpoints.keys())

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()


class FileCheckpointStore:
    """File-based checkpoint store for simple persistence."""

    def __init__(self, directory: str) -> None:
        """Initialize with a directory path for storing checkpoints.

        Args:
            directory: Path to directory where checkpoint JSON files are stored.
        """
        import os

        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, checkpoint_id: str) -> str:
        import os

        # Sanitize checkpoint_id to prevent path traversal
        safe_id = "".join(c for c in checkpoint_id if c.isalnum() or c in "-_")
        return os.path.join(self.directory, f"{safe_id}.json")

    def save(self, checkpoint: ExecutionCheckpoint) -> None:
        """Save a checkpoint to a JSON file."""
        checkpoint.touch()
        with open(self._path(checkpoint.checkpoint_id), "w") as f:
            f.write(checkpoint.model_dump_json(indent=2))

    def load(self, checkpoint_id: str) -> ExecutionCheckpoint | None:
        """Load a checkpoint from a JSON file."""
        import os

        path = self._path(checkpoint_id)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return ExecutionCheckpoint.model_validate_json(f.read())

    def delete(self, checkpoint_id: str) -> None:
        """Delete a checkpoint file."""
        import os

        path = self._path(checkpoint_id)
        if os.path.exists(path):
            os.remove(path)

    def list_by_status(self, status: CheckpointStatus) -> list[str]:
        """List checkpoint IDs with the given status."""
        import os

        result = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                checkpoint_id = filename[:-5]
                checkpoint = self.load(checkpoint_id)
                if checkpoint and checkpoint.status == status:
                    result.append(checkpoint_id)
        return result

    def list_all(self) -> list[str]:
        """List all checkpoint IDs."""
        import os

        return [f[:-5] for f in os.listdir(self.directory) if f.endswith(".json")]
