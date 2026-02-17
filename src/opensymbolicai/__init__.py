"""OpenSymbolicAI Core Runtime."""

from opensymbolicai.blueprints import DesignExecute, GoalSeeking, PlanExecute, Planner
from opensymbolicai.checkpoint import (
    CheckpointStatus,
    CheckpointStore,
    ExecutionCheckpoint,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    PendingMutation,
    PlanContext,
    SerializedValue,
    SerializerRegistry,
    default_serializer_registry,
)
from opensymbolicai.core import MethodType, decomposition, evaluator, primitive
from opensymbolicai.exceptions import (
    ExecutionError,
    OperationError,
    PreconditionError,
    ResourceError,
    RetryableError,
    ValidationError,
)
from opensymbolicai.models import (
    DesignExecuteConfig,
    ExecutionMetrics,
    ExecutionResult,
    ExecutionStep,
    ExecutionTrace,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
    GoalSeekingResult,
    GoalStatus,
    Iteration,
    MutationHook,
    MutationHookContext,
    OrchestrationResult,
    PlanAnalysis,
    PlanExecuteConfig,
    PlanResult,
    PrimitiveCall,
    TokenUsage,
)

__version__ = "0.3.0"

__all__ = [
    # Core
    "MethodType",
    "decomposition",
    "evaluator",
    "primitive",
    # Blueprints
    "DesignExecute",
    "GoalSeeking",
    "PlanExecute",
    "Planner",
    # Checkpoint (distributed execution)
    "CheckpointStatus",
    "CheckpointStore",
    "ExecutionCheckpoint",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "PendingMutation",
    "PlanContext",
    "SerializedValue",
    "SerializerRegistry",
    "default_serializer_registry",
    # Exceptions
    "ExecutionError",
    "OperationError",
    "PreconditionError",
    "ResourceError",
    "RetryableError",
    "ValidationError",
    # Models
    "DesignExecuteConfig",
    "ExecutionMetrics",
    "ExecutionResult",
    "ExecutionStep",
    "ExecutionTrace",
    "GoalContext",
    "GoalEvaluation",
    "GoalSeekingConfig",
    "GoalSeekingResult",
    "GoalStatus",
    "Iteration",
    "MutationHook",
    "MutationHookContext",
    "OrchestrationResult",
    "PlanAnalysis",
    "PlanExecuteConfig",
    "PlanResult",
    "PrimitiveCall",
    "TokenUsage",
]
