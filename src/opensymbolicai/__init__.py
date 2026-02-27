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
    PROMPT_CONTEXT_BEGIN,
    PROMPT_CONTEXT_END,
    PROMPT_DEFINITIONS_BEGIN,
    PROMPT_DEFINITIONS_END,
    PROMPT_INSTRUCTIONS_BEGIN,
    PROMPT_INSTRUCTIONS_END,
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
from opensymbolicai.observability import (
    EventType,
    FileTransport,
    HttpTransport,
    InMemoryTransport,
    ObservabilityConfig,
    TraceEvent,
    TraceTransport,
)
from opensymbolicai.prompt_utils import PromptSections, extract_context, split_prompt

__version__ = "0.6.0"

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
    # Observability
    "EventType",
    "FileTransport",
    "HttpTransport",
    "InMemoryTransport",
    "ObservabilityConfig",
    "TraceEvent",
    "TraceTransport",
    # Prompt utilities
    "PromptSections",
    "extract_context",
    "split_prompt",
    # Prompt section markers
    "PROMPT_CONTEXT_BEGIN",
    "PROMPT_CONTEXT_END",
    "PROMPT_DEFINITIONS_BEGIN",
    "PROMPT_DEFINITIONS_END",
    "PROMPT_INSTRUCTIONS_BEGIN",
    "PROMPT_INSTRUCTIONS_END",
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
