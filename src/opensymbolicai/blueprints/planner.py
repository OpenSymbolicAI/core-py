"""Planner blueprint: abstract base for plan-and-execute agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opensymbolicai.models import ExecutionResult, OrchestrationResult, PlanResult


class Planner(ABC):
    """Abstract base class for agents that plan and execute tasks.

    The Planner defines the interface for agents that:
    1. Generate plans from natural language tasks
    2. Execute those plans step-by-step
    3. Return structured results with execution traces

    Subclasses must implement:
    - plan(): Generate a plan for a given task
    - execute(): Execute a plan and return results
    - run(): Complete plan-and-execute cycle
    """

    @abstractmethod
    def plan(self, task: str) -> PlanResult:
        """Generate a plan (Python statements) for a task.

        Args:
            task: The task description to plan for.

        Returns:
            PlanResult containing the generated plan and metrics.
        """
        ...

    @abstractmethod
    def execute(self, plan: str) -> ExecutionResult:
        """Execute a plan step-by-step.

        Args:
            plan: The Python statements to execute.

        Returns:
            ExecutionResult containing the final value and execution trace.
        """
        ...

    @abstractmethod
    def run(self, task: str) -> OrchestrationResult:
        """Run the complete plan-and-execute cycle.

        Args:
            task: The task description to accomplish.

        Returns:
            OrchestrationResult containing the outcome and metrics.
        """
        ...
