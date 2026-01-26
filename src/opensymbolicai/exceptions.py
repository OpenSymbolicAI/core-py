"""Custom exceptions for primitive operations.

This module provides a hierarchy of exceptions that can be used in primitives
to stop execution with detailed error information.
"""

from __future__ import annotations

from typing import Any


class ExecutionError(Exception):
    """Base exception for all primitive execution errors.

    This is the root of the exception hierarchy. All custom exceptions
    should inherit from this class. When raised in a primitive, execution
    will stop and the error details will be captured in the execution trace.

    Attributes:
        message: Human-readable error message.
        code: Optional error code for programmatic handling.
        details: Optional dictionary with additional context.
        halt_execution: Whether this exception should stop plan execution.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        halt_execution: bool = True,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            code: Optional error code for programmatic handling (e.g., "INVALID_INPUT").
            details: Optional dictionary with additional context about the error.
            halt_execution: Whether this exception should stop plan execution.
                           Defaults to True.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.halt_execution = halt_execution

    def __str__(self) -> str:
        """Return string representation of the exception."""
        parts = [self.message]
        if self.code:
            parts.insert(0, f"[{self.code}]")
        return " ".join(parts)

    def __repr__(self) -> str:
        """Return detailed representation of the exception."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r}, "
            f"halt_execution={self.halt_execution!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to a dictionary for serialization.

        Returns:
            Dictionary containing all exception attributes.
        """
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
            "halt_execution": self.halt_execution,
        }


class ValidationError(ExecutionError):
    """Exception raised when input validation fails.

    Use this for invalid arguments, out-of-range values, or
    constraint violations in primitive inputs.

    Example:
        if number <= 0:
            raise ValidationError(
                "Logarithm requires positive input",
                code="INVALID_INPUT",
                details={"received": number, "expected": "positive number"}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
        field: str | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Human-readable error message.
            code: Error code, defaults to "VALIDATION_ERROR".
            details: Additional context about the error.
            field: Optional name of the field that failed validation.
        """
        if field and details is None:
            details = {}
        if field:
            details = details or {}
            details["field"] = field
        super().__init__(message, code=code, details=details)
        self.field = field


class PreconditionError(ExecutionError):
    """Exception raised when a precondition for an operation is not met.

    Use this when the operation cannot proceed due to missing prerequisites
    or invalid state (e.g., division by zero, empty collection).

    Example:
        if divisor == 0:
            raise PreconditionError(
                "Cannot divide by zero",
                code="DIVISION_BY_ZERO",
                details={"dividend": dividend, "divisor": divisor}
            )
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "PRECONDITION_FAILED",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize precondition error.

        Args:
            message: Human-readable error message.
            code: Error code, defaults to "PRECONDITION_FAILED".
            details: Additional context about the error.
        """
        super().__init__(message, code=code, details=details)


class ResourceError(ExecutionError):
    """Exception raised when a required resource is unavailable.

    Use this for missing files, network failures, or unavailable
    external services.

    Example:
        raise ResourceError(
            "Database connection failed",
            code="DB_UNAVAILABLE",
            details={"host": "localhost", "port": 5432}
        )
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "RESOURCE_ERROR",
        details: dict[str, Any] | None = None,
        resource: str | None = None,
    ) -> None:
        """Initialize resource error.

        Args:
            message: Human-readable error message.
            code: Error code, defaults to "RESOURCE_ERROR".
            details: Additional context about the error.
            resource: Optional identifier for the unavailable resource.
        """
        if resource:
            details = details or {}
            details["resource"] = resource
        super().__init__(message, code=code, details=details)
        self.resource = resource


class OperationError(ExecutionError):
    """Exception raised when an operation fails during execution.

    Use this for runtime errors that occur during the actual operation,
    not input validation.

    Example:
        raise OperationError(
            "File write failed",
            code="WRITE_FAILED",
            details={"path": "/tmp/output.txt", "reason": "disk full"}
        )
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "OPERATION_ERROR",
        details: dict[str, Any] | None = None,
        operation: str | None = None,
    ) -> None:
        """Initialize operation error.

        Args:
            message: Human-readable error message.
            code: Error code, defaults to "OPERATION_ERROR".
            details: Additional context about the error.
            operation: Optional name of the operation that failed.
        """
        if operation:
            details = details or {}
            details["operation"] = operation
        super().__init__(message, code=code, details=details)
        self.operation = operation


class RetryableError(ExecutionError):
    """Exception raised for errors that may succeed on retry.

    Use this for transient failures like network timeouts or rate limits.
    By default, this does NOT halt execution to allow potential recovery.

    Example:
        raise RetryableError(
            "API rate limit exceeded",
            code="RATE_LIMIT",
            details={"retry_after_seconds": 60}
        )
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = "RETRYABLE_ERROR",
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        """Initialize retryable error.

        Args:
            message: Human-readable error message.
            code: Error code, defaults to "RETRYABLE_ERROR".
            details: Additional context about the error.
            retry_after: Optional seconds to wait before retrying.
        """
        if retry_after is not None:
            details = details or {}
            details["retry_after_seconds"] = retry_after
        super().__init__(message, code=code, details=details, halt_execution=False)
        self.retry_after = retry_after
