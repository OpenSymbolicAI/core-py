"""Tests for the custom exception hierarchy."""

from __future__ import annotations

import pytest

from opensymbolicai.exceptions import (
    ExecutionError,
    OperationError,
    PreconditionError,
    ResourceError,
    RetryableError,
    ValidationError,
)


class TestExecutionError:
    """Tests for the base ExecutionError class."""

    def test_message_is_stored(self) -> None:
        exc = ExecutionError("Something went wrong")
        assert exc.message == "Something went wrong"

    def test_message_is_passed_to_base_exception(self) -> None:
        exc = ExecutionError("Something went wrong")
        assert str(exc) == "Something went wrong"

    def test_code_defaults_to_none(self) -> None:
        exc = ExecutionError("Error")
        assert exc.code is None

    def test_code_can_be_set(self) -> None:
        exc = ExecutionError("Error", code="ERR_001")
        assert exc.code == "ERR_001"

    def test_details_defaults_to_empty_dict(self) -> None:
        exc = ExecutionError("Error")
        assert exc.details == {}

    def test_details_can_be_set(self) -> None:
        exc = ExecutionError("Error", details={"key": "value"})
        assert exc.details == {"key": "value"}

    def test_halt_execution_defaults_to_true(self) -> None:
        exc = ExecutionError("Error")
        assert exc.halt_execution is True

    def test_halt_execution_can_be_set_to_false(self) -> None:
        exc = ExecutionError("Error", halt_execution=False)
        assert exc.halt_execution is False

    def test_str_includes_code_when_present(self) -> None:
        exc = ExecutionError("Something failed", code="ERR_001")
        assert str(exc) == "[ERR_001] Something failed"

    def test_str_without_code(self) -> None:
        exc = ExecutionError("Something failed")
        assert str(exc) == "Something failed"

    def test_repr_shows_all_attributes(self) -> None:
        exc = ExecutionError("Error", code="E1", details={"x": 1}, halt_execution=False)
        result = repr(exc)
        assert "ExecutionError" in result
        assert "message='Error'" in result
        assert "code='E1'" in result
        assert "details={'x': 1}" in result
        assert "halt_execution=False" in result

    def test_to_dict_returns_all_attributes(self) -> None:
        exc = ExecutionError("Error", code="E1", details={"x": 1}, halt_execution=False)
        result = exc.to_dict()
        assert result == {
            "type": "ExecutionError",
            "message": "Error",
            "code": "E1",
            "details": {"x": 1},
            "halt_execution": False,
        }

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(ExecutionError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(ExecutionError) as exc_info:
            raise ExecutionError("Test error", code="TEST")
        assert exc_info.value.code == "TEST"


class TestValidationError:
    """Tests for ValidationError."""

    def test_inherits_from_execution_error(self) -> None:
        assert issubclass(ValidationError, ExecutionError)

    def test_default_code_is_validation_error(self) -> None:
        exc = ValidationError("Invalid input")
        assert exc.code == "VALIDATION_ERROR"

    def test_custom_code_overrides_default(self) -> None:
        exc = ValidationError("Invalid input", code="CUSTOM_CODE")
        assert exc.code == "CUSTOM_CODE"

    def test_field_defaults_to_none(self) -> None:
        exc = ValidationError("Invalid input")
        assert exc.field is None

    def test_field_can_be_set(self) -> None:
        exc = ValidationError("Invalid input", field="email")
        assert exc.field == "email"

    def test_field_is_added_to_details(self) -> None:
        exc = ValidationError("Invalid input", field="email")
        assert exc.details["field"] == "email"

    def test_field_merges_with_existing_details(self) -> None:
        exc = ValidationError("Invalid", details={"expected": "int"}, field="age")
        assert exc.details == {"expected": "int", "field": "age"}

    def test_to_dict_shows_validation_error_type(self) -> None:
        exc = ValidationError("Invalid")
        assert exc.to_dict()["type"] == "ValidationError"


class TestPreconditionError:
    """Tests for PreconditionError."""

    def test_inherits_from_execution_error(self) -> None:
        assert issubclass(PreconditionError, ExecutionError)

    def test_default_code_is_precondition_failed(self) -> None:
        exc = PreconditionError("Precondition not met")
        assert exc.code == "PRECONDITION_FAILED"

    def test_custom_code_overrides_default(self) -> None:
        exc = PreconditionError("Division by zero", code="DIVISION_BY_ZERO")
        assert exc.code == "DIVISION_BY_ZERO"

    def test_details_can_include_context(self) -> None:
        exc = PreconditionError(
            "Cannot divide by zero",
            details={"dividend": 10, "divisor": 0},
        )
        assert exc.details == {"dividend": 10, "divisor": 0}


class TestResourceError:
    """Tests for ResourceError."""

    def test_inherits_from_execution_error(self) -> None:
        assert issubclass(ResourceError, ExecutionError)

    def test_default_code_is_resource_error(self) -> None:
        exc = ResourceError("Resource unavailable")
        assert exc.code == "RESOURCE_ERROR"

    def test_resource_defaults_to_none(self) -> None:
        exc = ResourceError("Resource unavailable")
        assert exc.resource is None

    def test_resource_can_be_set(self) -> None:
        exc = ResourceError("Database offline", resource="postgres")
        assert exc.resource == "postgres"

    def test_resource_is_added_to_details(self) -> None:
        exc = ResourceError("Database offline", resource="postgres")
        assert exc.details["resource"] == "postgres"

    def test_resource_merges_with_existing_details(self) -> None:
        exc = ResourceError(
            "Connection failed",
            details={"host": "localhost"},
            resource="db",
        )
        assert exc.details == {"host": "localhost", "resource": "db"}


class TestOperationError:
    """Tests for OperationError."""

    def test_inherits_from_execution_error(self) -> None:
        assert issubclass(OperationError, ExecutionError)

    def test_default_code_is_operation_error(self) -> None:
        exc = OperationError("Operation failed")
        assert exc.code == "OPERATION_ERROR"

    def test_operation_defaults_to_none(self) -> None:
        exc = OperationError("Operation failed")
        assert exc.operation is None

    def test_operation_can_be_set(self) -> None:
        exc = OperationError("Write failed", operation="file_write")
        assert exc.operation == "file_write"

    def test_operation_is_added_to_details(self) -> None:
        exc = OperationError("Write failed", operation="file_write")
        assert exc.details["operation"] == "file_write"


class TestRetryableError:
    """Tests for RetryableError."""

    def test_inherits_from_execution_error(self) -> None:
        assert issubclass(RetryableError, ExecutionError)

    def test_default_code_is_retryable_error(self) -> None:
        exc = RetryableError("Transient failure")
        assert exc.code == "RETRYABLE_ERROR"

    def test_halt_execution_defaults_to_false(self) -> None:
        exc = RetryableError("Transient failure")
        assert exc.halt_execution is False

    def test_retry_after_defaults_to_none(self) -> None:
        exc = RetryableError("Transient failure")
        assert exc.retry_after is None

    def test_retry_after_can_be_set(self) -> None:
        exc = RetryableError("Rate limited", retry_after=60.0)
        assert exc.retry_after == 60.0

    def test_retry_after_is_added_to_details(self) -> None:
        exc = RetryableError("Rate limited", retry_after=30.5)
        assert exc.details["retry_after_seconds"] == 30.5


class TestExceptionHierarchy:
    """Tests for the exception hierarchy and catching behavior."""

    def test_all_exceptions_caught_by_execution_error(self) -> None:
        exceptions = [
            ValidationError("test"),
            PreconditionError("test"),
            ResourceError("test"),
            OperationError("test"),
            RetryableError("test"),
        ]
        for exc in exceptions:
            with pytest.raises(ExecutionError):
                raise exc

    def test_all_exceptions_caught_by_base_exception(self) -> None:
        exceptions = [
            ExecutionError("test"),
            ValidationError("test"),
            PreconditionError("test"),
            ResourceError("test"),
            OperationError("test"),
            RetryableError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, Exception)

    def test_validation_error_not_caught_by_precondition_error(self) -> None:
        with pytest.raises(ValidationError):
            try:
                raise ValidationError("test")
            except PreconditionError:
                pytest.fail("Should not catch ValidationError as PreconditionError")
