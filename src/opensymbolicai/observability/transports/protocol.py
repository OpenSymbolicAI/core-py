"""Transport protocol for trace events."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from opensymbolicai.observability.events import TraceEvent


@runtime_checkable
class TraceTransport(Protocol):
    """Protocol for sending trace events to a backend.

    Implementations must provide ``send()`` and ``close()``.
    """

    def send(self, events: list[TraceEvent]) -> None:
        """Send a batch of trace events.

        Args:
            events: List of events to send.
        """
        ...

    def close(self) -> None:
        """Flush any buffered events and release resources."""
        ...
