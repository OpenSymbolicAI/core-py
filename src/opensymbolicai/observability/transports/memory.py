"""In-memory transport for testing and in-process inspection."""

from __future__ import annotations

from opensymbolicai.observability.events import TraceEvent


class InMemoryTransport:
    """Stores trace events in memory.

    Useful for testing and for inspecting events in the same process.
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def send(self, events: list[TraceEvent]) -> None:
        """Append events to the in-memory list."""
        self.events.extend(events)

    def close(self) -> None:
        """No-op for in-memory transport."""
