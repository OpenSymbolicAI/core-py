"""File-based transport that writes JSONL."""

from __future__ import annotations

from pathlib import Path
from typing import IO

from opensymbolicai.observability.events import TraceEvent


class FileTransport:
    """Appends events as newline-delimited JSON to a file.

    Each event is written as a single JSON line (JSONL format).

    Args:
        path: File path to write to. Parent directories are created if needed.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] | None = None

    def _ensure_open(self) -> IO[str]:
        if self._file is None or self._file.closed:
            self._file = open(self._path, "a")  # noqa: SIM115
        return self._file

    def send(self, events: list[TraceEvent]) -> None:
        """Append events as JSONL to the file."""
        f = self._ensure_open()
        for event in events:
            line = event.model_dump_json()
            f.write(line + "\n")
        f.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None and not self._file.closed:
            self._file.close()
