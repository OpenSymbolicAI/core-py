"""HTTP transport that batches events and sends them to a collector."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from queue import Empty, Queue

from opensymbolicai.observability.events import TraceEvent


class HttpTransport:
    """Batched HTTP transport using stdlib urllib.

    Events are queued in memory and flushed by a background thread
    either when the batch size is reached or on ``close()``.

    On HTTP failure the batch is dropped with a warning to stderr.
    No retries — keep it simple.

    Args:
        url: Collector endpoint (e.g. ``http://localhost:8100/events``).
        batch_size: Flush after this many events are queued.
        flush_interval_seconds: Periodic flush interval.
        headers: Extra HTTP headers sent with every request
            (e.g. ``{"X-API-Key": "os-…-am"}``).
    """

    def __init__(
        self,
        url: str,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._extra_headers = headers or {}
        self._queue: Queue[TraceEvent] = Queue()
        self._closed = False
        self._lock = threading.Lock()

        # Background flusher thread
        self._thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="observability-http"
        )
        self._thread.start()

    def send(self, events: list[TraceEvent]) -> None:
        """Queue events for batched sending."""
        if self._closed:
            return
        for event in events:
            self._queue.put(event)

    def flush(self) -> None:
        """Flush any buffered events without closing the transport."""
        self._flush_batch()

    def close(self) -> None:
        """Flush remaining events and stop the background thread."""
        self._closed = True
        self._flush_batch()
        self._thread.join(timeout=5.0)

    def _flush_loop(self) -> None:
        """Background loop that flushes batches periodically."""
        while not self._closed:
            try:
                event = self._queue.get(timeout=self._flush_interval)
                # Re-queue so _flush_batch drains it along with everything else.
                self._queue.put(event)
            except Empty:
                pass
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Drain the queue and send all buffered events."""
        batch: list[TraceEvent] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break

        if not batch:
            return

        self._send_http(batch)

    def _send_http(self, events: list[TraceEvent]) -> None:
        """POST a batch of events to the collector."""
        body = json.dumps(
            {"events": [e.model_dump(mode="json") for e in events]}
        ).encode("utf-8")

        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json", **self._extra_headers},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(
                f"[opensymbolicai.observability] Failed to send {len(events)} events: {e}",
                file=sys.stderr,
            )
