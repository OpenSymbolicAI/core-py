"""Anonymous usage telemetry for OpenSymbolicAI.

Sends lightweight, anonymous events to PostHog so we can understand
adoption and prioritise development.  **No prompts, responses, API keys,
or personal data are ever collected.**

Opt out by setting either environment variable:
    OPENSYMBOLICAI_TELEMETRY_DISABLED=1
    DO_NOT_TRACK=1
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import urllib.error
import urllib.request
import uuid

from pydantic import BaseModel, Field

import opensymbolicai

_POSTHOG_HOST = "https://us.i.posthog.com"
_POSTHOG_API_KEY = "phc_81jX7SpPZZxQkQX8Axxg8jKKwek0eT16DOG1KEkBDg8"


# ── Models ───────────────────────────────────────────────────────────────────


class TelemetryProperties(BaseModel):
    """Properties attached to every telemetry event."""

    framework_version: str = Field(default_factory=lambda: opensymbolicai.__version__)
    python_version: str = Field(default_factory=platform.python_version)
    os: str = Field(default_factory=lambda: sys.platform)

    # Agent-specific (populated per event)
    blueprint: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    success: bool | None = None
    primitive_count: int | None = None
    iterations: int | None = None


class TelemetryEvent(BaseModel):
    """PostHog /capture payload."""

    api_key: str = _POSTHOG_API_KEY
    event: str
    distinct_id: str = Field(default_factory=lambda: _anonymous_id())
    properties: TelemetryProperties = Field(default_factory=TelemetryProperties)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_disabled() -> bool:
    return os.environ.get("OPENSYMBOLICAI_TELEMETRY_DISABLED", "").strip() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("DO_NOT_TRACK", "").strip() in ("1", "true", "yes")


def _anonymous_id() -> str:
    """SHA-256 hash of the machine's MAC address. Not reversible."""
    mac = str(uuid.getnode())
    return hashlib.sha256(mac.encode()).hexdigest()


def _post(payload: TelemetryEvent) -> None:
    """POST JSON to PostHog. Swallows all errors."""
    try:
        body = json.dumps(
            payload.model_dump(mode="json", exclude_none=True)
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{_POSTHOG_HOST}/capture/",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            resp.read()
    except Exception:  # noqa: BLE001
        pass


# ── Public API ───────────────────────────────────────────────────────────────


def record_event(
    event: str, properties: TelemetryProperties | None = None
) -> None:
    """Fire-and-forget telemetry event.

    Runs in a daemon thread so it never blocks the caller.
    Silently no-ops when telemetry is disabled or on any failure.
    """
    if _is_disabled():
        return

    try:
        payload = TelemetryEvent(
            event=event,
            properties=properties or TelemetryProperties(),
        )
        thread = threading.Thread(target=_post, args=(payload,), daemon=True)
        thread.start()
    except Exception:  # noqa: BLE001
        pass
