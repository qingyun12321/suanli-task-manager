from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from suanli_task_manager.models import FileArtifact


@dataclass(slots=True)
class RuntimeResponse:
    payload: dict[str, Any]


@dataclass(slots=True)
class RuntimePending:
    runtime_request_id: str
    message: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    poll_interval_sec: float | None = None


__all__ = ["FileArtifact", "RuntimePending", "RuntimeResponse"]
