from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    SCALING = "SCALING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}


@dataclass(slots=True)
class StoredUpload:
    field_name: str
    filename: str
    content_type: str
    path: str
    size: int


@dataclass(slots=True)
class FileArtifact:
    name: str
    path: str
    content_type: str
    size: int
    url: str


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    request_id: str
    project: str
    model: str
    input_payload: dict[str, Any]
    parameter_payload: dict[str, Any]
    files: list[StoredUpload]
    workspace_dir: str
    submit_time: str
    task_status: TaskStatus = TaskStatus.PENDING
    scheduled_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    message: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    expires_at: float | None = None
    artifacts: dict[str, FileArtifact] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.task_status.is_terminal
