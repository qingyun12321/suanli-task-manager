from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from fastapi import HTTPException

from suanli_task_manager.adapters.types import RuntimePending
from suanli_task_manager.config import ProjectConfig
from suanli_task_manager.models import FileArtifact, TaskRecord

ArtifactWriter = Callable[[TaskRecord, str, bytes, str], FileArtifact]


def _stringify_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class ProjectAdapter(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def validate_request(
        self,
        input_payload: dict[str, Any],
        parameter_payload: dict[str, Any],
        uploads: list[Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def dispatch(
        self,
        service_url: str,
        task: TaskRecord,
        client: Any,
        project: ProjectConfig,
        artifact_writer: ArtifactWriter,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def poll(
        self,
        service_url: str,
        runtime_request_id: str,
        task: TaskRecord,
        client: Any,
        project: ProjectConfig,
        artifact_writer: ArtifactWriter,
    ) -> dict[str, Any] | RuntimePending:
        raise HTTPException(status_code=500, detail=f"Adapter does not support polling: {self.name}")

    def _ensure_file_count(
        self,
        uploads: list[Any],
        *,
        minimum: int = 0,
        maximum: int | None = None,
        detail: str = "Invalid upload count",
    ) -> None:
        if len(uploads) < minimum:
            raise HTTPException(status_code=400, detail=detail)
        if maximum is not None and len(uploads) > maximum:
            raise HTTPException(status_code=400, detail=detail)
