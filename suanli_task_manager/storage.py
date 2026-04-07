from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from suanli_task_manager.config import AppConfig
from suanli_task_manager.models import FileArtifact, StoredUpload, TaskRecord, TaskStatus


@dataclass(slots=True)
class UploadPayload:
    field_name: str
    filename: str
    content_type: str
    content: bytes


class TaskStore:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tasks: dict[str, TaskRecord] = {}
        self._root = config.server.temp_root
        self._root.mkdir(parents=True, exist_ok=True)

    def create_task(
        self,
        *,
        project: str,
        model: str,
        request_id: str,
        input_payload: dict[str, Any],
        parameter_payload: dict[str, Any],
        uploads: list[UploadPayload],
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex
        workspace_dir = Path(tempfile.mkdtemp(prefix=f"{task_id}-", dir=self._root))
        input_dir = workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        stored_uploads: list[StoredUpload] = []
        for index, upload in enumerate(uploads):
            target = input_dir / f"{index}-{Path(upload.filename).name}"
            target.write_bytes(upload.content)
            stored_uploads.append(
                StoredUpload(
                    field_name=upload.field_name,
                    filename=upload.filename,
                    content_type=upload.content_type,
                    path=str(target),
                    size=len(upload.content),
                )
            )

        record = TaskRecord(
            task_id=task_id,
            request_id=request_id,
            project=project,
            model=model,
            input_payload=dict(input_payload or {}),
            parameter_payload=dict(parameter_payload or {}),
            files=stored_uploads,
            workspace_dir=str(workspace_dir),
            submit_time=self._utc_now_str(),
        )
        self._tasks[task_id] = record
        return record

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus, *, message: str = "") -> TaskRecord:
        task = self._tasks[task_id]
        task.task_status = status
        task.message = message
        if status == TaskStatus.RUNNING:
            task.start_time = task.start_time or self._utc_now_str()
            task.scheduled_time = task.scheduled_time or self._utc_now_str()
        return task

    def mark_completed(self, task_id: str, output: dict[str, Any]) -> TaskRecord:
        task = self._tasks[task_id]
        task.task_status = TaskStatus.SUCCEEDED
        task.output = output
        task.message = ""
        task.end_time = self._utc_now_str()
        task.expires_at = time.time() + self._ttl_for(task.project)
        self.cleanup_inputs(task_id)
        return task

    def mark_failed(self, task_id: str, message: str) -> TaskRecord:
        task = self._tasks[task_id]
        task.task_status = TaskStatus.FAILED
        task.message = message
        task.end_time = self._utc_now_str()
        task.expires_at = time.time() + self._ttl_for(task.project)
        self.cleanup_inputs(task_id)
        return task

    def update_runtime_progress(
        self,
        task_id: str,
        *,
        message: str | None = None,
        output_updates: dict[str, Any] | None = None,
    ) -> TaskRecord:
        task = self._tasks[task_id]
        if task.is_terminal:
            return task
        if message is not None:
            task.message = message
        if output_updates:
            merged = dict(task.output)
            merged.update(output_updates)
            task.output = merged
        return task

    def cleanup_inputs(self, task_id: str) -> None:
        task = self._tasks[task_id]
        for upload in task.files:
            path = Path(upload.path)
            if path.exists():
                path.unlink()

    def write_artifact(self, task: TaskRecord, filename: str, content: bytes, content_type: str) -> FileArtifact:
        artifact_dir = Path(task.workspace_dir) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or f"{uuid.uuid4().hex}.bin"
        target = artifact_dir / safe_name
        target.write_bytes(content)
        url = f"/api/v1/tasks/{task.task_id}/artifacts/{safe_name}"
        artifact = FileArtifact(
            name=safe_name,
            path=str(target),
            content_type=content_type,
            size=len(content),
            url=url,
        )
        task.artifacts[safe_name] = artifact
        return artifact

    def get_artifact(self, task_id: str, artifact_name: str) -> FileArtifact | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return task.artifacts.get(artifact_name)

    def prune_expired(self, *, now: float | None = None) -> list[str]:
        timestamp = now if now is not None else time.time()
        pruned: list[str] = []
        for task_id, task in list(self._tasks.items()):
            if not task.is_terminal:
                continue
            if task.expires_at is None or task.expires_at > timestamp:
                continue
            workspace = Path(task.workspace_dir)
            if workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)
            self._tasks.pop(task_id, None)
            pruned.append(task_id)
        return pruned

    @staticmethod
    def format_task_output(task: TaskRecord) -> dict[str, Any]:
        payload = {
            "task_id": task.task_id,
            "task_status": task.task_status.value,
            "submit_time": task.submit_time,
            "scheduled_time": task.scheduled_time,
            "start_time": task.start_time,
            "end_time": task.end_time,
        }
        payload.update(task.output)
        return payload

    @staticmethod
    def _utc_now_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _ttl_for(self, project_name: str) -> int:
        project = self._config.projects[project_name]
        return int(project.task_ttl_sec)
