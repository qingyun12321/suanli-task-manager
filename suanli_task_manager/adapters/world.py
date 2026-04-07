from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .base import ProjectAdapter, _stringify_scalar
from .types import RuntimePending


class WorldAdapter(ProjectAdapter):
    def __init__(self) -> None:
        super().__init__("world")

    def validate_request(self, input_payload: dict[str, Any], parameter_payload: dict[str, Any], uploads: list[Any]) -> dict[str, Any]:
        self._ensure_file_count(uploads, minimum=1, detail="At least one file is required")
        return {"input_payload": input_payload, "parameter_payload": parameter_payload}

    async def dispatch(self, service_url: str, task, client, project, artifact_writer):
        form_data: dict[str, str] = {"request_id": task.request_id}
        frame_selector = task.input_payload.get("frame_selector")
        if frame_selector is not None:
            form_data["frame_selector"] = _stringify_scalar(frame_selector)

        for key, value in task.parameter_payload.items():
            if value is None:
                continue
            form_data[key] = _stringify_scalar(value)

        files = []
        for upload in task.files:
            files.append(
                (
                    upload.field_name,
                    (
                        upload.filename,
                        Path(upload.path).read_bytes(),
                        upload.content_type,
                    ),
                )
            )

        response = await client.post(f"{service_url}/run_with_files", data=form_data, files=files)
        if not response.is_success:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = getattr(response, "text", "") or f"runtime HTTP {response.status_code}"
            raise HTTPException(status_code=502, detail=detail)
        try:
            payload = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=502, detail=f"Invalid runtime response: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Runtime response must be a JSON object")

        runtime_request_id = str(payload.get("request_id") or task.request_id).strip()
        if not runtime_request_id:
            raise HTTPException(status_code=502, detail="World runtime request_id missing")

        output = {
            "runtime_request_id": runtime_request_id,
            "runtime_task_status": "PENDING",
        }
        position = payload.get("position")
        if position is not None:
            output["runtime_queue_position"] = position
        return RuntimePending(
            runtime_request_id=runtime_request_id,
            message="任务已提交到运行时队列，等待处理完成。",
            output=output,
        )

    async def poll(self, service_url: str, runtime_request_id: str, task, client, project, artifact_writer):
        response = await client.get(
            f"{service_url}/request_status",
            params={"request_id": runtime_request_id},
        )
        if not response.is_success:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = getattr(response, "text", "") or f"runtime HTTP {response.status_code}"
            raise HTTPException(status_code=502, detail=detail)

        try:
            payload = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=502, detail=f"Invalid runtime response: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Runtime response must be a JSON object")

        status = str(payload.get("status") or "").strip().lower()
        if status == "pending":
            output = {
                "runtime_request_id": runtime_request_id,
                "runtime_task_status": "PENDING",
            }
            if payload.get("position") is not None:
                output["runtime_queue_position"] = payload["position"]
            return RuntimePending(
                runtime_request_id=runtime_request_id,
                message="任务正在等待运行时执行。",
                output=output,
                poll_interval_sec=max(project.dispatch_retry_interval_ms / 1000.0, 0.05),
            )

        if status == "processing":
            return RuntimePending(
                runtime_request_id=runtime_request_id,
                message="运行时正在处理中。",
                output={
                    "runtime_request_id": runtime_request_id,
                    "runtime_task_status": "RUNNING",
                },
                poll_interval_sec=max(project.dispatch_retry_interval_ms / 1000.0, 0.05),
            )

        if status == "failed":
            raise HTTPException(status_code=502, detail=str(payload.get("error") or "World task failed"))

        if status != "completed":
            raise HTTPException(status_code=502, detail=f"Unexpected world runtime status: {status or 'unknown'}")

        result = payload.get("result") or {}
        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="World runtime result must be an object")
        result = dict(result)
        result["runtime_request_id"] = runtime_request_id
        return result
