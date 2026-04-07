from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .base import _stringify_scalar
from .types import RuntimePending
from .world import WorldAdapter


class UltraShapeAdapter(WorldAdapter):
    _PRECISION_OPTIONS = {"fast", "balanced", "quality"}
    _PARAMS = {
        "precision",
        "steps",
        "octree_res",
        "num_latents",
        "chunk_size",
        "seed",
        "remove_bg",
        "scale",
    }

    def __init__(self) -> None:
        super().__init__()
        self.name = "ultrashape"

    def validate_request(self, input_payload: dict[str, Any], parameter_payload: dict[str, Any], uploads: list[Any]) -> dict[str, Any]:
        self._ensure_file_count(uploads, minimum=1, maximum=1, detail="Exactly one image is required")
        upload = uploads[0]
        if not str(upload.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported")
        cleaned = {
            key: value
            for key, value in parameter_payload.items()
            if key in self._PARAMS or key in {"time_interval", "show_camera", "show_mesh", "filter_sky_bg", "filter_ambiguous"}
        }
        precision = cleaned.get("precision")
        if precision is None or str(precision).strip() == "":
            cleaned["precision"] = "balanced"
        else:
            normalized_precision = str(precision).strip().lower()
            if normalized_precision not in self._PRECISION_OPTIONS:
                choices = ", ".join(sorted(self._PRECISION_OPTIONS))
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid precision. Supported values: {choices}",
                )
            cleaned["precision"] = normalized_precision
        return {"input_payload": input_payload, "parameter_payload": cleaned}

    async def dispatch(self, service_url: str, task, client, project, artifact_writer):
        form_data: dict[str, str] = {"request_id": task.request_id}
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
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Invalid runtime response: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Runtime response must be a JSON object")

        runtime_request_id = str(payload.get("request_id") or task.request_id).strip()
        if not runtime_request_id:
            raise HTTPException(status_code=502, detail="kokoni-shape runtime request_id missing")

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
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Invalid runtime response: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Runtime response must be a JSON object")

        status = str(payload.get("status") or "").strip().lower()
        if status in {"pending", "processing"}:
            return RuntimePending(
                runtime_request_id=runtime_request_id,
                message="运行时正在处理中。" if status == "processing" else "任务正在等待运行时执行。",
                output={
                    "runtime_request_id": runtime_request_id,
                    "runtime_task_status": "RUNNING" if status == "processing" else "PENDING",
                },
                poll_interval_sec=max(project.dispatch_retry_interval_ms / 1000.0, 0.05),
            )

        if status == "failed":
            raise HTTPException(status_code=502, detail=str(payload.get("error") or "kokoni-shape task failed"))

        if status != "completed":
            raise HTTPException(status_code=502, detail=f"Unexpected kokoni-shape runtime status: {status or 'unknown'}")

        result = payload.get("result") or {}
        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="kokoni-shape runtime result must be an object")
        result = dict(result)
        result["runtime_request_id"] = runtime_request_id
        return result
