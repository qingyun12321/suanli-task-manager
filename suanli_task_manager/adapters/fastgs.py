from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import HTTPException

from .base import ProjectAdapter, _stringify_scalar


class FastGSAdapter(ProjectAdapter):
    def __init__(self) -> None:
        super().__init__("fastgs")

    def validate_request(self, input_payload: dict[str, Any], parameter_payload: dict[str, Any], uploads: list[Any]) -> dict[str, Any]:
        if not uploads:
            raise HTTPException(status_code=400, detail="FastGS requires at least one upload")
        return {"input_payload": input_payload, "parameter_payload": parameter_payload}

    async def dispatch(self, service_url: str, task, client, project, artifact_writer):
        session_resp = await client.post(f"{service_url}/create_session")
        if not session_resp.is_success:
            raise HTTPException(status_code=502, detail="Failed to create FastGS session")
        session_payload = session_resp.json()
        session_id = str(session_payload.get("session_id") or "").strip()
        if not session_id:
            raise HTTPException(status_code=502, detail="FastGS session_id missing")

        data: dict[str, str] = {
            "session_id": session_id,
            "request_id": task.request_id,
            "iterations": _stringify_scalar(task.parameter_payload.get("iterations", 30000)),
            "mult": _stringify_scalar(task.parameter_payload.get("mult", 0.5)),
            "white_background": _stringify_scalar(task.parameter_payload.get("white_background", False)),
            "eval": _stringify_scalar(task.parameter_payload.get("eval", False)),
            "data_device": _stringify_scalar(task.parameter_payload.get("data_device", "cuda")),
            "iteration": _stringify_scalar(task.parameter_payload.get("iteration", -1)),
            "video360": _stringify_scalar(task.parameter_payload.get("video360", False)),
            "use_dataset_cams": _stringify_scalar(task.parameter_payload.get("use_dataset_cams", False)),
            "use_test_cams": _stringify_scalar(task.parameter_payload.get("use_test_cams", False)),
            "interp_per_pair": _stringify_scalar(task.parameter_payload.get("interp_per_pair", 3)),
            "loop": _stringify_scalar(task.parameter_payload.get("loop", False)),
            "fps": _stringify_scalar(task.parameter_payload.get("fps", 30)),
            "frames": _stringify_scalar(task.parameter_payload.get("frames", 240)),
            "ease": _stringify_scalar(task.parameter_payload.get("ease", True)),
            "render_only": _stringify_scalar(task.parameter_payload.get("render_only", False)),
        }

        dataset_upload = None
        model_upload = None
        pose_upload = None
        for upload in task.files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix == ".json" and pose_upload is None:
                pose_upload = upload
            elif suffix == ".zip" and dataset_upload is None:
                dataset_upload = upload
            elif suffix == ".zip" and model_upload is None:
                model_upload = upload

        files: dict[str, tuple[str, bytes, str]] = {}
        if dataset_upload is not None:
            files["dataset_file"] = (
                dataset_upload.filename,
                Path(dataset_upload.path).read_bytes(),
                dataset_upload.content_type,
            )
        if model_upload is not None:
            files["model_file"] = (
                model_upload.filename,
                Path(model_upload.path).read_bytes(),
                model_upload.content_type,
            )
        if pose_upload is not None:
            files["pose_file"] = (
                pose_upload.filename,
                Path(pose_upload.path).read_bytes(),
                pose_upload.content_type,
            )

        run_resp = await client.post(f"{service_url}/run_with_files", data=data, files=files or None)
        if not run_resp.is_success:
            detail = getattr(run_resp, "text", "") or f"runtime HTTP {run_resp.status_code}"
            raise HTTPException(status_code=502, detail=detail)

        poll_interval = max(project.dispatch_retry_interval_ms / 1000.0, 0.01)
        while True:
            status_resp = await client.get(
                f"{service_url}/request_status?{urlencode({'request_id': task.request_id})}"
            )
            if not status_resp.is_success:
                raise HTTPException(status_code=502, detail="Failed to poll FastGS request_status")
            payload = status_resp.json()
            status = str(payload.get("status") or "").lower()
            if status == "completed":
                result = dict(payload.get("result") or {})
                result["session_id"] = result.get("session_id") or session_id
                return result
            if status == "failed":
                raise HTTPException(status_code=502, detail=str(payload.get("error") or "FastGS task failed"))
            await asyncio.sleep(poll_interval)
