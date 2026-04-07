from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .base import ProjectAdapter, _stringify_scalar


_FILENAME_RE = re.compile(r'filename="([^"]+)"')


class LightX2VAdapter(ProjectAdapter):
    def __init__(self) -> None:
        super().__init__("lightx2v")

    def validate_request(self, input_payload: dict[str, Any], parameter_payload: dict[str, Any], uploads: list[Any]) -> dict[str, Any]:
        self._ensure_file_count(uploads, maximum=1, detail="LightX2V supports at most one image upload")
        prompt = str(parameter_payload.get("prompt") or input_payload.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        return {
            "input_payload": input_payload,
            "parameter_payload": parameter_payload,
        }

    async def dispatch(self, service_url: str, task, client, project, artifact_writer):
        data = {
            "prompt": str(task.parameter_payload.get("prompt") or task.input_payload.get("prompt") or ""),
            "request_id": task.request_id,
        }
        for key in ("negative_prompt", "seed"):
            value = task.parameter_payload.get(key, task.input_payload.get(key))
            if value not in (None, ""):
                data[key] = _stringify_scalar(value)

        files = None
        if task.files:
            upload = task.files[0]
            files = {
                "image": (
                    upload.filename,
                    Path(upload.path).read_bytes(),
                    upload.content_type,
                )
            }

        response = await client.post(f"{service_url}/generate", data=data, files=files)
        if not response.is_success:
            detail = getattr(response, "text", "") or f"runtime HTTP {response.status_code}"
            raise HTTPException(status_code=502, detail=detail)

        content_type = str(response.headers.get("content-type") or "application/octet-stream")
        disposition = str(response.headers.get("content-disposition") or "")
        match = _FILENAME_RE.search(disposition)
        filename = match.group(1) if match else "result.mp4"
        artifact = artifact_writer(task, filename, response.content, content_type)
        return {
            "artifacts": {
                "video": {
                    "url": artifact.url,
                    "content_type": artifact.content_type,
                    "size": artifact.size,
                }
            }
        }
