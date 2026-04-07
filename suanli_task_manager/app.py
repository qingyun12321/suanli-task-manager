from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from suanli_task_manager.adapters import build_registry
from suanli_task_manager.config import AppConfig, load_config
from suanli_task_manager.platform import PlatformClient
from suanli_task_manager.service import TaskManagerService
from suanli_task_manager.storage import UploadPayload


def create_app(
    config: AppConfig,
    *,
    platform_client: PlatformClient | None = None,
    adapter_registry: dict[str, Any] | None = None,
) -> FastAPI:
    service = TaskManagerService(
        config=config,
        platform_client=platform_client,
        adapter_registry=adapter_registry or build_registry(),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(title="Suanli Task Manager", version="3.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _require_api_key(authorization: str | None) -> None:
        if not config.auth_enabled:
            return
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            raise HTTPException(status_code=401, detail="Authorization must use Bearer token")
        token = authorization[len(prefix):].strip()
        if token not in config.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _read_uploads(files: list[UploadFile]) -> list[UploadPayload]:
        payloads: list[UploadPayload] = []
        for upload in files:
            content = await upload.read()
            if not content:
                continue
            payloads.append(
                UploadPayload(
                    field_name="files",
                    filename=upload.filename or "upload.bin",
                    content_type=upload.content_type or "application/octet-stream",
                    content=content,
                )
            )
        return payloads

    @app.get("/api/projects")
    async def list_projects(authorization: str | None = Header(default=None)):
        _require_api_key(authorization)
        return {"projects": service.list_projects()}

    @app.post("/api/v1/services/aigc/3d-generation/reconstruction")
    async def create_reconstruction_task(
        request: str = Form(...),
        files: list[UploadFile] | None = File(None),
        authorization: str | None = Header(default=None),
    ):
        _require_api_key(authorization)
        try:
            payload = json.loads(request)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request must be a JSON object")
        model = str(payload.get("model") or "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        input_payload = payload.get("input") or {}
        parameter_payload = payload.get("parameters") or {}
        if not isinstance(input_payload, dict):
            raise HTTPException(status_code=400, detail="input must be an object")
        if not isinstance(parameter_payload, dict):
            raise HTTPException(status_code=400, detail="parameters must be an object")
        uploads = await _read_uploads(files or [])
        return await service.create_task(
            model=model,
            input_payload=input_payload,
            parameter_payload=parameter_payload,
            uploads=uploads,
        )

    @app.get("/api/v1/tasks/{task_id}")
    async def get_task(task_id: str, authorization: str | None = Header(default=None)):
        _require_api_key(authorization)
        return service.get_task(task_id)

    @app.get("/api/v1/tasks/{task_id}/artifacts/{artifact_name}")
    async def get_task_artifact(task_id: str, artifact_name: str, authorization: str | None = Header(default=None)):
        _require_api_key(authorization)
        artifact = service.store.get_artifact(task_id, artifact_name)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(artifact.path, media_type=artifact.content_type, filename=artifact.name)

    @app.post("/api/task/recover")
    async def recover_task(req: ProjectRequest, authorization: str | None = Header(default=None)):
        _require_api_key(authorization)
        project_name = config.resolve_project_name(req.project)
        return await service.recover_project(project_name)

    @app.post("/api/task/pause")
    async def pause_task(req: ProjectRequest, authorization: str | None = Header(default=None)):
        _require_api_key(authorization)
        project_name = config.resolve_project_name(req.project)
        return await service.pause_project(project_name)

    @app.get("/api/task/status")
    async def task_status(project: str, authorization: str | None = Header(default=None)):
        _require_api_key(authorization)
        project_name = config.resolve_project_name(project)
        return await service.project_status(project_name)

    app.state.task_manager_service = service
    return app


class ProjectRequest(BaseModel):
    project: str


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
