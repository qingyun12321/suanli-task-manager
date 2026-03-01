"""
suanli-task-manager -- Lightweight proxy for the Suanli compute-platform OpenAPI.

Keeps all sensitive credentials (tokens, task IDs) server-side so that
browser-based front-ends never touch them.  Designed for multi-project use:
each project entry in config.yaml maps a human-readable name to its platform
task configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"

_RECOVER_ALREADY_RUNNING = re.compile(
    r"already|running|active|运行中|已运行|无需恢复|未暂停|must.*paused|任务必须为暂停中",
    re.IGNORECASE,
)
_PAUSE_ALREADY_PAUSED = re.compile(
    r"already|paused|inactive|stopped|已暂停|无需暂停|must.*running|任务必须为运行中",
    re.IGNORECASE,
)
_RUNNING_STATUS = re.compile(
    r"running|ready|online|available|success|active|启动中|运行中",
    re.IGNORECASE,
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
OPENAPI_BASE: str = cfg.get("openapi", {}).get("base", "https://openapi.suanli.cn").rstrip("/")
OPENAPI_VERSION: str = cfg.get("openapi", {}).get("version", "1.0.0")
PROJECTS: dict[str, dict] = cfg.get("projects", {})

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Suanli Task Manager", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectRequest(BaseModel):
    project: str


# ---------------------------------------------------------------------------
# OpenAPI helpers
# ---------------------------------------------------------------------------
def _get_project(name: str) -> dict:
    proj = PROJECTS.get(name)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Unknown project: {name}")
    return proj


def _build_headers(token: str, *, content_type: bool = True) -> dict[str, str]:
    headers: dict[str, str] = {
        "token": token,
        "timestamp": str(int(time.time() * 1000)),
        "version": OPENAPI_VERSION,
    }
    if content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _normalize_code(code: Any) -> str:
    return str("" if code is None else code).strip().lower()


def _is_api_success(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    code = _normalize_code(payload.get("code"))
    return code in {"0000", "0", "200", "ok", "success"}


def _extract_message(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return ""
    msg = payload.get("message") or payload.get("msg") or payload.get("detail") or payload.get("error") or ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, (dict, list)):
        import json
        return json.dumps(msg, ensure_ascii=False)
    return str(msg or "")


def _extract_task_status(detail_payload: dict | None) -> str:
    if not isinstance(detail_payload, dict):
        return ""
    data = detail_payload.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("status") or data.get("task_status") or data.get("taskStatus") or "").strip()


def _is_running_status(status: str) -> bool:
    return bool(_RUNNING_STATUS.search(status or ""))


def _extract_service_url(detail_payload: dict | None, service_port: int) -> str:
    if not isinstance(detail_payload, dict):
        return ""
    data = detail_payload.get("data")
    if not isinstance(data, dict):
        return ""
    services = data.get("services") if isinstance(data.get("services"), list) else []
    fallback = ""
    for svc in services:
        remote_ports = svc.get("remote_ports") if isinstance(svc.get("remote_ports"), list) else []
        for port_item in remote_ports:
            url = str(port_item.get("url") or "").strip()
            if not url:
                continue
            if not fallback:
                fallback = url
            try:
                if int(port_item.get("service_port", 0)) == service_port:
                    return url.rstrip("/")
            except (ValueError, TypeError):
                pass
    return fallback.rstrip("/") if fallback else ""


# ---------------------------------------------------------------------------
# Core platform API calls
# ---------------------------------------------------------------------------
async def _call_task_detail(client: httpx.AsyncClient, proj: dict) -> dict:
    url = f"{OPENAPI_BASE}/api/deployment/task/detail"
    params = {"task_id": str(proj["task_id"])}
    headers = _build_headers(proj["token"], content_type=False)
    resp = await client.get(url, params=params, headers=headers)

    payload: dict | None = None
    try:
        payload = resp.json()
    except Exception:
        pass

    if not resp.is_success:
        detail = _extract_message(payload) or f"HTTP {resp.status_code}"
        raise HTTPException(status_code=502, detail=f"detail request failed: {detail}")

    if not _is_api_success(payload):
        msg = _extract_message(payload) or f"business code {payload.get('code') if payload else 'unknown'}"
        raise HTTPException(status_code=502, detail=f"detail failed: {msg}")

    return payload  # type: ignore[return-value]


async def _call_task_control(client: httpx.AsyncClient, proj: dict, action: str) -> dict:
    action_path = "recover" if action == "recover" else "pause"
    url = f"{OPENAPI_BASE}/api/deployment/task/{action_path}"
    headers = _build_headers(proj["token"])
    body = {"task_id": proj["task_id"]}
    resp = await client.post(url, json=body, headers=headers)

    payload: dict | None = None
    try:
        payload = resp.json()
    except Exception:
        pass

    if not resp.is_success:
        detail = _extract_message(payload) or f"HTTP {resp.status_code}"
        raise HTTPException(status_code=502, detail=f"{action_path} request failed: {detail}")

    if not _is_api_success(payload):
        msg = _extract_message(payload) or f"business code {payload.get('code') if payload else 'unknown'}"
        if action == "recover" and _RECOVER_ALREADY_RUNNING.search(msg):
            return payload  # type: ignore[return-value]
        if action == "pause" and _PAUSE_ALREADY_PAUSED.search(msg):
            return payload  # type: ignore[return-value]
        raise HTTPException(status_code=502, detail=f"{action_path} failed: {msg}")

    return payload  # type: ignore[return-value]


async def _poll_until_running(client: httpx.AsyncClient, proj: dict, timeout_sec: int) -> tuple[str, str]:
    """Poll task detail until status is running and service_url is available.

    Returns (status, service_url).
    """
    deadline = time.monotonic() + timeout_sec
    last_status = ""
    last_url = ""
    service_port = int(proj.get("service_port", 10085))

    while time.monotonic() < deadline:
        try:
            detail = await _call_task_detail(client, proj)
            status = _extract_task_status(detail)
            svc_url = _extract_service_url(detail, service_port)
            if status:
                last_status = status
            if svc_url:
                last_url = svc_url
            if _is_running_status(status) and svc_url:
                return status, svc_url
        except HTTPException:
            pass
        await asyncio.sleep(2)

    raise HTTPException(
        status_code=504,
        detail=f"Timeout waiting for task to run. last_status={last_status}, last_url={last_url}",
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/projects")
async def list_projects():
    return {"projects": list(PROJECTS.keys())}


@app.post("/api/task/recover")
async def recover_task(req: ProjectRequest):
    proj = _get_project(req.project)
    timeout_sec = int(proj.get("ready_timeout_sec", 120))
    service_port = int(proj.get("service_port", 10085))

    async with httpx.AsyncClient(timeout=30) as client:
        detail = await _call_task_detail(client, proj)
        status_before = _extract_task_status(detail)
        url_before = _extract_service_url(detail, service_port)

        recovered = False
        if _is_running_status(status_before) and url_before:
            return {
                "service_url": url_before,
                "status": status_before,
                "recovered": False,
            }

        await _call_task_control(client, proj, "recover")
        recovered = True

        status, service_url = await _poll_until_running(client, proj, timeout_sec)
        return {
            "service_url": service_url,
            "status": status,
            "recovered": recovered,
        }


@app.post("/api/task/pause")
async def pause_task(req: ProjectRequest):
    proj = _get_project(req.project)
    async with httpx.AsyncClient(timeout=30) as client:
        await _call_task_control(client, proj, "pause")
    return {"status": "paused"}


@app.get("/api/task/status")
async def task_status(project: str):
    proj = _get_project(project)
    service_port = int(proj.get("service_port", 10085))
    async with httpx.AsyncClient(timeout=30) as client:
        detail = await _call_task_detail(client, proj)
        status = _extract_task_status(detail)
        service_url = _extract_service_url(detail, service_port)
    return {
        "status": status,
        "service_url": service_url,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suanli Task Manager")
    server_cfg = cfg.get("server", {})
    parser.add_argument("--host", default=server_cfg.get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=server_cfg.get("port", 8090))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.config != str(CONFIG_PATH):
        cfg = load_config(Path(args.config))
        OPENAPI_BASE = cfg.get("openapi", {}).get("base", OPENAPI_BASE).rstrip("/")
        OPENAPI_VERSION = cfg.get("openapi", {}).get("version", OPENAPI_VERSION)
        PROJECTS = cfg.get("projects", {})
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
