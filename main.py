"""
suanli-task-manager -- Async facade and scaler for Suanli deployments.

This service exposes a Bailian-style async API to browsers, keeps platform
credentials server-side, and controls deployment recovery / pause / scaling
for long-running GPU inference tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
_BUSY_TEXT = re.compile(r"node\s+busy|busy|繁忙|忙", re.IGNORECASE)
_RUNTIME_RETRY_STATUS = {409, 429, 502, 503, 504}
_TERMINAL_TASK_STATUSES = {"SUCCEEDED", "FAILED"}


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
app = FastAPI(title="Suanli Task Manager", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectRequest(BaseModel):
    project: str


class RuntimeBusyError(Exception):
    """Raised when the runtime accepts only one job and is currently busy."""


@dataclass(slots=True)
class TaskFile:
    filename: str
    content_type: str
    content: bytes


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    request_id: str
    project: str
    model: str
    input_payload: dict[str, Any]
    parameter_payload: dict[str, Any]
    files: list[TaskFile]
    submit_time: str
    task_status: str = "PENDING"
    scheduled_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    message: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    scale_deadline: float | None = None


@dataclass(slots=True)
class ProjectRuntimeState:
    project: str
    service_url: str = ""
    desired_points: int = 0
    ready_capacity: int = 0
    occupied_slots: int = 0
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    pending_queue: deque[str] = field(default_factory=deque)
    recover_in_flight: bool = False
    scale_in_flight: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduler_task: asyncio.Task | None = None


PROJECT_STATES: dict[str, ProjectRuntimeState] = {
    name: ProjectRuntimeState(project=name) for name in PROJECTS
}
TASK_INDEX: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _utc_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _get_project(name: str) -> dict:
    proj = PROJECTS.get(name)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Unknown project: {name}")
    return proj


def _get_state(name: str) -> ProjectRuntimeState:
    state = PROJECT_STATES.get(name)
    if state is None:
        state = ProjectRuntimeState(project=name)
        PROJECT_STATES[name] = state
    return state


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


def _boolish(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _floatish(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _baillian_response(
    request_id: str,
    *,
    output: dict[str, Any],
    message: str = "",
    code: str | None = None,
    status_code: int = 200,
) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "request_id": request_id,
        "code": code,
        "message": message,
        "output": output,
    }


async def _save_uploads(files: list[UploadFile]) -> list[TaskFile]:
    saved: list[TaskFile] = []
    for f in files:
        name = (f.filename or "").strip()
        if not name:
            continue
        content = await f.read()
        if not content:
            continue
        saved.append(
            TaskFile(
                filename=name,
                content_type=(f.content_type or "application/octet-stream"),
                content=content,
            )
        )
    return saved


# ---------------------------------------------------------------------------
# Platform API helpers
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


async def _call_change_points(client: httpx.AsyncClient, proj: dict, points: int) -> dict:
    url = f"{OPENAPI_BASE}/api/deployment/task/change_points"
    headers = _build_headers(proj["token"])
    body = {"task_id": proj["task_id"], "points": points}
    resp = await client.post(url, json=body, headers=headers)

    payload: dict | None = None
    try:
        payload = resp.json()
    except Exception:
        pass

    if not resp.is_success:
        detail = _extract_message(payload) or f"HTTP {resp.status_code}"
        raise HTTPException(status_code=502, detail=f"change_points request failed: {detail}")

    if not _is_api_success(payload):
        msg = _extract_message(payload) or f"business code {payload.get('code') if payload else 'unknown'}"
        raise HTTPException(status_code=502, detail=f"change_points failed: {msg}")

    return payload  # type: ignore[return-value]


async def _wait_for_runtime_health(base_url: str, timeout_sec: int) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    async with httpx.AsyncClient(timeout=15) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.is_success:
                    return
                last_error = f"health HTTP {resp.status_code}"
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = str(exc)
            await asyncio.sleep(2)
    raise HTTPException(status_code=504, detail=f"Timeout waiting for runtime health: {last_error}")


async def _ensure_project_recovered(state: ProjectRuntimeState, proj: dict) -> str:
    timeout_sec = int(proj.get("ready_timeout_sec", 300))
    service_port = int(proj.get("service_port", 10085))

    async with httpx.AsyncClient(timeout=30) as client:
        detail = await _call_task_detail(client, proj)
        status_before = _extract_task_status(detail)
        url_before = _extract_service_url(detail, service_port)

        if not (_is_running_status(status_before) and url_before):
            await _call_task_control(client, proj, "recover")

        deadline = time.monotonic() + timeout_sec
        last_status = status_before
        last_url = url_before
        while time.monotonic() < deadline:
            detail = await _call_task_detail(client, proj)
            status = _extract_task_status(detail)
            service_url = _extract_service_url(detail, service_port)
            if status:
                last_status = status
            if service_url:
                last_url = service_url
            if _is_running_status(status) and service_url:
                await _wait_for_runtime_health(service_url.rstrip("/"), timeout_sec)
                state.service_url = service_url.rstrip("/")
                return state.service_url
            await asyncio.sleep(2)

    raise HTTPException(
        status_code=504,
        detail=f"Timeout waiting for task to run. last_status={last_status}, last_url={last_url}",
    )


async def _pause_project(state: ProjectRuntimeState, proj: dict) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        await _call_task_control(client, proj, "pause")
    state.service_url = ""
    state.desired_points = 0
    state.ready_capacity = 0


async def _set_project_points(state: ProjectRuntimeState, proj: dict, points: int) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        await _call_change_points(client, proj, points)
    state.desired_points = points
    if points < state.ready_capacity:
        state.ready_capacity = points


# ---------------------------------------------------------------------------
# Runtime dispatch
# ---------------------------------------------------------------------------
async def _runtime_reconstruct(state: ProjectRuntimeState, task: TaskRecord) -> dict[str, Any]:
    form_data = {
        "time_interval": str(_floatish(task.parameter_payload.get("time_interval"), 1.0)),
        "frame_selector": str(task.input_payload.get("frame_selector") or "All"),
        "show_camera": str(_boolish(task.parameter_payload.get("show_camera"), True)).lower(),
        "show_mesh": str(_boolish(task.parameter_payload.get("show_mesh"), True)).lower(),
        "filter_sky_bg": str(_boolish(task.parameter_payload.get("filter_sky_bg"), False)).lower(),
        "filter_ambiguous": str(_boolish(task.parameter_payload.get("filter_ambiguous"), True)).lower(),
        "request_id": task.request_id,
    }
    files = [
        ("files", (item.filename, item.content, item.content_type))
        for item in task.files
    ]

    timeout = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                f"{state.service_url}/reconstruct",
                data=form_data,
                files=files,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeBusyError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeBusyError(str(exc)) from exc

    if resp.status_code in _RUNTIME_RETRY_STATUS:
        detail = ""
        try:
            payload = resp.json()
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        except Exception:
            detail = resp.text
        if _BUSY_TEXT.search(detail or "") or resp.status_code in {409, 503}:
            raise RuntimeBusyError(detail or f"runtime busy: HTTP {resp.status_code}")
        raise HTTPException(status_code=502, detail=detail or f"runtime HTTP {resp.status_code}")

    if not resp.is_success:
        try:
            payload = resp.json()
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        except Exception:
            detail = resp.text
        raise HTTPException(status_code=502, detail=detail or f"runtime HTTP {resp.status_code}")

    try:
        payload = resp.json()
    except Exception as exc:  # pragma: no cover - upstream contract issue
        raise HTTPException(status_code=502, detail=f"Invalid runtime response: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="Runtime response must be a JSON object")
    return payload


def _format_task_output(task: TaskRecord) -> dict[str, Any]:
    output = {
        "task_id": task.task_id,
        "task_status": task.task_status,
        "submit_time": task.submit_time,
        "scheduled_time": task.scheduled_time,
        "start_time": task.start_time,
        "end_time": task.end_time,
    }
    output.update(task.output)
    return output


async def _mark_task_failed(
    state: ProjectRuntimeState,
    task: TaskRecord,
    proj: dict,
    message: str,
) -> None:
    task.task_status = "FAILED"
    task.message = message
    task.end_time = _utc_now_str()
    await _release_capacity_after_finish(state, proj)


async def _release_capacity_after_finish(state: ProjectRuntimeState, proj: dict) -> None:
    async with state.lock:
        if state.occupied_slots > 0:
            state.occupied_slots -= 1
        target_points = state.occupied_slots
        has_pending = bool(state.pending_queue)

    try:
        if target_points <= 0 and not has_pending:
            await _pause_project(state, proj)
        elif target_points > 0 and state.desired_points > target_points:
            await _set_project_points(state, proj, max(target_points, 1))
    finally:
        _ensure_scheduler(state)


async def _run_task_dispatch(state: ProjectRuntimeState, proj: dict, task_id: str) -> None:
    task = state.tasks[task_id]
    try:
        payload = await _runtime_reconstruct(state, task)
        task.output = payload
        task.task_status = "SUCCEEDED"
        task.message = ""
        task.end_time = _utc_now_str()
    except RuntimeBusyError:
        async with state.lock:
            if state.occupied_slots > 0:
                state.occupied_slots -= 1
            task.task_status = "SCALING"
            task.message = "节点冷启动中，等待可用容量。"
            if task_id not in state.pending_queue:
                state.pending_queue.appendleft(task_id)
        _ensure_scheduler(state)
        return
    except HTTPException as exc:
        await _mark_task_failed(state, task, proj, str(exc.detail))
        return
    except Exception as exc:  # pragma: no cover - defensive
        await _mark_task_failed(state, task, proj, f"Dispatch failed: {exc}")
        return

    await _release_capacity_after_finish(state, proj)


def _ensure_scheduler(state: ProjectRuntimeState) -> None:
    if state.scheduler_task and not state.scheduler_task.done():
        return
    state.scheduler_task = asyncio.create_task(_scheduler_loop(state.project))


async def _scheduler_loop(project_name: str) -> None:
    proj = _get_project(project_name)
    state = _get_state(project_name)

    while True:
        next_task: TaskRecord | None = None
        should_recover = False
        should_scale_to: int | None = None
        should_shrink_to: int | None = None
        should_pause = False
        now = time.monotonic()

        async with state.lock:
            while state.pending_queue:
                task_id = state.pending_queue[0]
                task = state.tasks.get(task_id)
                if task is None or task.task_status in _TERMINAL_TASK_STATUSES:
                    state.pending_queue.popleft()
                    continue
                if task.scale_deadline and task.scale_deadline < now:
                    state.pending_queue.popleft()
                    task.task_status = "FAILED"
                    task.message = "扩容超时/节点冷启动超时"
                    task.end_time = _utc_now_str()
                    if state.desired_points > state.occupied_slots:
                        if state.occupied_slots > 0:
                            should_shrink_to = state.occupied_slots
                        elif not state.pending_queue:
                            should_pause = True
                    continue
                next_task = task
                break

            if next_task is None:
                break

            target_points = min(
                max(1, state.occupied_slots + len(state.pending_queue)),
                int(proj.get("max_points", 1)),
            )

            if not state.service_url:
                state.recover_in_flight = True
                next_task.task_status = "SCALING"
                next_task.message = "正在恢复算力任务并等待节点就绪。"
                should_recover = True
            elif state.occupied_slots < state.ready_capacity:
                state.pending_queue.popleft()
                state.occupied_slots += 1
                next_task.task_status = "RUNNING"
                next_task.message = ""
                next_task.scheduled_time = next_task.scheduled_time or _utc_now_str()
                next_task.start_time = _utc_now_str()
                state.ready_capacity = max(state.ready_capacity, state.occupied_slots)
            else:
                if target_points > state.desired_points:
                    state.scale_in_flight = True
                    next_task.task_status = "SCALING"
                    next_task.message = "正在扩容并等待节点冷启动。"
                    should_scale_to = target_points
                elif state.desired_points > state.ready_capacity:
                    state.pending_queue.popleft()
                    state.occupied_slots += 1
                    next_task.task_status = "RUNNING"
                    next_task.message = ""
                    next_task.scheduled_time = next_task.scheduled_time or _utc_now_str()
                    next_task.start_time = _utc_now_str()
                else:
                    next_task.task_status = "PENDING"
                    next_task.message = "等待可用节点。"

        if should_recover:
            try:
                await _ensure_project_recovered(state, proj)
                async with state.lock:
                    state.recover_in_flight = False
                    state.desired_points = max(state.desired_points, 1)
                    state.ready_capacity = max(state.ready_capacity, 1)
            except HTTPException as exc:
                async with state.lock:
                    state.recover_in_flight = False
                    if next_task and next_task.task_status not in _TERMINAL_TASK_STATUSES:
                        next_task.task_status = "FAILED"
                        next_task.message = str(exc.detail)
                        next_task.end_time = _utc_now_str()
                        if state.pending_queue and state.pending_queue[0] == next_task.task_id:
                            state.pending_queue.popleft()
            await asyncio.sleep(0)
            continue

        if should_pause:
            await _pause_project(state, proj)
            await asyncio.sleep(0)
            continue

        if should_shrink_to is not None:
            await _set_project_points(state, proj, max(should_shrink_to, 1))
            await asyncio.sleep(0)
            continue

        if should_scale_to is not None:
            try:
                await _set_project_points(state, proj, should_scale_to)
                async with state.lock:
                    state.scale_in_flight = False
                    state.desired_points = should_scale_to
                    if next_task.scale_deadline is None:
                        next_task.scale_deadline = time.monotonic() + int(proj.get("scale_out_timeout_sec", 300))
            except HTTPException as exc:
                async with state.lock:
                    state.scale_in_flight = False
                    if next_task and next_task.task_status not in _TERMINAL_TASK_STATUSES:
                        next_task.task_status = "FAILED"
                        next_task.message = str(exc.detail)
                        next_task.end_time = _utc_now_str()
                        if state.pending_queue and state.pending_queue[0] == next_task.task_id:
                            state.pending_queue.popleft()
            await asyncio.sleep(0)
            continue

        if next_task and next_task.task_status == "RUNNING":
            asyncio.create_task(_run_task_dispatch(state, proj, next_task.task_id))
            await asyncio.sleep(0)
            continue

        await asyncio.sleep(min(int(proj.get("dispatch_retry_interval_ms", 2000)) / 1000.0, 5.0))

    async with state.lock:
        state.scheduler_task = None


# ---------------------------------------------------------------------------
# Public API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/projects")
async def list_projects():
    return {"projects": list(PROJECTS.keys())}


@app.post("/api/v1/services/aigc/3d-generation/reconstruction")
async def create_reconstruction_task(
    request: str = Form(...),
    files: list[UploadFile] = File(...),
):
    try:
        payload = json.loads(request)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid request JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request must be a JSON object")

    model = str(payload.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    proj = _get_project(model)
    state = _get_state(model)

    input_payload = payload.get("input")
    if input_payload is None:
        input_payload = {}
    if not isinstance(input_payload, dict):
        raise HTTPException(status_code=400, detail="input must be an object")

    parameter_payload = payload.get("parameters")
    if parameter_payload is None:
        parameter_payload = {}
    if not isinstance(parameter_payload, dict):
        raise HTTPException(status_code=400, detail="parameters must be an object")

    saved_files = await _save_uploads(files)
    if not saved_files:
        raise HTTPException(status_code=400, detail="At least one non-empty file is required")

    request_id = str(input_payload.get("request_id") or uuid.uuid4().hex)
    task_id = uuid.uuid4().hex
    now = _utc_now_str()
    task = TaskRecord(
        task_id=task_id,
        request_id=request_id,
        project=model,
        model=model,
        input_payload=input_payload,
        parameter_payload=parameter_payload,
        files=saved_files,
        submit_time=now,
    )

    async with state.lock:
        state.tasks[task_id] = task
        state.pending_queue.append(task_id)
        TASK_INDEX[task_id] = model

    _ensure_scheduler(state)

    return _baillian_response(
        request_id,
        output={
            "task_id": task_id,
            "task_status": "PENDING",
            "submit_time": now,
        },
    )


@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    project_name = TASK_INDEX.get(task_id)
    if not project_name:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    state = _get_state(project_name)
    task = state.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    return _baillian_response(
        task.request_id,
        output=_format_task_output(task),
        message=task.message,
    )


# ---------------------------------------------------------------------------
# Legacy compatibility endpoints
# ---------------------------------------------------------------------------
@app.post("/api/task/recover")
async def recover_task(req: ProjectRequest):
    proj = _get_project(req.project)
    state = _get_state(req.project)
    service_url = await _ensure_project_recovered(state, proj)
    async with state.lock:
        state.desired_points = max(state.desired_points, 1)
        state.ready_capacity = max(state.ready_capacity, 1)
    return {"service_url": service_url, "status": "running", "recovered": True}


@app.post("/api/task/pause")
async def pause_task(req: ProjectRequest):
    proj = _get_project(req.project)
    state = _get_state(req.project)
    async with state.lock:
        if state.occupied_slots > 0 or state.pending_queue:
            raise HTTPException(status_code=409, detail="Tasks are still running or queued")
    await _pause_project(state, proj)
    return {"status": "paused"}


@app.get("/api/task/status")
async def task_status(project: str):
    proj = _get_project(project)
    state = _get_state(project)
    service_port = int(proj.get("service_port", 10085))
    async with httpx.AsyncClient(timeout=30) as client:
        detail = await _call_task_detail(client, proj)
        status = _extract_task_status(detail)
        service_url = _extract_service_url(detail, service_port)
    async with state.lock:
        return {
            "status": status,
            "service_url": service_url,
            "desired_points": state.desired_points,
            "ready_capacity": state.ready_capacity,
            "occupied_slots": state.occupied_slots,
            "pending": len(state.pending_queue),
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
        PROJECT_STATES.clear()
        PROJECT_STATES.update({name: ProjectRuntimeState(project=name) for name in PROJECTS})
        TASK_INDEX.clear()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
