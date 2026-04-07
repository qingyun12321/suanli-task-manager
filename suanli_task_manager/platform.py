from __future__ import annotations

import re
import time
from typing import Any

import httpx
from fastapi import HTTPException

from suanli_task_manager.config import AppConfig, ProjectConfig

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


class PlatformClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    async def ensure_project_recovered(self, project: ProjectConfig) -> str:
        timeout_sec = int(project.ready_timeout_sec)
        service_port = int(project.service_port)
        async with httpx.AsyncClient(timeout=30) as client:
            detail = await self._call_task_detail(client, project)
            status_before = self._extract_task_status(detail)
            url_before = self._extract_service_url(detail, service_port)
            self._log_line(
                f"[task-manager] platform detail project={project.name} "
                f"status={status_before or '<empty>'} service_url={url_before or '<empty>'}"
            )
            if not (self._is_running_status(status_before) and url_before):
                await self._call_task_control(client, project, "recover")

            deadline = time.monotonic() + timeout_sec
            last_status = status_before
            last_url = url_before
            while time.monotonic() < deadline:
                detail = await self._call_task_detail(client, project)
                status = self._extract_task_status(detail)
                service_url = self._extract_service_url(detail, service_port)
                if status:
                    last_status = status
                if service_url:
                    last_url = service_url
                self._log_line(
                    f"[task-manager] platform detail project={project.name} "
                    f"status={status or '<empty>'} service_url={service_url or '<empty>'}"
                )
                if self._is_running_status(status) and service_url:
                    await self._wait_for_runtime_ready(service_url.rstrip("/"), project)
                    return service_url.rstrip("/")
                await asyncio_sleep(2)
        raise HTTPException(
            status_code=504,
            detail=f"Timeout waiting for task to run. last_status={last_status}, last_url={last_url}",
        )

    async def pause_project(self, project: ProjectConfig) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            await self._call_task_control(client, project, "pause")

    async def fetch_runtime_status(self, project: ProjectConfig) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            detail = await self._call_task_detail(client, project)
        return {
            "status": self._extract_task_status(detail),
            "service_url": self._extract_service_url(detail, project.service_port),
        }

    async def _call_task_detail(self, client: httpx.AsyncClient, project: ProjectConfig) -> dict[str, Any]:
        url = f"{self._config.openapi_base}/api/deployment/task/detail"
        response = await client.get(
            url,
            params={"task_id": str(project.task_id)},
            headers=self._build_headers(project.token, content_type=False),
        )
        payload = self._parse_payload(response)
        self._ensure_success(response, payload, "detail")
        return payload

    async def _call_task_control(self, client: httpx.AsyncClient, project: ProjectConfig, action: str) -> None:
        action_path = "recover" if action == "recover" else "pause"
        request_payload = {"task_id": project.task_id}
        self._log_line(
            f"[task-manager] platform request action={action_path} "
            f"project={project.name} task_id={project.task_id}"
        )
        response = await client.post(
            f"{self._config.openapi_base}/api/deployment/task/{action_path}",
            json=request_payload,
            headers=self._build_headers(project.token),
        )
        payload = self._parse_payload(response)
        self._log_line(
            f"[task-manager] platform response action={action_path} project={project.name} "
            f"status_code={response.status_code} message={self._extract_message(payload) or '<empty>'} "
            f"code={self._normalize_code(payload.get('code')) or '<empty>'}"
        )
        if response.is_success and self._is_api_success(payload):
            return
        message = self._extract_message(payload) or f"HTTP {response.status_code}"
        if action == "recover" and _RECOVER_ALREADY_RUNNING.search(message):
            return
        if action == "pause" and _PAUSE_ALREADY_PAUSED.search(message):
            return
        raise HTTPException(status_code=502, detail=f"{action_path} failed: {message}")

    async def _wait_for_runtime_ready(self, base_url: str, project: ProjectConfig) -> None:
        await self._wait_for_runtime_probe(
            base_url=base_url,
            timeout_sec=int(project.ready_timeout_sec),
            probe_path=project.ready_path,
            probe_name="runtime readiness",
        )

    async def _wait_for_runtime_probe(
        self,
        *,
        base_url: str,
        timeout_sec: int,
        probe_path: str,
        probe_name: str,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        normalized_path = self._normalize_probe_path(probe_path)
        async with httpx.AsyncClient(timeout=15) as client:
            last_error = ""
            while time.monotonic() < deadline:
                try:
                    response = await client.get(f"{base_url}{normalized_path}")
                    if response.is_success:
                        return
                    last_error = f"{normalized_path} HTTP {response.status_code}"
                except Exception as exc:  # pragma: no cover - network dependent
                    last_error = str(exc)
                await asyncio_sleep(2)
        raise HTTPException(status_code=504, detail=f"Timeout waiting for {probe_name}: {last_error}")

    def _build_headers(self, token: str, *, content_type: bool = True) -> dict[str, str]:
        headers = {
            "token": token,
            "timestamp": str(int(time.time() * 1000)),
            "version": self._config.openapi_version,
        }
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers

    @staticmethod
    def _parse_payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _normalize_code(code: Any) -> str:
        return str("" if code is None else code).strip().lower()

    def _is_api_success(self, payload: dict[str, Any]) -> bool:
        code = self._normalize_code(payload.get("code"))
        return code in {"0000", "0", "200", "ok", "success"}

    def _ensure_success(self, response: httpx.Response, payload: dict[str, Any], action: str) -> None:
        if not response.is_success:
            raise HTTPException(status_code=502, detail=f"{action} request failed: {self._extract_message(payload) or response.status_code}")
        if not self._is_api_success(payload):
            raise HTTPException(status_code=502, detail=f"{action} failed: {self._extract_message(payload)}")

    @staticmethod
    def _extract_message(payload: dict[str, Any]) -> str:
        msg = payload.get("message") or payload.get("msg") or payload.get("detail") or payload.get("error") or ""
        if isinstance(msg, str):
            return msg
        return str(msg or "")

    @staticmethod
    def _extract_task_status(detail_payload: dict[str, Any]) -> str:
        data = detail_payload.get("data")
        if not isinstance(data, dict):
            return ""
        return str(data.get("status") or data.get("task_status") or data.get("taskStatus") or "").strip()

    @staticmethod
    def _extract_service_url(detail_payload: dict[str, Any], service_port: int) -> str:
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
                    if int(port_item.get("service_port", 0)) == int(service_port):
                        return url.rstrip("/")
                except (TypeError, ValueError):
                    continue
        return fallback.rstrip("/") if fallback else ""

    @staticmethod
    def _is_running_status(status: str) -> bool:
        return bool(_RUNNING_STATUS.search(status or ""))

    @staticmethod
    def _normalize_probe_path(path: str) -> str:
        cleaned = str(path or "").strip()
        if not cleaned:
            return "/ready"
        return cleaned if cleaned.startswith("/") else f"/{cleaned}"

    @staticmethod
    def _log_line(message: str) -> None:
        print(message, flush=True)


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
