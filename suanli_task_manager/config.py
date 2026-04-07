from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ServerConfig:
    host: str
    port: int
    temp_root: Path
    cleanup_interval_sec: int
    default_task_ttl_sec: int


@dataclass(slots=True)
class ProjectConfig:
    name: str
    public_model: str
    adapter: str
    task_id: int
    service_port: int
    token: str
    ready_timeout_sec: int
    dispatch_retry_interval_ms: int
    dispatch_retry_backoff_max_ms: int
    queue_limit: int
    task_ttl_sec: int
    health_path: str = "/health"
    ready_path: str = "/ready"
    live_path: str = "/live"


@dataclass(slots=True)
class AppConfig:
    server: ServerConfig
    openapi_base: str
    openapi_version: str
    auth_enabled: bool
    api_keys: set[str]
    projects: dict[str, ProjectConfig]

    def resolve_project_name(self, name_or_model: str) -> str:
        key = (name_or_model or "").strip()
        if not key:
            raise KeyError("missing project/model")
        if key in self.projects:
            return key
        for name, project in self.projects.items():
            if project.public_model == key:
                return name
        raise KeyError(key)


def _default_adapter(project_name: str) -> str:
    lowered = project_name.strip().lower()
    if "ultrashape" in lowered or "shape" in lowered:
        return "ultrashape"
    if "fastgs" in lowered:
        return "fastgs"
    if "lightx2v" in lowered:
        return "lightx2v"
    return "world"


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    server_raw = raw.get("server") or {}
    cleanup_interval_sec = int(server_raw.get("cleanup_interval_sec", 60))
    default_task_ttl_sec = int(server_raw.get("default_task_ttl_sec", 3600))
    temp_root = Path(server_raw.get("temp_root") or "/data/tmp/suanli-task-manager")
    server = ServerConfig(
        host=str(server_raw.get("host", "0.0.0.0")),
        port=int(server_raw.get("port", 8090)),
        temp_root=temp_root,
        cleanup_interval_sec=cleanup_interval_sec,
        default_task_ttl_sec=default_task_ttl_sec,
    )

    auth_raw = raw.get("auth") or {}
    projects_raw = raw.get("projects") or {}
    projects: dict[str, ProjectConfig] = {}
    for name, proj_raw in projects_raw.items():
        proj = proj_raw or {}
        projects[name] = ProjectConfig(
            name=name,
            public_model=str(proj.get("public_model") or name),
            adapter=str(proj.get("adapter") or _default_adapter(name)),
            task_id=int(proj["task_id"]),
            service_port=int(proj.get("service_port", 10085)),
            token=str(proj.get("token") or ""),
            ready_timeout_sec=int(proj.get("ready_timeout_sec", 300)),
            dispatch_retry_interval_ms=int(proj.get("dispatch_retry_interval_ms", 2000)),
            dispatch_retry_backoff_max_ms=int(proj.get("dispatch_retry_backoff_max_ms", 5000)),
            queue_limit=int(proj.get("queue_limit", 8)),
            task_ttl_sec=int(proj.get("task_ttl_sec", default_task_ttl_sec)),
            health_path=str(proj.get("health_path") or "/health"),
            ready_path=str(proj.get("ready_path") or "/ready"),
            live_path=str(proj.get("live_path") or "/live"),
        )

    api_keys = {
        str(item).strip()
        for item in (auth_raw.get("api_keys") or [])
        if str(item).strip()
    }
    return AppConfig(
        server=server,
        openapi_base=str((raw.get("openapi") or {}).get("base", "https://openapi.suanli.cn")).rstrip("/"),
        openapi_version=str((raw.get("openapi") or {}).get("version", "1.0.0")),
        auth_enabled=bool(auth_raw.get("enabled", False)),
        api_keys=api_keys,
        projects=projects,
    )
