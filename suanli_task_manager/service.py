from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import HTTPException

from suanli_task_manager.adapters.types import RuntimePending
from suanli_task_manager.config import AppConfig, ProjectConfig
from suanli_task_manager.models import TaskRecord, TaskStatus
from suanli_task_manager.platform import PlatformClient
from suanli_task_manager.storage import TaskStore, UploadPayload


@dataclass(slots=True)
class ProjectRuntimeState:
    project: str
    service_url: str = ""
    desired_points: int = 0
    ready_capacity: int = 0
    occupied_slots: int = 0
    pending_queue: deque[str] = field(default_factory=deque)
    active_tasks: set[str] = field(default_factory=set)
    recover_in_flight: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    scheduler_task: asyncio.Task | None = None


class RuntimeBusyError(Exception):
    pass


class TaskManagerService:
    def __init__(
        self,
        *,
        config: AppConfig,
        platform_client: PlatformClient | None = None,
        adapter_registry: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.platform_client = platform_client or PlatformClient(config)
        self.adapter_registry = adapter_registry or {}
        self.store = TaskStore(config)
        self.project_states = {
            name: ProjectRuntimeState(project=name) for name in config.projects
        }
        self.cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        for state in self.project_states.values():
            if state.scheduler_task is not None:
                state.scheduler_task.cancel()
                try:
                    await state.scheduler_task
                except asyncio.CancelledError:
                    pass

    def list_projects(self) -> list[str]:
        return [project.public_model or name for name, project in self.config.projects.items()]

    async def create_task(
        self,
        *,
        model: str,
        input_payload: dict[str, Any],
        parameter_payload: dict[str, Any],
        uploads: list[UploadPayload],
    ) -> dict[str, Any]:
        try:
            project_name = self.config.resolve_project_name(model)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown project/model: {model}") from None
        project = self.config.projects[project_name]
        adapter = self._get_adapter(project)
        normalized = adapter.validate_request(input_payload, parameter_payload, uploads) or {}
        request_id = str(input_payload.get("request_id") or self._new_request_id())
        state = self.project_states[project_name]

        async with state.lock:
            active_count = len(state.pending_queue) + len(state.active_tasks)
            if active_count >= project.queue_limit:
                raise HTTPException(status_code=429, detail="Queue is full, try again later")

            task = self.store.create_task(
                project=project_name,
                model=model,
                request_id=request_id,
                input_payload=normalized.get("input_payload", input_payload),
                parameter_payload=normalized.get("parameter_payload", parameter_payload),
                uploads=uploads,
            )
            state.pending_queue.append(task.task_id)

        self._ensure_scheduler(project_name)
        return self._bailian_response(
            request_id,
            output={
                "task_id": task.task_id,
                "task_status": TaskStatus.PENDING.value,
                "submit_time": task.submit_time,
            },
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
        return self._bailian_response(
            task.request_id,
            output=self.store.format_task_output(task),
            message=task.message,
        )

    async def recover_project(self, project_name: str) -> dict[str, Any]:
        state = self.project_states[project_name]
        project = self.config.projects[project_name]
        service_url = await self.platform_client.ensure_project_recovered(project)
        async with state.lock:
            state.service_url = service_url
            state.desired_points = max(state.desired_points, 1)
            state.ready_capacity = max(state.ready_capacity, 1)
        return {"service_url": service_url, "status": "running", "recovered": True}

    async def pause_project(self, project_name: str) -> dict[str, Any]:
        state = self.project_states[project_name]
        project = self.config.projects[project_name]
        async with state.lock:
            if state.active_tasks or state.pending_queue:
                raise HTTPException(status_code=409, detail="Tasks are still running or queued")
        await self.platform_client.pause_project(project)
        async with state.lock:
            state.service_url = ""
            state.desired_points = 0
            state.ready_capacity = 0
        return {"status": "paused"}

    async def project_status(self, project_name: str) -> dict[str, Any]:
        state = self.project_states[project_name]
        status_payload = await self.platform_client.fetch_runtime_status(self.config.projects[project_name])
        async with state.lock:
            return {
                "status": status_payload.get("status", ""),
                "service_url": status_payload.get("service_url", ""),
                "desired_points": state.desired_points,
                "ready_capacity": state.ready_capacity,
                "occupied_slots": len(state.active_tasks),
                "pending": len(state.pending_queue),
            }

    async def _cleanup_loop(self) -> None:
        interval = max(self.config.server.cleanup_interval_sec, 1)
        while True:
            self.store.prune_expired()
            await self._reconcile_idle_projects_once()
            await asyncio.sleep(interval)

    def _ensure_scheduler(self, project_name: str) -> None:
        state = self.project_states[project_name]
        if state.scheduler_task and not state.scheduler_task.done():
            return
        state.scheduler_task = asyncio.create_task(self._scheduler_loop(project_name))

    async def _scheduler_loop(self, project_name: str) -> None:
        project = self.config.projects[project_name]
        state = self.project_states[project_name]

        while True:
            next_task: TaskRecord | None = None
            should_recover = False
            should_pause = False

            async with state.lock:
                while state.pending_queue:
                    task_id = state.pending_queue[0]
                    task = self.store.get_task(task_id)
                    if task is None or task.is_terminal:
                        state.pending_queue.popleft()
                        continue
                    next_task = task
                    break

                if next_task is None:
                    if state.service_url and not state.active_tasks and not state.recover_in_flight:
                        should_pause = True
                    else:
                        break

                if should_pause:
                    pass
                elif not state.service_url:
                    state.recover_in_flight = True
                    self.store.update_status(next_task.task_id, TaskStatus.SCALING, message="正在恢复算力任务并等待节点就绪。")
                    should_recover = True
                elif len(state.active_tasks) < state.ready_capacity:
                    state.pending_queue.popleft()
                    state.active_tasks.add(next_task.task_id)
                    state.occupied_slots = len(state.active_tasks)
                    state.ready_capacity = max(state.ready_capacity, state.occupied_slots)
                    self.store.update_status(next_task.task_id, TaskStatus.RUNNING, message="")
                else:
                    self.store.update_status(next_task.task_id, TaskStatus.PENDING, message="等待可用节点。")

            if should_recover:
                try:
                    service_url = await self.platform_client.ensure_project_recovered(project)
                    async with state.lock:
                        state.recover_in_flight = False
                        state.service_url = service_url
                        state.desired_points = max(state.desired_points, 1)
                        state.ready_capacity = max(state.ready_capacity, 1)
                except HTTPException as exc:
                    async with state.lock:
                        state.recover_in_flight = False
                        if next_task and not next_task.is_terminal:
                            self.store.mark_failed(next_task.task_id, str(exc.detail))
                            if state.pending_queue and state.pending_queue[0] == next_task.task_id:
                                state.pending_queue.popleft()
                await asyncio.sleep(0)
                continue

            if should_pause:
                paused = await self._pause_idle_project(project_name)
                if not paused:
                    await asyncio.sleep(min(project.dispatch_retry_interval_ms / 1000.0, 5.0))
                await asyncio.sleep(0)
                continue

            if next_task and next_task.task_status == TaskStatus.RUNNING:
                asyncio.create_task(self._run_task_dispatch(project_name, next_task.task_id))
                await asyncio.sleep(0)
                continue

            await asyncio.sleep(min(project.dispatch_retry_interval_ms / 1000.0, 5.0))

        async with state.lock:
            state.scheduler_task = None

    async def _run_task_dispatch(self, project_name: str, task_id: str) -> None:
        project = self.config.projects[project_name]
        state = self.project_states[project_name]
        task = self.store.get_task(task_id)
        if task is None:
            return
        adapter = self._get_adapter(project)
        try:
            print(f"[task-manager] dispatch start project={project_name} task_id={task_id}", flush=True)
            timeout = httpx.Timeout(connect=30.0, read=None, write=120.0, pool=None)
            async with httpx.AsyncClient(timeout=timeout) as client:
                result = await adapter.dispatch(
                    service_url=state.service_url,
                    task=task,
                    client=client,
                    project=project,
                    artifact_writer=self.store.write_artifact,
                )
                payload = await self._resolve_adapter_result(
                    adapter=adapter,
                    service_url=state.service_url,
                    task=task,
                    client=client,
                    project=project,
                    initial_result=result,
                )
            self.store.mark_completed(task.task_id, payload)
            print(f"[task-manager] task completed project={project_name} task_id={task_id}", flush=True)
        except HTTPException as exc:
            self.store.mark_failed(task.task_id, str(exc.detail))
            print(
                f"[task-manager] task failed project={project_name} task_id={task_id} detail={exc.detail}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.store.mark_failed(task.task_id, f"Dispatch failed: {exc}")
            print(
                f"[task-manager] dispatch exception project={project_name} task_id={task_id} detail={exc}",
                flush=True,
            )
        finally:
            await self._release_capacity_after_finish(project_name, task_id)

    async def _resolve_adapter_result(
        self,
        *,
        adapter,
        service_url: str,
        task: TaskRecord,
        client: httpx.AsyncClient,
        project: ProjectConfig,
        initial_result: dict[str, Any] | RuntimePending,
    ) -> dict[str, Any]:
        result: dict[str, Any] | RuntimePending = initial_result
        default_poll_interval = max(project.dispatch_retry_interval_ms / 1000.0, 0.05)

        while isinstance(result, RuntimePending):
            progress_output = dict(result.output)
            progress_output.setdefault("runtime_request_id", result.runtime_request_id)
            self.store.update_runtime_progress(
                task.task_id,
                message=result.message or "任务已提交到运行时队列，等待处理完成。",
                output_updates=progress_output,
            )
            print(
                f"[task-manager] runtime pending project={project.name} task_id={task.task_id} "
                f"runtime_request_id={result.runtime_request_id} status={progress_output.get('runtime_task_status', '')}",
                flush=True,
            )
            await asyncio.sleep(result.poll_interval_sec or default_poll_interval)
            result = await adapter.poll(
                service_url=service_url,
                runtime_request_id=result.runtime_request_id,
                task=task,
                client=client,
                project=project,
                artifact_writer=self.store.write_artifact,
            )

        return result

    async def _release_capacity_after_finish(self, project_name: str, task_id: str) -> None:
        state = self.project_states[project_name]
        async with state.lock:
            state.active_tasks.discard(task_id)
            state.occupied_slots = len(state.active_tasks)
            has_pending = bool(state.pending_queue)

        try:
            if not has_pending:
                await self._pause_idle_project(project_name)
        finally:
            self._ensure_scheduler(project_name)

    async def _pause_idle_project(self, project_name: str) -> bool:
        project = self.config.projects[project_name]
        state = self.project_states[project_name]

        async with state.lock:
            should_pause = bool(state.service_url) and not state.active_tasks and not state.pending_queue
            if not should_pause:
                return False

        try:
            await self.platform_client.pause_project(project)
        except Exception as exc:
            print(f"[task-manager] pause failed for {project_name}: {exc}", flush=True)
            return False

        async with state.lock:
            state.service_url = ""
            state.desired_points = 0
            state.ready_capacity = 0
        print(f"[task-manager] project paused: {project_name}", flush=True)
        return True

    async def _reconcile_idle_projects_once(self) -> None:
        for project_name, state in self.project_states.items():
            async with state.lock:
                should_pause = bool(state.service_url) and not state.active_tasks and not state.pending_queue and not state.recover_in_flight
            if should_pause:
                print(f"[task-manager] idle reconcile project={project_name}", flush=True)
                await self._pause_idle_project(project_name)

    def _get_adapter(self, project: ProjectConfig):
        adapter = self.adapter_registry.get(project.adapter)
        if adapter is None:
            raise HTTPException(status_code=500, detail=f"Adapter not registered: {project.adapter}")
        return adapter

    @staticmethod
    def _bailian_response(
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

    @staticmethod
    def _new_request_id() -> str:
        import uuid

        return uuid.uuid4().hex
