"""Python client for NPU Server."""

from __future__ import annotations

from typing import Any

import requests

from server.nvgpu.client import TaskResult


class NPUClient:
    """Client for interacting with NPU Server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def submit_task(
        self,
        script_path: str,
        task_mode: str | None = None,
        task_type: str | None = None,
        task_label: str | None = None,
        work_dir: str = ".",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        npu_id: int | None = None,
        gpu_id: int | None = None,  # compatibility alias
    ) -> str:
        payload = {
            "script_path": script_path,
            "work_dir": work_dir,
            "args": args or [],
        }
        if task_mode is not None:
            payload["task_mode"] = task_mode
        if task_type is not None:
            payload["task_type"] = task_type
        if task_label is not None:
            payload["task_label"] = task_label
        if env:
            payload["env"] = env

        resolved_npu_id = npu_id if npu_id is not None else gpu_id
        if resolved_npu_id is not None:
            payload["npu_id"] = resolved_npu_id

        response = self.session.post(f"{self.base_url}/tasks", json=payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to submit task: {response.text}")
        return response.json()["task_id"]

    def submit_task_in_script_dir(
        self,
        script_path: str,
        task_mode: str | None = None,
        task_type: str | None = None,
        task_label: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        npu_id: int | None = None,
        gpu_id: int | None = None,  # compatibility alias
    ) -> str:
        from pathlib import Path

        work_dir = str(Path(script_path).parent.absolute())
        return self.submit_task(
            script_path=script_path,
            task_mode=task_mode,
            task_type=task_type,
            task_label=task_label,
            work_dir=work_dir,
            args=args,
            env=env,
            npu_id=npu_id,
            gpu_id=gpu_id,
        )

    def get_task(self, task_id: str) -> TaskResult:
        response = self.session.get(f"{self.base_url}/tasks/{task_id}", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get task: {response.text}")
        data = response.json()
        assigned = data.get("assigned_npu", data.get("assigned_gpu"))
        return TaskResult(
            task_id=data["task_id"],
            status=data["status"],
            task_mode=data.get("task_mode"),
            task_type=data.get("task_type"),
            task_label=data.get("task_label"),
            exit_code=data.get("exit_code"),
            log_file=data.get("log_file"),
            error_message=data.get("error_message"),
            submit_timestamp=data.get("submit_timestamp"),
            queued_timestamp=data.get("queued_timestamp"),
            start_timestamp=data.get("start_timestamp"),
            end_timestamp=data.get("end_timestamp"),
            stdout_size=data.get("stdout_size", 0),
            stderr_size=data.get("stderr_size", 0),
            gpu_id=assigned,
            pending_duration_ms=data.get("pending_duration_ms"),
            queue_duration_ms=data.get("queue_duration_ms"),
            waiting_duration_ms=data.get("waiting_duration_ms"),
            running_duration_ms=data.get("running_duration_ms"),
            total_duration_ms=data.get("total_duration_ms"),
            execution_duration_ms=data.get("execution_duration_ms"),
        )

    def cancel_task(self, task_id: str, force: bool = False) -> bool:
        response = self.session.post(
            f"{self.base_url}/tasks/{task_id}/cancel",
            params={"force": force},
            timeout=10,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to cancel task: {response.text}")
        return True

    def list_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        url = f"{self.base_url}/tasks"
        if status:
            url += f"?status={status}"
        response = self.session.get(url, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list tasks: {response.text}")
        return response.json()["tasks"]

    def list_npus(self) -> list[dict[str, Any]]:
        response = self.session.get(f"{self.base_url}/npus", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list NPUs: {response.text}")
        return response.json()["npus"]

    def get_npu(self, npu_id: int) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/npus/{npu_id}", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get NPU: {response.text}")
        return response.json()

    # Compatibility aliases for old GPU-based callers
    def list_gpus(self) -> list[dict[str, Any]]:
        return self.list_npus()

    def get_gpu(self, gpu_id: int) -> dict[str, Any]:
        return self.get_npu(gpu_id)

    def get_stats(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/stats", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get stats: {response.text}")
        return response.json()

    def get_task_log(
        self,
        task_id: str,
        log_type: str = "summary",
        offset: int = 0,
        limit: int = 102400,
    ) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/tasks/{task_id}/log",
            params={"log_type": log_type, "offset": offset, "limit": limit},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get task log: {response.text}")
        return response.json()

    def get_full_task_log(
        self,
        task_id: str,
        log_type: str = "summary",
        chunk_size: int = 1024 * 1024,
    ) -> str:
        content_parts = []
        offset = 0
        while True:
            result = self.get_task_log(task_id, log_type, offset, chunk_size)
            content_parts.append(result["content"])
            if not result.get("has_more", False):
                break
            offset += result["size"]
        return "".join(content_parts)

