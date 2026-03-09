"""REST API server for NPU service."""
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from server.npu.config import TaskMode, config
from server.npu.logger import setup_logger
from server.npu.models import NPUMode, NPUStatus, Task, TaskType
from server.npu.npu_manager import NPUManager
from server.npu.scheduler import Scheduler
from server.npu.task_queue import TaskQueue

logger = setup_logger("api_server")

app = FastAPI(title="NPU Server", description="NPU Task Scheduling Server")

npu_manager: NPUManager = None
task_queue: TaskQueue = None
scheduler: Scheduler = None
task_runner = None


def init_app(nm: NPUManager, tq: TaskQueue, sched: Scheduler, tr=None):
    """Initialize FastAPI app with managers."""
    global npu_manager, task_queue, scheduler, task_runner
    npu_manager = nm
    task_queue = tq
    scheduler = sched
    task_runner = tr


def _task_dict_with_npu_fields(task: Task) -> dict[str, Any]:
    payload = task.to_dict()
    payload["npu_id"] = payload.pop("gpu_id", None)
    payload["assigned_npu"] = payload.pop("assigned_gpu", None)
    return payload


class NPUStatusUpdate(BaseModel):
    status: str


class NPUModeUpdate(BaseModel):
    mode: str
    manual: bool = True


class NPUMemoryThresholdUpdate(BaseModel):
    threshold: float


class NPUMaxConcurrentTasksUpdate(BaseModel):
    max_tasks: int


class NPURegister(BaseModel):
    npu_id: int
    mode: str | None = None
    memory_threshold: float | None = None
    max_concurrent_tasks: int | None = None


class NPUError(BaseModel):
    error_message: str = "Manual error trigger"


class TaskSubmit(BaseModel):
    script_path: str
    task_mode: str | None = None
    task_type: str | None = None
    task_label: str | None = None
    work_dir: str = "."
    args: list[str] = []
    env: dict[str, str] | None = None
    npu_id: int | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/npus")
async def list_npus():
    npus = npu_manager.list_npus()
    return {
        "npus": [
            {
                "npu_id": n.gpu_id,
                "mode": n.mode.value,
                "manual_mode": n.manual_mode.value if n.manual_mode else None,
                "mode_locked_by": n.mode_locked_by,
                "status": n.status.value,
                "memory_threshold": n.memory_threshold,
                "max_concurrent_tasks": n.max_concurrent_tasks,
                "current_memory_usage": n.current_memory_usage,
                "running_tasks": n.running_tasks,
                "running_task_count": len(n.running_tasks),
                "error_message": n.error_message,
            }
            for n in npus
        ]
    }


@app.get("/npus/{npu_id}")
async def get_npu(npu_id: int):
    npu = npu_manager.get_npu(npu_id)
    if not npu:
        raise HTTPException(status_code=404, detail="NPU not found")
    return {
        "npu_id": npu.gpu_id,
        "mode": npu.mode.value,
        "manual_mode": npu.manual_mode.value if npu.manual_mode else None,
        "mode_locked_by": npu.mode_locked_by,
        "status": npu.status.value,
        "memory_threshold": npu.memory_threshold,
        "max_concurrent_tasks": npu.max_concurrent_tasks,
        "current_memory_usage": npu.current_memory_usage,
        "running_tasks": npu.running_tasks,
        "running_task_count": len(npu.running_tasks),
        "error_message": npu.error_message,
    }


@app.put("/npus/{npu_id}/status")
async def set_npu_status(npu_id: int, data: NPUStatusUpdate):
    try:
        status = NPUStatus(data.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {data.status}")
    if npu_manager.set_npu_status(npu_id, status):
        return {"success": True, "npu_id": npu_id, "status": status.value}
    raise HTTPException(status_code=400, detail="Failed to set NPU status")


@app.put("/npus/{npu_id}/mode")
async def set_npu_mode(npu_id: int, data: NPUModeUpdate):
    try:
        mode = NPUMode(data.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {data.mode}")
    if npu_manager.set_npu_mode(npu_id, mode, manual=data.manual):
        return {"success": True, "npu_id": npu_id, "mode": mode.value, "manual": data.manual}
    raise HTTPException(status_code=400, detail="Failed to set NPU mode")


@app.delete("/npus/{npu_id}/mode")
async def clear_npu_manual_mode(npu_id: int):
    if npu_manager.clear_npu_manual_mode(npu_id):
        return {"success": True, "npu_id": npu_id, "message": "Manual mode cleared"}
    raise HTTPException(status_code=400, detail="Failed to clear NPU manual mode")


@app.put("/npus/{npu_id}/memory_threshold")
async def set_npu_memory_threshold(npu_id: int, data: NPUMemoryThresholdUpdate):
    if npu_manager.set_npu_memory_threshold(npu_id, data.threshold):
        return {"success": True, "npu_id": npu_id, "threshold": data.threshold}
    raise HTTPException(status_code=400, detail="Failed to set memory threshold")


@app.put("/npus/{npu_id}/max_concurrent_tasks")
async def set_npu_max_concurrent_tasks(npu_id: int, data: NPUMaxConcurrentTasksUpdate):
    if npu_manager.set_npu_max_concurrent_tasks(npu_id, data.max_tasks):
        return {"success": True, "npu_id": npu_id, "max_tasks": data.max_tasks}
    raise HTTPException(status_code=400, detail="Failed to set max concurrent tasks")


@app.post("/npus/register")
async def register_npu(data: NPURegister):
    mode = None
    if data.mode:
        try:
            mode = NPUMode(data.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {data.mode}")
    if npu_manager.register_npu(data.npu_id, mode, data.memory_threshold, data.max_concurrent_tasks):
        return {"success": True, "npu_id": data.npu_id}
    raise HTTPException(status_code=400, detail="Failed to register NPU")


@app.post("/npus/{npu_id}/unregister")
async def unregister_npu(npu_id: int):
    if npu_manager.unregister_npu(npu_id):
        return {"success": True, "npu_id": npu_id}
    raise HTTPException(status_code=400, detail="Failed to unregister NPU")


@app.post("/npus/{npu_id}/error")
async def trigger_npu_error(npu_id: int, data: NPUError):
    npu_manager.trigger_severe_error(npu_id, data.error_message)
    return {"success": True, "npu_id": npu_id}


@app.post("/npus/clear_error")
async def clear_severe_error(npu_id: int | None = Query(None, description="NPU ID to clear; omit to clear all")):
    npu_manager.clear_severe_error(npu_id)
    return {"success": True, "npu_id": npu_id}


@app.post("/tasks")
async def submit_task(data: TaskSubmit):
    task_type = None
    if data.task_type:
        try:
            task_type = TaskType(data.task_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid task_type: {data.task_type}")

    if data.task_mode is None:
        if task_type == TaskType.FUNCTIONAL:
            task_mode_str = "shared"
        elif task_type in (TaskType.PERFORMANCE, TaskType.BOTH):
            task_mode_str = "exclusive"
        else:
            task_mode_str = "shared"
    else:
        task_mode_str = data.task_mode

    try:
        task_mode = TaskMode(task_mode_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task_mode: {task_mode_str}")

    task = Task(
        task_mode=task_mode,
        task_type=task_type,
        task_label=data.task_label,
        script_path=data.script_path,
        work_dir=data.work_dir,
        args=data.args,
        env=data.env,
        gpu_id=data.npu_id,
    )
    task_id = task_queue.submit_task(task)
    return {"success": True, "task_id": task_id, "task": _task_dict_with_npu_fields(task)}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_dict_with_npu_fields(task)


@app.get("/tasks")
async def list_tasks(status: str | None = Query(None)):
    task_status = None
    if status:
        from server.npu.models import TaskStatus

        try:
            task_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    tasks = task_queue.list_tasks(task_status)
    return {"tasks": [_task_dict_with_npu_fields(t) for t in tasks]}


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, force: bool = Query(False, description="Force cancel running tasks")):
    if force and task_runner:
        if task_queue.force_cancel_task(task_id, task_runner):
            return {"success": True, "task_id": task_id, "forced": True}
        raise HTTPException(status_code=400, detail="Failed to force cancel task")
    if task_queue.cancel_task(task_id):
        return {"success": True, "task_id": task_id, "forced": False}
    raise HTTPException(status_code=400, detail="Failed to cancel task (task may be running, use force=true)")


@app.get("/stats")
async def get_statistics():
    queue_stats = task_queue.get_statistics()
    npu_stats = {
        "total_npus": len(npu_manager.list_npus()),
        "online_npus": sum(1 for n in npu_manager.list_npus() if n.status == NPUStatus.ONLINE),
        "severe_error_active": npu_manager.severe_error_active,
        "paused_npus": npu_manager.list_paused_gpus(),
    }
    return {"queue": queue_stats, "npus": npu_stats}


@app.get("/tasks/{task_id}/log")
async def get_task_log(
    task_id: str,
    log_type: str = Query("summary", description="Log type: summary, stdout, stderr"),
    offset: int = Query(0, description="Byte offset to start reading from"),
    limit: int = Query(102400, description="Maximum bytes to read (default 100KB)"),
):
    from pathlib import Path

    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not task.log_file:
        raise HTTPException(status_code=404, detail="Log file not available yet")

    log_dir = Path(task.log_file).parent
    if log_type == "summary":
        log_path = Path(task.log_file)
    elif log_type == "stdout":
        log_path = log_dir / f"{task_id}.stdout"
    elif log_type == "stderr":
        log_path = log_dir / f"{task_id}.stderr"
    else:
        raise HTTPException(status_code=400, detail="Invalid log_type. Must be: summary, stdout, or stderr")

    if not log_path.exists():
        return {
            "task_id": task_id,
            "log_type": log_type,
            "content": "",
            "total_size": 0,
            "offset": 0,
            "size": 0,
            "truncated": False,
        }

    total_size = log_path.stat().st_size
    if offset < 0:
        offset = 0
    if offset >= total_size:
        offset = total_size

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            content = f.read(limit)
            actual_size = len(content.encode("utf-8"))
            truncated = (offset + actual_size) < total_size
            return {
                "task_id": task_id,
                "log_type": log_type,
                "log_path": str(log_path),
                "content": content,
                "total_size": total_size,
                "offset": offset,
                "size": actual_size,
                "truncated": truncated,
                "has_more": truncated,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {str(e)}")
