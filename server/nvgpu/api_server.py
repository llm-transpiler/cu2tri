"""REST API server for NVGPU service."""
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from gpu_manager import GPUManager
from task_queue import TaskQueue
from scheduler import Scheduler
from models import Task, TaskType, GPUMode, GPUStatus
from config import config
from logger import setup_logger

logger = setup_logger("api_server")

app = FastAPI(title="NVGPU Server", description="GPU Task Scheduling Server")

# Global references (set by main)
gpu_manager: GPUManager = None
task_queue: TaskQueue = None
scheduler: Scheduler = None
task_runner = None  # Added for force cancel


def init_app(gm: GPUManager, tq: TaskQueue, sched: Scheduler, tr=None):
    """Initialize FastAPI app with managers."""
    global gpu_manager, task_queue, scheduler, task_runner
    gpu_manager = gm
    task_queue = tq
    scheduler = sched
    task_runner = tr


# Pydantic models for request bodies
class GPUStatusUpdate(BaseModel):
    status: str

class GPUModeUpdate(BaseModel):
    mode: str

class GPUMemoryThresholdUpdate(BaseModel):
    threshold: float

class GPUMaxConcurrentTasksUpdate(BaseModel):
    max_tasks: int

class GPURegister(BaseModel):
    gpu_id: int
    mode: Optional[str] = None
    memory_threshold: Optional[float] = None
    max_concurrent_tasks: Optional[int] = None

class GPUError(BaseModel):
    error_message: str = "Manual error trigger"

class TaskSubmit(BaseModel):
    script_path: str
    task_type: str = "functional"
    work_dir: str = "."
    args: List[str] = []
    env: Optional[Dict[str, str]] = None
    gpu_id: Optional[int] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/gpus")
async def list_gpus():
    """List all GPUs and their status."""
    gpus = gpu_manager.list_gpus()
    return {
        "gpus": [
            {
                "gpu_id": g.gpu_id,
                "mode": g.mode.value,
                "status": g.status.value,
                "memory_threshold": g.memory_threshold,
                "max_concurrent_tasks": g.max_concurrent_tasks,
                "current_memory_usage": g.current_memory_usage,
                "running_tasks": g.running_tasks,
                "running_task_count": len(g.running_tasks),
                "error_message": g.error_message,
            }
            for g in gpus
        ]
    }


@app.get("/gpus/{gpu_id}")
async def get_gpu(gpu_id: int):
    """Get specific GPU info."""
    gpu = gpu_manager.get_gpu(gpu_id)
    if not gpu:
        raise HTTPException(status_code=404, detail="GPU not found")
    
    return {
        "gpu_id": gpu.gpu_id,
        "mode": gpu.mode.value,
        "status": gpu.status.value,
        "memory_threshold": gpu.memory_threshold,
        "max_concurrent_tasks": gpu.max_concurrent_tasks,
        "current_memory_usage": gpu.current_memory_usage,
        "running_tasks": gpu.running_tasks,
        "running_task_count": len(gpu.running_tasks),
        "error_message": gpu.error_message,
    }


@app.put("/gpus/{gpu_id}/status")
async def set_gpu_status(gpu_id: int, data: GPUStatusUpdate):
    """Set GPU status (online/offline/maintenance)."""
    try:
        status = GPUStatus(data.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {data.status}")
    
    if gpu_manager.set_gpu_status(gpu_id, status):
        return {"success": True, "gpu_id": gpu_id, "status": status.value}
    raise HTTPException(status_code=400, detail="Failed to set GPU status")


@app.put("/gpus/{gpu_id}/mode")
async def set_gpu_mode(gpu_id: int, data: GPUModeUpdate):
    """Set GPU execution mode."""
    try:
        mode = GPUMode(data.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {data.mode}")
    
    if gpu_manager.set_gpu_mode(gpu_id, mode):
        return {"success": True, "gpu_id": gpu_id, "mode": mode.value}
    raise HTTPException(status_code=400, detail="Failed to set GPU mode")


@app.put("/gpus/{gpu_id}/memory_threshold")
async def set_gpu_memory_threshold(gpu_id: int, data: GPUMemoryThresholdUpdate):
    """Set GPU memory threshold."""
    if gpu_manager.set_gpu_memory_threshold(gpu_id, data.threshold):
        return {"success": True, "gpu_id": gpu_id, "threshold": data.threshold}
    raise HTTPException(status_code=400, detail="Failed to set memory threshold")


@app.put("/gpus/{gpu_id}/max_concurrent_tasks")
async def set_gpu_max_concurrent_tasks(gpu_id: int, data: GPUMaxConcurrentTasksUpdate):
    """Set GPU maximum concurrent tasks."""
    if gpu_manager.set_gpu_max_concurrent_tasks(gpu_id, data.max_tasks):
        return {"success": True, "gpu_id": gpu_id, "max_tasks": data.max_tasks}
    raise HTTPException(status_code=400, detail="Failed to set max concurrent tasks")


@app.post("/gpus/register")
async def register_gpu(data: GPURegister):
    """Register a new GPU."""
    mode = None
    if data.mode:
        try:
            mode = GPUMode(data.mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {data.mode}")
    
    if gpu_manager.register_gpu(data.gpu_id, mode, data.memory_threshold, data.max_concurrent_tasks):
        return {"success": True, "gpu_id": data.gpu_id}
    raise HTTPException(status_code=400, detail="Failed to register GPU")


@app.post("/gpus/{gpu_id}/unregister")
async def unregister_gpu(gpu_id: int):
    """Unregister a GPU."""
    if gpu_manager.unregister_gpu(gpu_id):
        return {"success": True, "gpu_id": gpu_id}
    raise HTTPException(status_code=400, detail="Failed to unregister GPU")


@app.post("/gpus/{gpu_id}/error")
async def trigger_gpu_error(gpu_id: int, data: GPUError):
    """Manually trigger a severe GPU error."""
    gpu_manager.trigger_severe_error(gpu_id, data.error_message)
    return {"success": True, "gpu_id": gpu_id}


@app.post("/gpus/clear_error")
async def clear_severe_error():
    """Clear severe error state."""
    gpu_manager.clear_severe_error()
    return {"success": True}


@app.post("/tasks")
async def submit_task(data: TaskSubmit):
    """Submit a new task."""
    # Parse task type
    try:
        task_type = TaskType(data.task_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid task_type: {data.task_type}")
    
    # Create task
    task = Task(
        task_type=task_type,
        script_path=data.script_path,
        work_dir=data.work_dir,
        args=data.args,
        env=data.env,
        gpu_id=data.gpu_id,
    )
    
    # Submit to queue
    task_id = task_queue.submit_task(task)
    
    return {
        "success": True,
        "task_id": task_id,
        "task": task.to_dict()
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and results."""
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()


@app.get("/tasks")
async def list_tasks(status: Optional[str] = Query(None)):
    """List all tasks with optional status filter."""
    task_status = None
    
    if status:
        try:
            from models import TaskStatus
            task_status = TaskStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    tasks = task_queue.list_tasks(task_status)
    return {
        "tasks": [t.to_dict() for t in tasks]
    }


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, force: bool = Query(False, description="Force cancel running tasks")):
    """Cancel a task.
    
    Args:
        task_id: Task ID to cancel
        force: If True, will kill running tasks. If False, only cancels pending/queued tasks.
    """
    if force and task_runner:
        # Force cancel (including running tasks)
        if task_queue.force_cancel_task(task_id, task_runner):
            return {"success": True, "task_id": task_id, "forced": True}
        raise HTTPException(status_code=400, detail="Failed to force cancel task")
    else:
        # Regular cancel (only pending/queued)
        if task_queue.cancel_task(task_id):
            return {"success": True, "task_id": task_id, "forced": False}
        raise HTTPException(status_code=400, detail="Failed to cancel task (task may be running, use force=true)")


@app.get("/stats")
async def get_statistics():
    """Get server statistics."""
    queue_stats = task_queue.get_statistics()
    gpu_stats = {
        "total_gpus": len(gpu_manager.list_gpus()),
        "online_gpus": sum(1 for g in gpu_manager.list_gpus() if g.status == GPUStatus.ONLINE),
        "severe_error_active": gpu_manager.severe_error_active,
    }
    
    return {
        "queue": queue_stats,
        "gpus": gpu_stats
    }


@app.get("/tasks/{task_id}/log")
async def get_task_log(
    task_id: str,
    log_type: str = Query("summary", description="Log type: summary, stdout, stderr"),
    offset: int = Query(0, description="Byte offset to start reading from"),
    limit: int = Query(102400, description="Maximum bytes to read (default 100KB)")
):
    """Get task log content.
    
    Args:
        task_id: Task ID
        log_type: Type of log (summary, stdout, stderr)
        offset: Byte offset to start reading from
        limit: Maximum bytes to read
        
    Returns:
        Log content and metadata
    """
    from pathlib import Path
    
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if not task.log_file:
        raise HTTPException(status_code=404, detail="Log file not available yet")
    
    log_dir = Path(task.log_file).parent
    
    # Determine which file to read
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
            "truncated": False
        }
    
    # Get file size
    total_size = log_path.stat().st_size
    
    # Validate offset
    if offset < 0:
        offset = 0
    if offset >= total_size:
        offset = total_size
    
    # Read content
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(offset)
            content = f.read(limit)
            actual_size = len(content.encode('utf-8'))
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
                "has_more": truncated
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {str(e)}")
