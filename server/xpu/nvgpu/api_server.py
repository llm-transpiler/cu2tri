"""
NVIDIA GPU Management API Server

Provides REST API for GPU resource management and task scheduling.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .gpu_manager import GPUManager
from .task_queue import TaskType
from .gpu_info import GPUType

# 添加项目根目录到 Python 路径以支持导入
from pathlib import Path
from utils.set_env import PROJECT_ROOT

from eval_.kernelbench_c.gpu_task_executor import (
    create_correctness_task,
    create_performance_task,
    create_comprehensive_task
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global GPU manager instance
gpu_manager: Optional[GPUManager] = None

# FastAPI app
app = FastAPI(
    title="NVIDIA GPU Management API",
    description="API for managing GPU resources and kernel testing tasks",
    version="1.0.0"
)


# Pydantic models for API requests/responses
class TaskSubmissionRequest(BaseModel):
    task_type: str = Field(..., description="Task type: 'functional' or 'performance'")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    kernel_dir: str = Field(..., description="Directory containing kernel files")
    comparison_type: Optional[str] = Field(None, description="Comparison type for kernel tasks")
    model_name: Optional[str] = Field("gpu_task", description="Model name for logging")
    estimated_memory_mb: int = Field(0, description="Estimated memory usage in MB")
    max_wait_time_minutes: Optional[int] = Field(None, description="Maximum wait time in minutes")


class TaskStatusResponse(BaseModel):
    task_id: str
    name: str
    description: str
    task_type: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    wait_time_seconds: float
    execution_time_seconds: float
    gpu_id: Optional[int]
    error: Optional[str]
    estimated_memory_mb: int


class GPUStatusResponse(BaseModel):
    device_id: int
    name: str
    gpu_type: str
    memory_total_mb: int
    memory_used_mb: int
    memory_usage_ratio: float
    utilization_percent: int
    temperature_c: int
    power_draw_w: int
    available_for_functional: bool
    available_for_performance: bool


class SystemStatusResponse(BaseModel):
    timestamp: str
    gpu_manager_running: bool
    last_gpu_refresh: Optional[str]
    gpus: Dict[str, GPUStatusResponse]
    queue: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """Initialize GPU manager on startup"""
    global gpu_manager
    
    logger.info("Starting GPU Management API Server")
    
    # Create log directory
    log_dir = Path("server/logs/xpu/nvgpu")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GPU manager
    gpu_manager = GPUManager(
        refresh_interval_seconds=10,
        max_task_wait_minutes=30,
        log_dir=str(log_dir)
    )
    
    await gpu_manager.start()
    logger.info("GPU Manager started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global gpu_manager
    
    logger.info("Shutting down GPU Management API Server")
    
    if gpu_manager:
        await gpu_manager.stop()
        logger.info("GPU Manager stopped")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "NVIDIA GPU Management API",
        "version": "1.0.0",
        "status": "running" if gpu_manager and gpu_manager._running else "stopped"
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get overall system status"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        status = await gpu_manager.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpus", response_model=Dict[str, GPUStatusResponse])
async def get_gpu_status():
    """Get GPU status information"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        gpu_status = await gpu_manager.get_gpu_status()
        return {
            gpu_id: GPUStatusResponse(**info) 
            for gpu_id, info in gpu_status.items()
        }
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue", response_model=Dict[str, Any])
async def get_queue_status():
    """Get task queue status"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        return await gpu_manager.get_queue_status()
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/submit", response_model=Dict[str, str])
async def submit_task(request: TaskSubmissionRequest):
    """Submit a new task"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        # Validate task type
        if request.task_type.lower() not in ["functional", "performance"]:
            raise HTTPException(status_code=400, detail="Invalid task type. Must be 'functional' or 'performance'")
        
        task_type = TaskType.FUNCTIONAL if request.task_type.lower() == "functional" else TaskType.PERFORMANCE
        
        # Determine execution function based on comparison type
        if request.comparison_type == "comprehensive":
            execute_func = create_comprehensive_task
            args = (request.kernel_dir, request.model_name)
        elif request.comparison_type in ["triton_vs_torch", "triton_vs_cuda", "cuda_vs_torch"]:
            if task_type == TaskType.FUNCTIONAL:
                execute_func = create_correctness_task
            else:
                execute_func = create_performance_task
            args = (request.kernel_dir, request.comparison_type, request.model_name)
        else:
            raise HTTPException(status_code=400, detail="Invalid comparison type")
        
        # Submit task
        task_id = await gpu_manager.submit_task(
            task_type=task_type,
            name=request.name,
            description=request.description,
            execute_func=execute_func,
            args=args,
            estimated_memory_mb=request.estimated_memory_mb,
            max_wait_time_minutes=request.max_wait_time_minutes
        )
        
        return {"task_id": task_id, "status": "submitted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        status = await gpu_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/tasks/{task_id}", response_model=Dict[str, str])
async def cancel_task(task_id: str):
    """Cancel a pending task"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        success = await gpu_manager.cancel_task(task_id)
        if success:
            return {"task_id": task_id, "status": "cancelled"}
        else:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpus/set-visible", response_model=Dict[str, str])
async def set_visible_gpus(gpu_ids: List[int]):
    """Set visible GPUs for the current process"""
    if not gpu_manager:
        raise HTTPException(status_code=503, detail="GPU Manager not initialized")
    
    try:
        gpu_manager.set_visible_gpus(gpu_ids)
        return {"status": "success", "visible_gpus": str(gpu_ids)}
        
    except Exception as e:
        logger.error(f"Failed to set visible GPUs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    if not gpu_manager or not gpu_manager._running:
        raise HTTPException(status_code=503, detail="GPU Manager not running")
    
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
