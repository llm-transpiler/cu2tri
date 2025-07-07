"""
GPU API Server Module

FastAPI-based REST API server for GPU task management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core.task_dispatcher import TaskDispatcher
from ..core.task import TaskType


# Request/Response Models
class TaskSubmissionRequest(BaseModel):
    """Request model for task submission"""
    task_type: str = Field(..., description="Task type: 'exclusive' or 'shared'")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    module_path: str = Field(..., description="Python module path containing the task function")
    function_name: str = Field(..., description="Function name to execute")
    args: List[Any] = Field(default=[], description="Function arguments")
    kwargs: Dict[str, Any] = Field(default={}, description="Function keyword arguments")
    max_wait_time_minutes: Optional[int] = Field(30, description="Maximum wait time in minutes")
    gpu_id: Optional[int] = Field(None, description="Specific GPU CUDA device ID (None for auto-select)")


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
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
    result: Optional[Any] = None


class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    dispatcher_running: bool
    total_gpus: int
    available_gpus: int
    cooldown_gpus: int
    unhealthy_gpus: int
    available_gpu_ids: List[int]
    global_task_count: int
    queue_statistics: Dict[str, Any]
    configuration: Dict[str, Any]
    gpus: Dict[str, Any]


class GPUAPIServer:
    """GPU Management API Server"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 available_gpu_ids: Optional[List[int]] = None,
                 max_parallel_task_num: int = 4,
                 error_cooldown_seconds: int = 30,
                 log_dir: Optional[str] = None,
                 log_level: str = "INFO",
                 logger: Optional[logging.Logger] = None):
        
        self.host = host
        self.port = port
        self.available_gpu_ids = available_gpu_ids or []
        self.max_parallel_task_num = max_parallel_task_num
        self.error_cooldown_seconds = error_cooldown_seconds
        self.log_level = log_level
        
        # Setup logging
        self._setup_logging(log_dir, logger)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="GPU Task Management API",
            description="Advanced GPU Task Management System with task-based execution",
            version="4.0.0",
            lifespan=self.lifespan
        )
        
        # Task dispatcher will be initialized in lifespan
        self.task_dispatcher: Optional[TaskDispatcher] = None
        
        # Setup routes
        self._setup_routes()
        
        if self.available_gpu_ids:
            self.logger.info(f"[GPUAPIServer] Initialized with available GPUs: {self.available_gpu_ids}")
        else:
            self.logger.info(f"[GPUAPIServer] Initialized with all available GPUs")
        self.logger.info(f"[GPUAPIServer] Max parallel tasks: {self.max_parallel_task_num}")
        self.logger.info(f"[GPUAPIServer] Error cooldown: {self.error_cooldown_seconds} seconds")
    
    def _setup_logging(self, log_dir: Optional[str], logger: Optional[logging.Logger]):
        """Setup logging configuration"""
        if logger:
            self.logger = logger
            return
        
        # Import Path for directory management
        from pathlib import Path
        
        # Determine log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            from utils.set_env import PROJECT_ROOT
            self.log_dir = Path(PROJECT_ROOT) / "server" / "xpu" / "nvgpu_new" / "logs"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter('[%(levelname)s] - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"api_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation
        self.logger.propagate = False
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan event handler"""
        # Startup
        self.logger.info("[GPUAPIServer] Starting GPU Management API Server v4.0")
        
        # Initialize task dispatcher
        self.task_dispatcher = TaskDispatcher(
            available_gpu_ids=self.available_gpu_ids,
            max_parallel_task_num=self.max_parallel_task_num,
            error_cooldown_seconds=self.error_cooldown_seconds,
            logger=self.logger
        )
        
        await self.task_dispatcher.start()
        self.logger.info(f"[GPUAPIServer] Task dispatcher started with {len(self.task_dispatcher.gpu_managers)} GPUs")
        
        yield
        
        # Shutdown
        self.logger.info("[GPUAPIServer] Shutting down GPU Management API Server")
        if self.task_dispatcher:
            await self.task_dispatcher.stop()
            self.logger.info("[GPUAPIServer] Task dispatcher stopped")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "service": "NVIDIA GPU Management API",
                "version": "4.0.0", 
                "architecture": "Task-based execution with exclusive/shared GPU access",
                "status": "running"
            }
        
        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get comprehensive system status"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            status = await self.task_dispatcher.get_system_status()
            return SystemStatusResponse(**status)
        
        @self.app.get("/gpus", response_model=Dict[str, Any])
        async def get_gpu_status():
            """Get GPU status information"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            system_status = await self.task_dispatcher.get_system_status()
            return system_status["gpus"]
        
        @self.app.post("/tasks/submit", response_model=Dict[str, str])
        async def submit_task(request: TaskSubmissionRequest):
            """Submit a task for execution"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            # Validate task type
            try:
                task_type = TaskType(request.task_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid task type: {request.task_type}. Must be 'exclusive' or 'shared'"
                )
            
            try:
                # Create execution function
                async def execute_user_task():
                    """Execute the user-defined task"""
                    try:
                        # Import the module
                        import importlib
                        module = importlib.import_module(request.module_path)
                        
                        # Get the function
                        if not hasattr(module, request.function_name):
                            raise AttributeError(f"Function '{request.function_name}' not found in module '{request.module_path}'")
                        
                        func = getattr(module, request.function_name)
                        
                        # TODO 这里似乎一定是else, 决定了是单进程单线程执行
                        # Execute the function
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*request.args, **request.kwargs)
                        else:
                            result = func(*request.args, **request.kwargs)
                        
                        return result
                        
                    except Exception as e:
                        self.logger.error(f"[GPUAPIServer] Task execution error: {e}")
                        raise
                
                # Submit task to dispatcher
                task_id = await self.task_dispatcher.submit_task(
                    task_type=task_type,
                    name=request.name,
                    description=request.description,
                    execute_func=execute_user_task,
                    max_wait_time_minutes=request.max_wait_time_minutes or 15,
                    preferred_gpu_id=request.gpu_id
                )
                
                return {"task_id": task_id, "status": "submitted"}
                
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to submit task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
        async def get_task_status(task_id: str):
            """Get task status by ID"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            status = await self.task_dispatcher.get_task_status(task_id)
            
            if not status:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            
            return TaskStatusResponse(**status)
        
        @self.app.delete("/tasks/{task_id}", response_model=Dict[str, str])
        async def cancel_task(task_id: str):
            """Cancel a pending task"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            success = await self.task_dispatcher.cancel_task(task_id)
            
            if not success:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")
            
            return {"task_id": task_id, "status": "cancelled"}
        
        @self.app.delete("/tasks/{task_id}/kill", response_model=Dict[str, str])
        async def kill_task(task_id: str):
            """Kill a running task"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            success = await self.task_dispatcher.kill_task(task_id)
            
            if not success:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be killed")
            
            return {"task_id": task_id, "status": "killed"}
        
        @self.app.get("/gpus/available", response_model=Dict[str, Any])
        async def get_available_gpus():
            """Get available GPU IDs and basic info"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            gpu_ids = self.task_dispatcher.get_available_gpu_ids()
            return {
                "available_gpu_ids": gpu_ids,
                "total_gpus": len(gpu_ids),
                "configured_gpu_ids": self.available_gpu_ids
            }
        
        @self.app.get("/task-types", response_model=Dict[str, Any])
        async def get_task_types():
            """Get available task types"""
            return {
                "available_types": [task_type.value for task_type in TaskType],
                "descriptions": {
                    TaskType.EXCLUSIVE.value: "Requires exclusive GPU access",
                    TaskType.SHARED.value: "Can share GPU with other shared tasks"
                }
            }
        
        @self.app.get("/tasks", response_model=Dict[str, Any])
        async def get_all_tasks():
            """Get system-wide task statistics"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            system_status = await self.task_dispatcher.get_system_status()
            return {
                "queue_statistics": system_status["queue_statistics"],
                "gpu_count": system_status["total_gpus"],
                "available_gpus": system_status["available_gpus"],
                "cooldown_gpus": system_status["cooldown_gpus"],
                "task_registry_size": len(self.task_dispatcher.global_task_registry)
            }
        
        @self.app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint"""
            if not self.task_dispatcher or not self.task_dispatcher._running:
                raise HTTPException(status_code=503, detail="Service unavailable")
            
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.post("/gpus/{gpu_id}/health-check", response_model=Dict[str, Any])
        async def perform_gpu_health_check(gpu_id: int):
            """
            Perform health check on a specific GPU
            Only allowed when GPU has no running tasks
            """
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            try:
                if gpu_id not in self.task_dispatcher.gpu_managers:
                    raise HTTPException(status_code=404, detail=f"GPU {gpu_id} not found")
                
                gpu_manager = self.task_dispatcher.gpu_managers[gpu_id]
                
                # Perform health check
                is_healthy = await gpu_manager.perform_health_check()
                
                return {
                    "gpu_id": gpu_id,
                    "health_check_passed": is_healthy,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Health check completed successfully" if is_healthy else "Health check failed"
                }
                
            except RuntimeError as e:
                if "has running tasks" in str(e):
                    raise HTTPException(status_code=409, detail=str(e))
                else:
                    raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                self.logger.error(f"[APIServer] Health check error for GPU {gpu_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

        @self.app.get("/gpus/{gpu_id}/health", response_model=Dict[str, Any])
        async def get_gpu_health_status(gpu_id: int):
            """Get health status of a specific GPU"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            try:
                if gpu_id not in self.task_dispatcher.gpu_managers:
                    raise HTTPException(status_code=404, detail=f"GPU {gpu_id} not found")
                
                gpu_manager = self.task_dispatcher.gpu_managers[gpu_id]
                status = await gpu_manager.get_status()
                
                return {
                    "gpu_id": gpu_id,
                    "health_status": {
                        "last_health_check": status["last_health_check"],
                        "last_health_check_passed": status["last_health_check_passed"],
                        "last_health_error": status["last_health_error"],
                        "is_available": status["is_available"],
                        "is_in_cooldown": status["is_in_cooldown"],
                        "cooldown_reason": status["cooldown_reason"],
                        "cooldown_expires_at": status["cooldown_expires_at"],
                        "cuda_error_count": status["cuda_error_count"],
                        "has_running_tasks": gpu_manager.task_queue.has_running_tasks()
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"[APIServer] Error getting health status for GPU {gpu_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/gpus/configuration", response_model=Dict[str, Any])
        async def update_gpu_configuration(gpu_ids: List[int]):
            """Update the list of available GPUs dynamically"""
            if not self.task_dispatcher:
                raise HTTPException(status_code=503, detail="Task dispatcher not initialized")
            
            try:
                result = await self.task_dispatcher.update_available_gpus(gpu_ids)
                
                # Update server configuration
                self.available_gpu_ids = gpu_ids
                
                return {
                    "status": "updated",
                    "timestamp": datetime.now().isoformat(),
                    "configuration_changes": result
                }
                
            except Exception as e:
                self.logger.error(f"[APIServer] Error updating GPU configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/gpus/system", response_model=Dict[str, Any])
        async def get_system_gpu_info():
            """Get information about all system GPUs (regardless of availability configuration)"""
            try:
                from ..base.gpu_info import get_all_gpus_info
                all_gpus = get_all_gpus_info()
                
                return {
                    "system_gpus": {
                        str(gpu_id): {
                            "name": gpu_info.name,
                            "nvidia_smi_id": gpu_info.nvidia_smi_id,
                            "memory_total_mb": gpu_info.memory_total_mb,
                            "memory_used_mb": gpu_info.memory_used_mb,
                            "memory_free_mb": gpu_info.memory_free_mb,
                            "utilization_percent": gpu_info.utilization_percent,
                            "temperature_c": gpu_info.temperature_c,
                            "power_draw_w": gpu_info.power_draw_w,
                            "gpu_type": gpu_info.gpu_type.value,
                            "is_available_for_functional": gpu_info.is_available_for_functional,
                            "is_available_for_performance": gpu_info.is_available_for_performance,
                            "is_managed": gpu_id in self.task_dispatcher.gpu_managers if self.task_dispatcher else False
                        }
                        for gpu_id, gpu_info in all_gpus.items()
                    },
                    "current_configuration": {
                        "available_gpu_ids": self.available_gpu_ids,
                        "managed_gpu_ids": list(self.task_dispatcher.gpu_managers.keys()) if self.task_dispatcher else []
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"[APIServer] Error getting system GPU info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self):
        """Start the API server programmatically"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level.lower()
        )
        server = uvicorn.Server(config)
        await server.serve()


def create_app(available_gpu_ids: Optional[List[int]] = None,
               max_parallel_task_num: int = 4,
               log_dir: Optional[str] = None) -> FastAPI:
    """Create FastAPI app instance"""
    server = GPUAPIServer(
        available_gpu_ids=available_gpu_ids,
        max_parallel_task_num=max_parallel_task_num,
        log_dir=log_dir
    )
    return server.app


async def run_gpu_api_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    available_gpu_ids: Optional[List[int]] = None,
    max_parallel_task_num: int = 4,
    log_dir: Optional[str] = None
):
    """Run the GPU API server"""
    server = GPUAPIServer(
        host=host,
        port=port,
        available_gpu_ids=available_gpu_ids,
        max_parallel_task_num=max_parallel_task_num,
        log_dir=log_dir
    )
    
    await server.start() 