"""
NVIDIA GPU Management API Server

Provides REST API for GPU resource management and task scheduling.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .gpu_manager import GPUManager
from .task_queue import TaskType


class TaskSubmissionRequest(BaseModel):
    task_type: str = Field(..., description="Task type: 'functional' or 'performance'")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    module_path: str = Field(..., description="Python module path containing the task function")
    function_name: str = Field(..., description="Function name to execute")
    args: List[Any] = Field(default=[], description="Function arguments")
    kwargs: Dict[str, Any] = Field(default={}, description="Function keyword arguments")
    max_wait_time_minutes: Optional[int] = Field(None, description="Maximum wait time in minutes")
    preferred_gpu_id: Optional[int] = Field(None, description="Preferred GPU ID (if available)")
    allow_fallback: bool = Field(False, description="Allow fallback to other GPUs if preferred is unavailable")
    require_same_gpu_type: bool = Field(True, description="Require fallback GPU to have same type as preferred GPU")


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
    result: Optional[Any] = None


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


class KernelTaskRequest(BaseModel):
    """Convenience model for kernel testing tasks"""
    task_type: str = Field(..., description="Task type: 'functional' or 'performance'")
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    kernel_dir: str = Field(..., description="Directory containing kernel files")
    model_name: Optional[str] = Field("gpu_task", description="Model name for logging")
    max_wait_time_minutes: Optional[int] = Field(None, description="Maximum wait time in minutes")
    preferred_gpu_id: Optional[int] = Field(None, description="Preferred GPU ID (if available)")
    allow_fallback: bool = Field(False, description="Allow fallback to other GPUs if preferred is unavailable")
    require_same_gpu_type: bool = Field(True, description="Require fallback GPU to have same type as preferred GPU")


class GPUAPIServer:
    """GPU Management API Server"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 log_dir: Optional[str] = None,
                 log_file: Optional[str] = None,
                 log_level: str = "INFO",
                 logger: Optional[logging.Logger] = None,
                 ):
        """Initialize GPU API Server"""
        self.host = host
        self.port = port
        self.log_level = log_level
        self.log_file = None
        if log_dir is not None and log_file is not None:
            self.log_dir = Path(log_dir)
            self.log_file = Path(log_file)
            if not self.log_file.parent == self.log_dir:
                raise ValueError(f"log_file{self.log_file} must be in log_dir{self.log_dir}")
        elif log_dir is not None and log_file is None:
            self.log_dir = Path(log_dir)
            self.log_file = self.log_dir / f"gpu_resource_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            from utils.set_env import PROJECT_ROOT
            self.log_dir = Path(PROJECT_ROOT / "server" / "logs" / "xpu" / "nvgpu")
            self.log_file = self.log_dir / f"gpu_resource_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logger or logging.getLogger(__name__)
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # # 清理旧的时间戳命名的日志文件
        # self._cleanup_old_log_files()
        
        # 如果没有传入自定义logger，则配置日志
        if self.logger.hasHandlers():
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        formatter = logging.Formatter('[%(levelname)s] - %(message)s')
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        # console_handler.setLevel(self.log_level)
        self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setFormatter(formatter)
        # file_handler.setLevel(self.log_level)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(self.log_level)
        
        # 防止日志向上传播到根日志器，避免重复
        self.logger.propagate = False
        
        # GPU manager instance
        self.gpu_manager: Optional[GPUManager] = None
        
        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="NVIDIA GPU Management API",
            description="API for managing GPU resources and kernel testing tasks",
            version="1.0.0",
            lifespan=self.lifespan
        )
        
        # Setup routes
        self._setup_routes()
    
    def _cleanup_old_log_files(self, keep_days: int = 7):
        """清理旧的时间戳命名的日志文件和空文件"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (keep_days * 24 * 3600)
        
        for log_file in self.log_dir.glob("gpu_resource_server_*.log"):
            try:
                # 删除旧的时间戳文件或空文件
                file_stat = log_file.stat()
                if file_stat.st_mtime < cutoff_time or file_stat.st_size == 0:
                    log_file.unlink()
                    print(f"Cleaned up old log file: {log_file.name}")
            except Exception as e:
                print(f"Failed to clean up log file {log_file.name}: {e}")
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan event handler"""
        # Startup
        self.logger.info("[GPUAPIServer] Starting GPU Management API Server")
        
        # Initialize GPU manager
        self.gpu_manager = GPUManager(log_dir=str(self.log_dir), logger=self.logger)
        await self.gpu_manager.start()
        self.logger.info("[GPUAPIServer] GPU Manager started successfully")
        
        yield
        
        # Shutdown
        self.logger.info("[GPUAPIServer] Shutting down GPU Management API Server")
        if self.gpu_manager:
            await self.gpu_manager.stop()
            self.logger.info("[GPUAPIServer] GPU Manager stopped")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "NVIDIA GPU Management API",
                "version": "0.0.10",
                "status": "running" if self.gpu_manager and self.gpu_manager._running else "stopped"
            }

        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get overall system status"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                status = await self.gpu_manager.get_system_status()
                return SystemStatusResponse(**status)
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to get system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/gpus", response_model=Dict[str, GPUStatusResponse])
        async def get_gpu_status():
            """Get GPU status information"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                gpu_status = await self.gpu_manager.get_gpu_status()
                return {
                    gpu_id: GPUStatusResponse(**info) 
                    for gpu_id, info in gpu_status.items()
                }
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to get GPU status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/queue", response_model=Dict[str, Any])
        async def get_queue_status():
            """Get task queue status"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                return await self.gpu_manager.get_queue_status()
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to get queue status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/tasks/submit", response_model=Dict[str, str])
        async def submit_task(request: TaskSubmissionRequest):
            """Submit a new task"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                # Validate task type - only functional or performance
                if request.task_type.lower() not in ["functional", "performance"]:
                    raise HTTPException(status_code=400, detail="Invalid task type. Must be 'functional' or 'performance'")
                
                task_type = TaskType.FUNCTIONAL if request.task_type.lower() == "functional" else TaskType.PERFORMANCE
                
                # Dynamic task function creation
                async def execute_user_task():
                    """Dynamically import and execute user-defined task"""
                    import importlib
                    # 强制重新加载模块以避免缓存
                    import sys
                    if request.module_path in sys.modules: # important for hot-reload for task function
                        importlib.reload(sys.modules[request.module_path])
                    
                    # Import the module
                    module = importlib.import_module(request.module_path)
                    
                    # Get the function
                    if not hasattr(module, request.function_name):
                        raise AttributeError(f"Function '{request.function_name}' not found in module '{request.module_path}'")
                    
                    func = getattr(module, request.function_name)
                    
                    # Execute the function
                    if asyncio.iscoroutinefunction(func):
                        return await func(*request.args, **request.kwargs)
                    else:
                        return func(*request.args, **request.kwargs)
                
                # Submit task
                task_id = await self.gpu_manager.submit_task(
                    task_type=task_type,
                    name=request.name,
                    description=request.description,
                    execute_func=execute_user_task,
                    max_wait_time_minutes=request.max_wait_time_minutes,
                    preferred_gpu_id=request.preferred_gpu_id,
                    allow_fallback=request.allow_fallback,
                    require_same_gpu_type=request.require_same_gpu_type
                )
                
                return {"task_id": task_id, "status": "submitted"}
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to submit task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
        async def get_task_status(task_id: str):
            """Get task status"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                status = await self.gpu_manager.get_task_status(task_id)
                if not status:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                return TaskStatusResponse(**status)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to get task status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/tasks/{task_id}", response_model=Dict[str, str])
        async def cancel_task(task_id: str):
            """Cancel a pending task"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                success = await self.gpu_manager.cancel_task(task_id)
                if success:
                    return {"task_id": task_id, "status": "cancelled"}
                else:
                    raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
                    
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to cancel task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/gpus/set-visible", response_model=Dict[str, str])
        async def set_visible_gpus(gpu_ids: List[int]):
            """Set visible GPUs for the current process"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                self.gpu_manager.set_visible_gpus(gpu_ids)
                return {"status": "success", "visible_gpus": str(gpu_ids)}
                
            except Exception as e:
                self.logger.error(f"[GPUAPIServer] Failed to set visible GPUs: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health", response_model=Dict[str, str])
        async def health_check():
            """Health check endpoint"""
            if not self.gpu_manager or not self.gpu_manager._running:
                raise HTTPException(status_code=503, detail="GPU Manager not running")
            
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.post("/tasks/submit-kernel", response_model=Dict[str, str])
        async def submit_kernel_task(request: KernelTaskRequest):
            """Submit a kernel testing task (convenience endpoint)"""
            if not self.gpu_manager:
                raise HTTPException(status_code=503, detail="GPU Manager not initialized")
            
            try:
                # Validate task type
                if request.task_type.lower() not in ["functional", "performance"]:
                    raise HTTPException(status_code=400, detail="Invalid task type. Must be 'functional' or 'performance'")
                
                task_type = TaskType.FUNCTIONAL if request.task_type.lower() == "functional" else TaskType.PERFORMANCE
                
                # Use comprehensive task for kernel testing
                async def execute_kernel_task():
                    """Execute kernel testing task"""
                    from eval_.kernelbench_c.gpu_task_executor import create_kernel_task
                    return await create_kernel_task(request.kernel_dir, request.model_name)
                
                # Submit task
                task_id = await self.gpu_manager.submit_task(
                    task_type=task_type,
                    name=request.name,
                    description=request.description,
                    execute_func=execute_kernel_task,
                    max_wait_time_minutes=request.max_wait_time_minutes,
                    preferred_gpu_id=request.preferred_gpu_id,
                    allow_fallback=request.allow_fallback,
                    require_same_gpu_type=request.require_same_gpu_type
                )
                
                return {"task_id": task_id, "status": "submitted"}
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to submit kernel task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start(self):
        """Start API server"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# Create global app instance for direct uvicorn usage  
def create_app(log_dir: Optional[str] = None) -> FastAPI:
    """Create FastAPI app instance"""
    server = GPUAPIServer(log_dir=log_dir)
    return server.app


## Default app instance
# app = create_app() # bug here, 会导致反复生成新的log文件


# Convenience function for running the server
async def run_gpu_api_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    log_dir: Optional[str] = None
):
    """Run GPU API server"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = GPUAPIServer(host=host, port=port, log_dir=log_dir)
    await server.start()


if __name__ == "__main__":
    asyncio.run(run_gpu_api_server())
