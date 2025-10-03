"""Data models for NVGPU server."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from config import GPUMode, GPUStatus, TaskType, TaskStatus


@dataclass
class GPU:
    """GPU resource representation."""
    gpu_id: int
    mode: GPUMode = GPUMode.SHARED
    status: GPUStatus = GPUStatus.ONLINE
    memory_threshold: float = 0.75
    max_concurrent_tasks: int = 3  # Maximum concurrent tasks in shared mode
    current_memory_usage: float = 0.0
    running_tasks: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    def can_accept_task(self) -> bool:
        """Check if GPU can accept a new task."""
        if self.status != GPUStatus.ONLINE:
            return False
        
        if self.mode == GPUMode.EXCLUSIVE:
            return len(self.running_tasks) == 0
        
        # Shared mode: check both task count and memory threshold
        # Task count check is important because tasks may not use GPU memory immediately
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            return False
        
        return self.current_memory_usage < self.memory_threshold
    
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self.status == GPUStatus.ONLINE


@dataclass
class Task:
    """Task representation."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.FUNCTIONAL
    script_path: str = ""
    work_dir: str = "."
    args: list[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None
    
    gpu_id: Optional[int] = None  # Specific GPU, or None for auto-assign
    status: TaskStatus = TaskStatus.PENDING
    
    # Execution info
    assigned_gpu: Optional[int] = None
    submit_time: datetime = field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    exit_code: Optional[int] = None
    log_file: Optional[str] = None
    error_message: Optional[str] = None
    stdout_size: int = 0  # Size of stdout in bytes
    stderr_size: int = 0  # Size of stderr in bytes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "script_path": self.script_path,
            "work_dir": self.work_dir,
            "args": self.args,
            "gpu_id": self.gpu_id,
            "assigned_gpu": self.assigned_gpu,
            "status": self.status.value,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exit_code": self.exit_code,
            "log_file": self.log_file,
            "error_message": self.error_message,
            "stdout_size": self.stdout_size,
            "stderr_size": self.stderr_size,
        }

