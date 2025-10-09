"""Data models for NVGPU server."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from config import GPUMode, GPUStatus, TaskType, TaskMode, TaskStatus


@dataclass
class GPU:
    """GPU resource representation."""
    gpu_id: int
    mode: GPUMode = GPUMode.SHARED  # Current mode (dynamic, task-driven)
    status: GPUStatus = GPUStatus.ONLINE
    memory_threshold: float = 0.75
    max_concurrent_tasks: int = 3  # Maximum concurrent tasks in shared mode
    current_memory_usage: float = 0.0
    running_tasks: list[str] = field(default_factory=list)
    error_message: str | None = None
    last_error_time: datetime | None = None
    
    # Mode management (new)
    manual_mode: GPUMode | None = None  # Manual mode override (if set)
    mode_locked_by: str | None = None   # Task ID that locked the mode (exclusive tasks)
    
    def can_accept_task(self, task_mode: TaskMode | None = None) -> bool:
        """Check if GPU can accept a new task.
        
        Args:
            task_mode: Optional task mode to check compatibility.
                      If None, checks based on current GPU mode.
        
        Returns:
            True if GPU can accept the task, False otherwise.
        """
        if self.status != GPUStatus.ONLINE:
            return False
        
        # If manual mode is set, it takes precedence
        effective_mode = self.manual_mode if self.manual_mode else self.mode
        
        # If task requires exclusive access
        if task_mode == TaskMode.EXCLUSIVE:
            # Need GPU to be completely free
            return len(self.running_tasks) == 0
        
        # If task can share
        if task_mode == TaskMode.SHARED:
            # Cannot share if GPU is in exclusive mode and has running tasks
            # This applies to both manual exclusive mode and task-driven exclusive mode
            if effective_mode == GPUMode.EXCLUSIVE and len(self.running_tasks) > 0:
                return False
            # Check task count and memory constraints
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                return False
            return self.current_memory_usage < self.memory_threshold
        
        # Backward compatibility: if no task_mode specified, check current mode
        if effective_mode == GPUMode.EXCLUSIVE:
            return len(self.running_tasks) == 0
        
        # Shared mode: check both task count and memory threshold
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
    task_mode: TaskMode = TaskMode.SHARED  # Controls GPU behavior (exclusive/shared)
    task_type: TaskType | None = None  # Business categorization (functional/performance/both)
    task_label: str | None = None  # Specific identification tag (e.g., "xpiler_cuda/add_3_3_256/cuda_vs_triton")
    script_path: str = ""
    work_dir: str = "."
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    
    gpu_id: int | None = None  # Specific GPU, or None for auto-assign
    status: TaskStatus = TaskStatus.PENDING
    
    # Execution info
    assigned_gpu: int | None = None
    submit_time: datetime = field(default_factory=datetime.now)
    queued_time: datetime | None = None  # Time when assigned to GPU queue
    start_time: datetime | None = None
    end_time: datetime | None = None
    
    # Results
    exit_code: int | None = None
    log_file: str | None = None
    error_message: str | None = None
    stdout_size: int = 0  # Size of stdout in bytes
    stderr_size: int = 0  # Size of stderr in bytes
    
    @property
    def pending_time_ms(self) -> float | None:
        """Time spent in PENDING state (submit to GPU assignment), in milliseconds with 2 decimal places."""
        if self.queued_time:
            return round((self.queued_time - self.submit_time).total_seconds() * 1000, 2)
        return None
    
    @property
    def queue_time_ms(self) -> float | None:
        """Time spent in QUEUED state (GPU assignment to execution start), in milliseconds with 2 decimal places."""
        if self.queued_time and self.start_time:
            return round((self.start_time - self.queued_time).total_seconds() * 1000, 2)
        return None
    
    @property
    def waiting_time_ms(self) -> float | None:
        """Total waiting time (submit to execution start), in milliseconds with 2 decimal places."""
        if self.start_time:
            return round((self.start_time - self.submit_time).total_seconds() * 1000, 2)
        return None
    
    @property
    def execution_time_ms(self) -> float | None:
        """Execution time (start to end), in milliseconds with 2 decimal places."""
        if self.start_time and self.end_time:
            return round((self.end_time - self.start_time).total_seconds() * 1000, 2)
        return None
    
    @property
    def total_time_ms(self) -> float | None:
        """Total time (submit to end), in milliseconds with 2 decimal places."""
        if self.end_time:
            return round((self.end_time - self.submit_time).total_seconds() * 1000, 2)
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary."""
        result = {
            "task_id": self.task_id,
            "task_mode": self.task_mode.value,
            "script_path": self.script_path,
            "work_dir": self.work_dir,
            "args": self.args,
            "gpu_id": self.gpu_id,
            "assigned_gpu": self.assigned_gpu,
            "status": self.status.value,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "queued_time": self.queued_time.isoformat() if self.queued_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "exit_code": self.exit_code,
            "log_file": self.log_file,
            "error_message": self.error_message,
            "stdout_size": self.stdout_size,
            "stderr_size": self.stderr_size,
        }
        
        # Only include task_type if it's set
        if self.task_type is not None:
            result["task_type"] = self.task_type.value
        
        # Only include task_label if it's set
        if self.task_label is not None:
            result["task_label"] = self.task_label
        
        # Add computed timing fields (in milliseconds)
        result["pending_time_ms"] = self.pending_time_ms
        result["queue_time_ms"] = self.queue_time_ms
        result["waiting_time_ms"] = self.waiting_time_ms
        result["execution_time_ms"] = self.execution_time_ms
        result["total_time_ms"] = self.total_time_ms
        
        return result

