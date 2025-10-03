"""Configuration for NVGPU server."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GPUMode(str, Enum):
    """GPU execution mode."""
    EXCLUSIVE = "exclusive"  # Only one task at a time (exclusive access)
    SHARED = "shared"        # Multiple tasks allowed (shared access)


class TaskType(str, Enum):
    """Task type."""
    FUNCTIONAL = "functional"  # Functional test
    PERFORMANCE = "performance"  # Performance test


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GPUStatus(str, Enum):
    """GPU status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"      # Severe error, all tasks paused
    MAINTENANCE = "maintenance"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    log_file: Optional[str] = "logs/nvgpu_server.log"
    log_level: str = "INFO"
    
    # Default GPU settings
    default_gpu_mode: GPUMode = GPUMode.SHARED
    default_memory_threshold: float = 0.75  # 75% memory usage threshold
    default_max_concurrent_tasks: int = 3   # Maximum concurrent tasks in shared mode
    
    # Task settings
    task_timeout: int = 600  # 10 minutes default timeout
    error_pause_duration: int = 60  # 1 minute pause on severe error (auto-resume)
    
    # Scheduler settings
    scheduler_interval: float = 1.0  # Check every 1 second
    gpu_monitor_interval: float = 5.0  # Monitor GPU every 5 seconds


# Global server configuration instance
config = ServerConfig()

