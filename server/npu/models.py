"""Data models for NPU server."""
from server.nvgpu.models import GPU as NPU, Task
from server.nvgpu.config import GPUMode as NPUMode
from server.nvgpu.config import GPUStatus as NPUStatus
from server.nvgpu.config import TaskMode, TaskStatus, TaskType

__all__ = [
    "NPU",
    "NPUMode",
    "NPUStatus",
    "Task",
    "TaskMode",
    "TaskStatus",
    "TaskType",
]
