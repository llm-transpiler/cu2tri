"""Configuration for NPU server."""
from server.nvgpu.config import (  # Reuse stable scheduling/task enums.
    GPUMode as NPUMode,
    TaskType,
    TaskMode,
    TaskStatus,
    GPUStatus as NPUStatus,
    LoadBalancingStrategy,
    ServerConfig,
)


# Global server configuration instance (reuse existing structure/fields).
config = ServerConfig()
