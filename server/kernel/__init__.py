"""
CUDA/Triton 内核开发与测试服务
包含编译、功能测试、性能测试和与LLM交互的完整服务
"""

__version__ = "1.0.0"
__author__ = "Kernel Development Team"

from .models import (
    KernelRequest, KernelResponse, ErrorResponse,
    CompileRequest, TestRequest, PerfRequest,
    KernelType, GPUType, TestStage
)

from .service import KernelDevelopmentService
from .gpu_manager import GPUResourceManager
from .queue_manager import KernelQueueManager

__all__ = [
    "KernelRequest", "KernelResponse", "ErrorResponse",
    "CompileRequest", "TestRequest", "PerfRequest", 
    "KernelType", "GPUType", "TestStage",
    "KernelDevelopmentService",
    "GPUResourceManager", 
    "KernelQueueManager"
] 