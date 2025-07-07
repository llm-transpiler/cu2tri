"""
API server components for GPU task management system.

This module contains the HTTP API server that provides:
- RESTful endpoints for task management
- GPU status and monitoring APIs
- System health and configuration APIs
"""

from .server import GPUAPIServer

# Convenience alias
APIServer = GPUAPIServer

__all__ = [
    'GPUAPIServer',
    'APIServer',
] 