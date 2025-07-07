"""
GPU Task Management System - Modular Architecture

This module provides a comprehensive GPU task management system with:
- Task-based GPU access control (exclusive/shared)
- Dynamic GPU resource allocation  
- Error handling and recovery mechanisms
- Health monitoring and cooldown protection
- RESTful API interface

The system is organized into the following modules:
- base: Fundamental classes and functions (GPUInfo, health checking)
- core: Core management components (Task, GPUManager, TaskDispatcher, GPUTaskQueue)
- api: HTTP API server (APIServer)
- config: Configuration management (settings, config loading)
- examples: Usage examples and demos
- tests: Test suite
- docs: Documentation
"""

__version__ = "2.1.0"

# Import from reorganized modules
from .base import *
from .core import *  
from .api import *
from .config import *

# Convenience aliases for backward compatibility  
from .api.server import GPUAPIServer
APIServer = GPUAPIServer

__all__ = [
    # Base module exports
    'GPUInfo', 'GPUType', 'GPUSpec', 
    'query_gpu_info', 'get_available_gpus', 'get_all_gpus_info', 'set_visible_gpus',
    'GPUHealthChecker',
    
    # Core module exports
    'Task', 'TaskType', 'TaskStatus',
    'GPUManager', 'TaskDispatcher', 'GPUTaskQueue',
    
    # API module exports  
    'APIServer', 'GPUAPIServer',
    
    # Config module exports
    'load_config', 'get_config_path', 'validate_config', 'get_default_config',
    
    # Version
    '__version__',
] 