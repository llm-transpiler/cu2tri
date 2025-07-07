"""
Configuration Settings Management

Centralized configuration loading and validation for the GPU task management system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "config.yaml"


def get_config_path() -> Path:
    """获取配置文件路径"""
    return CONFIG_FILE


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"[Config] Configuration loaded from {CONFIG_FILE}")
        return config or {}
    except Exception as e:
        logger.warning(f"[Config] Failed to load config file {CONFIG_FILE}: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'api_server': {
            'host': '0.0.0.0',
            'port': 8080,
            'log_level': 'INFO',
            'log_dir': 'logs/xpu/nvgpu_new'
        },
        'task_dispatcher': {
            'available_gpu_ids': [],
            'max_parallel_task_num': 4,
            'gpu_manager_refresh_interval': 30,
            'error_cooldown_seconds': 30
        },
        'gpu_manager': {
            'status_refresh_interval': 30
        },
        'cuda_error_protection': {
            'enabled': True,
            'cooldown_seconds': 30,
            'error_patterns': [
                'cuda', 'gpu', 'device', 'out of memory', 'nvml',
                'runtime error', 'driver error', 'kernel launch',
                'device assert', 'illegal memory access'
            ],
            'max_error_count': 10,
            'extended_cooldown_multiplier': 2.0
        },
        'gpu_health_check': {
            'enabled': True,
            'timeout_seconds': 30,
            'isolation': {
                'use_subprocess': True,
                'clean_environment': True,
                'force_cleanup': True
            },
            'operations': {
                'tensor_add': True,
                'matrix_multiply': True,
                'device_properties': True,
                'memory_test': True,
                'test_tensor_size': [2, 2]
            }
        },
        'task_queue': {
            'max_completed_history': 1000,
            'default_max_wait_time_minutes': 30
        },
        'gpu_availability': {
            'max_memory_usage_ratio': 0.8,
            'max_temperature_c': 85
        },
        'task_types': {
            'exclusive': {
                'description': 'Requires exclusive GPU access (独占GPU)',
                'examples': ['Model training', 'Performance benchmarking', 'Memory-intensive computations']
            },
            'shared': {
                'description': 'Can share GPU with other shared tasks (共享GPU)',
                'examples': ['Model inference', 'Batch processing', 'Development and testing']
            }
        },
        'logging': {
            'format': '[%(levelname)s] - %(message)s',
            'console': {
                'enabled': True,
                'level': 'INFO'
            },
            'file': {
                'enabled': True,
                'level': 'DEBUG',
                'max_bytes': 10485760,  # 10MB
                'backup_count': 5
            }
        },
        'load_balancing': {
            'memory_weight': 50,
            'utilization_weight': 0.3,
            'temperature_weight': 0.2,
            'exclusive_queue_weight': 20,
            'shared_queue_weight': 5,
            'running_exclusive_weight': 50,
            'running_shared_weight': 10
        },
        'gpu_id_mapping': {},
        'gpu_specs': {
            'H100_PCIE_80G': {
                'memory_gb': 80,
                'compute_capability': '9.0',
                'tensor_cores': True,
                'max_power_w': 350,
                'memory_bandwidth_gbps': 2000
            },
            'A100_PCIE_40G': {
                'memory_gb': 40,
                'compute_capability': '8.0',
                'tensor_cores': True,
                'max_power_w': 250,
                'memory_bandwidth_gbps': 1555
            },
            'RTX_6000_Ada': {
                'memory_gb': 48,
                'compute_capability': '8.9',
                'tensor_cores': True,
                'max_power_w': 300,
                'memory_bandwidth_gbps': 960
            },
            'L20': {
                'memory_gb': 48,
                'compute_capability': '8.9',
                'tensor_cores': True,
                'max_power_w': 275,
                'memory_bandwidth_gbps': 864
            },
            'UNKNOWN': {
                'memory_gb': 0,
                'compute_capability': '0.0',
                'tensor_cores': False,
                'max_power_w': 0,
                'memory_bandwidth_gbps': 0
            }
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的有效性"""
    required_sections = [
        'api_server', 'task_dispatcher', 'gpu_manager', 
        'cuda_error_protection', 'gpu_health_check'
    ]
    
    for section in required_sections:
        if section not in config:
            logger.error(f"[Config] Missing required section: {section}")
            return False
    
    # 验证API服务器配置
    api_config = config.get('api_server', {})
    if not isinstance(api_config.get('port'), int):
        logger.error("[Config] API server port must be an integer")
        return False
    
    if not (1 <= api_config.get('port', 0) <= 65535):
        logger.error("[Config] API server port must be between 1 and 65535")
        return False
    
    # 验证任务分发器配置
    dispatcher_config = config.get('task_dispatcher', {})
    max_parallel = dispatcher_config.get('max_parallel_task_num', 4)
    if not isinstance(max_parallel, int) or max_parallel < 1:
        logger.error("[Config] max_parallel_task_num must be a positive integer")
        return False
    
    logger.info("[Config] Configuration validation passed")
    return True


def get_gpu_id_mapping() -> Dict[int, int]:
    """获取GPU ID映射关系 (CUDA device ID -> nvidia-smi device ID)"""
    config = load_config()
    return config.get('gpu_id_mapping', {})


def reverse_gpu_id_mapping() -> Dict[int, int]:
    """获取反向GPU ID映射关系 (nvidia-smi device ID -> CUDA device ID)"""
    mapping = get_gpu_id_mapping()
    return {v: k for k, v in mapping.items()}


def get_gpu_specs() -> Dict[str, Dict[str, Any]]:
    """获取GPU规格配置"""
    config = load_config()
    return config.get('gpu_specs', {})


def get_task_dispatcher_config() -> Dict[str, Any]:
    """获取任务分发器配置"""
    config = load_config()
    return config.get('task_dispatcher', {})


def get_api_server_config() -> Dict[str, Any]:
    """获取API服务器配置"""
    config = load_config()
    return config.get('api_server', {})


def get_cuda_error_protection_config() -> Dict[str, Any]:
    """获取CUDA错误保护配置"""
    config = load_config()
    return config.get('cuda_error_protection', {})


def get_gpu_health_check_config() -> Dict[str, Any]:
    """获取GPU健康检查配置"""
    config = load_config()
    return config.get('gpu_health_check', {}) 