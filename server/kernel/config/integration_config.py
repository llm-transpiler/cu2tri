#!/usr/bin/env python3
"""
集成配置文件
定义内核开发服务器与现有系统的集成配置
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import dotenv

dotenv.load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# GPU配置
GPU_CONFIG = {
    "development": {
        "gpu_ids": [0, 1],      # L20 GPUs
        "max_concurrent": 2,
        "priority": "low",
        "memory_limit_gb": 24,
        "idle_threshold": 10    # 10% utilization threshold
    },
    "production": {
        "gpu_ids": [2, 3, 4, 5],  # H100 GPUs
        "max_concurrent": 4,
        "priority": "high",
        "memory_limit_gb": 80,
        "idle_threshold": 5     # 5% utilization threshold
    }
}

# 集成路径配置
INTEGRATION_PATHS = {
    "cu2tri": PROJECT_ROOT / "cu2tri",
    "eval_triton": PROJECT_ROOT / "eval_triton.py",
    "llm_providers": PROJECT_ROOT / "llm" / "providers",
    "chat_service": PROJECT_ROOT / "server" / "chat",
    "third_party": PROJECT_ROOT / "third_party"
}

# LLM服务配置
LLM_CONFIG = {
    "providers": {
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "models": {
                "code_generation": "anthropic/claude-3.5-sonnet",
                "optimization": "openai/gpt-4-turbo-preview",
                "debugging": "anthropic/claude-3-haiku"
            }
        },
        "gemini": {
            "api_key_env": "GEMINI_API_KEY",
            "models": {
                "code_review": "gemini-pro",
                "performance_analysis": "gemini-pro"
            }
        }
    },
    "conversation_settings": {
        "max_history": 10,
        "context_window": 32000,
        "temperature": 0.35,
        "max_tokens": 4096 * 4
    }
}

# 编译器配置
COMPILER_CONFIG = {
    "cuda": {
        "nvcc_path": "/usr/local/cuda/bin/nvcc",
        "arch_flags": ["sm_80", "sm_86", "sm_89", "sm_90"],
        "optimization_flags": ["-O3", "-use_fast_math"],
        "include_paths": [
            "/usr/local/cuda/include",
            "/opt/conda/include"
        ]
    },
    "triton": {
        "optimization_level": "O3",
        "enable_profiling": True,
        "cache_dir": PROJECT_ROOT / "cache" / "triton",
        "compile_timeout": 300  # seconds
    }
}

# 测试配置
TEST_CONFIG = {
    "functional": {
        "tolerance": 1e-5,
        "max_test_size": 10000,
        "timeout": 60  # seconds
    },
    "performance": {
        "warmup_runs": 10,
        "benchmark_runs": 100,
        "timeout": 300,  # seconds
        "memory_check": True,
        "profile_kernels": True
    }
}

# 队列配置
QUEUE_CONFIG = {
    "max_workers": 8,
    "retry_attempts": 3,
    "retry_delay": 5,  # seconds
    "task_timeout": 600,  # seconds
    "priority_levels": 5,
    "cleanup_interval": 300  # seconds
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "files": {
        "main": PROJECT_ROOT / "server" / "kernel" / "logs" / "kernel_service.log",
        "gpu": PROJECT_ROOT / "server" / "kernel" / "logs" / "gpu_manager.log",
        "queue": PROJECT_ROOT / "server" / "kernel" / "logs" / "queue_manager.log",
        "integration": PROJECT_ROOT / "server" / "kernel" / "logs" / "integration.log"
    },
    "rotation": {
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
}

# 监控配置
MONITORING_CONFIG = {
    "gpu_check_interval": 5,  # seconds
    "health_check_interval": 30,  # seconds
    "metrics_retention": 24 * 3600,  # 24 hours in seconds
    "alerts": {
        "gpu_temperature_threshold": 85,  # celsius
        "memory_usage_threshold": 0.9,   # 90%
        "queue_size_threshold": 100
    }
}

# API服务器配置
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "cors_origins": ["*"],
    "max_request_size": 100 * 1024 * 1024,  # 100MB
    "request_timeout": 300,  # seconds
    "websocket_timeout": 3600  # seconds
}

# 安全配置
SECURITY_CONFIG = {
    "api_keys": {
        "admin": os.getenv("KERNEL_ADMIN_KEY"),
        "user": os.getenv("KERNEL_USER_KEY")
    },
    "allowed_ips": ["127.0.0.1", "localhost"],
    "rate_limiting": {
        "requests_per_minute": 60,
        "burst_size": 10
    }
}

# 数据库配置（如果需要持久化）
DATABASE_CONFIG = {
    "type": "sqlite",  # 或 "postgresql", "mysql"
    "path": PROJECT_ROOT / "server" / "kernel" / "data" / "kernel_db.sqlite",
    "connection_pool_size": 10,
    "echo": False  # SQLAlchemy echo
}

# 缓存配置
CACHE_CONFIG = {
    "type": "redis",  # 或 "memory", "file"
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": os.getenv("REDIS_PASSWORD")
    },
    "ttl": {
        "compile_cache": 3600,  # 1 hour
        "test_results": 1800,   # 30 minutes
        "gpu_stats": 60         # 1 minute
    }
}

# 集成钩子配置
INTEGRATION_HOOKS = {
    "pre_compile": [
        "check_gpu_availability",
        "validate_code_syntax",
        "check_resource_limits"
    ],
    "post_compile": [
        "update_compile_cache",
        "send_notifications",
        "log_metrics"
    ],
    "pre_test": [
        "allocate_gpu_resources",
        "prepare_test_data"
    ],
    "post_test": [
        "release_gpu_resources",
        "archive_test_results"
    ]
}

# 实验性特性配置
EXPERIMENTAL_CONFIG = {
    "auto_optimization": {
        "enabled": True,
        "optimization_passes": 3,
        "genetic_algorithm": False
    },
    "ai_debugging": {
        "enabled": True,
        "error_analysis": True,
        "suggestion_generation": True
    },
    "distributed_compilation": {
        "enabled": False,
        "cluster_nodes": []
    }
}

def get_config(section: str) -> Dict:
    """获取指定section的配置"""
    config_map = {
        "gpu": GPU_CONFIG,
        "integration": INTEGRATION_PATHS,
        "llm": LLM_CONFIG,
        "compiler": COMPILER_CONFIG,
        "test": TEST_CONFIG,
        "queue": QUEUE_CONFIG,
        "logging": LOGGING_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "api": API_CONFIG,
        "security": SECURITY_CONFIG,
        "database": DATABASE_CONFIG,
        "cache": CACHE_CONFIG,
        "hooks": INTEGRATION_HOOKS,
        "experimental": EXPERIMENTAL_CONFIG
    }
    
    return config_map.get(section, {})

def validate_config() -> bool:
    """验证配置有效性"""
    errors = []
    
    # 检查GPU配置
    if not GPU_CONFIG["development"]["gpu_ids"] or not GPU_CONFIG["production"]["gpu_ids"]:
        errors.append("GPU配置缺少gpu_ids")
    
    # 检查路径配置
    for path_name, path in INTEGRATION_PATHS.items():
        if path_name in ["cu2tri", "eval_triton"] and not path.exists():
            errors.append(f"集成路径不存在: {path_name} -> {path}")
    
    # 检查API密钥
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        errors.append("缺少LLM API密钥")
    
    # 检查CUDA环境
    cuda_nvcc = Path(COMPILER_CONFIG["cuda"]["nvcc_path"])
    if not cuda_nvcc.exists():
        errors.append(f"NVCC编译器不存在: {cuda_nvcc}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def create_directories():
    """创建必要的目录"""
    dirs_to_create = [
        PROJECT_ROOT / "server" / "kernel" / "logs",
        PROJECT_ROOT / "server" / "kernel" / "data",
        PROJECT_ROOT / "cache" / "triton",
        PROJECT_ROOT / "cache" / "compiled",
        PROJECT_ROOT / "temp" / "kernels"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_environment_info() -> Dict:
    """获取环境信息"""
    import torch
    import platform
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "project_root": str(PROJECT_ROOT)
    }
    
    if torch.cuda.is_available():
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info

if __name__ == "__main__":
    print("内核开发服务器集成配置")
    print("=" * 50)
    
    # 验证配置
    if validate_config():
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败")
        exit(1)
    
    # 创建目录
    create_directories()
    print("✅ 目录创建完成")
    
    # 显示环境信息
    env_info = get_environment_info()
    print("\n环境信息:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    
    print("\n配置验证完成!") 