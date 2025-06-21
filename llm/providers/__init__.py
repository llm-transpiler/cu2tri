# -*- coding: utf-8 -*-
"""
LLM提供商模块 - 三层架构
平台 → 厂商 → 模型的优雅分级设计
"""

# 核心类型定义
from .types import (
    PlatformType, PlatformCategory, PlatformInfo,
    VendorType, ModelSpec, SDKType,
)

# 配置系统
from .config import (
    ConfigManager, PlatformConfig, LLMConfig,
    DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, get_config_manager
)

# 基础接口
from .base import (
    # 统一数据模型
    ChatMessage, ChatRequest, ChatResponse, StreamChunk,
    # Provider基类
    Provider, OpenAICompatibleProvider,
    # 异常类
    ProviderError, AuthenticationError, RateLimitError, 
    ModelNotFoundError, ValidationError, NetworkError, ServiceUnavailableError,
    # 工具类
    RequestValidator, ExceptionConverter,
    # 兼容性接口
    FileManager, Message, ChatHistory
)

# 工厂和管理器
from .factory import (
    ProviderFactory, ProviderManager, get_provider_manager
)

# 注册中心
from .registry import (
    ProviderRegistry, get_provider_registry,
    # 便捷函数
    get_platform_info, list_platforms, list_models,
    # 注册表
    PLATFORM_REGISTRY, VENDOR_MODEL_REGISTRY, SDK_REGISTRY
)

# 具体实现（自动注册）
from . import impl

# 历史兼容
from ..history.parts import Part, TextPart, ThoughtPart, ContentPart, FilePart, ImagePart

# 导出所有公共接口
__all__ = [
    # === 核心类型 ===
    'PlatformType', 'PlatformCategory', 'PlatformInfo',
    'VendorType', 'ModelSpec', 'SDKType',
    
    # === 注册表 ===
    'PLATFORM_REGISTRY', 'VENDOR_MODEL_REGISTRY', 'SDK_REGISTRY',
    
    # === 配置系统 ===
    'ConfigManager', 'PlatformConfig', 'LLMConfig',
    'DEFAULT_TIMEOUT', 'DEFAULT_MAX_RETRIES', 'get_config_manager',
    
    # === 统一接口 ===
    'ChatMessage', 'ChatRequest', 'ChatResponse', 'StreamChunk',
    'Provider', 'OpenAICompatibleProvider',
    
    # === 工具类 ===
    'RequestValidator', 'ExceptionConverter',
    
    # === 异常处理 ===
    'ProviderError', 'AuthenticationError', 'RateLimitError',
    'ModelNotFoundError', 'ValidationError', 'NetworkError', 'ServiceUnavailableError',
    
    # === 工厂管理 ===
    'ProviderFactory', 'ProviderManager', 'get_provider_manager',
    
    # === 注册中心 ===
    'ProviderRegistry', 'get_provider_registry',
    'get_platform_info', 'list_platforms', 'list_models',
    
    # === 兼容性 ===
    'FileManager', 'Message', 'ChatHistory',
    'Part', 'TextPart', 'ThoughtPart', 'ContentPart', 'FilePart', 'ImagePart',
]

# 模块版本
__version__ = "1.0.0" 