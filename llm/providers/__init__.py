# -*- coding: utf-8 -*-
"""
LLM提供商模块 - 三层架构
平台 → 厂商 → 模型的优雅分级设计
"""

# 核心类型定义
from .types import (
    # 平台相关
    PlatformType, PlatformCategory, PlatformInfo, PLATFORM_REGISTRY,
    # 厂商相关  
    VendorType, VENDOR_MODEL_REGISTRY,
    # 模型相关
    ModelSpec,
    # SDK相关
    SDKType, SDK_REGISTRY,
    # 工具函数
    get_platform_info, get_model_spec, get_supported_models,
    get_platforms_by_category, get_vendors_by_platform
)

# 配置系统
from .config import (
    ConfigManager, PlatformConfig, LLMConfig,
    DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, get_config_manager
)

# 基础接口
from .base import (
    # 新统一接口
    ChatMessage, ChatRequest, ChatResponse, StreamChunk,
    Provider, OpenAICompatibleProvider,
    # 异常类
    ProviderError, AuthenticationError, RateLimitError, 
    ModelNotFoundError, ValidationError,
    # 兼容性接口
    FileManager, Message, ChatHistory
)

# 工厂和管理器
from .factory import (
    ProviderFactory, ProviderManager, get_provider_manager
)

# 具体实现（自动注册）
from . import impl

# 历史兼容
from ..history.parts import Part, TextPart, ThoughtPart, ContentPart, FilePart, ImagePart

# 导出所有公共接口
__all__ = [
    # === 核心类型 ===
    'PlatformType', 'PlatformCategory', 'PlatformInfo', 'PLATFORM_REGISTRY',
    'VendorType', 'VENDOR_MODEL_REGISTRY',
    'ModelSpec',
    'SDKType', 'SDK_REGISTRY',
    
    # === 工具函数 ===
    'get_platform_info', 'get_model_spec', 'get_supported_models',
    'get_platforms_by_category', 'get_vendors_by_platform',
    
    # === 配置系统 ===
    'ConfigManager', 'PlatformConfig', 'LLMConfig',
    'DEFAULT_TIMEOUT', 'DEFAULT_MAX_RETRIES', 'get_config_manager',
    
    # === 统一接口 ===
    'ChatMessage', 'ChatRequest', 'ChatResponse', 'StreamChunk',
    'Provider', 'OpenAICompatibleProvider',
    
    # === 异常处理 ===
    'ProviderError', 'AuthenticationError', 'RateLimitError',
    'ModelNotFoundError', 'ValidationError',
    
    # === 工厂管理 ===
    'ProviderFactory', 'ProviderManager', 'get_provider_manager',
    
    # === 兼容性 ===
    'FileManager', 'Message', 'ChatHistory',
    'Part', 'TextPart', 'ThoughtPart', 'ContentPart', 'FilePart', 'ImagePart',
]

# 模块版本
__version__ = "1.0.0" 