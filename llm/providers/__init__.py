# -*- coding: utf-8 -*-
"""
LLM提供商集成模块
提供统一的多平台LLM服务接口，支持流式对话和异步操作
"""

# 导入基础类型和接口
from .types import PlatformType, ModelSpec, PlatformInfo

# 导入核心基类
from .base import (
    Provider,
    Message, ChatMessage, ChatHistory, FileManager,
    ChatRequest, ChatResponse, StreamChunk,
    ProviderError, AuthenticationError, RateLimitError, 
    ModelNotFoundError, ValidationError, NetworkError, ServiceUnavailableError
)

# 导入具体实现
from .impl import (
    OpenAICompatibleProvider,
    OpenAIProvider, OpenRouterProvider,
    DeepSeekProvider, GoogleProvider,
)

# 导入配置管理
from .config import (
    PlatformConfig, ConfigManager, get_config_manager,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT
)

# 导入注册中心
from .registry import (
    ProviderRegistry,
    get_provider_registry,
)

# 导入工厂类
from .factory import (
    ProviderFactory, ProviderManager, get_provider_manager, get_provider
)

# 多模态支持
from .multimodal import (
    GeminiMessage, GeminiChatHistory, GeminiFileManager,
    OpenRouterMessage, OpenRouterChatHistory, OpenRouterFileManager,
    OpenAIMessage, OpenAIChatHistory,
)

# 便捷函数
def create_provider(platform_type, api_key=None, api_base=None, **kwargs) -> Provider:
    """便捷创建提供商函数"""
    config = PlatformConfig(
        platform_type=platform_type,
        api_key=api_key,
        api_base=api_base,
        **kwargs
    )
    return ProviderFactory.create_provider(config)

# 版本信息
__version__ = "1.0.0"

# 平台类型快速访问
PLATFORM_TYPES = PlatformType

# 公开的API
__all__ = [
    # 核心类型
    'PlatformType', 'ModelSpec', 'PlatformInfo',
    
    # 基础类
    'Provider', 'OpenAICompatibleProvider',
    'Message', 'ChatMessage', 'ChatHistory', 'FileManager',
    'ChatRequest', 'ChatResponse', 'StreamChunk',
    
    # 异常类
    'ProviderError', 'AuthenticationError', 'RateLimitError',
    'ModelNotFoundError', 'ValidationError', 'NetworkError', 'ServiceUnavailableError',
    
    # 提供商实现
    'OpenAIProvider', 'GoogleProvider', 'OpenRouterProvider',
    'DeepSeekProvider',
    
    # 配置管理
    'PlatformConfig', 'ConfigManager', 'get_config_manager',
    'DEFAULT_MAX_TOKENS', 'DEFAULT_TEMPERATURE', 'DEFAULT_TIMEOUT',
    
    # 注册中心
    'ProviderRegistry', 'get_provider_registry',
    
    # 工厂类
    'ProviderFactory', 'ProviderManager', 'get_provider_manager',
    
    # 便捷函数
    'get_provider',
    
    # 便捷函数
    'create_provider',
    
    # 多模态支持
    'GeminiMessage', 'GeminiChatHistory', 'GeminiFileManager',
    'OpenRouterMessage', 'OpenRouterChatHistory', 'OpenRouterFileManager',
    'OpenAIMessage', 'OpenAIChatHistory',
    
    # 快捷访问
    'PLATFORM_TYPES',
] 