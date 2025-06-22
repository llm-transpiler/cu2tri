# -*- coding: utf-8 -*-
"""
LLM提供商工厂类 - 精简版
"""
from typing import Dict, List, Optional, Type, Union
import logging

from .config import PlatformConfig, create_config
from .types import PlatformType
from .base import Provider

class ProviderFactory:
    """LLM提供商工厂类"""
    
    # 提供商类型到实现类的映射
    _PROVIDER_CLASSES: Dict[PlatformType, Type[Provider]] = {}
    _initialized = False
    
    @classmethod
    def _initialize_provider_classes(cls):
        """初始化提供商类映射"""
        if cls._initialized:
            return
            
        logger = logging.getLogger(__name__)
        
        # 导入所有可用的提供商实现
        provider_mappings = {}
        
        try:
            from .impl import (
                OpenAIProvider, OpenRouterProvider, DeepSeekProvider,
                GenaiProvider, AnthropicProvider, ZhipuProvider, VLLMProvider
            )
            
            provider_mappings.update({
                PlatformType.OPENAI_OFFICIAL: OpenAIProvider,
                PlatformType.OPENROUTER: OpenRouterProvider,
                PlatformType.DEEPSEEK_OFFICIAL: DeepSeekProvider,
                PlatformType.GOOGLE_OFFICIAL: GenaiProvider,
                PlatformType.ANTHROPIC_OFFICIAL: AnthropicProvider,
                PlatformType.ZHIPU_OFFICIAL: ZhipuProvider,
                PlatformType.VLLM: VLLMProvider,
            })
            
        except ImportError as e:
            logger.warning(f"Some providers not available: {e}")
        
        cls._PROVIDER_CLASSES.update(provider_mappings)
        cls._initialized = True
        
        logger.info(f"Initialized {len(cls._PROVIDER_CLASSES)} provider classes")
    
    @classmethod
    def register_provider(cls, platform_type: Union[str, PlatformType], provider_class: Type[Provider]) -> None:
        """注册新的提供商类型"""
        if isinstance(platform_type, str):
            platform_type = PlatformType(platform_type)
        
        cls._PROVIDER_CLASSES[platform_type] = provider_class
        logging.getLogger(__name__).info(f"Registered provider: {platform_type.value} -> {provider_class.__name__}")
    
    @classmethod
    def get_provider_class(cls, platform_type: Union[str, PlatformType]) -> Optional[Type[Provider]]:
        """获取提供商类"""
        cls._initialize_provider_classes()
        
        if isinstance(platform_type, str):
            try:
                platform_type = PlatformType(platform_type)
            except ValueError:
                return None
        
        return cls._PROVIDER_CLASSES.get(platform_type)
    
    @classmethod
    def list_supported_providers(cls) -> List[PlatformType]:
        """列出支持的提供商类型"""
        cls._initialize_provider_classes()
        return list(cls._PROVIDER_CLASSES.keys())
    
    @classmethod
    def create_provider(cls, platform_type: Union[str, PlatformType], 
                       api_key: Optional[str] = None,
                       api_base: Optional[str] = None,
                       preferred_models: Optional[List[str]] = None,
                       **kwargs) -> Provider:
        """创建提供商实例"""
        if isinstance(platform_type, str):
            platform_type = PlatformType(platform_type)
        
        provider_class = cls.get_provider_class(platform_type)
        if provider_class is None:
            available = [pt.value for pt in cls.list_supported_providers()]
            raise ValueError(f"Unsupported platform type: {platform_type}. Available: {available}")
        
        # 创建配置
        config = create_config(
            platform_type=platform_type,
            api_key=api_key,
            api_base=api_base,
            preferred_models=preferred_models,
            **kwargs
        )
        
        logger = logging.getLogger(f"{__name__}.{platform_type}")
        return provider_class(config, logger)

# 全局提供商缓存
_provider_cache: Dict[PlatformType, Provider] = {}

def get_provider(platform_type: Union[str, PlatformType], 
                api_key: Optional[str] = None,
                api_base: Optional[str] = None,
                preferred_models: Optional[List[str]] = None,
                use_cache: bool = True,
                **kwargs) -> Provider:
    """获取提供商实例的便捷函数
    
    这是主要的入口函数，用于获取提供商实例
    """
    if isinstance(platform_type, str):
        platform_type = PlatformType(platform_type)
    
    # 如果使用缓存且缓存中有实例，直接返回
    if use_cache and platform_type in _provider_cache:
        return _provider_cache[platform_type]
    
    # 创建新的提供商实例
    provider = ProviderFactory.create_provider(
        platform_type=platform_type,
        api_key=api_key,
        api_base=api_base,
        preferred_models=preferred_models,
        **kwargs
    )
    
    # 缓存实例
    if use_cache:
        _provider_cache[platform_type] = provider
    
    return provider

def clear_provider_cache():
    """清空提供商缓存"""
    global _provider_cache
    _provider_cache.clear()

# 导出主要接口
__all__ = [
    'ProviderFactory',
    'get_provider', 
    'clear_provider_cache'
]