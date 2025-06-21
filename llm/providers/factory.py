# -*- coding: utf-8 -*-
"""
LLM提供商工厂类 - 简化版本
支持动态创建和管理各种类型的提供商
"""
from typing import Dict, List, Optional, Type, Union, Any
import logging
import asyncio

from .config import PlatformConfig, ConfigManager, get_config_manager
from .types import PlatformType
from .registry import get_provider_registry
from .base import Provider

class ProviderFactory:
    """LLM提供商工厂类 - 负责创建和注册提供商"""
    
    # 提供商类型到实现类的映射
    _PROVIDER_CLASSES: Dict[PlatformType, Type[Provider]] = {}
    
    @classmethod
    def register_provider(cls, platform_type: Union[str, PlatformType], provider_class: Type[Provider]) -> None:
        """注册新的提供商类型
        
        Args:
            platform_type: 平台类型
            provider_class: 提供商实现类
        """
        if isinstance(platform_type, str):
            platform_type = PlatformType(platform_type)
        
        cls._PROVIDER_CLASSES[platform_type] = provider_class
        logging.getLogger(__name__).info(f"Registered provider: {platform_type.value} -> {provider_class.__name__}")
    
    @classmethod
    def get_provider_class(cls, platform_type: Union[str, PlatformType]) -> Optional[Type[Provider]]:
        """获取提供商类"""
        if isinstance(platform_type, str):
            try:
                platform_type = PlatformType(platform_type)
            except ValueError:
                return None
        
        return cls._PROVIDER_CLASSES.get(platform_type)
    
    @classmethod
    def list_supported_providers(cls) -> List[PlatformType]:
        """列出支持的提供商类型"""
        return list(cls._PROVIDER_CLASSES.keys())
    
    @classmethod
    def list_available_platforms(cls) -> List[PlatformType]:
        """列出所有可用的平台类型（从注册中心获取）"""
        registry = get_provider_registry()
        return registry.list_platforms()
    
    @classmethod
    def get_platform_info(cls, platform_type: PlatformType):
        """获取平台信息"""
        registry = get_provider_registry()
        return registry.get_platform_info(platform_type)
    
    @classmethod
    def get_supported_models(cls, platform_type: PlatformType, vendor=None) -> List[str]:
        """获取平台支持的模型列表"""
        registry = get_provider_registry()
        return registry.get_models_by_platform(platform_type, vendor)
    
    @classmethod
    def create_provider(cls, config: PlatformConfig, logger: Optional[logging.Logger] = None) -> Provider:
        """创建提供商实例
        
        Args:
            config: 平台配置
            logger: 日志记录器
            
        Returns:
            Provider实例
            
        Raises:
            ValueError: 不支持的平台类型
        """
        platform_type = config.platform_type
        if isinstance(platform_type, str):
            try:
                platform_type = PlatformType(platform_type)
            except ValueError:
                raise ValueError(f"Unsupported platform type: {platform_type}")
        
        provider_class = cls.get_provider_class(platform_type)
        if provider_class is None:
            supported = [pt.value for pt in cls.list_supported_providers()]
            raise ValueError(f"Unsupported platform type: {platform_type}, supported types: {supported}")
        
        if logger is None:
            logger = logging.getLogger(f"{__name__}.{platform_type}")
        
        return provider_class(config, logger)
    
    @classmethod
    def create_from_config_name(cls, config_name: str, 
                               config_manager: Optional[ConfigManager] = None) -> Provider:
        """从配置名称创建提供商"""
        if config_manager is None:
            config_manager = get_config_manager()
        
        config = config_manager.get_config(config_name)
        if config is None:
            raise ValueError(f"Configuration not found: {config_name}")
        
        return cls.create_provider(config)

class ProviderManager:
    """提供商管理器 - 简化版本"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None, 
                 logger: Optional[logging.Logger] = None):
        self.config_manager = config_manager or get_config_manager()
        self.logger = logger or logging.getLogger(__name__)
        self.registry = get_provider_registry()
        self._providers: Dict[str, Provider] = {}
        self._initialized_providers: Dict[str, Provider] = {}
    
    # ============ 提供商管理 ============
    
    def add_provider(self, name: str, config: PlatformConfig) -> None:
        """添加提供商
        
        Args:
            name: 提供商名称
            config: 平台配置
        """
        try:
            provider = ProviderFactory.create_provider(config, self.logger)
            self._providers[name] = provider
            self.logger.info(f"Added provider: {name} ({config.platform_type})")
        except Exception as e:
            self.logger.error(f"Failed to add provider {name}: {e}")
            raise
    
    def get_provider(self, name: str) -> Optional[Provider]:
        """获取提供商实例"""
        return self._providers.get(name)
    
    def get_initialized_provider(self, name: str) -> Optional[Provider]:
        """获取已初始化的提供商实例"""
        return self._initialized_providers.get(name)
    
    def list_providers(self, initialized_only: bool = False) -> List[str]:
        """列出提供商名称"""
        if initialized_only:
            return list(self._initialized_providers.keys())
        return list(self._providers.keys())
    
    def remove_provider(self, name: str) -> bool:
        """移除提供商"""
        if name not in self._providers:
            return False
        
        provider = self._providers.pop(name)
        if name in self._initialized_providers:
            self._initialized_providers.pop(name)
        
        # 异步关闭提供商
        asyncio.create_task(self._safe_close_provider(provider, name))
        
        self.logger.info(f"Removed provider: {name}")
        return True
    
    async def _safe_close_provider(self, provider: Provider, name: str):
        """安全关闭提供商"""
        try:
            await provider.close()
        except Exception as e:
            self.logger.warning(f"Error closing provider {name}: {e}")
    
    # ============ 初始化管理 ============
    
    async def initialize_provider(self, name: str) -> bool:
        """初始化单个提供商"""
        provider = self._providers.get(name)
        if provider is None:
            self.logger.error(f"Provider does not exist: {name}")
            return False
        
        if name in self._initialized_providers:
            return True  # 已经初始化
        
        try:
            await provider.initialize()
            self._initialized_providers[name] = provider
            self.logger.info(f"Provider initialized successfully: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Provider initialization failed {name}: {e}")
            return False
    
    # ============ 配置管理 ============
    
    def load_from_config_manager(self) -> None:
        """从配置管理器加载所有配置"""
        configs = self.config_manager.list_configs()
        loaded_count = 0
        
        for config in configs:
            if config.enabled and config.name not in self._providers:
                try:
                    self.add_provider(config.name, config)
                    loaded_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to load configuration for {config.name}: {e}")
        
        self.logger.info(f"Loaded {loaded_count} provider configurations from config manager")
    
    # ============ 注册中心访问方法 ============
    
    def get_platform_info(self, platform_type: PlatformType):
        """获取平台信息"""
        return self.registry.get_platform_info(platform_type)
    
    def list_available_platforms(self, category=None) -> List[PlatformType]:
        """列出可用的平台"""
        return self.registry.list_platforms(category)
    
    def list_available_models(self, platform_type=None, vendor=None, thinking_only=False) -> List[str]:
        """列出可用的模型"""
        return self.registry.list_models(vendor, platform_type, thinking_only)
    
    def recommend_model(self, requirements: Dict[str, Any]) -> Optional[str]:
        """推荐模型"""
        return self.registry.recommend_model(requirements)
    
    def get_thinking_models(self, platform_type=None) -> List[str]:
        """获取思维链模型"""
        return self.registry.get_thinking_models(platform_type)
    
    def get_vision_models(self, platform_type=None) -> List[str]:
        """获取视觉模型"""
        return self.registry.get_vision_models(platform_type)
    
    def get_platform_summary(self, platform_type: PlatformType) -> Dict[str, Any]:
        """获取平台摘要"""
        return self.registry.get_platform_summary(platform_type)
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """获取注册中心统计信息"""
        return self.registry.get_statistics()
    
    # ============ 生命周期管理 ============
    
    async def close_all(self) -> None:
        """关闭所有提供商"""
        if not self._initialized_providers:
            return
        
        self.logger.info(f"Starting to close {len(self._initialized_providers)} active providers")
        
        # 并发关闭所有提供商
        tasks = [provider.close() for provider in self._initialized_providers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._initialized_providers.clear()
        self.logger.info("All providers have been closed")
    
    # ============ 工具方法 ============
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()

# 全局提供商管理器实例
_default_manager = None

def get_provider_manager(config_path: Optional[str] = None) -> ProviderManager:
    """获取全局提供商管理器实例"""
    global _default_manager
    
    if _default_manager is None or config_path is not None:
        config_manager = get_config_manager(config_path)
        _default_manager = ProviderManager(config_manager)
        _default_manager.load_from_config_manager()
    
    return _default_manager 