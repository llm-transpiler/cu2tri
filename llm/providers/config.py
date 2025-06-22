# -*- coding: utf-8 -*-
"""
LLM提供商配置系统 - 精简版
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import yaml
import os
import logging

from .types import PlatformType

# 默认配置
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_TOKENS = 4096 * 4
DEFAULT_TEMPERATURE = 0.3

@dataclass
class PlatformConfig:
    """平台配置"""
    platform_type: Union[str, PlatformType]      # 平台类型
    enabled: bool = True                         # 是否启用
    
    # ============ 认证配置 ============
    api_key: Optional[str] = None                # API密钥
    api_base: Optional[str] = None               # API基础URL（会覆盖默认的）
    organization: Optional[str] = None           # 组织ID（如OpenAI）
    
    # ============ 请求配置 ============
    timeout: int = DEFAULT_TIMEOUT              # 请求超时时间
    max_retries: int = DEFAULT_MAX_RETRIES      # 最大重试次数
    
    # ============ 模型偏好配置 ============
    preferred_models: List[str] = field(default_factory=list)  # 首选模型列表
    model_aliases: Dict[str, str] = field(default_factory=dict) # 模型别名映射
    
    # ============ 默认参数 ============
    temperature: Optional[float] = None          # 默认温度
    max_tokens: Optional[int] = None            # 默认最大token数
    top_p: Optional[float] = None               # 默认top_p
    other_params: Optional[Dict[str, Any]] = None # 其他参数
    
    # ============ 特性配置 ============
    supports_streaming: bool = True              # 是否支持流式输出
    supports_function_calling: bool = False     # 是否支持函数调用
    supports_vision: bool = False               # 是否支持视觉
    
    # ============ 自定义配置 ============
    custom_headers: Dict[str, str] = field(default_factory=dict)  # 自定义HTTP头
    custom_params: Dict[str, Any] = field(default_factory=dict)   # 自定义参数
    
    def __post_init__(self):
        """转换platform_type为枚举类型并初始化默认值"""
        if isinstance(self.platform_type, str):
            try:
                self.platform_type = PlatformType(self.platform_type)
            except ValueError:
                pass  # 保持原字符串值，在使用时处理
        
        # 如果没有提供 api_key，尝试从环境变量获取
        if not self.api_key:
            self.api_key = self._get_api_key_from_env()
        
        # 如果没有提供 api_base，从注册中心获取默认值
        if not self.api_base:
            self.api_base = self._get_default_api_base()
        
        # 设置默认值
        if self.temperature is None:
            self.temperature = DEFAULT_TEMPERATURE
        if self.max_tokens is None:
            self.max_tokens = DEFAULT_MAX_TOKENS
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        try:
            from .registry import get_platform_info
            platform_info = get_platform_info(self.platform_type)
            if platform_info and platform_info.api_key_name:
                return os.getenv(platform_info.api_key_name)
        except:
            pass
        return None
    
    def _get_default_api_base(self) -> Optional[str]:
        """从注册中心获取默认API基础URL"""
        try:
            from .registry import get_platform_info
            platform_info = get_platform_info(self.platform_type)
            return platform_info.base_url if platform_info else None
        except:
            return None
    
    @property
    def name(self) -> str:
        """配置名称"""
        return str(self.platform_type)

def create_config(platform_type: Union[str, PlatformType], 
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 preferred_models: Optional[List[str]] = None,
                 **kwargs) -> PlatformConfig:
    """创建平台配置的便捷函数"""
    if isinstance(platform_type, str):
        platform_type = PlatformType(platform_type)
    
    config = PlatformConfig(
        platform_type=platform_type,
        api_key=api_key,
        api_base=api_base,
        preferred_models=preferred_models or [],
        **kwargs
    )
    
    return config 