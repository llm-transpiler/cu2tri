# -*- coding: utf-8 -*-
"""
LLM提供商配置系统 - 简化版本
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
DEFAULT_CONFIG_PATH = "llm_providers_config.yaml"
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
    
    # ============ 特性配置 ============
    supports_streaming: bool = True              # 是否支持流式输出
    supports_function_calling: bool = False     # 是否支持函数调用
    supports_vision: bool = False               # 是否支持视觉
    
    # ============ 自定义配置 ============
    custom_headers: Dict[str, str] = field(default_factory=dict)  # 自定义HTTP头
    custom_params: Dict[str, Any] = field(default_factory=dict)   # 自定义参数
    
    def __post_init__(self):
        """转换platform_type为枚举类型"""
        if isinstance(self.platform_type, str):
            try:
                self.platform_type = PlatformType(self.platform_type)
            except ValueError:
                pass  # 保持原字符串值，在使用时处理
    
    @property
    def name(self) -> str:
        """配置名称"""
        return str(self.platform_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 确保platform_type是字符串
        data['platform_type'] = str(self.platform_type)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlatformConfig':
        """从字典创建"""
        # 处理嵌套的默认值
        for key in ['preferred_models']:
            if key not in data:
                data[key] = []
        for key in ['model_aliases', 'custom_headers', 'custom_params']:
            if key not in data:
                data[key] = {}
        
        return cls(**data)
    
    def get_effective_api_base(self) -> Optional[str]:
        """获取有效的API基础URL"""
        if self.api_base:
            return self.api_base
        
        # 从types.py获取默认的base_url
        try:
            from .registry import get_platform_info
            platform_type = PlatformType(str(self.platform_type))
            platform_info = get_platform_info(platform_type)
            return platform_info.base_url if platform_info else None
        except:
            return None

@dataclass
class LLMConfig:
    """整个LLM系统配置"""
    platforms: Dict[str, PlatformConfig] = field(default_factory=dict)  # 平台配置
    default_platform: Optional[str] = None      # 默认平台
    default_model: Optional[str] = None          # 默认模型
    fallback_models: List[str] = field(default_factory=list)  # 备选模型
    log_level: str = "INFO"                      # 日志级别
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'platforms': {k: v.to_dict() for k, v in self.platforms.items()},
            'default_platform': self.default_platform,
            'default_model': self.default_model,
            'fallback_models': self.fallback_models,
            'log_level': self.log_level,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        platforms = {k: PlatformConfig.from_dict(v) for k, v in data.get('platforms', {}).items()}
        
        return cls(
            platforms=platforms,
            default_platform=data.get('default_platform'),
            default_model=data.get('default_model'),
            fallback_models=data.get('fallback_models', []),
            log_level=data.get('log_level', 'INFO'),
        )
    
    def get_enabled_platforms(self) -> List[str]:
        """获取启用的平台列表"""
        return [platform_name for platform_name, config in self.platforms.items() if config.enabled]

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = LLMConfig()
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data:
                        self.config = LLMConfig.from_dict(data)
                        self.logger.info(f"Config file loaded successfully: {self.config_path}")
                    else:
                        self.logger.warning(f"Config file is empty: {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
        else:
            self.logger.info(f"Config file not found, using default config: {self.config_path}")
    
    def save_config(self):
        """保存配置文件"""
        try:
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.to_dict(), f, 
                         default_flow_style=False, 
                         allow_unicode=True,
                         indent=2)
            self.logger.info(f"Config file saved successfully: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config file: {e}")
    
    # ============ 平台配置管理 ============
    
    def get_config(self, platform_name: str) -> Optional[PlatformConfig]:
        """获取平台配置"""
        return self.config.platforms.get(platform_name)
    
    def set_config(self, platform_name: str, config: PlatformConfig):
        """设置平台配置"""
        self.config.platforms[platform_name] = config
    
    def remove_config(self, platform_name: str) -> bool:
        """删除平台配置"""
        if platform_name in self.config.platforms:
            del self.config.platforms[platform_name]
            return True
        return False
    
    def list_configs(self) -> List[PlatformConfig]:
        """列出所有平台配置"""
        return list(self.config.platforms.values())
    
    def generate_sample_config(self):
        """生成示例配置"""
        # 添加一些示例平台配置
        self.set_config("openai", PlatformConfig(
            platform_type="openai",
            api_key="your-openai-api-key",
            preferred_models=["gpt-4o", "gpt-4o-mini"],
            temperature=0.7,
            max_tokens=4096
        ))
        
        self.set_config("openrouter", PlatformConfig(
            platform_type="openrouter", 
            api_key="your-openrouter-api-key",
            preferred_models=["openai/gpt-4o", "deepseek/deepseek-r1-0528:free"],
        ))
        
        # 设置默认值
        self.config.default_platform = "openai"
        self.config.default_model = "gpt-4o-mini"
        self.config.fallback_models = ["gpt-4o-mini", "deepseek/deepseek-r1-0528:free"]

# 全局配置管理器实例
_global_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取配置管理器实例"""
    global _global_config_manager
    
    if config_path:
        return ConfigManager(config_path)
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    
    return _global_config_manager 