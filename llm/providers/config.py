# -*- coding: utf-8 -*-
"""
LLM提供商配置系统 - 基于注册中心的版本
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
    other_params: Optional[Dict[str, Any]] = None # 其他参数
    
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
    
    def generate_default_config(self):
        """根据注册中心生成完整的示例配置"""
        import dotenv
        dotenv.load_dotenv()
        from .registry import get_provider_registry
        
        registry = get_provider_registry()
        
        # ============ 官方平台配置 ============
        
        # OpenAI 官方
        self.set_config("openai_official", PlatformConfig(
            platform_type=PlatformType.OPENAI_OFFICIAL,
            enabled=False,  # 默认禁用，需要用户填入API密钥
            api_key="your-openai-api-key-here",
            preferred_models=["gpt-o3-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
            model_aliases={
                "gpt-4": "gpt-4o",
                "gpt-4-turbo": "gpt-4o",
                "gpt-3.5-turbo": "gpt-4o-mini"
            },
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            supports_vision=True,
            supports_function_calling=True,
            custom_headers={"User-Agent": "monocases-llm-client/1.0"}
        ))
        
        # Google 官方 (Gemini)
        self.set_config("google_official", PlatformConfig(
            platform_type=PlatformType.GOOGLE_OFFICIAL,
            enabled=False,
            api_key="your-google-genai-api-key-here",
            preferred_models=["gemini-2.5-pro", "gemini-2.5-flash"],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            supports_vision=True,
            supports_function_calling=True,
            custom_params={"safety_settings": "BLOCK_NONE"}
        ))
        
        # Anthropic 官方 (Claude)
        self.set_config("anthropic_official", PlatformConfig(
            platform_type=PlatformType.ANTHROPIC_OFFICIAL,
            enabled=False,
            api_key="your-anthropic-api-key-here",
            preferred_models=["claude-4-sonnet", "claude-4-opus"],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            supports_vision=True,
            supports_function_calling=True,
        ))
        
        # DeepSeek 官方
        self.set_config("deepseek_official", PlatformConfig(
            platform_type=PlatformType.DEEPSEEK_OFFICIAL,
            enabled=False,
            api_key="your-deepseek-api-key-here",
            preferred_models=["deepseek-r1-0528", "deepseek-v3-0324"],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        ))
        
        # 智谱AI 官方
        self.set_config("zhipu_official", PlatformConfig(
            platform_type=PlatformType.ZHIPU_OFFICIAL,
            enabled=False,
            api_key="your-zhipu-api-key-here",
            preferred_models=["glm-4-plus", "glm-4-flash"],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        ))
        
        # ============ 聚合平台配置 ============
        
        # OpenRouter 聚合平台
        self.set_config("openrouter", PlatformConfig(
            platform_type=PlatformType.OPENROUTER,
            enabled=True,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            preferred_models=[
                # DeepSeek 免费模型
                "deepseek/deepseek-r1-0528:free",
                "deepseek/deepseek-chat-v3-0324:free"
                # OpenAI 模型
                # "openai/gpt-4o",
                # "openai/gpt-4o-mini", 
                # "openai/gpt-o3-mini",
                # Google 模型
                "google/gemini-2.5-pro",
                # "google/gemini-2.5-flash",
                # Anthropic 模型
                "anthropic/claude-4-sonnet",
            ],
            model_aliases={
                "gpt-4": "openai/gpt-4o",
                "claude": "anthropic/claude-4-sonnet",
                "gemini": "google/gemini-2.5-flash",
                "deepseek": "deepseek/deepseek-r1-0528:free"
            },
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            supports_function_calling=True,
        ))
        
        # ============ 本地平台配置 ============
        
        # vLLM 本地推理
        self.set_config("vllm_local", PlatformConfig(
            platform_type=PlatformType.VLLM,
            enabled=False,  # 默认禁用，需要本地启动服务
            api_base="http://localhost:8000/v1",
            preferred_models=["custom-model"],  # 用户需要根据实际部署的模型修改
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout=600,  # 本地推理可能需要更长时间
            custom_params={"local_deployment": True}
        ))
        
        # ============ 全局配置 ============
        
        # 设置默认配置 - 优先级：本地免费 > 聚合平台 > 官方平台
        thinking_models = registry.get_thinking_models()
        free_models = [model for model in thinking_models if ":free" in model]
        
        if free_models:
            self.config.default_platform = "openrouter"
            self.config.default_model = free_models[0]  # deepseek/deepseek-r1-0528:free
        else:
            self.config.default_platform = "openai_official"
            self.config.default_model = "gpt-4o-mini"
        
        # 设置备选模型列表 - 按成本和性能平衡
        self.config.fallback_models = [
            # 免费模型优先
            "deepseek/deepseek-r1-0528:free",
            "deepseek/deepseek-chat-v3-0324:free",
            # 高性价比模型
            "gpt-4o-mini",
            "gemini-2.5-flash",
            # 高性能模型
            "gpt-o3-mini",
            "claude-4-sonnet",
            "gemini-2.5-pro",
        ]
        
        # 日志级别
        self.config.log_level = "INFO"
        
        self.logger.info("Generated comprehensive sample configuration based on registry data")
        self.logger.info(f"Default platform: {self.config.default_platform}")
        self.logger.info(f"Default model: {self.config.default_model}")
        self.logger.info(f"Configured platforms: {list(self.config.platforms.keys())}")
    
    def generate_minimal_config(self):
        import dotenv
        dotenv.load_dotenv()
        """生成最小化配置 - 只包含最常用的平台"""
        # 只配置最常用的几个平台
        self.set_config("openrouter", PlatformConfig(
            platform_type=PlatformType.OPENROUTER,
            enabled=True,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            preferred_models=["deepseek/deepseek-r1-0528:free", "openai/gpt-4o-mini"],
            temperature=DEFAULT_TEMPERATURE
        ))
        
        self.set_config("google_official", PlatformConfig(
            platform_type=PlatformType.GOOGLE_OFFICIAL,
            enabled=False,
            api_key=os.getenv("GEMINI_API_KEY"),
            preferred_models=["gemini-2.5-pro", "gemini-2.5-flash"],
            temperature=DEFAULT_TEMPERATURE
        ))
        
        self.config.default_platform = "openrouter"
        self.config.default_model = "deepseek/deepseek-r1-0528:free"
        self.config.fallback_models = ["deepseek/deepseek-r1-0528:free", "deepseek/deepseek-chat-v3-0324:free"]

    def auto_initialize_from_registry(self):
        """从注册中心自动初始化所有平台配置"""
        import dotenv
        dotenv.load_dotenv()
        
        from .registry import get_provider_registry
        registry = get_provider_registry()
        
        # 清空现有配置
        self.config.platforms = {}
        
        # 为每个平台创建基础配置
        for platform_type in registry.list_platforms():
            platform_info = registry.get_platform_info(platform_type)
            if not platform_info:
                continue
            
            # 配置名称
            config_name = platform_info.type_.lower().replace(' ', '_').replace('.', '_').replace('-', '_').replace('/', '_').replace(':', '_')
            
            api_key = os.getenv(platform_info.api_key_name)
            
            # 获取该平台的推荐模型
            thinking_models = registry.get_thinking_models(platform_type)
            all_models = registry.get_models_by_platform(platform_type)
            
            # 选择首选模型（优先免费模型）
            preferred_models = []
            free_models = [m for m in all_models if ':free' in m]
            if free_models:
                preferred_models.extend(free_models[:3])
            else:
                preferred_models.extend(thinking_models[:2])
                preferred_models.extend([m for m in all_models if m not in preferred_models][:3])
            
            # 创建配置
            config = PlatformConfig(
                platform_type=platform_type,
                enabled=bool(api_key),  # 有API密钥就启用
                api_key=api_key,
                api_base=platform_info.base_url,
                preferred_models=preferred_models,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            
            self.set_config(config_name, config)
        
        # # 设置默认配置
        # self._set_auto_defaults()
        
        self.logger.info(f"Initialized {len(self.config.platforms)} platform configurations from registry")
    
    # def _set_auto_defaults(self):
    #     """自动设置默认配置"""
    #     from .registry import get_provider_registry
    #     registry = get_provider_registry()
        
    #     # 查找最优默认平台和模型
    #     enabled_configs = [(name, config) for name, config in self.config.platforms.items() if config.enabled]
        
    #     if enabled_configs:
    #         # 优先选择有免费模型的平台
    #         free_platform = None
    #         for name, config in enabled_configs:
    #             if any(':free' in model for model in config.preferred_models):
    #                 free_platform = name
    #                 break
            
    #         if free_platform:
    #             self.config.default_platform = free_platform
    #             config = self.config.platforms[free_platform]
    #             free_model = next((m for m in config.preferred_models if ':free' in m), None)
    #             self.config.default_model = free_model or config.preferred_models[0]
    #         else:
    #             # 选择第一个启用的平台
    #             name, config = enabled_configs[0]
    #             self.config.default_platform = name
    #             self.config.default_model = config.preferred_models[0] if config.preferred_models else None
        
    #     # 设置备选模型
    #     all_free_models = []
    #     all_thinking_models = []
        
    #     for config in self.config.platforms.values():
    #         if config.enabled:
    #             for model in config.preferred_models:
    #                 if ':free' in model and model not in all_free_models:
    #                     all_free_models.append(model)
    #                 elif 'thinking' in model.lower() and model not in all_thinking_models:
    #                     all_thinking_models.append(model)
        
    #     self.config.fallback_models = all_free_models[:3] + all_thinking_models[:3]

# 全局配置管理器实例
_global_config_manager = None

def get_config_manager(config_path: Optional[str] = None, 
                      auto_init_from_registry: bool = True) -> ConfigManager:
    """获取配置管理器实例"""
    global _global_config_manager
    
    if config_path:
        return ConfigManager(config_path)
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
        
        # 如果配置文件不存在，自动从注册中心初始化
        if auto_init_from_registry and not os.path.exists(_global_config_manager.config_path):
            _global_config_manager.auto_initialize_from_registry()
    
    return _global_config_manager 