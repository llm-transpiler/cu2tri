# -*- coding: utf-8 -*-
"""
LLM提供商注册中心 - 统一管理平台、模型、厂商信息
"""
from typing import Dict, List, Optional, Set, Union, Any
import logging
from dataclasses import dataclass

from .types import (
    PlatformType, PlatformInfo, VendorType, ModelSpec, PlatformCategory, SDKType,
)

# 平台注册表
PLATFORM_REGISTRY: Dict[PlatformType, PlatformInfo] = {
    # ============ 聚合平台 ============
    PlatformType.OPENROUTER: PlatformInfo(
        type_=PlatformType.OPENROUTER,
        category=PlatformCategory.AGGREGATOR,
        name="OpenRouter",
        description="OpenRouter aggregation platform supporting multi-vendor models",
        http_base_url="https://openrouter.ai/api/v1",
        base_url="https://openrouter.ai/api/v1",
        api_key_name="OPENROUTER_API_KEY",
        supported_vendors={
            VendorType.OPENAI, VendorType.DEEPSEEK, VendorType.OPENROUTER_DEEPSEEK,
            VendorType.GOOGLE,VendorType.ANTHROPIC
        },
        supported_sdks={SDKType.OPENAI},
        model_name_format="{vendor}/{model}"
    ),
    
    # ============ 模型厂商官方部署平台 ============
    PlatformType.OPENAI_OFFICIAL: PlatformInfo(
        type_=PlatformType.OPENAI_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="OpenAI Official",
        description="OpenAI官方API",
        http_base_url="https://api.openai.com/v1",
        base_url="https://api.openai.com/v1",
        api_key_name="OPENAI_API_KEY",
        supported_vendors={VendorType.OPENAI},
        supported_sdks={SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.ANTHROPIC_OFFICIAL: PlatformInfo(
        type_=PlatformType.ANTHROPIC_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="Anthropic Official",
        description="Anthropic官方API (Claude)",
        http_base_url="https://api.anthropic.com/v1/", # https://docs.anthropic.com/en/api/overview#curl
        base_url="https://api.anthropic.com/v1/", # https://docs.anthropic.com/en/api/openai-sdk#getting-started-with-the-openai-sdk
        api_key_name="ANTHROPIC_API_KEY",
        supported_vendors={VendorType.ANTHROPIC},
        supported_sdks={SDKType.ANTHROPIC, SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.GOOGLE_OFFICIAL: PlatformInfo(
        type_=PlatformType.GOOGLE_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="Google Official",
        description="Google genai official API",
        http_base_url="https://generativelanguage.googleapis.com/v1beta", # https://ai.google.dev/gemini-api/docs/quickstart#rest
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # https://ai.google.dev/gemini-api/docs/openai?hl=zh-cn
        api_key_name="GOOGLE_API_KEY",
        supported_vendors={VendorType.GOOGLE},
        supported_sdks={SDKType.GENAI, SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.DEEPSEEK_OFFICIAL: PlatformInfo(
        type_=PlatformType.DEEPSEEK_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="DeepSeek Official",
        description="DeepSeek official API",
        http_base_url="https://api.deepseek.com", # https://api-docs.deepseek.com/zh-cn/
        base_url="https://api.deepseek.com", # https://api-docs.deepseek.com/zh-cn/
        api_key_name="DEEPSEEK_API_KEY",
        supported_vendors={VendorType.DEEPSEEK},
        supported_sdks={SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    PlatformType.ZHIPU_OFFICIAL: PlatformInfo(
        type_=PlatformType.ZHIPU_OFFICIAL,
        category=PlatformCategory.OFFICIAL,
        name="Zhipu AI Official",
        description="智谱AI官方API",
        http_base_url="https://open.bigmodel.cn/api/paas/v4/",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_name="ZHIPU_API_KEY",
        supported_vendors={VendorType.ZHIPU},
        supported_sdks={SDKType.OPENAI},
        model_name_format="{model}"
    ),
    
    # ============ 本地平台 ============
    PlatformType.VLLM: PlatformInfo(
        type_=PlatformType.VLLM,
        category=PlatformCategory.LOCAL,
        name="vLLM",
        description="vLLM local inference server",
        http_base_url="http://localhost:8000/v1",
        base_url="http://localhost:8000/v1",
        api_key_name="VLLM_API_KEY",
        supported_vendors=set(),  # vLLM 可以运行任何厂商的模型
        supported_sdks={SDKType.OPENAI, SDKType.VLLM},
        model_name_format="{model}"
    ),
}

# SDKType -> PlatformType
SDK_REGISTRY: Dict[SDKType, List[PlatformType]] = {
    SDKType.OPENAI: [PlatformType.OPENAI_OFFICIAL, PlatformType.ANTHROPIC_OFFICIAL, PlatformType.GOOGLE_OFFICIAL, PlatformType.DEEPSEEK_OFFICIAL, PlatformType.OPENROUTER],
    SDKType.ANTHROPIC: [PlatformType.ANTHROPIC_OFFICIAL],
    SDKType.GENAI: [PlatformType.GOOGLE_OFFICIAL],
    SDKType.VLLM: [PlatformType.VLLM],
}

# ================== Per vendor ==================

GEMINI_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="gemini-2.5-pro",
        vendor=VendorType.GOOGLE,
        is_thinking=True,
        input_token_limit=1048576, # 1M
        output_token_limit=65536, # 66K
        supports_streaming_output=True,
        supports_picture_input=True,
        supports_structured_output=True,
        supports_pdf_input=True,
        supports_audio_input=True,
        supports_video_input=True,
        supports_function_calling_input=True,
    ),
    ModelSpec(
        name="gemini-2.5-flash",
        vendor=VendorType.GOOGLE,
        is_thinking=True,
        input_token_limit=1048576, # 1M
        output_token_limit=65536, # 66K
        supports_streaming_output=True,
        supports_picture_input=True,
        supports_structured_output=True,
        supports_pdf_input=True,
        supports_audio_input=True,
        supports_video_input=False,
        supports_function_calling_input=True,
    ),
]

OPENAI_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="gpt-o3-mini",
        vendor=VendorType.OPENAI,
        is_thinking=True,
        supports_picture_input=True
    ),
    ModelSpec(
        name="gpt-o3",
        vendor=VendorType.OPENAI,
        is_thinking=True,
        supports_picture_input=True
    ),
    ModelSpec(
        name="gpt-4.1",
        vendor=VendorType.OPENAI,
        is_thinking=True,
        supports_picture_input=True
    ),
    ModelSpec(
        name="gpt-4o",
        vendor=VendorType.OPENAI,
        supports_picture_input=True,
        aliases=["gpt-4-omni"]
    ),
    ModelSpec(
        name="gpt-4o-mini",
        vendor=VendorType.OPENAI,
        supports_picture_input=True,
        aliases=["gpt-4-omni-mini"]
    ),
]

ANTHROPIC_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="claude-4-sonnet",
        vendor=VendorType.ANTHROPIC,
        is_thinking=True,
        supports_picture_input=True,
    ),
    ModelSpec(
        name="claude-4-opus",
        vendor=VendorType.ANTHROPIC,
        is_thinking=True,
        supports_picture_input=True,
    ),
]

DEEPSEEK_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="deepseek-r1-0528",
        vendor=VendorType.DEEPSEEK,
        is_thinking=True,
    ),
    ModelSpec(
        name="deepseek-v3-0324",
        vendor=VendorType.DEEPSEEK,
    ),
]

# VendorType -> List[ModelSpec]
OPENROUTER_EXCLUSIVE_MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="deepseek-r1-0528:free",
        vendor=VendorType.OPENROUTER_DEEPSEEK,
    ),
    ModelSpec(
        name="deepseek-chat-v3-0324:free",
        vendor=VendorType.OPENROUTER_DEEPSEEK,
    ),
]

# 构建厂商模型注册表
def _build_vendor_model_registry() -> Dict[VendorType, List[ModelSpec]]:
    """
    构建厂商模型注册表
    return {
        VendorType.OPENAI: OPENAI_MODEL_REGISTRY,
        VendorType.ANTHROPIC: ANTHROPIC_MODEL_REGISTRY,
        VendorType.GOOGLE: GEMINI_MODEL_REGISTRY,
        VendorType.DEEPSEEK: DEEPSEEK_MODEL_REGISTRY,
        VendorType.OPENROUTER_DEEPSEEK: OPENROUTER_EXCLUSIVE_MODEL_REGISTRY,
    }
    """
    registry = {}
    model_lists = [
        OPENAI_MODEL_REGISTRY,
        ANTHROPIC_MODEL_REGISTRY,
        GEMINI_MODEL_REGISTRY,
        DEEPSEEK_MODEL_REGISTRY,
        OPENROUTER_EXCLUSIVE_MODEL_REGISTRY,
    ]
    
    for model_list in model_lists:
        for spec in model_list:
            if spec.vendor not in registry:
                registry[spec.vendor] = []
            registry[spec.vendor].append(spec)
    
    return registry

VENDOR_MODEL_REGISTRY = _build_vendor_model_registry()


class ProviderRegistry:
    """LLM提供商注册中心 - 统一管理所有注册信息"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 直接在构造函数中完成初始化
        self.logger.info("Initializing LLM provider registry...")
        self.logger.info(f"Registered {len(PLATFORM_REGISTRY)} platforms")
        self.logger.info(f"Registered {len(VENDOR_MODEL_REGISTRY)} vendors")
        
        total_models = sum(len(models) for models in VENDOR_MODEL_REGISTRY.values())
        self.logger.info(f"Registered {total_models} models")
        
        # 动态填充每个平台的 supported_models
        self._populate_platform_models()
    
    def _populate_platform_models(self):
        """动态填充每个平台的supported_models字段"""
        for platform_type, platform_info in PLATFORM_REGISTRY.items():
            models = set()
            
            # 遍历所有厂商模型，找到该平台支持的模型
            for vendor_type, vendor_models in VENDOR_MODEL_REGISTRY.items():
                if vendor_type in platform_info.supported_vendors:
                    for spec in vendor_models:
                        # 格式化模型名称
                        formatted_name = platform_info.model_name_format.format(
                            vendor=str(spec.vendor),
                            model=spec.name
                        )
                        models.add(formatted_name)
            
            platform_info.supported_models = models
    
    # ============ 平台信息查询 ============
    
    def get_platform_info(self, platform_type: Union[str, PlatformType]) -> Optional[PlatformInfo]:
        """获取平台信息"""
        if platform_type is None:
            return None
        if isinstance(platform_type, str):
            try:
                platform_type = PlatformType(platform_type)
            except ValueError:
                return None
        return PLATFORM_REGISTRY.get(platform_type)
    
    def list_platforms(self, category: Optional[PlatformCategory] = None) -> List[PlatformType]:
        """列出平台类型"""
        if category is None:
            return list(PLATFORM_REGISTRY.keys())
        return [info.type_ for info in PLATFORM_REGISTRY.values() if info.category == category]
    
    def get_platform_categories(self) -> List[PlatformCategory]:
        """获取所有平台类别"""
        return list(set(info.category for info in PLATFORM_REGISTRY.values()))
    
    def get_platforms_by_sdk(self, sdk_type: SDKType) -> List[PlatformType]:
        """根据SDK类型获取支持的平台"""
        return SDK_REGISTRY.get(sdk_type, [])
    
    # ============ 厂商和模型查询 ============
    
    def get_model_spec(self, model_name: str, vendor: Optional[VendorType] = None) -> Optional[ModelSpec]:
        """获取模型规格"""
        if not model_name:
            return None
        
        # 如果指定了厂商，直接在该厂商下查找
        if vendor and vendor in VENDOR_MODEL_REGISTRY:
            for spec in VENDOR_MODEL_REGISTRY[vendor]:
                if spec.name == model_name or model_name in spec.aliases:
                    return spec
        
        # 在所有厂商中查找
        for vendor_models in VENDOR_MODEL_REGISTRY.values():
            for spec in vendor_models:
                if spec.name == model_name or model_name in spec.aliases:
                    return spec
        
        return None
    
    def list_vendors(self, platform_type: Optional[PlatformType] = None) -> List[VendorType]:
        """列出厂商"""
        if platform_type is None:
            return list(VENDOR_MODEL_REGISTRY.keys())
        
        platform_info = self.get_platform_info(platform_type)
        return list(platform_info.supported_vendors) if platform_info else []
    
    def list_models(self, vendor: Optional[VendorType] = None, 
                   platform_type: Optional[PlatformType] = None,
                   thinking_only: bool = False) -> List[str]:
        """列出模型"""
        if platform_type:
            return self._get_platform_models(platform_type, vendor, thinking_only)
        
        # 原始模型名称
        models = []
        vendor_models = VENDOR_MODEL_REGISTRY
        if vendor:
            vendor_models = {vendor: vendor_models.get(vendor, [])}
        
        for vendor_type, model_specs in vendor_models.items():
            for spec in model_specs:
                if thinking_only and not spec.is_thinking:
                    continue
                models.append(spec.name)
        
        return sorted(models)
    
    def get_models_by_vendor(self, vendor: VendorType) -> List[ModelSpec]:
        """获取指定厂商的所有模型规格"""
        return VENDOR_MODEL_REGISTRY.get(vendor, [])
    
    def get_models_by_platform(self, platform_type: PlatformType, vendor: Optional[VendorType] = None) -> List[str]:
        """获取指定平台支持的模型列表"""
        return self._get_platform_models(platform_type, vendor, False)
    
    def _get_platform_models(self, platform_type: PlatformType, vendor: Optional[VendorType] = None, thinking_only: bool = False) -> List[str]:
        """获取平台支持的模型列表（内部方法）"""
        platform_info = self.get_platform_info(platform_type)
        if not platform_info:
            return []
        
        models = []
        for vendor_type, vendor_models in VENDOR_MODEL_REGISTRY.items():
            # 检查厂商是否支持
            if vendor_type not in platform_info.supported_vendors:
                continue
            
            # 如果指定了厂商，检查是否匹配
            if vendor and vendor_type != vendor:
                continue
            
            for spec in vendor_models:
                if thinking_only and not spec.is_thinking:
                    continue
                
                # 格式化模型名称
                formatted_name = platform_info.model_name_format.format(
                    vendor=str(spec.vendor),
                    model=spec.name
                )
                models.append(formatted_name)
        
        return sorted(models)
    
    # ============ 能力查询 ============
    
    def get_thinking_models(self, platform_type: Optional[PlatformType] = None) -> List[str]:
        """获取支持思维链的模型"""
        return self.list_models(platform_type=platform_type, thinking_only=True)
    
    def get_vision_models(self, platform_type: Optional[PlatformType] = None) -> List[str]:
        """获取支持视觉的模型"""
        return self._get_models_by_capability(platform_type, lambda spec: spec.supports_picture_input)
    
    def get_function_calling_models(self, platform_type: Optional[PlatformType] = None) -> List[str]:
        """获取支持函数调用的模型"""
        return self._get_models_by_capability(platform_type, lambda spec: spec.supports_function_calling_input)
    
    def _get_models_by_capability(self, platform_type: Optional[PlatformType], capability_check) -> List[str]:
        """根据能力获取模型列表（内部方法）"""
        models = []
        for vendor_type, model_specs in VENDOR_MODEL_REGISTRY.items():
            if platform_type:
                platform_info = self.get_platform_info(platform_type)
                if platform_info and vendor_type not in platform_info.supported_vendors:
                    continue
            
            for spec in model_specs:
                if capability_check(spec):
                    if platform_type:
                        platform_info = self.get_platform_info(platform_type)
                        formatted_name = platform_info.model_name_format.format(
                            vendor=str(spec.vendor), model=spec.name
                        )
                        models.append(formatted_name)
                    else:
                        models.append(spec.name)
        
        return sorted(models)
    
    def get_platform_summary(self, platform_type: PlatformType) -> Dict[str, Any]:
        """获取平台摘要信息"""
        platform_info = self.get_platform_info(platform_type)
        if not platform_info:
            return {}
        
        vendors = self.list_vendors(platform_type)
        models = self.get_models_by_platform(platform_type)
        thinking_models = self.get_thinking_models(platform_type)
        vision_models = self.get_vision_models(platform_type)
        function_calling_models = self.get_function_calling_models(platform_type)
        
        return {
            'platform_info': platform_info,
            'supported_vendors': len(vendors),
            'vendor_list': [str(v) for v in vendors],
            'total_models': len(models),
            'thinking_models': len(thinking_models),
            'vision_models': len(vision_models),
            'function_calling_models': len(function_calling_models),
            'sample_models': models[:5],  # 显示前5个作为示例
        }
    
    # ============ 统计信息 ============
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册中心统计信息"""
        total_platforms = len(PLATFORM_REGISTRY)
        total_vendors = len(VENDOR_MODEL_REGISTRY)
        total_models = sum(len(models) for models in VENDOR_MODEL_REGISTRY.values())
        
        thinking_models = sum(
            1 for models in VENDOR_MODEL_REGISTRY.values() 
            for spec in models if spec.is_thinking
        )
        
        vision_models = sum(
            1 for models in VENDOR_MODEL_REGISTRY.values() 
            for spec in models if spec.supports_picture_input
        )
        
        return {
            'total_platforms': total_platforms,
            'total_vendors': total_vendors,
            'total_models': total_models,
            'thinking_models': thinking_models,
            'vision_models': vision_models,
            'platform_categories': len(self.get_platform_categories()),
            'supported_sdks': len(SDK_REGISTRY),
        }


# 全局注册中心实例
_global_registry = None

def get_provider_registry() -> ProviderRegistry:
    """获取全局注册中心实例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ProviderRegistry()
    return _global_registry

# 便捷函数 - 直接使用全局注册中心
def get_platform_info(platform_type: Union[str, PlatformType]) -> Optional[PlatformInfo]:
    """获取平台信息"""
    return get_provider_registry().get_platform_info(platform_type)

def list_platforms(category: Optional[PlatformCategory] = None) -> List[PlatformType]:
    """列出平台类型"""
    return get_provider_registry().list_platforms(category)

def list_models(vendor: Optional[VendorType] = None, 
               platform_type: Optional[PlatformType] = None,
               thinking_only: bool = False) -> List[str]:
    """列出模型"""
    return get_provider_registry().list_models(vendor, platform_type, thinking_only)

__all__ = [
    "get_provider_registry",
    "get_platform_info",
    "list_platforms",
    "list_models",
]