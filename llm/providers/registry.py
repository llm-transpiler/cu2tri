# -*- coding: utf-8 -*-
"""
LLM提供商注册中心 - 统一管理平台、模型、厂商信息
"""
from typing import Dict, List, Optional, Set, Union, Any
import logging
from dataclasses import dataclass

from .types import (
    PlatformType, PlatformInfo, VendorType, ModelSpec, PlatformCategory, SDKType,
    PLATFORM_REGISTRY, VENDOR_MODEL_REGISTRY, SDK_REGISTRY,
    get_platform_info as _get_platform_info,
    get_model_spec as _get_model_spec,
    get_supported_models as _get_supported_models,
    get_platforms_by_category as _get_platforms_by_category,
    get_vendors_by_platform as _get_vendors_by_platform,
)

class ProviderRegistry:
    """LLM提供商注册中心 - 统一管理所有注册信息"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    def initialize(self):
        """初始化注册中心"""
        if self._initialized:
            return
        
        self.logger.info("初始化LLM提供商注册中心...")
        self.logger.info(f"已注册 {len(PLATFORM_REGISTRY)} 个平台")
        self.logger.info(f"已注册 {len(VENDOR_MODEL_REGISTRY)} 个厂商")
        
        total_models = sum(len(models) for models in VENDOR_MODEL_REGISTRY.values())
        self.logger.info(f"已注册 {total_models} 个模型")
        
        self._initialized = True
    
    # ============ 平台信息查询 ============
    
    def get_platform_info(self, platform_type: Union[str, PlatformType]) -> Optional[PlatformInfo]:
        """获取平台信息"""
        return _get_platform_info(platform_type)
    
    def list_platforms(self, category: Optional[PlatformCategory] = None) -> List[PlatformType]:
        """列出平台类型
        
        Args:
            category: 可选的平台类别筛选
        """
        if category is None:
            return list(PLATFORM_REGISTRY.keys())
        return _get_platforms_by_category(category)
    
    def get_platform_categories(self) -> List[PlatformCategory]:
        """获取所有平台类别"""
        return list(set(info.category for info in PLATFORM_REGISTRY.values()))
    
    def get_platforms_by_sdk(self, sdk_type: SDKType) -> List[PlatformType]:
        """根据SDK类型获取支持的平台"""
        return SDK_REGISTRY.get(sdk_type, [])
    
    # ============ 厂商和模型查询 ============
    
    def get_model_spec(self, model_name: str, vendor: Optional[VendorType] = None) -> Optional[ModelSpec]:
        """获取模型规格"""
        return _get_model_spec(model_name, vendor)
    
    def list_vendors(self, platform_type: Optional[PlatformType] = None) -> List[VendorType]:
        """列出厂商
        
        Args:
            platform_type: 可选的平台类型筛选
        """
        if platform_type is None:
            return list(VENDOR_MODEL_REGISTRY.keys())
        
        return list(_get_vendors_by_platform(platform_type))
    
    def list_models(self, vendor: Optional[VendorType] = None, 
                   platform_type: Optional[PlatformType] = None,
                   thinking_only: bool = False) -> List[str]:
        """列出模型
        
        Args:
            vendor: 可选的厂商筛选
            platform_type: 可选的平台类型筛选  
            thinking_only: 仅返回思维链模型
        """
        models = []
        
        if platform_type:
            # 基于平台的格式化模型名称
            return _get_supported_models(platform_type, vendor)
        
        # 原始模型名称
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
        """获取指定平台支持的模型列表（格式化后的）"""
        return _get_supported_models(platform_type, vendor)
    
    # ============ 能力查询 ============
    
    def get_thinking_models(self, platform_type: Optional[PlatformType] = None) -> List[str]:
        """获取支持思维链的模型"""
        return self.list_models(platform_type=platform_type, thinking_only=True)
    
    def get_vision_models(self, platform_type: Optional[PlatformType] = None) -> List[str]:
        """获取支持视觉的模型"""
        models = []
        for vendor_type, model_specs in VENDOR_MODEL_REGISTRY.items():
            if platform_type:
                platform_info = self.get_platform_info(platform_type)
                if platform_info and vendor_type not in platform_info.supported_vendors:
                    continue
            
            for spec in model_specs:
                if spec.supports_picture_input:
                    if platform_type:
                        platform_info = self.get_platform_info(platform_type)
                        formatted_name = platform_info.model_name_format.format(
                            vendor=str(spec.vendor), model=spec.name
                        )
                        models.append(formatted_name)
                    else:
                        models.append(spec.name)
        
        return sorted(models)
    
    def get_function_calling_models(self, platform_type: Optional[PlatformType] = None) -> List[str]:
        """获取支持函数调用的模型"""
        models = []
        for vendor_type, model_specs in VENDOR_MODEL_REGISTRY.items():
            if platform_type:
                platform_info = self.get_platform_info(platform_type)
                if platform_info and vendor_type not in platform_info.supported_vendors:
                    continue
            
            for spec in model_specs:
                if spec.supports_function_calling_input:
                    if platform_type:
                        platform_info = self.get_platform_info(platform_type)
                        formatted_name = platform_info.model_name_format.format(
                            vendor=str(spec.vendor), model=spec.name
                        )
                        models.append(formatted_name)
                    else:
                        models.append(spec.name)
        
        return sorted(models)
    
    # ============ 推荐和建议 ============
    
    # def recommend_model(self, requirements: Dict[str, Any]) -> Optional[str]:
    #     """根据需求推荐模型
        
    #     Args:
    #         requirements: 需求字典，可包含：
    #             - platform: 平台类型
    #             - thinking: 是否需要思维链
    #             - vision: 是否需要视觉
    #             - function_calling: 是否需要函数调用
    #             - max_tokens: 最大token需求
    #     """
    #     platform_type = requirements.get('platform')
    #     need_thinking = requirements.get('thinking', False)
    #     need_vision = requirements.get('vision', False) 
    #     need_function_calling = requirements.get('function_calling', False)
    #     max_tokens_needed = requirements.get('max_tokens', 0)
        
    #     candidates = []
        
    #     for vendor_type, model_specs in VENDOR_MODEL_REGISTRY.items():
    #         if platform_type:
    #             platform_info = self.get_platform_info(platform_type)
    #             if platform_info and vendor_type not in platform_info.supported_vendors:
    #                 continue
            
    #         for spec in model_specs:
    #             # 检查需求匹配
    #             if need_thinking and not spec.is_thinking:
    #                 continue
    #             if need_vision and not spec.supports_picture_input:
    #                 continue
    #             if need_function_calling and not spec.supports_function_calling_input:
    #                 continue
    #             if max_tokens_needed > 0 and spec.output_token_limit and spec.output_token_limit < max_tokens_needed:
    #                 continue
                
    #             # 计算得分（简单的启发式）
    #             score = 0
    #             if spec.is_thinking:
    #                 score += 10
    #             if spec.supports_picture_input:
    #                 score += 5
    #             if spec.supports_function_calling_input:
    #                 score += 5
    #             if spec.output_token_limit:
    #                 score += min(spec.output_token_limit // 1000, 50)  # token限制也是加分项
                
    #             if platform_type:
    #                 platform_info = self.get_platform_info(platform_type)
    #                 formatted_name = platform_info.model_name_format.format(
    #                     vendor=str(spec.vendor), model=spec.name
    #                 )
    #                 candidates.append((formatted_name, score))
    #             else:
    #                 candidates.append((spec.name, score))
        
    #     if not candidates:
    #         return None
        
    #     # 返回得分最高的模型
    #     candidates.sort(key=lambda x: x[1], reverse=True) # type: ignore
    #     return candidates[0][0] # type: ignore
    
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
        
        platforms_by_category = {}
        for category in self.get_platform_categories():
            platforms_by_category[str(category)] = len(self.list_platforms(category))
        
        return {
            'total_platforms': total_platforms,
            'total_vendors': total_vendors,
            'total_models': total_models,
            'thinking_models': thinking_models,
            'vision_models': vision_models,
            'platforms_by_category': platforms_by_category,
        }

# ============ 全局注册中心实例 ============

_global_registry: Optional[ProviderRegistry] = None

def get_provider_registry() -> ProviderRegistry:
    """获取全局提供商注册中心"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ProviderRegistry()
        _global_registry.initialize()
    return _global_registry

# ============ 便捷函数 ============

def get_platform_info(platform_type: Union[str, PlatformType]) -> Optional[PlatformInfo]:
    """获取平台信息 - 便捷函数"""
    return get_provider_registry().get_platform_info(platform_type)

def list_platforms(category: Optional[PlatformCategory] = None) -> List[PlatformType]:
    """列出平台 - 便捷函数"""
    return get_provider_registry().list_platforms(category)

def list_models(vendor: Optional[VendorType] = None, 
               platform_type: Optional[PlatformType] = None,
               thinking_only: bool = False) -> List[str]:
    """列出模型 - 便捷函数"""
    return get_provider_registry().list_models(vendor, platform_type, thinking_only)

# def recommend_model(requirements: Dict[str, Any]) -> Optional[str]:
#     """推荐模型 - 便捷函数"""
#     return get_provider_registry().recommend_model(requirements) 