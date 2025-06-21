# -*- coding: utf-8 -*-
"""
注册中心使用示例
展示如何查询和使用注册中心的平台、模型信息
"""
import logging
from llm.providers.registry import get_provider_registry
from llm.providers.types import PlatformType, PlatformCategory

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def query_registry_statistics():
    """查询注册中心统计信息"""
    print("=" * 50)
    print("注册中心统计信息")
    print("=" * 50)
    
    registry = get_provider_registry()
    stats = registry.get_statistics()
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

def explore_platforms():
    """探索可用平台"""
    print("=" * 50)
    print("平台信息探索")
    print("=" * 50)
    
    registry = get_provider_registry()
    
    # 按类别显示平台
    categories = registry.get_platform_categories()
    for category in categories:
        print(f"\n{category.value} 平台:")
        platforms = registry.list_platforms(category)
        
        for platform in platforms:
            platform_info = registry.get_platform_info(platform)
            if platform_info:
                print(f"  • {platform_info.name}")
                print(f"    - 类型: {platform.value}")
                print(f"    - 描述: {platform_info.description}")
                print(f"    - API地址: {platform_info.base_url}")
                print(f"    - 支持的厂商: {len(platform_info.supported_vendors)}")
                print()

def explore_models():
    """探索可用模型"""
    print("=" * 50)
    print("模型信息探索")
    print("=" * 50)
    
    registry = get_provider_registry()
    
    # 显示思维链模型
    thinking_models = registry.get_thinking_models()
    print(f"支持思维链的模型 (共{len(thinking_models)}个):")
    for i, model in enumerate(thinking_models[:10], 1):
        print(f"  {i}. {model}")
    if len(thinking_models) > 10:
        print(f"     ... 以及其他 {len(thinking_models) - 10} 个模型")
    print()
    
    # 显示视觉模型
    vision_models = registry.get_vision_models()
    print(f"支持视觉的模型 (共{len(vision_models)}个):")
    for i, model in enumerate(vision_models[:8], 1):
        print(f"  {i}. {model}")
    if len(vision_models) > 8:
        print(f"     ... 以及其他 {len(vision_models) - 8} 个模型")
    print()

def compare_platforms():
    """比较不同平台的模型"""
    print("=" * 50)
    print("平台模型比较")
    print("=" * 50)
    
    registry = get_provider_registry()
    
    # 比较几个主要平台
    platforms_to_compare = [
        PlatformType.OPENROUTER,
        PlatformType.OPENAI_OFFICIAL,
        PlatformType.GOOGLE_OFFICIAL,
    ]
    
    for platform in platforms_to_compare:
        print(f"\n{platform.value} 平台:")
        
        try:
            # 获取平台摘要
            summary = registry.get_platform_summary(platform)
            
            print(f"  支持厂商数: {summary.get('supported_vendors', 0)}")
            print(f"  总模型数: {summary.get('total_models', 0)}")
            print(f"  思维链模型数: {summary.get('thinking_models', 0)}")
            print(f"  视觉模型数: {summary.get('vision_models', 0)}")
            
            # 显示示例模型
            sample_models = summary.get('sample_models', [])
            if sample_models:
                print("  示例模型:")
                for model in sample_models:
                    print(f"    - {model}")
                    
        except Exception as e:
            print(f"  获取信息失败: {e}")

def search_specific_models():
    """搜索特定模型"""
    print("=" * 50)
    print("特定模型搜索")
    print("=" * 50)
    
    registry = get_provider_registry()
    
    # 搜索免费模型
    print("免费模型 (包含 ':free' 的模型):")
    all_models = registry.list_models()
    free_models = [model for model in all_models if ':free' in model.lower()]
    
    for i, model in enumerate(free_models, 1):
        print(f"  {i}. {model}")
        # 尝试获取模型详细信息
        spec = registry.get_model_spec(model.split('/')[-1] if '/' in model else model)
        if spec:
            print(f"     厂商: {spec.vendor.value}, 思维链: {'是' if spec.is_thinking else '否'}")
    
    if not free_models:
        print("  未找到免费模型")
    print()
    
    # 搜索特定厂商的模型
    print("DeepSeek 模型:")
    from llm.providers.types import VendorType
    deepseek_models = registry.list_models(vendor=VendorType.DEEPSEEK)
    for i, model in enumerate(deepseek_models, 1):
        print(f"  {i}. {model}")
    print()

def demonstrate_model_selection():
    """演示智能模型选择"""
    print("=" * 50)
    print("智能模型选择建议")
    print("=" * 50)
    
    registry = get_provider_registry()
    
    # 不同用途的模型推荐
    scenarios = {
        "代码生成": {"thinking": True, "vision": False},
        "图像分析": {"thinking": False, "vision": True},
        "复杂推理": {"thinking": True, "vision": False},
        "快速问答": {"thinking": False, "vision": False},
    }
    
    for scenario, requirements in scenarios.items():
        print(f"\n{scenario} 推荐模型:")
        
        if requirements["thinking"]:
            candidates = registry.get_thinking_models(PlatformType.OPENROUTER)
        elif requirements["vision"]:
            candidates = registry.get_vision_models(PlatformType.OPENROUTER)
        else:
            candidates = registry.list_models(platform_type=PlatformType.OPENROUTER)
        
        # 优先推荐免费模型
        free_candidates = [m for m in candidates if ':free' in m]
        paid_candidates = [m for m in candidates if ':free' not in m]
        
        print("  免费选项:")
        for model in free_candidates[:3]:
            print(f"    - {model}")
        
        print("  付费选项:")
        for model in paid_candidates[:3]:
            print(f"    - {model}")

def main():
    """主函数"""
    print("LLM提供商注册中心使用示例")
    print("=" * 80)
    
    try:
        # 1. 基础统计信息
        query_registry_statistics()
        
        # 2. 平台信息探索
        explore_platforms()
        
        # 3. 模型信息探索
        explore_models()
        
        # 4. 平台比较
        compare_platforms()
        
        # 5. 特定模型搜索
        search_specific_models()
        
        # 6. 智能模型选择
        demonstrate_model_selection()
        
        print("=" * 80)
        print("注册中心探索完成!")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 