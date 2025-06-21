# -*- coding: utf-8 -*-
"""
注册中心演示脚本 - 展示如何使用新的统一注册中心系统
"""
import asyncio
from llm.providers import (
    get_provider_registry, get_provider_manager,
    PlatformType, VendorType, PlatformCategory
)

async def demo_registry_system():
    """演示注册中心系统的功能"""
    print("=== LLM提供商注册中心演示 ===\n")
    
    # 获取注册中心
    registry = get_provider_registry()
    
    # 1. 基础统计信息
    print("1. 注册中心统计信息:")
    stats = registry.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 2. 列出所有平台
    print("\n2. 所有可用平台:")
    platforms = registry.list_platforms()
    for platform in platforms:
        print(f"   - {platform.value}")
    
    # 3. 按类别列出平台
    print("\n3. 按类别列出平台:")
    for category in [PlatformCategory.OFFICIAL, PlatformCategory.AGGREGATOR, PlatformCategory.LOCAL]:
        platforms = registry.list_platforms(category)
        print(f"   {category.value}: {[p.value for p in platforms]}")
    
    # 4. 获取平台详细信息
    print("\n4. 平台详细信息 (OpenRouter):")
    platform_info = registry.get_platform_info(PlatformType.OPENROUTER)
    if platform_info:
        print(f"   名称: {platform_info.name}")
        print(f"   描述: {platform_info.description}")
        print(f"   基础URL: {platform_info.base_url}")
        print(f"   支持的厂商: {[v.value for v in platform_info.supported_vendors]}")
    
    # 5. 列出思维链模型
    print("\n5. 支持思维链的模型:")
    thinking_models = registry.get_thinking_models()
    for model in thinking_models[:10]:  # 只显示前10个
        print(f"   - {model}")
    
    # 6. 列出视觉模型
    print("\n6. 支持视觉的模型:")
    vision_models = registry.get_vision_models()
    for model in vision_models[:10]:  # 只显示前10个
        print(f"   - {model}")
    
    # 7. 模型推荐
    print("\n7. 模型推荐:")
    
    # 推荐一个支持思维链的模型
    # rec1 = registry.recommend_model({'thinking': True})
    # print(f"   需要思维链功能 -> 推荐: {rec1}")
    
    # # 推荐一个支持视觉的模型
    # rec2 = registry.recommend_model({'vision': True})
    # print(f"   需要视觉功能 -> 推荐: {rec2}")
    
    # # 推荐OpenRouter平台上的思维链模型
    # rec3 = registry.recommend_model({
    #     'platform': PlatformType.OPENROUTER, 
    #     'thinking': True
    # })
    # print(f"   OpenRouter + 思维链 -> 推荐: {rec3}")
    
    # 8. 获取平台摘要
    print("\n8. 平台摘要 (Google Official):")
    summary = registry.get_platform_summary(PlatformType.GOOGLE_OFFICIAL)
    for key, value in summary.items():
        if key != 'platform_info':  # 跳过复杂的对象
            print(f"   {key}: {value}")

async def demo_provider_manager_integration():
    """演示ProviderManager与注册中心的整合"""
    print("\n\n=== ProviderManager 与注册中心整合演示 ===\n")
    
    # 获取管理器
    manager = get_provider_manager()
    
    # 1. 通过管理器访问注册中心功能
    print("1. 通过管理器查询可用平台:")
    platforms = manager.list_available_platforms()
    for platform in platforms[:5]:  # 只显示前5个
        print(f"   - {platform.value}")
    
    # # 2. 通过管理器获取模型推荐
    # print("\n2. 通过管理器获取模型推荐:")
    # rec = manager.recommend_model({'thinking': True, 'vision': True})
    # print(f"   需要思维链+视觉 -> 推荐: {rec}")
    
    # 3. 通过管理器获取统计信息
    print("\n3. 通过管理器获取统计信息:")
    stats = manager.get_registry_statistics()
    print(f"   总平台数: {stats['total_platforms']}")
    print(f"   总模型数: {stats['total_models']}")
    print(f"   思维链模型数: {stats['thinking_models']}")

def demo_direct_imports():
    """演示直接导入便捷函数的使用"""
    print("\n\n=== 直接导入便捷函数演示 ===\n")
    
    from llm.providers.registry import (
        get_platform_info, list_platforms, list_models, #recommend_model
    )
    
    # 1. 直接获取平台信息
    print("1. 直接获取平台信息:")
    platform_info = get_platform_info(PlatformType.DEEPSEEK_OFFICIAL)
    if platform_info:
        print(f"   {platform_info.name}: {platform_info.description}")
    
    # 2. 直接列出平台
    print("\n2. 直接列出官方平台:")
    official_platforms = list_platforms(PlatformCategory.OFFICIAL)
    for platform in official_platforms:
        print(f"   - {platform.value}")
    
    # 3. 直接列出模型
    print("\n3. 直接列出DeepSeek的模型:")
    deepseek_models = list_models(vendor=VendorType.DEEPSEEK)
    for model in deepseek_models:
        print(f"   - {model}")
    
    # # 4. 直接推荐模型
    # print("\n4. 直接推荐模型:")
    # rec = recommend_model({'platform': PlatformType.OPENAI_OFFICIAL, 'thinking': True})
    # print(f"   OpenAI官方平台思维链模型 -> 推荐: {rec}")

async def main():
    """主函数"""
    try:
        await demo_registry_system()
        await demo_provider_manager_integration()
        demo_direct_imports()
        
        print("\n=== 演示完成 ===")
        print("现在你可以通过以下方式使用注册中心:")
        print("1. 通过 get_provider_registry() 获取注册中心实例")
        print("2. 通过 ProviderManager 访问注册中心功能")
        print("3. 直接导入便捷函数使用")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 