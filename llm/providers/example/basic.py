# -*- coding: utf-8 -*-
"""
LLM Providers 基础使用示例
展示如何使用 LLM 提供商系统
"""
import os
import asyncio
import dotenv

# 加载环境变量
dotenv.load_dotenv()

from llm.providers import (
    # 配置相关
    PlatformConfig, PlatformType, get_config_manager,
    # 工厂和管理器
    ProviderFactory, get_provider_manager,
    # 请求响应模型
    ChatMessage, ChatRequest,
    # 注册中心
    get_provider_registry
)

async def basic_usage_example():
    """基础使用示例"""
    print("=== LLM Providers 基础使用示例 ===\n")
    
    # 1. 直接创建配置和提供商
    print("1. 直接创建提供商:")
    config = PlatformConfig(
        platform_type=PlatformType.OPENROUTER,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        preferred_models=["deepseek/deepseek-r1-0528:free"],
        temperature=0.7,
        max_tokens=4096
    )
    
    provider = ProviderFactory.create_provider(config)
    await provider.initialize()
    
    # 发送请求
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="你好！用中文回答。")],
        model="deepseek/deepseek-r1-0528:free"
    )
    
    response = await provider.chat(request)
    print(f"响应: {response.content[:100]}...\n")
    
    # 2. 使用配置管理器
    print("2. 使用配置管理器:")
    config_manager = get_config_manager()
    config_manager.set_config("openrouter", config)
    config_manager.save_config()
    print("配置已保存到文件\n")
    
    # 3. 使用提供商管理器
    print("3. 使用提供商管理器:")
    manager = get_provider_manager()
    manager.load_from_config_manager()
    
    print(f"已加载的提供商: {manager.list_providers()}")
    print(f"可用平台: {manager.list_available_platforms()[:3]}")
    print(f"可用模型: {manager.list_available_models()[:5]}\n")
    
    # 4. 流式请求示例
    print("4. 流式请求示例:")
    print("流式响应: ", end="", flush=True)
    async for chunk in provider.stream_chat(request):
        print(chunk.content, end="", flush=True)
    print("\n")
    
    await provider.close()

async def registry_example():
    """注册中心使用示例"""
    print("=== 注册中心使用示例 ===\n")
    
    registry = get_provider_registry()
    
    # 查询平台信息
    platform_info = registry.get_platform_info(PlatformType.OPENROUTER)
    print(f"OpenRouter 平台信息:")
    print(f"  名称: {platform_info.name}")
    print(f"  描述: {platform_info.description}")
    print(f"  基础URL: {platform_info.base_url}")
    print(f"  支持的厂商: {[v.value for v in platform_info.supported_vendors]}\n")
    
    # 查询模型信息
    thinking_models = registry.get_thinking_models()
    print(f"支持思维链的模型: {thinking_models[:3]}")
    
    vision_models = registry.get_vision_models()
    print(f"支持视觉的模型: {vision_models[:3]}")
    
    # 统计信息
    stats = registry.get_statistics()
    print(f"\n注册中心统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

async def main():
    """主函数"""
    try:
        await basic_usage_example()
        await registry_example()
        
        print("\n=== 示例完成 ===")
        print("你现在可以:")
        print("1. 通过 ProviderFactory 直接创建提供商")
        print("2. 通过 ConfigManager 管理配置文件")
        print("3. 通过 ProviderManager 统一管理多个提供商")
        print("4. 通过 ProviderRegistry 查询平台和模型信息")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())