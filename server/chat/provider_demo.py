# -*- coding: utf-8 -*-
"""
提供商分类演示脚本
展示OpenRouter和Google genai的区分功能
"""
import sys
import os

# 添加父目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from server.chat.deprecated.models import ModelName, get_provider_type_by_model_name
from chat.providers import ProviderFactory

def main():
    print("=== 聊天服务提供商分类演示 ===\n")
    
    # 1. 显示所有可用模型，按提供商分类
    print("1. 按提供商分类的所有可用模型:")
    available_models = ProviderFactory.list_available_models()
    
    for provider, models in available_models.items():
        print(f"\n📍 {provider.upper()} 提供商:")
        for model in models[:5]:  # 只显示前5个，避免输出过长
            print(f"   • {model}")
        if len(models) > 5:
            print(f"   ... 还有 {len(models) - 5} 个模型")
    
    # 2. 测试提供商类型判断
    print("\n\n2. 提供商类型判断测试:")
    test_models = [
        # OpenRouter 模型
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash", 
        "deepseek/deepseek-r1-0528:free",
        "anthropic/claude-sonnet-4",
        
        # Google genai 模型
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        
        # 边界情况
        "unknown-model",
        "gemini/something",  # 包含"/"的gemini模型应该是OpenRouter
    ]
    
    for model in test_models:
        provider_type = get_provider_type_by_model_name(model)
        print(f"   {model:<35} → {provider_type}")
    
    # 3. 展示枚举中的新命名规范
    print("\n\n3. 新的模型命名规范示例:")
    
    openrouter_examples = [
        ModelName.OPENROUTER_GPT_4O_MINI,
        ModelName.OPENROUTER_DEEPSEEK_R1_0528_FREE,
        ModelName.OPENROUTER_GEMINI_2_0_FLASH,
        ModelName.OPENROUTER_CLAUDE_SONNET_4,
    ]
    
    genai_examples = [
        ModelName.GOOGLE_GENAI_GEMINI_2_5_PRO,
        ModelName.GOOGLE_GENAI_GEMINI_2_0_FLASH,
        ModelName.GOOGLE_GENAI_GEMINI_1_5_FLASH,
    ]
    
    print("\n   OpenRouter 模型命名:")
    for model in openrouter_examples:
        print(f"   {model.name:<40} = '{model.value}'")
    
    print("\n   Google genai 模型命名:")
    for model in genai_examples:
        print(f"   {model.name:<40} = '{model.value}'")
    
    # 4. 向后兼容性测试
    print("\n\n4. 向后兼容性 (旧名称仍然可用):")
    compatible_models = [
        ModelName.GPT_4O_MINI,
        ModelName.DEEPSEEK_R1_0528_FREE,
        ModelName.GEMINI_2_0_FLASH,
        ModelName.GENAI_GEMINI_2_0_FLASH,
    ]
    
    for model in compatible_models:
        provider_type = get_provider_type_by_model_name(model.value)
        print(f"   {model.name:<25} = '{model.value}' → {provider_type}")

if __name__ == "__main__":
    main() 