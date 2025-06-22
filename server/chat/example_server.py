#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天服务示例启动脚本
演示如何配置和启动聊天服务
"""
import asyncio
import logging
import os
from typing import Dict

from .api_server import create_chat_api_server
import dotenv

dotenv.load_dotenv()

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('chat_service.log')
        ]
    )
    return logging.getLogger(__name__)

def create_model_configs() -> Dict[str, ModelConfig]:
    """创建模型配置
    
    注意：这里使用环境变量获取API密钥，实际使用时请设置相应的环境变量
    """
    configs = {}
    
    # OpenRouter API密钥（用于所有模型）
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key")
    
    if not openrouter_api_key or openrouter_api_key == "your-openrouter-api-key":
        print("⚠️  警告: OPENROUTER_API_KEY 环境变量未设置或使用默认值")
    
    # OpenAI模型（通过OpenRouter）
    configs["openai/gpt-4o-mini"] = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="openai/gpt-4o-mini",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    configs["openai/o3-pro"] = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="openai/o3-pro",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    # Claude模型（通过OpenRouter）
    configs["anthropic/claude-3.7-sonnet"] = ModelConfig(
        provider=ModelProvider.CLAUDE,
        model_name="anthropic/claude-3.7-sonnet",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    configs["anthropic/claude-sonnet-4"] = ModelConfig(
        provider=ModelProvider.CLAUDE,
        model_name="anthropic/claude-sonnet-4",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    # DeepSeek模型（通过OpenRouter - 使用免费版本）
    configs["deepseek/deepseek-chat-v3-0324:free"] = ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek/deepseek-chat-v3-0324:free",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    configs["deepseek/deepseek-r1-0528:free"] = ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek/deepseek-r1-0528:free",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    # Gemini模型（通过OpenRouter）
    configs["google/gemini-2.5-pro"] = ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="google/gemini-2.5-pro",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    configs["google/gemini-2.0-flash-001"] = ModelConfig(
        provider=ModelProvider.GEMINI,
        model_name="google/gemini-2.0-flash-001",
        api_key=openrouter_api_key,
        max_tokens=4096,
        temperature=0.7
    )
    
    return configs

async def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("Starting chat service...")
    
    # 创建模型配置
    model_configs = create_model_configs()
    logger.info(f"Configured {len(model_configs)} models")
    
    # 创建API服务器
    api_server = create_chat_api_server(
        model_configs=model_configs,
        host="0.0.0.0",
        port=8000,
        max_workers=20,  # 20个工作协程
        logger=logger
    )
    
    try:
        # 启动服务器
        await api_server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # 停止服务器
        await api_server.stop()
        logger.info("Chat service stopped")

if __name__ == "__main__":
    print("🚀 启动聊天服务...")
    print("📖 使用说明:")
    print("   1. 设置环境变量:")
    print("      export OPENROUTER_API_KEY=your-openrouter-api-key")
    print("   2. 安装依赖:")
    print("      pip install fastapi uvicorn aiohttp openai google-genai")
    print("   3. 访问 http://localhost:8000/docs 查看API文档")
    print("   4. 使用 Ctrl+C 停止服务")
    print()
    
    asyncio.run(main()) 