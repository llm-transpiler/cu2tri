# -*- coding: utf-8 -*-
"""
基础使用示例
展示如何使用LLM提供商进行简单的对话
"""
import asyncio
import logging
import os
import dotenv
from pathlib import Path

# 核心导入
from llm.providers import (
    create_provider,
    PlatformType,
    Message, ChatRequest,
)
from llm.providers.impl import OpenRouterChatTree

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

async def basic_chat_example():
    """基础聊天示例"""
    try:
        from llm.providers.factory import get_provider
        provider = get_provider(PlatformType.OPENROUTER)
        tree = OpenRouterChatTree()
        tree.add_user_message("你好！用中文回答。")
        tree.add_assistant_message("好的。")
        tree.add_user_message("你叫什么名字？")
        
        # 创建聊天请求
        request = ChatRequest(
            messages=tree.get_history().to_native(),
            model="anthropic/claude-3.7-sonnet",
            include_thinking=True,
            thinking_budget=1000,
            reasoning_effort="high",
        )
        
        # 发送请求
        response = await provider.chat(request)
        tree.add_assistant_message(response.content)
        tree.pretty_print()
        print(tree.get_history().simple_print())
        
        # 清理资源
        await provider.close()
        
    except Exception as e:
        logger.error(f"聊天失败: {e}")

async def stream_chat_example():
    """流式聊天示例"""
    try:
        provider = create_provider(
            platform_type=PlatformType.OPENROUTER,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            preferred_models=["anthropic/claude-3.5-sonnet"]
        )
        
        # 初始化provider
        await provider.initialize()
        
        request = ChatRequest(
            messages=[Message(role="user", content="请写一首关于编程的诗")],
            model="anthropic/claude-3.5-sonnet",
            max_tokens=200,
            stream=True
        )
        
        print("流式回复: ", end="")
        resp = await provider.stream_chat_completion(request)
        print(resp)
        print()  # 换行
        
        await provider.close()
        
    except Exception as e:
        logger.error(f"流式聊天失败: {e}")

async def gemini_chat_example():
    """gemini聊天示例"""
    try:
        from llm.providers.factory import get_provider
        from llm.providers.impl import GeminiChatTree
        provider = get_provider(PlatformType.GOOGLE_OFFICIAL)
        print(provider.list_models())
        tree = GeminiChatTree()
        tree.add_user_message("你好！用中文回答。")
        tree.add_assistant_message("好的。")
        tree.add_user_message("你叫什么名字？")
        
        request = ChatRequest(
            system_prompt=tree.system_prompt,
            messages=tree.get_history().messages,
            model="gemini-2.5-flash",
        )
        
        response = await provider.chat(request)
        tree.add_assistant_message(response.content)
        tree.pretty_print()
        print(tree.get_history().simple_print())
        
    except Exception as e:
        logger.error(f"gemini聊天失败: {e}")
        
if __name__ == "__main__":
    print("=== 基础聊天示例 ===")
    asyncio.run(basic_chat_example())
    
    print("=== gemini chat tree ===")
    asyncio.run(gemini_chat_example())
    # print("\n=== 流式聊天示例 ===")
    # asyncio.run(stream_chat_example())