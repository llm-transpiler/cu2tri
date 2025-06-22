#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天服务简单测试
基于精简后的 /llm 模块
"""
import asyncio
import logging
import os
import aiohttp
import json
from typing import Dict, Any

# 设置环境变量
import dotenv
dotenv.load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatTestClient:
    """聊天测试客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def get_models(self) -> Dict[str, Any]:
        """获取支持的模型"""
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def chat(self, conversation_id: str, message: str, 
                  model: str = "deepseek/deepseek-r1-0528:free",
                  **kwargs) -> Dict[str, Any]:
        """发送聊天请求"""
        data = {
            "conversation_id": conversation_id,
            "message": message,
            "model": model,
            **kwargs
        }
        
        async with self.session.post(f"{self.base_url}/chat", json=data) as response:
            return await response.json()
    
    async def stream_chat(self, conversation_id: str, message: str,
                         model: str = "deepseek/deepseek-r1-0528:free",
                         **kwargs):
        """流式聊天"""
        data = {
            "conversation_id": conversation_id,
            "message": message,
            "model": model,
            "stream": True,
            **kwargs
        }
        
        async with self.session.post(f"{self.base_url}/stream_chat", json=data) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue
    
    async def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话历史"""
        async with self.session.get(f"{self.base_url}/conversations/{conversation_id}/history") as response:
            return await response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        async with self.session.get(f"{self.base_url}/stats") as response:
            return await response.json()

async def test_basic_functionality():
    """测试基本功能"""
    logger.info("🧪 Testing basic functionality...")
    
    async with ChatTestClient() as client:
        # 1. 健康检查
        logger.info("1️⃣ Health check...")
        try:
            health = await client.health_check()
            logger.info(f"✅ Health status: {health.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
        
        # 2. 获取支持的模型
        logger.info("2️⃣ Getting supported models...")
        try:
            models = await client.get_models()
            logger.info(f"✅ Supported models: {len(models.get('models', []))}")
            logger.info(f"   Models: {models.get('models', [])[:3]}...")
        except Exception as e:
            logger.error(f"❌ Get models failed: {e}")
            return False
        
        # 3. 单轮对话测试
        logger.info("3️⃣ Single turn chat...")
        try:
            response = await client.chat(
                conversation_id="test_conv_1",
                message="你好，请用中文简短回答",
                model="deepseek/deepseek-r1-0528:free",
                max_tokens=100
            )
            
            if "error" in response:
                logger.error(f"❌ Chat failed: {response['error']}")
                return False
            else:
                logger.info(f"✅ Chat response: {response.get('message', '')[:50]}...")
        except Exception as e:
            logger.error(f"❌ Chat failed: {e}")
            return False
        
        # 4. 多轮对话测试
        logger.info("4️⃣ Multi-turn conversation...")
        try:
            # 第二轮对话
            response2 = await client.chat(
                conversation_id="test_conv_1",
                message="我刚才问了什么？",
                model="deepseek/deepseek-r1-0528:free",
                max_tokens=50
            )
            
            if "error" in response2:
                logger.error(f"❌ Multi-turn chat failed: {response2['error']}")
            else:
                logger.info(f"✅ Multi-turn response: {response2.get('message', '')[:50]}...")
        except Exception as e:
            logger.error(f"❌ Multi-turn chat failed: {e}")
        
        # 5. 对话历史测试
        logger.info("5️⃣ Conversation history...")
        try:
            history = await client.get_conversation_history("test_conv_1")
            messages = history.get('messages', [])
            logger.info(f"✅ History has {len(messages)} messages")
        except Exception as e:
            logger.error(f"❌ Get history failed: {e}")
        
        # 6. 服务统计
        logger.info("6️⃣ Service statistics...")
        try:
            stats = await client.get_stats()
            logger.info(f"✅ Total requests: {stats.get('total_requests', 0)}")
            logger.info(f"   Successful: {stats.get('successful_requests', 0)}")
            logger.info(f"   Failed: {stats.get('failed_requests', 0)}")
        except Exception as e:
            logger.error(f"❌ Get stats failed: {e}")
        
        return True

async def test_stream_chat():
    """测试流式对话"""
    logger.info("🌊 Testing stream chat...")
    
    async with ChatTestClient() as client:
        try:
            logger.info("Sending stream request...")
            chunks = []
            
            async for chunk in client.stream_chat(
                conversation_id="test_stream_1",
                message="请写一首关于编程的短诗",
                model="deepseek/deepseek-r1-0528:free",
                max_tokens=200
            ):
                content = chunk.get('content', '')
                if content:
                    chunks.append(content)
                    print(content, end='', flush=True)
                
                error = chunk.get('error')
                if error:
                    logger.error(f"❌ Stream error: {error}")
                    return False
            
            print()  # 换行
            logger.info(f"✅ Stream completed with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Stream test failed: {e}")
            return False

async def test_thinking_models():
    """测试思考功能"""
    logger.info("🤔 Testing thinking models...")
    
    async with ChatTestClient() as client:
        try:
            response = await client.chat(
                conversation_id="test_thinking_1",
                message="请解释一下什么是递归",
                model="deepseek/deepseek-r1-0528:free",
                include_thinking=True,
                thinking_budget=500,
                max_tokens=200
            )
            
            if "error" in response:
                logger.error(f"❌ Thinking test failed: {response['error']}")
                return False
            
            thoughts = response.get('thoughts', '')
            message = response.get('message', '')
            
            logger.info(f"✅ Response: {message[:50]}...")
            if thoughts:
                logger.info(f"   Thoughts: {thoughts[:50]}...")
            else:
                logger.info("   No thoughts captured")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Thinking test failed: {e}")
            return False

async def main():
    """主测试函数"""
    logger.info("🚀 Chat Service Test Suite v2.0")
    logger.info("=" * 50)
    
    # 检查服务是否可用
    async with ChatTestClient() as client:
        try:
            await client.health_check()
        except Exception as e:
            logger.error(f"❌ Cannot connect to chat service: {e}")
            logger.error("Please make sure the chat service is running on http://localhost:8000")
            return
    
    test_results = []
    
    # 运行测试
    test_results.append(await test_basic_functionality())
    test_results.append(await test_stream_chat())
    test_results.append(await test_thinking_models())
    
    # 结果总结
    passed = sum(test_results)
    total = len(test_results)
    
    logger.info("=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")

if __name__ == "__main__":
    asyncio.run(main()) 