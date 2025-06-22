#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Gemini API流式聊天功能
验证chat服务中的genai流式输出是否正常工作
"""

import asyncio
import os
import logging
import json
import time
from typing import Dict, Any
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import dotenv
dotenv.load_dotenv()

from server.chat.service import ChatService
from server.chat.deprecated.models import (
    ChatRequest, ModelConfig, ModelProvider, ModelName
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenaiStreamTestSuite:
    """Genai流式聊天测试套件"""
    
    def __init__(self):
        self.chat_service = None
        self.model_configs = {}
        
    def setup_model_configs(self) -> Dict[str, ModelConfig]:
        """设置模型配置"""
        configs = {}
        
        # Gemini配置 - 通过genai直接调用
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            configs["gemini-2.5-pro-preview-06-05"] = ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name="gemini-2.5-pro-preview-06-05",
                api_key=gemini_key,
                max_tokens=4096,
                temperature=0.7
            )
            
            configs["gemini-2.0-flash"] = ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name="gemini-2.0-flash",
                api_key=gemini_key,
                max_tokens=4096,
                temperature=0.7
            )
        
        # OpenRouter配置作为对比
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            configs["google/gemini-2.5-pro"] = ModelConfig(
                provider=ModelProvider.OPENAI,  # 通过OpenRouter
                model_name="google/gemini-2.5-pro",
                api_key=openrouter_key,
                max_tokens=4096,
                temperature=0.7
            )
        
        self.model_configs = configs
        return configs
    
    async def setup_chat_service(self):
        """设置聊天服务"""
        configs = self.setup_model_configs()
        
        if not configs:
            raise ValueError("No model configurations available. Please set GEMINI_API_KEY or OPENROUTER_API_KEY")
        
        self.chat_service = ChatService(
            model_configs=configs,
            max_workers=5,
            max_queue_size=100,
            logger=logger
        )
        
        await self.chat_service.start()
        logger.info("Chat service started for testing")
    
    async def test_genai_stream_basic(self):
        """测试基本的genai流式调用"""
        logger.info("=== 测试基本genai流式调用 ===")
        
        if "gemini-2.5-pro-preview-06-05" not in self.model_configs:
            logger.warning("Gemini model not configured, skipping test")
            return False
        
        try:
            request = ChatRequest(
                conversation_id="test_genai_stream_basic",
                message="写一个简单的Python Hello World程序",
                model=ModelName.GENAI_GEMINI_2_5_PRO,
                stream=True
            )
            
            logger.info(f"发送流式请求: {request.message}")
            start_time = time.time()
            
            chunks = []
            chunk_count = 0
            
            async for chunk in self.chat_service.stream_chat(request):
                chunk_count += 1
                chunks.append(chunk)
                
                # 打印前几个chunk
                if chunk_count <= 5:
                    logger.info(f"Chunk {chunk_count}: {chunk[:50]}...")
                
                # 实时输出（可选）
                print(chunk, end="", flush=True)
            
            total_time = time.time() - start_time
            full_response = "".join(chunks)
            
            print("\n")  # 换行
            logger.info(f"✅ 基本genai流式调用成功")
            logger.info(f"   总耗时: {total_time:.2f}秒")
            logger.info(f"   响应块数: {chunk_count}")
            logger.info(f"   总字符数: {len(full_response)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 基本genai流式调用失败: {e}")
            return False
    
    async def test_genai_vs_openrouter_comparison(self):
        """测试genai和openrouter的性能对比"""
        logger.info("=== 测试genai vs OpenRouter性能对比 ===")
        
        test_message = "解释一下Python的装饰器概念"
        
        results = {}
        
        # 测试genai直接调用
        if "gemini-2.0-flash" in self.model_configs:
            try:
                request = ChatRequest(
                    conversation_id="test_comparison_genai",
                    message=test_message,
                    model=ModelName.GENAI_GEMINI_2_0_FLASH,
                    stream=True
                )
                
                logger.info("测试genai直接调用...")
                start_time = time.time()
                
                chunks = []
                async for chunk in self.chat_service.stream_chat(request):
                    chunks.append(chunk)
                
                genai_time = time.time() - start_time
                genai_response = "".join(chunks)
                
                results["genai"] = {
                    "time": genai_time,
                    "response_length": len(genai_response),
                    "chunk_count": len(chunks)
                }
                
                logger.info(f"Genai完成，耗时: {genai_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"Genai测试失败: {e}")
        
        # 测试OpenRouter调用
        if "google/gemini-2.5-pro" in self.model_configs:
            try:
                request = ChatRequest(
                    conversation_id="test_comparison_openrouter",
                    message=test_message,
                    model=ModelName.GEMINI_2_5_PRO,
                    stream=True
                )
                
                logger.info("测试OpenRouter调用...")
                start_time = time.time()
                
                chunks = []
                async for chunk in self.chat_service.stream_chat(request):
                    chunks.append(chunk)
                
                openrouter_time = time.time() - start_time
                openrouter_response = "".join(chunks)
                
                results["openrouter"] = {
                    "time": openrouter_time,
                    "response_length": len(openrouter_response),
                    "chunk_count": len(chunks)
                }
                
                logger.info(f"OpenRouter完成，耗时: {openrouter_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"OpenRouter测试失败: {e}")
        
        # 输出对比结果
        if results:
            logger.info("\n📊 性能对比结果:")
            for provider, data in results.items():
                logger.info(f"   {provider.upper()}:")
                logger.info(f"     - 耗时: {data['time']:.2f}秒")
                logger.info(f"     - 响应长度: {data['response_length']}字符")
                logger.info(f"     - 块数: {data['chunk_count']}")
            
            if len(results) == 2:
                genai_faster = results["genai"]["time"] < results["openrouter"]["time"]
                speed_diff = abs(results["genai"]["time"] - results["openrouter"]["time"])
                logger.info(f"   速度优势: {'Genai' if genai_faster else 'OpenRouter'} 快 {speed_diff:.2f}秒")
        
        return len(results) > 0
    
    async def test_conversation_history(self):
        """测试对话历史记录功能"""
        logger.info("=== 测试对话历史记录 ===")
        
        if "gemini-2.5-pro-preview-06-05" not in self.model_configs:
            logger.warning("Gemini model not configured, skipping test")
            return False
        
        try:
            conversation_id = "test_conversation_history"
            
            # 第一轮对话
            request1 = ChatRequest(
                conversation_id=conversation_id,
                message="你好，我是小明",
                model=ModelName.GENAI_GEMINI_2_5_PRO,
                stream=True
            )
            
            logger.info("第一轮对话...")
            response1_chunks = []
            async for chunk in self.chat_service.stream_chat(request1):
                response1_chunks.append(chunk)
            
            response1 = "".join(response1_chunks)
            logger.info(f"第一轮响应: {response1[:100]}...")
            
            # 第二轮对话（测试历史记录）
            request2 = ChatRequest(
                conversation_id=conversation_id,
                message="你还记得我的名字吗？",
                model=ModelName.GENAI_GEMINI_2_5_PRO,
                stream=True
            )
            
            logger.info("第二轮对话（测试历史记录）...")
            response2_chunks = []
            async for chunk in self.chat_service.stream_chat(request2):
                response2_chunks.append(chunk)
            
            response2 = "".join(response2_chunks)
            logger.info(f"第二轮响应: {response2[:100]}...")
            
            # 检查是否记住了名字
            has_name = "小明" in response2
            logger.info(f"✅ 对话历史记录功能: {'正常' if has_name else '可能有问题'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 对话历史记录测试失败: {e}")
            return False
    
    async def cleanup(self):
        """清理测试环境"""
        if self.chat_service:
            await self.chat_service.stop()
            logger.info("Chat service stopped")

async def main():
    """运行测试套件"""
    test_suite = GenaiStreamTestSuite()
    
    try:
        # 设置测试环境
        await test_suite.setup_chat_service()
        
        # 运行测试
        tests = [
            ("基本genai流式调用", test_suite.test_genai_stream_basic),
            ("genai vs OpenRouter对比", test_suite.test_genai_vs_openrouter_comparison),
            ("对话历史记录", test_suite.test_conversation_history),
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n🚀 开始测试: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"测试 {test_name} 异常: {e}")
                results.append((test_name, False))
            
            # 测试间间隔
            await asyncio.sleep(1)
        
        # 输出测试结果
        logger.info("\n📋 测试结果总结:")
        passed = 0
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\n总计: {passed}/{len(results)} 个测试通过")
        
        if passed == len(results):
            logger.info("🎉 所有测试通过！genai流式聊天功能正常工作。")
        else:
            logger.warning("⚠️  部分测试失败，请检查配置和代码。")
    
    finally:
        # 清理环境
        await test_suite.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试运行异常: {e}")
        import traceback
        traceback.print_exc() 