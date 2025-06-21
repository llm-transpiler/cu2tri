#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天服务简单测试脚本
验证服务基本功能
"""
import asyncio
import aiohttp
import json
import time

async def test_chat_service():
    """测试聊天服务基本功能"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        print("🧪 开始测试聊天服务...")
        
        # 1. 健康检查
        print("\n1️⃣ 健康检查...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ 服务状态: {health_data['status']}")
                else:
                    print(f"❌ 健康检查失败: {response.status}")
                    return
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return
        
        # 2. 获取支持的模型
        print("\n2️⃣ 获取支持的模型...")
        try:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    print(f"✅ 支持的模型: {models_data['models'][:3]}...")  # 只显示前3个
                else:
                    print(f"❌ 获取模型失败: {response.status}")
        except Exception as e:
            print(f"❌ 获取模型失败: {e}")
        
        # 3. 测试单轮对话
        print("\n3️⃣ 测试单轮对话...")
        conversation_id = f"test_conv_{int(time.time())}"
        
        chat_request = {
            "conversation_id": conversation_id,
            "message": "Hello! Please respond with a simple greeting.",
            "model": "openai/gpt-4o-mini",
            "thought": "Testing basic chat functionality",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/chat", json=chat_request) as response:
                end_time = time.time()
                
                if response.status == 200:
                    chat_data = await response.json()
                    print(f"✅ 对话成功!")
                    print(f"   响应时间: {end_time - start_time:.2f}秒")
                    print(f"   请求ID: {chat_data['request_id']}")
                    print(f"   模型: {chat_data['model']}")
                    print(f"   回复: {chat_data['message'][:100]}...")
                    if chat_data.get('thought'):
                        print(f"   思考: {chat_data['thought'][:50]}...")
                else:
                    error_data = await response.text()
                    print(f"❌ 对话失败: {response.status} - {error_data}")
                    return
        except Exception as e:
            print(f"❌ 对话请求失败: {e}")
            return
        
        # 4. 测试多轮对话
        print("\n4️⃣ 测试多轮对话...")
        
        messages = [
            "What's your name?",
            "Can you remember what I just asked?",
            "What's 2+2?"
        ]
        
        for i, message in enumerate(messages, 1):
            chat_request = {
                "conversation_id": conversation_id,
                "message": message,
                "model": "openai/gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            try:
                async with session.post(f"{base_url}/chat", json=chat_request) as response:
                    if response.status == 200:
                        chat_data = await response.json()
                        print(f"   轮次 {i}: ✅ {message}")
                        print(f"           回复: {chat_data['message'][:80]}...")
                    else:
                        print(f"   轮次 {i}: ❌ {message}")
            except Exception as e:
                print(f"   轮次 {i}: ❌ {message} - {e}")
            
            # 短暂延迟
            await asyncio.sleep(0.5)
        
        # 5. 获取对话历史
        print("\n5️⃣ 获取对话历史...")
        try:
            async with session.get(
                f"{base_url}/conversations/{conversation_id}/history",
                params={"model": "openai/gpt-4o-mini"}
            ) as response:
                if response.status == 200:
                    history_data = await response.json()
                    print(f"✅ 对话历史获取成功")
                    print(f"   对话ID: {history_data['conversation_id']}")
                    print(f"   消息数量: {len(history_data['history'])}")
                    
                    # 显示最后几条消息
                    for msg in history_data['history'][-3:]:
                        role = msg['role'].upper()
                        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                        print(f"   [{role}]: {content}")
                else:
                    print(f"❌ 获取历史失败: {response.status}")
        except Exception as e:
            print(f"❌ 获取历史失败: {e}")
        
        # 6. 获取服务统计
        print("\n6️⃣ 获取服务统计...")
        try:
            async with session.get(f"{base_url}/stats") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    print(f"✅ 服务统计:")
                    print(f"   服务状态: {stats_data['service_running']}")
                    print(f"   支持模型数: {len(stats_data['supported_models'])}")
                    queue_stats = stats_data['queue']
                    print(f"   队列统计:")
                    print(f"     总请求: {queue_stats['total_requests']}")
                    print(f"     成功请求: {queue_stats['completed_requests']}")
                    print(f"     失败请求: {queue_stats['failed_requests']}")
                    print(f"     平均处理时间: {queue_stats['average_processing_time']:.3f}秒")
                    
                    conv_stats = stats_data['conversations']
                    print(f"   对话统计:")
                    print(f"     总对话数: {conv_stats['total_conversations']}")
                else:
                    print(f"❌ 获取统计失败: {response.status}")
        except Exception as e:
            print(f"❌ 获取统计失败: {e}")
        
        print("\n🎉 测试完成!")

async def test_concurrent_requests():
    """测试并发请求"""
    print("\n🔄 测试并发请求...")
    base_url = "http://localhost:8000"
    
    async def send_request(session, request_id):
        chat_request = {
            "conversation_id": f"concurrent_test_{request_id}",
            "message": f"This is concurrent request {request_id}. Please respond briefly.",
            "model": "openai/gpt-4o-mini",
            "max_tokens": 30,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{base_url}/chat", json=chat_request) as response:
                end_time = time.time()
                if response.status == 200:
                    return {
                        "success": True,
                        "request_id": request_id,
                        "response_time": end_time - start_time
                    }
                else:
                    return {
                        "success": False,
                        "request_id": request_id,
                        "response_time": end_time - start_time,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "success": False,
                "request_id": request_id,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    async with aiohttp.ClientSession() as session:
        # 发送10个并发请求
        tasks = [send_request(session, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # 统计结果
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        response_times = [r["response_time"] for r in results if r["success"]]
        
        print(f"✅ 并发测试结果:")
        print(f"   成功: {successful}/10")
        print(f"   失败: {failed}/10")
        if response_times:
            print(f"   平均响应时间: {sum(response_times)/len(response_times):.3f}秒")
            print(f"   最快响应: {min(response_times):.3f}秒")
            print(f"   最慢响应: {max(response_times):.3f}秒")

async def main():
    """主函数"""
    print("🚀 聊天服务测试")
    print("=" * 50)
    
    # 基本功能测试
    await test_chat_service()
    
    # 并发测试
    await test_concurrent_requests()
    
    print("\n" + "=" * 50)
    print("✨ 所有测试完成!")

if __name__ == "__main__":
    print("请确保聊天服务已启动在 http://localhost:8000")
    print("启动命令: python -m server.chat.example_server")
    print()
    
    asyncio.run(main()) 