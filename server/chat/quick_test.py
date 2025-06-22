#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速聊天服务测试工具
专门针对频率限制优化，可以立即使用
"""
import asyncio
import aiohttp
import time
import json
import sys
from datetime import datetime

class QuickTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def single_request(self, conversation_id, message, model):
        """发送单个请求"""
        request_data = {
            "conversation_id": conversation_id,
            "message": message,
            "model": model,
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "message_length": len(data.get("message", "")),
                        "model": model
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status}: {error_text[:100]}"
                    }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    async def sequential_test(self, num_requests=10, delay=4.0):
        """顺序测试 - 避免频率限制"""
        print(f"🧪 开始顺序测试: {num_requests} 个请求，间隔 {delay} 秒")
        print("=" * 50)
        
        # 使用免费模型，但控制频率
        models = ["deepseek/deepseek-r1-0528:free"]
        messages = [
            "Hello!",
            "What's 2+2?", 
            "Tell me a joke",
            "What's Python?",
            "How are you?"
        ]
        
        results = []
        for i in range(num_requests):
            model = models[i % len(models)]
            message = messages[i % len(messages)]
            conversation_id = f"test_conv_{i // 3}"  # 每3个请求一个对话
            
            print(f"[{i+1}/{num_requests}] 发送请求... ", end="", flush=True)
            
            result = await self.single_request(conversation_id, message, model)
            results.append(result)
            
            if result["success"]:
                print(f"✅ 成功 ({result['response_time']:.2f}s)")
            else:
                print(f"❌ 失败: {result['error']}")
            
            # 等待间隔（除了最后一个请求）
            if i < num_requests - 1:
                print(f"⏳ 等待 {delay}s...", end="", flush=True)
                await asyncio.sleep(delay)
                print(" ✓")
        
        return results
    
    async def batch_test(self, batch_size=5, num_batches=3, batch_delay=60):
        """批次测试 - 分批发送以避免频率限制"""
        print(f"🔄 开始批次测试: {num_batches} 批，每批 {batch_size} 个请求")
        print(f"批次间隔: {batch_delay} 秒")
        print("=" * 50)
        
        all_results = []
        
        for batch_num in range(num_batches):
            print(f"\n📦 批次 {batch_num + 1}/{num_batches}")
            print("-" * 30)
            
            # 并发发送一批请求
            tasks = []
            for i in range(batch_size):
                req_id = batch_num * batch_size + i
                conversation_id = f"batch_conv_{req_id // 2}"
                message = f"Batch {batch_num + 1}, Request {i + 1}"
                model = "deepseek/deepseek-r1-0528:free"
                
                task = asyncio.create_task(
                    self.single_request(conversation_id, message, model)
                )
                tasks.append(task)
            
            # 等待批次完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果 
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"  [{i+1}] ❌ 异常: {result}")
                    all_results.append({"success": False, "error": str(result)})
                else:
                    if result["success"]:
                        print(f"  [{i+1}] ✅ 成功 ({result['response_time']:.2f}s)")
                    else:
                        print(f"  [{i+1}] ❌ 失败: {result['error']}")
                    all_results.append(result)
            
            # 批次间等待
            if batch_num < num_batches - 1:
                print(f"\n⏳ 批次间等待 {batch_delay}s...")
                await asyncio.sleep(batch_delay)
        
        return all_results
    
    def print_summary(self, results, test_name):
        """打印测试总结"""
        print(f"\n{'='*50}")
        print(f"📊 {test_name} - 测试总结")
        print(f"{'='*50}")
        
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful
        
        print(f"总请求数: {total}")
        print(f"成功请求: {successful}")
        print(f"失败请求: {failed}")
        print(f"成功率: {successful/total*100:.1f}%")
        
        if successful > 0:
            response_times = [r["response_time"] for r in results if r.get("success")]  # type: ignore
            print(f"平均响应时间: {sum(response_times)/len(response_times):.2f}s")
            print(f"最快响应: {min(response_times):.2f}s")
            print(f"最慢响应: {max(response_times):.2f}s")
        
        # 错误统计
        if failed > 0:
            print(f"\n❌ 错误类型:")
            error_counts = {}
            for r in results:
                if not r.get("success", False):
                    error = r.get("error", "Unknown")
                    # 简化错误信息
                    if "429" in error:
                        error_type = "频率限制 (429)"
                    elif "timeout" in error.lower():
                        error_type = "请求超时"
                    elif "connection" in error.lower():
                        error_type = "连接错误"
                    else:
                        error_type = "其他错误"
                    
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in error_counts.items():
                print(f"  {error_type}: {count} 次")

async def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("""
🚀 快速聊天服务测试工具

用法:
  python quick_test.py sequential [请求数] [间隔秒]
  python quick_test.py batch [批次大小] [批次数] [批次间隔]

示例:
  python quick_test.py sequential 10 3      # 10个请求，间隔3秒
  python quick_test.py batch 5 3 60         # 3批，每批5个，间隔60秒

推荐配置（避免频率限制）:
  python quick_test.py sequential 15 4      # 安全的顺序测试
  python quick_test.py batch 3 5 90         # 安全的批次测试
        """)
        return
    
    test_type = sys.argv[1]
    
    async with QuickTester() as tester:
        if test_type == "sequential":
            num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            delay = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
            
            results = await tester.sequential_test(num_requests, delay)
            tester.print_summary(results, "顺序测试")
            
        elif test_type == "batch":
            batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            num_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 3
            batch_delay = int(sys.argv[4]) if len(sys.argv) > 4 else 60
            
            results = await tester.batch_test(batch_size, num_batches, batch_delay)
            tester.print_summary(results, "批次测试")
            
        else:
            print("❌ 未知的测试类型。请使用 'sequential' 或 'batch'")

if __name__ == "__main__":
    asyncio.run(main()) 