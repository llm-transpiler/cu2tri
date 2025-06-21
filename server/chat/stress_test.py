# -*- coding: utf-8 -*-
"""
聊天服务压力测试
测试服务在高并发下的性能表现
"""
import asyncio
import aiohttp
import time
import random
import statistics
from typing import List, Dict, Any
import json
import logging
from dataclasses import dataclass, field
import argparse

@dataclass
class TestResult:
    """测试结果"""
    request_id: str
    conversation_id: str
    model: str
    start_time: float
    end_time: float
    response_time: float
    success: bool
    error: str = ""
    response_length: int = 0

@dataclass
class TestStats:
    """测试统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100
    
    @property
    def requests_per_second(self) -> float:
        if self.duration == 0:
            return 0.0
        return self.total_requests / self.duration
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def median_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile

class ChatStressTester:
    """聊天服务压力测试器"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 logger: logging.Logger = None):
        """初始化压力测试器
        
        Args:
            base_url: 服务器基础URL
            logger: 日志记录器
        """
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        self.session: aiohttp.ClientSession = None
        
        # 测试消息模板
        self.test_messages = [
            "Hello, how are you today?",
            "What's the weather like?",
            "Can you help me with a math problem?",
            "Tell me a joke.",
            "What's the capital of France?",
            "Explain quantum physics in simple terms.",
            "Write a short poem about nature.",
            "What are the benefits of exercise?",
            "How do computers work?",
            "What's your favorite color and why?",
            "Can you summarize the history of AI?",
            "What's the difference between machine learning and deep learning?",
            "How can I improve my productivity?",
            "What are some healthy breakfast ideas?",
            "Explain the concept of blockchain.",
        ]
        
        # 可用的模型列表（使用正确的模型名称）
        self.models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.7-sonnet", 
            "google/gemini-2.0-flash-001",
            "deepseek/deepseek-chat-v3-0324:free",
            "deepseek/deepseek-r1-0528:free"
        ]
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def send_chat_request(self, 
                               conversation_id: str,
                               message: str,
                               model: str,
                               thought: str = None) -> TestResult:
        """发送单个聊天请求"""
        request_data = {
            "conversation_id": conversation_id,
            "message": message,
            "model": model,
            "thought": thought,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    return TestResult(
                        request_id=response_data.get("request_id", ""),
                        conversation_id=conversation_id,
                        model=model,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        success=True,
                        response_length=len(response_data.get("message", ""))
                    )
                else:
                    error_data = await response.text()
                    return TestResult(
                        request_id="",
                        conversation_id=conversation_id,
                        model=model,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_data}"
                    )
                    
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            return TestResult(
                request_id="",
                conversation_id=conversation_id,
                model=model,
                start_time=start_time,
                end_time=end_time,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def run_concurrent_test(self,
                                 num_requests: int = 100,
                                 concurrency: int = 10,
                                 num_conversations: int = 5,
                                 delay_between_requests: float = 0.1) -> TestStats:
        """运行并发测试
        
        Args:
            num_requests: 总请求数
            concurrency: 并发数
            num_conversations: 对话数量
            delay_between_requests: 请求间延迟（秒）
            
        Returns:
            TestStats: 测试统计结果
        """
        self.logger.info(f"Starting concurrent test: {num_requests} requests, {concurrency} concurrent, {num_conversations} conversations")
        
        stats = TestStats()
        stats.start_time = time.time()
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        
        async def worker(request_num: int) -> TestResult:
            async with semaphore:
                # 随机选择对话ID、消息和模型
                conversation_id = f"test_conv_{request_num % num_conversations}"
                message = random.choice(self.test_messages)
                model = random.choice(self.models)
                
                # 随机添加thought
                thought = None
                if random.random() < 0.3:  # 30%概率添加thought
                    thought = f"This is request {request_num}, testing {model}"
                
                result = await self.send_chat_request(conversation_id, message, model, thought)
                
                # 添加延迟
                if delay_between_requests > 0:
                    await asyncio.sleep(delay_between_requests)
                
                return result
        
        # 创建所有任务
        tasks = [worker(i) for i in range(num_requests)]
        
        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stats.end_time = time.time()
        
        # 处理结果
        for result in results:
            if isinstance(result, Exception):
                stats.failed_requests += 1
                stats.errors.append(str(result))
            elif isinstance(result, TestResult):
                stats.total_requests += 1
                if result.success:
                    stats.successful_requests += 1
                    stats.response_times.append(result.response_time)
                else:
                    stats.failed_requests += 1
                    stats.errors.append(result.error)
        
        return stats
    
    async def run_load_test(self,
                           duration_seconds: int = 60,
                           requests_per_second: int = 10,
                           num_conversations: int = 5) -> TestStats:
        """运行负载测试
        
        Args:
            duration_seconds: 测试持续时间（秒）
            requests_per_second: 每秒请求数
            num_conversations: 对话数量
            
        Returns:
            TestStats: 测试统计结果
        """
        self.logger.info(f"Starting load test: {duration_seconds}s duration, {requests_per_second} RPS, {num_conversations} conversations")
        
        stats = TestStats()
        stats.start_time = time.time()
        
        request_interval = 1.0 / requests_per_second
        request_count = 0
        
        async def request_worker():
            nonlocal request_count
            conversation_id = f"load_conv_{request_count % num_conversations}"
            message = random.choice(self.test_messages)
            model = random.choice(self.models)
            
            result = await self.send_chat_request(conversation_id, message, model)
            request_count += 1
            
            return result
        
        # 运行测试
        end_time = stats.start_time + duration_seconds
        results = []
        
        while time.time() < end_time:
            # 启动请求
            task = asyncio.create_task(request_worker())
            results.append(task)
            
            # 等待下一个请求间隔
            await asyncio.sleep(request_interval)
        
        # 等待所有请求完成
        completed_results = await asyncio.gather(*results, return_exceptions=True)
        
        stats.end_time = time.time()
        
        # 处理结果
        for result in completed_results:
            if isinstance(result, Exception):
                stats.failed_requests += 1
                stats.errors.append(str(result))
            elif isinstance(result, TestResult):
                stats.total_requests += 1
                if result.success:
                    stats.successful_requests += 1
                    stats.response_times.append(result.response_time)
                else:
                    stats.failed_requests += 1
                    stats.errors.append(result.error)
        
        return stats
    
    async def run_conversation_test(self,
                                   num_conversations: int = 3,
                                   messages_per_conversation: int = 5) -> TestStats:
        """运行多轮对话测试
        
        Args:
            num_conversations: 对话数量
            messages_per_conversation: 每个对话的消息数
            
        Returns:
            TestStats: 测试统计结果
        """
        self.logger.info(f"Starting conversation test: {num_conversations} conversations, {messages_per_conversation} messages each")
        
        stats = TestStats()
        stats.start_time = time.time()
        
        async def conversation_worker(conv_id: int) -> List[TestResult]:
            conversation_id = f"multi_conv_{conv_id}"
            model = random.choice(self.models)
            results = []
            
            for msg_num in range(messages_per_conversation):
                message = f"Message {msg_num + 1}: {random.choice(self.test_messages)}"
                thought = f"This is message {msg_num + 1} in conversation {conv_id}"
                
                result = await self.send_chat_request(conversation_id, message, model, thought)
                results.append(result)
                
                # 短暂延迟模拟真实对话
                await asyncio.sleep(0.5)
            
            return results
        
        # 创建所有对话任务
        tasks = [conversation_worker(i) for i in range(num_conversations)]
        
        # 执行所有对话
        conversation_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stats.end_time = time.time()
        
        # 处理结果
        for conv_results in conversation_results:
            if isinstance(conv_results, Exception):
                stats.failed_requests += 1
                stats.errors.append(str(conv_results))
            else:
                for result in conv_results:
                    stats.total_requests += 1
                    if result.success:
                        stats.successful_requests += 1
                        stats.response_times.append(result.response_time)
                    else:
                        stats.failed_requests += 1
                        stats.errors.append(result.error)
        
        return stats
    
    def print_stats(self, stats: TestStats, test_name: str):
        """打印测试统计结果"""
        print(f"\n{'='*60}")
        print(f"测试结果: {test_name}")
        print(f"{'='*60}")
        print(f"总请求数: {stats.total_requests}")
        print(f"成功请求: {stats.successful_requests}")
        print(f"失败请求: {stats.failed_requests}")
        print(f"成功率: {stats.success_rate:.2f}%")
        print(f"测试持续时间: {stats.duration:.2f}秒")
        print(f"每秒请求数: {stats.requests_per_second:.2f}")
        
        if stats.response_times:
            print(f"\n响应时间统计:")
            print(f"  平均响应时间: {stats.avg_response_time:.3f}秒")
            print(f"  中位数响应时间: {stats.median_response_time:.3f}秒")
            print(f"  95%响应时间: {stats.p95_response_time:.3f}秒")
            print(f"  99%响应时间: {stats.p99_response_time:.3f}秒")
            print(f"  最快响应时间: {min(stats.response_times):.3f}秒")
            print(f"  最慢响应时间: {max(stats.response_times):.3f}秒")
        
        if stats.errors:
            print(f"\n错误统计:")
            error_counts = {}
            for error in stats.errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error}: {count}次")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="聊天服务压力测试")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--test", choices=["concurrent", "load", "conversation", "all"], 
                       default="all", help="测试类型")
    parser.add_argument("--requests", type=int, default=100, help="并发测试请求数")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--duration", type=int, default=60, help="负载测试持续时间（秒）")
    parser.add_argument("--rps", type=int, default=10, help="负载测试每秒请求数")
    parser.add_argument("--conversations", type=int, default=5, help="对话数量")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    async with ChatStressTester(args.url, logger) as tester:
        if args.test in ["concurrent", "all"]:
            stats = await tester.run_concurrent_test(
                num_requests=args.requests,
                concurrency=args.concurrency,
                num_conversations=args.conversations
            )
            tester.print_stats(stats, "并发测试")
        
        if args.test in ["load", "all"]:
            stats = await tester.run_load_test(
                duration_seconds=args.duration,
                requests_per_second=args.rps,
                num_conversations=args.conversations
            )
            tester.print_stats(stats, "负载测试")
        
        if args.test in ["conversation", "all"]:
            stats = await tester.run_conversation_test(
                num_conversations=args.conversations,
                messages_per_conversation=5
            )
            tester.print_stats(stats, "多轮对话测试")

if __name__ == "__main__":
    asyncio.run(main()) 