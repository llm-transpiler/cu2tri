# -*- coding: utf-8 -*-
"""
智能聊天服务压力测试
包含频率限制感知、智能重试和更好的错误处理
"""
import asyncio
import aiohttp
import time
import random
import statistics
from typing import List, Dict, Any, Optional
import json
import logging
from dataclasses import dataclass, field
import argparse
from datetime import datetime, timedelta

@dataclass
class RateLimitInfo:
    """频率限制信息"""
    limit: int = 0
    remaining: int = 0
    reset_time: int = 0
    provider: str = ""

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
    retry_count: int = 0
    rate_limit_info: Optional[RateLimitInfo] = None

@dataclass
class TestStats:
    """测试统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    retried_requests: int = 0
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

class SmartChatStressTester:
    """智能聊天服务压力测试器"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 logger: logging.Logger = None,
                 enable_rate_limit_handling: bool = True,
                 max_retries: int = 3):
        """初始化压力测试器
        
        Args:
            base_url: 服务器基础URL
            logger: 日志记录器
            enable_rate_limit_handling: 启用频率限制处理
            max_retries: 最大重试次数
        """
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        self.session: aiohttp.ClientSession = None
        self.enable_rate_limit_handling = enable_rate_limit_handling
        self.max_retries = max_retries
        
        # 频率限制跟踪
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.last_request_times: Dict[str, float] = {}
        
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
        
        # 模型配置（按频率限制分类）
        self.model_configs = {
            "free": {
                "models": [
                    "deepseek/deepseek-chat-v3-0324:free",
                    "deepseek/deepseek-r1-0528:free"
                ],
                "max_rpm": 20,  # 每分钟最大请求数
                "min_interval": 3.0  # 最小请求间隔（秒）
            },
            "paid": {
                "models": [
                    "openai/gpt-4o-mini", 
                    "google/gemini-2.0-flash-001"
                ],
                "max_rpm": 300,  # 较高限制（需要API密钥）
                "min_interval": 0.2  # 较短间隔
            }
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def extract_rate_limit_info(self, response_headers: Dict[str, str]) -> RateLimitInfo:
        """从响应头提取频率限制信息"""
        return RateLimitInfo(
            limit=int(response_headers.get('X-RateLimit-Limit', 0)),
            remaining=int(response_headers.get('X-RateLimit-Remaining', 0)),
            reset_time=int(response_headers.get('X-RateLimit-Reset', 0)),
            provider=response_headers.get('provider_name', 'unknown')
        )
    
    def get_model_category(self, model: str) -> str:
        """获取模型类别"""
        for category, config in self.model_configs.items():
            if model in config["models"]:
                return category
        return "paid"  # 默认为付费模型
    
    async def wait_for_rate_limit(self, model: str):
        """等待频率限制重置"""
        if not self.enable_rate_limit_handling:
            return
            
        category = self.get_model_category(model)
        config = self.model_configs[category]
        
        # 检查上次请求时间
        last_time = self.last_request_times.get(category, 0)
        elapsed = time.time() - last_time
        min_interval = config["min_interval"]
        
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {category} models")
            await asyncio.sleep(wait_time)
        
        self.last_request_times[category] = time.time()
    
    async def send_chat_request_with_retry(self, 
                                          conversation_id: str,
                                          message: str,
                                          model: str,
                                          thought: str = None) -> TestResult:
        """发送聊天请求（带重试）"""
        for retry in range(self.max_retries + 1):
            # 频率限制等待
            await self.wait_for_rate_limit(model)
            
            result = await self.send_chat_request(conversation_id, message, model, thought)
            result.retry_count = retry
            
            # 如果成功或非频率限制错误，直接返回
            if result.success or "429" not in result.error:
                return result
            
            # 频率限制错误处理
            if retry < self.max_retries:
                wait_time = (2 ** retry) * 5  # 指数退避：5s, 10s, 20s
                self.logger.warning(f"Rate limited, retrying in {wait_time}s (attempt {retry + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
            else:
                self.logger.error(f"Max retries exceeded for request")
        
        return result
    
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
                
                # 提取频率限制信息
                rate_limit_info = self.extract_rate_limit_info(dict(response.headers))
                
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
                        response_length=len(response_data.get("message", "")),
                        rate_limit_info=rate_limit_info
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
                        error=f"HTTP {response.status}: {error_data}",
                        rate_limit_info=rate_limit_info
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
    
    async def run_rate_aware_test(self,
                                 num_requests: int = 50,
                                 model_category: str = "free",
                                 num_conversations: int = 5) -> TestStats:
        """运行频率限制感知测试"""
        config = self.model_configs[model_category]
        models = config["models"]
        max_rpm = config["max_rpm"]
        
        self.logger.info(f"Starting rate-aware test: {num_requests} requests, {model_category} models (max {max_rpm} RPM)")
        
        stats = TestStats()
        stats.start_time = time.time()
        
        # 计算请求间隔以不超过频率限制
        request_interval = 60.0 / (max_rpm * 0.8)  # 使用80%的限制以留出缓冲
        
        results = []
        for i in range(num_requests):
            conversation_id = f"rate_conv_{i % num_conversations}"
            message = random.choice(self.test_messages)
            model = random.choice(models)
            
            # 异步发送请求
            task = asyncio.create_task(
                self.send_chat_request_with_retry(conversation_id, message, model)
            )
            results.append(task)
            
            # 等待下一个请求间隔
            if i < num_requests - 1:  # 最后一个请求不需要等待
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
                if result.retry_count > 0:
                    stats.retried_requests += 1
                if "429" in result.error:
                    stats.rate_limited_requests += 1
                
                if result.success:
                    stats.successful_requests += 1
                    stats.response_times.append(result.response_time)
                else:
                    stats.failed_requests += 1
                    stats.errors.append(result.error)
        
        return stats
    
    async def run_mixed_model_test(self,
                                  num_requests: int = 50,
                                  free_ratio: float = 0.3) -> TestStats:
        """运行混合模型测试（免费+付费模型）"""
        self.logger.info(f"Starting mixed model test: {num_requests} requests, {free_ratio*100}% free models")
        
        stats = TestStats()
        stats.start_time = time.time()
        
        # 分配请求到不同模型类型
        free_requests = int(num_requests * free_ratio)
        paid_requests = num_requests - free_requests
        
        tasks = []
        
        # 免费模型请求（较慢速率）
        free_models = self.model_configs["free"]["models"]
        free_interval = 60.0 / (self.model_configs["free"]["max_rpm"] * 0.8)
        
        for i in range(free_requests):
            conversation_id = f"free_conv_{i % 3}"
            message = random.choice(self.test_messages)
            model = random.choice(free_models)
            
            # 延迟启动以分散请求
            delay = i * free_interval
            task = asyncio.create_task(self._delayed_request(
                delay, conversation_id, message, model
            ))
            tasks.append(task)
        
        # 付费模型请求（较快速率）
        paid_models = self.model_configs["paid"]["models"]
        paid_interval = 60.0 / (self.model_configs["paid"]["max_rpm"] * 0.8)
        
        for i in range(paid_requests):
            conversation_id = f"paid_conv_{i % 3}"
            message = random.choice(self.test_messages)
            model = random.choice(paid_models)
            
            # 延迟启动
            delay = i * paid_interval
            task = asyncio.create_task(self._delayed_request(
                delay, conversation_id, message, model
            ))
            tasks.append(task)
        
        # 等待所有请求完成
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stats.end_time = time.time()
        
        # 处理结果（与之前相同的逻辑）
        for result in completed_results:
            if isinstance(result, Exception):
                stats.failed_requests += 1
                stats.errors.append(str(result))
            elif isinstance(result, TestResult):
                stats.total_requests += 1
                if result.retry_count > 0:
                    stats.retried_requests += 1
                if "429" in result.error:
                    stats.rate_limited_requests += 1
                
                if result.success:
                    stats.successful_requests += 1
                    stats.response_times.append(result.response_time)
                else:
                    stats.failed_requests += 1
                    stats.errors.append(result.error)
        
        return stats
    
    async def _delayed_request(self, delay: float, conversation_id: str, message: str, model: str) -> TestResult:
        """延迟发送请求"""
        await asyncio.sleep(delay)
        return await self.send_chat_request_with_retry(conversation_id, message, model)
    
    def print_stats(self, stats: TestStats, test_name: str):
        """打印测试统计结果"""
        print(f"\n{'='*60}")
        print(f"🧪 测试结果: {test_name}")
        print(f"{'='*60}")
        print(f"📊 基本统计:")
        print(f"  总请求数: {stats.total_requests}")
        print(f"  成功请求: {stats.successful_requests}")
        print(f"  失败请求: {stats.failed_requests}")
        print(f"  频率限制: {stats.rate_limited_requests}")
        print(f"  重试请求: {stats.retried_requests}")
        print(f"  成功率: {stats.success_rate:.2f}%")
        print(f"  测试持续时间: {stats.duration:.2f}秒")
        print(f"  每秒请求数: {stats.requests_per_second:.2f}")
        
        if stats.response_times:
            print(f"\n⏱️  响应时间统计:")
            print(f"  平均响应时间: {stats.avg_response_time:.3f}秒")
            print(f"  中位数响应时间: {stats.median_response_time:.3f}秒")
            print(f"  95%响应时间: {stats.p95_response_time:.3f}秒")
            print(f"  99%响应时间: {stats.p99_response_time:.3f}秒")
            print(f"  最快响应时间: {min(stats.response_times):.3f}秒")
            print(f"  最慢响应时间: {max(stats.response_times):.3f}秒")
        
        if stats.errors:
            print(f"\n❌ 错误统计:")
            error_counts = {}
            for error in stats.errors:
                # 简化错误消息显示
                simplified_error = error[:100] + "..." if len(error) > 100 else error
                error_counts[simplified_error] = error_counts.get(simplified_error, 0) + 1
            
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:  # 只显示前5个
                print(f"  {error}: {count}次")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能聊天服务压力测试")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--test", choices=["rate-aware", "mixed", "all"], 
                       default="rate-aware", help="测试类型")
    parser.add_argument("--requests", type=int, default=30, help="请求数（适配频率限制）")
    parser.add_argument("--model-category", choices=["free", "paid"], 
                       default="free", help="模型类别")
    parser.add_argument("--free-ratio", type=float, default=0.5, 
                       help="混合测试中免费模型比例")
    parser.add_argument("--disable-rate-handling", action="store_true", 
                       help="禁用频率限制处理")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    async with SmartChatStressTester(
        args.url, 
        logger,
        enable_rate_limit_handling=not args.disable_rate_handling,
        max_retries=args.max_retries
    ) as tester:
        
        if args.test in ["rate-aware", "all"]:
            stats = await tester.run_rate_aware_test(
                num_requests=args.requests,
                model_category=args.model_category
            )
            tester.print_stats(stats, f"频率限制感知测试 ({args.model_category}模型)")
        
        if args.test in ["mixed", "all"]:
            stats = await tester.run_mixed_model_test(
                num_requests=args.requests,
                free_ratio=args.free_ratio
            )
            tester.print_stats(stats, "混合模型测试")

if __name__ == "__main__":
    asyncio.run(main()) 