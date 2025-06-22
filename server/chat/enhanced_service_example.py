#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版聊天服务使用示例
展示如何集成速率限制和token管理
"""
import asyncio
import logging
import os
from pathlib import Path

try:
    from .enhanced_queue_manager import EnhancedChatQueueManager, RateLimitConfig
    from .rate_limit_config import OpenRouterRateLimits, ModelClassifier, TokenEstimator, get_env_config
    from .deprecated.models import ChatRequest, ChatResponse
    from .service import ChatService
except ImportError:
    # 当作为独立模块运行时的后备导入
    from enhanced_queue_manager import EnhancedChatQueueManager, RateLimitConfig
    from rate_limit_config import OpenRouterRateLimits, ModelClassifier, TokenEstimator, get_env_config
    from server.chat.deprecated.models import ChatRequest, ChatResponse
    from service import ChatService

class EnhancedChatService:
    """增强版聊天服务"""
    
    def __init__(self):
        # 获取环境配置
        self.env_config = get_env_config()
        
        # 设置日志
        self.setup_logging()
        
        # 创建速率限制配置
        self.rate_config = RateLimitConfig(**OpenRouterRateLimits.get_config(
            has_paid_credits=self.env_config['has_paid_credits']
        ))
        
        # 创建增强队列管理器
        self.queue_manager = EnhancedChatQueueManager(
            max_workers=self.env_config['max_workers'],
            max_queue_size=self.env_config['max_queue_size'],
            rate_limit_config=self.rate_config,
            openrouter_api_key=self.env_config['openrouter_api_key'],
            log_dir=self.env_config['log_dir'],
            logger=self.logger
        )
        
        # 创建基础聊天服务
        self.chat_service = ChatService(logger=self.logger)
        
        # 设置请求处理器
        self.queue_manager.set_request_processor(self.process_chat_request)
    
    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.env_config['log_level'].upper())
        
        # 创建日志目录
        log_dir = Path(self.env_config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(log_dir / 'enhanced_service.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # 配置logger
        self.logger = logging.getLogger('enhanced_chat_service')
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求（由队列管理器调用）"""
        # 记录请求开始
        estimated_tokens = TokenEstimator.estimate_tokens(
            request.message, 
            request.thought, 
            request.max_tokens or 1000
        )
        estimated_cost = TokenEstimator.estimate_cost(estimated_tokens, request.model)
        
        self.logger.info(
            f"🚀 Processing request {request.request_id} - "
            f"Model: {request.model}, "
            f"Estimated tokens: {estimated_tokens}, "
            f"Estimated cost: ${estimated_cost:.4f}"
        )
        
        # 使用基础服务处理请求
        response = await self.chat_service.process_request(request)
        
        # 记录响应
        if hasattr(response, 'usage') and response.usage:
            actual_tokens = response.usage.get('total_tokens', 0)
            actual_cost = TokenEstimator.estimate_cost(actual_tokens, request.model)
            
            self.logger.info(
                f"✅ Completed request {request.request_id} - "
                f"Actual tokens: {actual_tokens}, "
                f"Actual cost: ${actual_cost:.4f}"
            )
        
        return response
    
    async def start(self):
        """启动服务"""
        self.logger.info("🚀 Starting Enhanced Chat Service")
        
        # 显示配置信息
        self.logger.info(f"📋 Configuration:")
        self.logger.info(f"  - Max Workers: {self.env_config['max_workers']}")
        self.logger.info(f"  - Max Queue Size: {self.env_config['max_queue_size']}")
        self.logger.info(f"  - Free Model RPM: {self.rate_config.free_model_rpm}")
        self.logger.info(f"  - Free Model Daily: {self.rate_config.free_model_daily}")
        self.logger.info(f"  - Paid Model RPM: {self.rate_config.paid_model_rpm}")
        self.logger.info(f"  - Rate Limit Buffer: {self.rate_config.rate_limit_buffer}")
        
        # 显示推荐模型
        recommended = ModelClassifier.get_recommended_models()
        self.logger.info(f"🎯 Recommended Models:")
        self.logger.info(f"  - Free: {', '.join(recommended['free'])}")
        self.logger.info(f"  - Paid: {', '.join(recommended['paid'])}")
        
        # 启动队列管理器
        await self.queue_manager.start()
        
        self.logger.info("✅ Enhanced Chat Service started successfully")
    
    async def stop(self):
        """停止服务"""
        self.logger.info("🛑 Stopping Enhanced Chat Service")
        await self.queue_manager.stop()
        self.logger.info("✅ Enhanced Chat Service stopped")
    
    async def chat(self, request: ChatRequest, priority: int = 0) -> ChatResponse:
        """处理聊天请求（公共接口）"""
        return await self.queue_manager.submit_request(request, priority)
    
    def get_stats(self):
        """获取服务统计"""
        return self.queue_manager.get_stats()
    
    def get_model_recommendations(self, use_free: bool = None):
        """获取模型推荐"""
        recommended = ModelClassifier.get_recommended_models()
        
        if use_free is True:
            return recommended['free']
        elif use_free is False:
            return recommended['paid']
        else:
            return recommended

async def demo():
    """演示功能"""
    # 创建服务
    service = EnhancedChatService()
    
    try:
        # 启动服务
        await service.start()
        
        # 创建测试请求
        requests = [
            ChatRequest(
                request_id=f"demo_{i}",
                conversation_id="demo_conversation",
                message=f"Hello, this is test message {i}",
                model="deepseek/deepseek-r1-0528:free",
                max_tokens=100
            )
            for i in range(5)
        ]
        
        print("🧪 Sending demo requests...")
        
        # 发送请求
        tasks = [service.chat(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 显示结果
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"❌ Request {i} failed: {response}")
            else:
                print(f"✅ Request {i} succeeded: {response.message[:50]}...")
        
        # 显示统计
        stats = service.get_stats()
        print(f"\n📊 Final Stats:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Completed: {stats['completed_requests']}")
        print(f"  Failed: {stats['failed_requests']}")
        print(f"  Rate Limited: {stats['rate_limited_requests']}")
        print(f"  Current RPM: {stats['current_rpm']}")
        print(f"  Tokens/min: {stats['tokens_per_minute']}")
        
    finally:
        # 停止服务
        await service.stop()

async def monitoring_demo():
    """监控演示"""
    service = EnhancedChatService()
    
    try:
        await service.start()
        
        # 监控循环
        for i in range(10):
            stats = service.get_stats()
            rate_limits = stats.get('rate_limits', {})
            
            print(f"📊 Monitor Update {i+1}:")
            print(f"  Queue Size: {stats['queue_size']}")
            print(f"  Active Workers: {stats['active_workers']}")
            print(f"  Free RPM: {rate_limits.get('free_rpm_current', 0)}/{rate_limits.get('free_rpm_limit', 0)}")
            print(f"  Paid RPM: {rate_limits.get('paid_rpm_current', 0)}/{rate_limits.get('paid_rpm_limit', 0)}")
            print(f"  Daily Free Usage: {rate_limits.get('daily_free_usage', 0)}/{rate_limits.get('daily_free_limit', 0)}")
            print("-" * 50)
            
            await asyncio.sleep(5)
            
    finally:
        await service.stop()

if __name__ == "__main__":
    # 设置环境变量（如果需要）
    if not os.getenv('OPENROUTER_API_KEY'):
        print("⚠️  Warning: OPENROUTER_API_KEY not set")
    
    # 运行演示
    print("🎯 Running Enhanced Chat Service Demo")
    asyncio.run(demo())
    
    print("\n🔍 Running Monitoring Demo")
    asyncio.run(monitoring_demo()) 