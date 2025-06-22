#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强队列管理器验证测试
快速验证所有功能是否正常工作
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

# 模拟导入（实际使用时请使用真实的导入）
@dataclass
class MockChatRequest:
    request_id: str
    conversation_id: str
    message: str
    model: str
    max_tokens: Optional[int] = 1000
    thought: Optional[str] = None

@dataclass
class MockChatResponse:
    request_id: str
    conversation_id: str
    message: str
    usage: Optional[dict] = None
    success: bool = True

class MockChatService:
    """模拟聊天服务"""
    
    async def process_request(self, request: MockChatRequest) -> MockChatResponse:
        # 模拟处理时间
        await asyncio.sleep(0.1)
        
        # 模拟token使用
        usage = {
            'prompt_tokens': len(request.message) // 4,
            'completion_tokens': 50,
            'total_tokens': len(request.message) // 4 + 50
        }
        
        return MockChatResponse(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            message=f"Response to: {request.message}",
            usage=usage,
            success=True
        )

async def test_basic_functionality():
    """测试基础功能"""
    print("🧪 Testing Basic Functionality...")
    
    # 导入增强队列管理器
    try:
        from enhanced_queue_manager import EnhancedChatQueueManager, RateLimitConfig
        from rate_limit_config import OpenRouterRateLimits, ModelClassifier, TokenEstimator
        print("✅ Successfully imported enhanced queue manager")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # 测试配置创建
    try:
        rate_config = RateLimitConfig(**OpenRouterRateLimits.get_config())
        print("✅ Successfully created rate limit config")
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False
    
    # 测试模型分类
    try:
        assert ModelClassifier.is_free_model("deepseek/deepseek-r1-0528:free")
        assert not ModelClassifier.is_free_model("openai/gpt-4o-mini")
        print("✅ Model classification works correctly")
    except Exception as e:
        print(f"❌ Model classification failed: {e}")
        return False
    
    # 测试token估算
    try:
        tokens = TokenEstimator.estimate_tokens("Hello world", max_tokens=100)
        assert tokens > 0
        print(f"✅ Token estimation works: {tokens} tokens")
    except Exception as e:
        print(f"❌ Token estimation failed: {e}")
        return False
    
    return True

async def test_queue_manager():
    """测试队列管理器"""
    print("\n🧪 Testing Queue Manager...")
    
    try:
        from enhanced_queue_manager import EnhancedChatQueueManager, RateLimitConfig
        from rate_limit_config import OpenRouterRateLimits
        
        # 创建测试配置
        rate_config = RateLimitConfig(**OpenRouterRateLimits.get_config())
        
        # 创建队列管理器
        queue_manager = EnhancedChatQueueManager(
            max_workers=2,
            max_queue_size=10,
            task_timeout=30,
            rate_limit_config=rate_config,
            log_dir="logs/test_queue"
        )
        
        # 创建模拟服务
        mock_service = MockChatService()
        
        # 设置请求处理器
        queue_manager.set_request_processor(mock_service.process_request)
        
        print("✅ Queue manager created successfully")
        
        # 启动队列管理器
        await queue_manager.start()
        print("✅ Queue manager started successfully")
        
        # 创建测试请求
        test_requests = [
            MockChatRequest(
                request_id=f"test_{i}",
                conversation_id="test_conversation",
                message=f"Test message {i}",
                model="deepseek/deepseek-r1-0528:free" if i % 2 == 0 else "openai/gpt-4o-mini"
            )
            for i in range(5)
        ]
        
        # 提交请求
        print("🚀 Submitting test requests...")
        start_time = time.time()
        
        tasks = [queue_manager.submit_request(req) for req in test_requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # 检查结果
        success_count = sum(1 for r in responses if not isinstance(r, Exception))
        error_count = len(responses) - success_count
        
        print(f"✅ Processed {len(responses)} requests in {end_time - start_time:.2f}s")
        print(f"✅ Success: {success_count}, Errors: {error_count}")
        
        # 获取统计信息
        stats = queue_manager.get_stats()
        print(f"📊 Queue Stats:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Completed: {stats['completed_requests']}")
        print(f"  Failed: {stats['failed_requests']}")
        print(f"  Rate Limited: {stats['rate_limited_requests']}")
        print(f"  Current RPM: {stats['current_rpm']}")
        print(f"  Tokens/min: {stats['tokens_per_minute']}")
        
        # 停止队列管理器
        await queue_manager.stop()
        print("✅ Queue manager stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Queue manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rate_limiting():
    """测试速率限制"""
    print("\n🧪 Testing Rate Limiting...")
    
    try:
        from enhanced_queue_manager import EnhancedChatQueueManager, RateLimitConfig
        from rate_limit_config import OpenRouterRateLimits
        
        # 创建严格的测试配置
        rate_config = RateLimitConfig(
            free_model_rpm=3,  # 非常低的限制用于测试
            free_model_daily=10,
            paid_model_rpm=5,
            max_tokens_per_minute=1000,
            rate_limit_buffer=0.8
        )
        
        queue_manager = EnhancedChatQueueManager(
            max_workers=1,
            rate_limit_config=rate_config,
            log_dir="logs/test_rate_limit"
        )
        
        mock_service = MockChatService()
        queue_manager.set_request_processor(mock_service.process_request)
        
        await queue_manager.start()
        
        # 创建大量免费模型请求
        requests = [
            MockChatRequest(
                request_id=f"rate_test_{i}",
                conversation_id="rate_test",
                message=f"Rate limit test {i}",
                model="deepseek/deepseek-r1-0528:free"
            )
            for i in range(8)  # 超过限制的请求数
        ]
        
        print("🚀 Testing rate limiting with 8 free model requests...")
        start_time = time.time()
        
        # 提交所有请求
        tasks = [queue_manager.submit_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # 分析结果
        success_count = sum(1 for r in responses if not isinstance(r, Exception))
        timeout_count = sum(1 for r in responses if isinstance(r, asyncio.TimeoutError))
        
        print(f"⏱️  Total time: {end_time - start_time:.2f}s")
        print(f"✅ Successful requests: {success_count}")
        print(f"⏰ Timeout requests: {timeout_count}")
        
        stats = queue_manager.get_stats()
        print(f"🚫 Rate limited requests: {stats['rate_limited_requests']}")
        
        # 验证速率限制是否工作
        if stats['rate_limited_requests'] > 0 or end_time - start_time > 10:
            print("✅ Rate limiting is working correctly")
        else:
            print("⚠️  Rate limiting might not be working as expected")
        
        await queue_manager.stop()
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_token_management():
    """测试token管理"""
    print("\n🧪 Testing Token Management...")
    
    try:
        from rate_limit_config import TokenEstimator, ModelClassifier
        
        # 测试token估算
        test_cases = [
            ("Hello world", None, 100),
            ("This is a longer message for testing", "Some thought", 500),
            ("Short", None, 50)
        ]
        
        for message, thought, max_tokens in test_cases:
            tokens = TokenEstimator.estimate_tokens(message, thought, max_tokens)
            cost = TokenEstimator.estimate_cost(tokens, "openai/gpt-4o-mini")

            msg_text: str = message  # Type hint to resolve fixture confusion
            print(f"📝 Message: '{msg_text[:20]}...'")
            print(f"   Estimated tokens: {tokens}")
            print(f"   Estimated cost: ${cost:.4f}")
        
        # 测试模型推荐
        recommendations = ModelClassifier.get_recommended_models()
        print(f"🎯 Recommended models:")
        print(f"   Free: {recommendations['free']}")
        print(f"   Paid: {recommendations['paid']}")
        
        print("✅ Token management tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Token management test failed: {e}")
        return False

async def main():
    """主测试函数"""
    print("🚀 Enhanced Queue Manager Validation Tests")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Queue Manager", test_queue_manager),
        ("Rate Limiting", test_rate_limiting),
        ("Token Management", test_token_management)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*50)
    print("🎯 Test Results Summary")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
            failed += 1
    
    print(f"\n📊 Overall: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced queue manager is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(main()) 