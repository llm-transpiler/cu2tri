# -*- coding: utf-8 -*-
"""
增强版队列管理器
包含OpenRouter速率限制、token计算和智能日志维护
"""
import asyncio
import logging
import json
import time
import aiohttp
from typing import Dict, Optional, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import os
from pathlib import Path

try:
    from .deprecated.models import ChatRequest, ChatResponse, ErrorResponse
except ImportError:
    # 当作为独立模块运行时的后备导入
    from server.chat.deprecated.models import ChatRequest, ChatResponse, ErrorResponse

@dataclass
class RateLimitConfig:
    """速率限制配置"""
    # OpenRouter限制配置
    free_model_rpm: int = 20  # 免费模型每分钟请求数
    free_model_daily: int = 1000  # 免费模型每日请求数（购买10+credits后）
    paid_model_rpm: int = 300  # 付费模型每分钟请求数
    
    # Token限制配置
    max_tokens_per_minute: int = 100000  # 每分钟最大token数
    max_tokens_per_request: int = 4096   # 单次请求最大token数
    
    # 缓冲配置
    rate_limit_buffer: float = 0.8  # 使用限制的80%作为缓冲

@dataclass
class TokenUsage:
    """Token使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class RequestMetrics:
    """请求指标"""
    request_id: str
    model: str
    is_free_model: bool
    tokens_used: TokenUsage
    response_time: float
    timestamp: float
    success: bool
    error_type: Optional[str] = None

@dataclass
class QueueTask:
    """队列任务"""
    request: ChatRequest
    future: asyncio.Future
    timestamp: float
    priority: int = 0
    estimated_tokens: int = 0  # 预估token数

class EnhancedChatQueueManager:
    """增强版聊天队列管理器"""
    
    def __init__(self, 
                 max_workers: int = 10,
                 max_queue_size: int = 1000,
                 task_timeout: int = 300,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 openrouter_api_key: Optional[str] = None,
                 log_dir: str = "logs/queue",
                 logger: Optional[logging.Logger] = None):
        """初始化增强队列管理器"""
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_timeout = task_timeout
        self.rate_config = rate_limit_config or RateLimitConfig()
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.log_dir = Path(log_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 队列和工作者
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.workers = []
        self.running = False
        
        # 速率限制跟踪
        self.request_times = {
            'free': deque(),      # 免费模型请求时间
            'paid': deque(),      # 付费模型请求时间
        }
        self.token_usage_history = deque()  # Token使用历史
        self.daily_free_requests = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 指标收集
        self.request_metrics: List[RequestMetrics] = []
        self.current_api_limits = {}  # 当前API限制信息
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'queue_size': 0,
            'active_workers': 0,
            'average_processing_time': 0.0,
            'tokens_per_minute': 0,
            'current_rpm': 0,
            'free_model_daily_usage': 0
        }
        
        # 处理器函数
        self.request_processor: Optional[Callable] = None
        
        # 线程安全锁
        self._stats_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()
        
        # HTTP会话
        self.session: Optional[aiohttp.ClientSession] = None
    
    def set_request_processor(self, processor: Callable):
        """设置请求处理器"""
        self.request_processor = processor
    
    def _is_free_model(self, model: str) -> bool:
        """判断是否为免费模型"""
        return model.endswith(':free')
    
    def _estimate_tokens(self, request: ChatRequest) -> int:
        """估算请求token数"""
        # 简单的token估算：大约4个字符=1个token
        message_length = len(request.message)
        thought_length = len(request.thought or "")
        estimated = (message_length + thought_length) // 4
        
        # 加上系统消息和响应token的估算
        estimated += 100  # 系统消息
        estimated += (request.max_tokens or 1000) // 2  # 预期响应长度的一半
        
        return min(estimated, self.rate_config.max_tokens_per_request)
    
    async def _check_rate_limit(self, model: str, estimated_tokens: int) -> Tuple[bool, str]:
        """检查速率限制"""
        with self._rate_limit_lock:
            current_time = time.time()
            is_free = self._is_free_model(model)
            model_type = 'free' if is_free else 'paid'
            
            # 检查每日限制（仅免费模型）
            if is_free:
                # 检查是否需要重置每日计数
                now = datetime.now()
                if now.date() > self.daily_reset_time.date():
                    self.daily_free_requests = 0
                    self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                
                if self.daily_free_requests >= self.rate_config.free_model_daily:
                    return False, f"Daily free model limit exceeded ({self.rate_config.free_model_daily})"
            
            # 检查每分钟请求数限制
            rpm_limit = self.rate_config.free_model_rpm if is_free else self.rate_config.paid_model_rpm
            effective_limit = int(rpm_limit * self.rate_config.rate_limit_buffer)
            
            # 清理1分钟前的记录
            minute_ago = current_time - 60
            request_times = self.request_times[model_type]
            while request_times and request_times[0] < minute_ago:
                request_times.popleft()
            
            if len(request_times) >= effective_limit:
                return False, f"Rate limit exceeded: {len(request_times)}/{effective_limit} RPM"
            
            # 检查token限制
            minute_tokens = sum(
                usage.total_tokens for usage in self.token_usage_history
                if current_time - usage.timestamp < 60
            )
            
            if minute_tokens + estimated_tokens > self.rate_config.max_tokens_per_minute:
                return False, f"Token rate limit exceeded: {minute_tokens + estimated_tokens}/{self.rate_config.max_tokens_per_minute}"
            
            return True, "OK"
    
    async def start(self):
        """启动队列管理器"""
        if self.running:
            return
        
        self.running = True
        self.logger.info(f"Starting enhanced queue manager with {self.max_workers} workers")
        
        # 创建HTTP会话
        self.session = aiohttp.ClientSession()
        
        # 检查API限制
        await self._check_api_limits()
        
        # 启动工作者协程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # 启动后台任务
        asyncio.create_task(self._stats_updater())
        asyncio.create_task(self._rate_limit_cleaner())
        asyncio.create_task(self._log_maintainer())
        asyncio.create_task(self._api_limit_checker())
    
    async def stop(self):
        """停止队列管理器"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping enhanced queue manager...")
        
        # 关闭HTTP会话
        if self.session:
            await self.session.close()
        
        # 取消所有工作者
        for worker in self.workers:
            worker.cancel()
        
        # 等待所有工作者完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        # 保存最终指标
        await self._save_metrics()
        
        self.logger.info("Enhanced queue manager stopped")
    
    async def submit_request(self, request: ChatRequest, priority: int = 0) -> ChatResponse:
        """提交请求到队列"""
        if not self.running:
            raise RuntimeError("Queue manager is not running")
        
        if not self.request_processor:
            raise RuntimeError("Request processor not set")
        
        # 估算token使用
        estimated_tokens = self._estimate_tokens(request)
        
        # 检查速率限制
        allowed, reason = await self._check_rate_limit(request.model, estimated_tokens)
        if not allowed:
            self.logger.warning(f"Request {request.request_id} rate limited: {reason}")
            with self._stats_lock:
                self.stats['rate_limited_requests'] += 1
            
            # 等待速率限制
            wait_time = await self._wait_for_rate_limit(request.model)
            if wait_time > 0:
                self.logger.info(f"Waiting {wait_time:.1f}s for rate limit")
                await asyncio.sleep(wait_time)
        
        # 创建任务
        future = asyncio.Future()
        task = QueueTask(
            request=request,
            future=future,
            timestamp=time.time(),
            priority=priority,
            estimated_tokens=estimated_tokens
        )
        
        try:
            # 将任务添加到队列
            await asyncio.wait_for(
                self.task_queue.put((priority, task.timestamp, task)),
                timeout=5.0
            )
            
            with self._stats_lock:
                self.stats['total_requests'] += 1
                self.stats['queue_size'] = self.task_queue.qsize()
            
            # 等待结果
            result = await asyncio.wait_for(future, timeout=self.task_timeout)
            return result
            
        except asyncio.QueueFull:
            raise asyncio.QueueFull("Request queue is full")
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Request timeout")
    
    async def _wait_for_rate_limit(self, model: str) -> float:
        """等待速率限制，返回等待时间"""
        is_free = self._is_free_model(model)
        model_type = 'free' if is_free else 'paid'
        
        with self._rate_limit_lock:
            request_times = self.request_times[model_type]
            if not request_times:
                return 0.0
            
            # 计算需要等待的时间
            rpm_limit = self.rate_config.free_model_rpm if is_free else self.rate_config.paid_model_rpm
            effective_limit = int(rpm_limit * self.rate_config.rate_limit_buffer)
            
            if len(request_times) < effective_limit:
                return 0.0
            
            # 等待最早的请求过期
            oldest_request = request_times[0]
            wait_time = 60 - (time.time() - oldest_request)
            return max(0.0, wait_time)
    
    async def _worker(self, worker_name: str):
        """工作者协程"""
        self.logger.info(f"Worker {worker_name} started")
        
        with self._stats_lock:
            self.stats['active_workers'] += 1
        
        try:
            while self.running:
                try:
                    # 从队列获取任务
                    priority, timestamp, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                    
                    with self._stats_lock:
                        self.stats['queue_size'] = self.task_queue.qsize()
                    
                    # 检查任务是否超时
                    if time.time() - task.timestamp > self.task_timeout:
                        error_response = ErrorResponse(
                            request_id=task.request.request_id,
                            conversation_id=task.request.conversation_id,
                            error="Task timeout",
                            error_type="TimeoutError"
                        )
                        task.future.set_result(error_response)
                        
                        with self._stats_lock:
                            self.stats['failed_requests'] += 1
                        continue
                    
                    # 记录请求时间（用于速率限制）
                    with self._rate_limit_lock:
                        model_type = 'free' if self._is_free_model(task.request.model) else 'paid'
                        self.request_times[model_type].append(time.time())
                        
                        if self._is_free_model(task.request.model):
                            self.daily_free_requests += 1
                    
                    # 处理请求
                    start_time = time.time()
                    try:
                        result = await self.request_processor(task.request)
                        processing_time = time.time() - start_time
                        
                        # 提取token使用信息
                        token_usage = self._extract_token_usage(result)
                        
                        # 记录指标
                        metrics = RequestMetrics(
                            request_id=task.request.request_id,
                            model=task.request.model,
                            is_free_model=self._is_free_model(task.request.model),
                            tokens_used=token_usage,
                            response_time=processing_time,
                            timestamp=start_time,
                            success=True
                        )
                        self.request_metrics.append(metrics)
                        
                        # 更新token使用历史
                        self.token_usage_history.append(token_usage)
                        
                        task.future.set_result(result)
                        
                        with self._stats_lock:
                            self.stats['completed_requests'] += 1
                            total_completed = self.stats['completed_requests']
                            current_avg = self.stats['average_processing_time']
                            self.stats['average_processing_time'] = (
                                (current_avg * (total_completed - 1) + processing_time) / total_completed
                            )
                        
                        self.logger.debug(f"Worker {worker_name} completed request {task.request.request_id} in {processing_time:.2f}s, tokens: {token_usage.total_tokens}")
                        
                    except Exception as e:
                        processing_time = time.time() - start_time
                        self.logger.error(f"Worker {worker_name} error processing request {task.request.request_id}: {e}")
                        
                        # 记录错误指标
                        metrics = RequestMetrics(
                            request_id=task.request.request_id,
                            model=task.request.model,
                            is_free_model=self._is_free_model(task.request.model),
                            tokens_used=TokenUsage(),
                            response_time=processing_time,
                            timestamp=start_time,
                            success=False,
                            error_type=type(e).__name__
                        )
                        self.request_metrics.append(metrics)
                        
                        error_response = ErrorResponse(
                            request_id=task.request.request_id,
                            conversation_id=task.request.conversation_id,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        task.future.set_result(error_response)
                        
                        with self._stats_lock:
                            self.stats['failed_requests'] += 1
                    
                    # 标记任务完成
                    self.task_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Worker {worker_name} unexpected error: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            self.logger.info(f"Worker {worker_name} cancelled")
        finally:
            with self._stats_lock:
                self.stats['active_workers'] -= 1
            self.logger.info(f"Worker {worker_name} stopped")
    
    def _extract_token_usage(self, response) -> TokenUsage:
        """从响应中提取token使用信息"""
        if isinstance(response, ChatResponse) and hasattr(response, 'usage'):
            usage = response.usage or {}
            return TokenUsage(
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0),
                total_tokens=usage.get('total_tokens', 0)
            )
        return TokenUsage()
    
    async def _check_api_limits(self):
        """检查API限制"""
        if not self.openrouter_api_key:
            self.logger.warning("No OpenRouter API key provided, cannot check limits")
            return
        
        try:
            async with self.session.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.openrouter_api_key}"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.current_api_limits = data.get('data', {})
                    self.logger.info(f"API Limits - Usage: {self.current_api_limits.get('usage', 0)}, Limit: {self.current_api_limits.get('limit', 'unlimited')}")
                else:
                    self.logger.error(f"Failed to check API limits: {response.status}")
        except Exception as e:
            self.logger.error(f"Error checking API limits: {e}")
    
    async def _rate_limit_cleaner(self):
        """清理过期的速率限制记录"""
        while self.running:
            try:
                await asyncio.sleep(30)  # 每30秒清理一次
                
                current_time = time.time()
                minute_ago = current_time - 60
                
                with self._rate_limit_lock:
                    # 清理请求时间记录
                    for model_type in self.request_times:
                        request_times = self.request_times[model_type]
                        while request_times and request_times[0] < minute_ago:
                            request_times.popleft()
                    
                    # 清理token使用历史
                    while self.token_usage_history and current_time - self.token_usage_history[0].timestamp > 60:
                        self.token_usage_history.popleft()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Rate limit cleaner error: {e}")
    
    async def _stats_updater(self):
        """统计信息更新器"""
        while self.running:
            try:
                await asyncio.sleep(10)  # 每10秒更新一次
                
                current_time = time.time()
                
                with self._stats_lock:
                    stats = self.stats.copy()
                
                # 计算当前RPM
                with self._rate_limit_lock:
                    minute_ago = current_time - 60
                    total_recent_requests = sum(
                        len([t for t in times if t > minute_ago])
                        for times in self.request_times.values()
                    )
                    stats['current_rpm'] = total_recent_requests
                    
                    # 计算每分钟token数
                    minute_tokens = sum(
                        usage.total_tokens for usage in self.token_usage_history
                        if current_time - usage.timestamp < 60
                    )
                    stats['tokens_per_minute'] = minute_tokens
                    stats['free_model_daily_usage'] = self.daily_free_requests
                
                self.logger.info(
                    f"📊 Queue Stats - Total: {stats['total_requests']}, "
                    f"✅ Completed: {stats['completed_requests']}, "
                    f"❌ Failed: {stats['failed_requests']}, "
                    f"🚫 Rate Limited: {stats['rate_limited_requests']}, "
                    f"📋 Queue: {stats['queue_size']}, "
                    f"👷 Workers: {stats['active_workers']}, "
                    f"⚡ RPM: {stats['current_rpm']}, "
                    f"🎯 Tokens/min: {stats['tokens_per_minute']}, "
                    f"🆓 Daily Free: {stats['free_model_daily_usage']}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats updater error: {e}")
    
    async def _api_limit_checker(self):
        """定期检查API限制"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 每5分钟检查一次
                await self._check_api_limits()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"API limit checker error: {e}")
    
    async def _log_maintainer(self):
        """日志维护器"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # 每小时维护一次
                
                # 保存指标
                await self._save_metrics()
                
                # 清理旧指标（保留24小时）
                cutoff_time = time.time() - 24 * 3600
                self.request_metrics = [
                    m for m in self.request_metrics 
                    if m.timestamp > cutoff_time
                ]
                
                self.logger.info(f"🧹 Log maintenance completed, {len(self.request_metrics)} metrics retained")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Log maintainer error: {e}")
    
    async def _save_metrics(self):
        """保存指标到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H")
            metrics_file = self.log_dir / f"metrics_{timestamp}.json"
            
            # 准备指标数据
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.get_stats(),
                'api_limits': self.current_api_limits,
                'recent_metrics': [
                    {
                        'request_id': m.request_id,
                        'model': m.model,
                        'is_free_model': m.is_free_model,
                        'tokens_used': {
                            'prompt_tokens': m.tokens_used.prompt_tokens,
                            'completion_tokens': m.tokens_used.completion_tokens,
                            'total_tokens': m.tokens_used.total_tokens
                        },
                        'response_time': m.response_time,
                        'timestamp': m.timestamp,
                        'success': m.success,
                        'error_type': m.error_type
                    }
                    for m in self.request_metrics[-100:]  # 保存最近100个指标
                ]
            }
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"💾 Metrics saved to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self.stats.copy()
        
        stats['queue_size'] = self.task_queue.qsize()
        
        # 添加速率限制信息
        current_time = time.time()
        minute_ago = current_time - 60
        
        with self._rate_limit_lock:
            stats['rate_limits'] = {
                'free_rpm_current': len([t for t in self.request_times['free'] if t > minute_ago]),
                'free_rpm_limit': int(self.rate_config.free_model_rpm * self.rate_config.rate_limit_buffer),
                'paid_rpm_current': len([t for t in self.request_times['paid'] if t > minute_ago]),
                'paid_rpm_limit': int(self.rate_config.paid_model_rpm * self.rate_config.rate_limit_buffer),
                'daily_free_usage': self.daily_free_requests,
                'daily_free_limit': self.rate_config.free_model_daily
            }
        
        return stats
    
    async def wait_for_completion(self):
        """等待所有任务完成"""
        await self.task_queue.join() 