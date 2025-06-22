# -*- coding: utf-8 -*-
"""
队列管理器
实现生产者消费者模式的任务队列
"""
import asyncio
import logging
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from .deprecated.models import ChatRequest, ChatResponse, ErrorResponse

@dataclass
class QueueTask:
    """队列任务"""
    request: ChatRequest
    future: asyncio.Future
    timestamp: float
    priority: int = 0  # 优先级，数字越小优先级越高

class ChatQueueManager:
    """聊天队列管理器"""
    
    def __init__(self, 
                 max_workers: int = 10,
                 max_queue_size: int = 1000,
                 task_timeout: int = 300,
                 logger: Optional[logging.Logger] = None):
        """初始化队列管理器
        
        Args:
            max_workers: 最大工作线程数
            max_queue_size: 最大队列大小
            task_timeout: 任务超时时间（秒）
            logger: 日志记录器
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_timeout = task_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # 队列和工作者
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.workers = []
        self.running = False
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'queue_size': 0,
            'active_workers': 0,
            'average_processing_time': 0.0
        }
        
        # 处理器函数
        self.request_processor: Optional[Callable] = None
        
        # 线程安全锁
        self._stats_lock = threading.Lock()
    
    def set_request_processor(self, processor: Callable):
        """设置请求处理器"""
        self.request_processor = processor
    
    async def start(self):
        """启动队列管理器"""
        if self.running:
            return
        
        self.running = True
        self.logger.info(f"Starting queue manager with {self.max_workers} workers")
        
        # 启动工作者协程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # 启动统计协程
        asyncio.create_task(self._stats_updater())
    
    async def stop(self):
        """停止队列管理器"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping queue manager...")
        
        # 取消所有工作者
        for worker in self.workers:
            worker.cancel()
        
        # 等待所有工作者完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        self.logger.info("Queue manager stopped")
    
    async def submit_request(self, request: ChatRequest, priority: int = 0) -> ChatResponse:
        """提交请求到队列
        
        Args:
            request: 聊天请求
            priority: 优先级（数字越小优先级越高）
            
        Returns:
            ChatResponse: 聊天响应
            
        Raises:
            asyncio.QueueFull: 队列已满
            asyncio.TimeoutError: 请求超时
        """
        if not self.running:
            raise RuntimeError("Queue manager is not running")
        
        if not self.request_processor:
            raise RuntimeError("Request processor not set")
        
        # 创建Future用于接收结果
        future = asyncio.Future()
        task = QueueTask(
            request=request,
            future=future,
            timestamp=time.time(),
            priority=priority
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
                    
                    # 处理请求
                    start_time = time.time()
                    try:
                        result = await self.request_processor(task.request)
                        task.future.set_result(result)
                        
                        processing_time = time.time() - start_time
                        
                        with self._stats_lock:
                            self.stats['completed_requests'] += 1
                            # 更新平均处理时间
                            total_completed = self.stats['completed_requests']
                            current_avg = self.stats['average_processing_time']
                            self.stats['average_processing_time'] = (
                                (current_avg * (total_completed - 1) + processing_time) / total_completed
                            )
                        
                        self.logger.debug(f"Worker {worker_name} completed request {task.request.request_id} in {processing_time:.2f}s")
                        
                    except Exception as e:
                        self.logger.error(f"Worker {worker_name} error processing request {task.request.request_id}: {e}")
                        
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
                    # 等待任务超时，继续循环
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
    
    async def _stats_updater(self):
        """统计信息更新器"""
        while self.running:
            try:
                await asyncio.sleep(10)  # 每10秒更新一次统计
                
                with self._stats_lock:
                    stats = self.stats.copy()
                
                self.logger.info(
                    f"Queue Stats - Total: {stats['total_requests']}, "
                    f"Completed: {stats['completed_requests']}, "
                    f"Failed: {stats['failed_requests']}, "
                    f"Queue Size: {stats['queue_size']}, "
                    f"Active Workers: {stats['active_workers']}, "
                    f"Avg Processing Time: {stats['average_processing_time']:.2f}s"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats updater error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self.stats.copy()
        
        stats['queue_size'] = self.task_queue.qsize()
        return stats
    
    async def wait_for_completion(self):
        """等待所有任务完成"""
        await self.task_queue.join() 