# -*- coding: utf-8 -*-
"""
内核开发任务队列管理器
负责管理编译、测试、性能测试、LLM交互等任务的队列
"""
import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import heapq
import uuid

from .models import (
    KernelRequest, KernelResponse, ErrorResponse,
    TaskType, TaskStatus, TestStage,
    CompileRequest, TestRequest, PerfRequest, LLMRequest,
    RAGRequest, WebSearchRequest, SMTRequest
)
from .gpu_manager import GPUResourceManager

@dataclass
class QueueTask:
    """队列任务"""
    priority: int
    timestamp: float
    request: KernelRequest
    future: asyncio.Future
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    estimated_duration: float = 300.0  # 预计执行时间（秒）
    gpu_requirements: List[int] = field(default_factory=list)
    requires_gpu: bool = False
    stage: TestStage = TestStage.DEVELOPMENT
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """优先级队列排序"""
        if self.priority != other.priority:
            return self.priority < other.priority  # 数字越小优先级越高
        return self.timestamp < other.timestamp  # 相同优先级按时间排序

@dataclass
class TaskMetrics:
    """任务指标"""
    task_id: str
    request_id: str
    conversation_id: str
    task_type: TaskType
    status: TaskStatus
    start_time: float
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    gpu_ids: List[int] = field(default_factory=list)
    error_info: Optional[str] = None
    retry_count: int = 0

class KernelQueueManager:
    """内核开发任务队列管理器"""
    
    def __init__(self,
                 gpu_manager: GPUResourceManager,
                 max_workers: int = 8,
                 max_queue_size: int = 500,
                 task_timeout: int = 600,
                 logger: Optional[logging.Logger] = None):
        """初始化队列管理器"""
        self.gpu_manager = gpu_manager
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.task_timeout = task_timeout
        self.logger = logger or logging.getLogger(__name__)
        
        # 队列系统
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.workers = []
        self.running = False
        
        # 任务处理器
        self.processors: Dict[TaskType, Callable] = {}
        
        # 任务追踪
        self.active_tasks: Dict[str, QueueTask] = {}
        self.task_metrics: List[TaskMetrics] = []
        self.conversation_contexts: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'retried_tasks': 0,
            'queue_size': 0,
            'active_workers': 0,
            'average_processing_time': 0.0,
            'task_type_stats': defaultdict(lambda: {
                'total': 0, 'completed': 0, 'failed': 0, 'avg_time': 0.0
            })
        }
        
        # 线程安全锁
        self._stats_lock = threading.Lock()
        self._context_lock = threading.Lock()
        
        # 任务历史（用于清理）
        self.max_history_size = 10000
        
    def register_processor(self, task_type: TaskType, processor: Callable):
        """注册任务处理器"""
        self.processors[task_type] = processor
        self.logger.info(f"Registered processor for task type: {task_type.value}")
    
    async def start(self):
        """启动队列管理器"""
        if self.running:
            return
        
        self.logger.info("Starting kernel queue manager...")
        
        # 启动工作线程
        self.running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_task())
        
        self.logger.info(f"Kernel queue manager started with {self.max_workers} workers")
    
    async def stop(self):
        """停止队列管理器"""
        if not self.running:
            return
        
        self.logger.info("Stopping kernel queue manager...")
        self.running = False
        
        # 停止工作线程
        for worker in self.workers:
            worker.cancel()
        
        # 停止清理任务
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
        
        # 等待所有任务完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        if hasattr(self, 'cleanup_task'):
            await asyncio.gather(self.cleanup_task, return_exceptions=True)
        
        # 清理队列
        while not self.task_queue.empty():
            try:
                _, task = self.task_queue.get_nowait()
                if not task.future.done():
                    task.future.set_exception(asyncio.CancelledError("Queue manager stopped"))
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("Kernel queue manager stopped")
    
    async def submit_request(self, 
                           request: KernelRequest, 
                           priority: int = 5,
                           estimated_duration: float = 300.0) -> KernelResponse:
        """提交任务请求"""
        if not self.running:
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=self._get_task_type(request),
                error="Queue manager is not running",
                error_type="ServiceError"
            )
        
        # 检查队列容量
        if self.task_queue.qsize() >= self.max_queue_size:
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=self._get_task_type(request),
                error="Queue is full",
                error_type="QueueFullError"
            )
        
        # 创建任务
        task = QueueTask(
            priority=priority,
            timestamp=time.time(),
            request=request,
            future=asyncio.Future(),
            estimated_duration=estimated_duration,
            requires_gpu=self._requires_gpu(request),
            stage=getattr(request, 'stage', TestStage.DEVELOPMENT),
            gpu_requirements=getattr(request, 'gpu_requirements', [])
        )
        
        # 添加到队列
        try:
            await self.task_queue.put((priority, task))
            
            with self._stats_lock:
                self.stats['total_tasks'] += 1
                self.stats['queue_size'] = self.task_queue.qsize()
            
            self.logger.debug(f"Task {task.task_id} queued for request {request.request_id}")
            
            # 等待任务完成
            return await task.future
            
        except asyncio.QueueFull:
            return ErrorResponse(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                task_type=self._get_task_type(request),
                error="Queue is full",
                error_type="QueueFullError"
            )
    
    def _get_task_type(self, request: KernelRequest) -> TaskType:
        """获取任务类型"""
        if isinstance(request, CompileRequest):
            return TaskType.COMPILE
        elif isinstance(request, TestRequest):
            return TaskType.FUNCTIONAL_TEST
        elif isinstance(request, PerfRequest):
            return TaskType.PERFORMANCE_TEST
        elif isinstance(request, LLMRequest):
            return TaskType.LLM_GENERATION
        elif isinstance(request, RAGRequest):
            return TaskType.RAG_QUERY
        elif isinstance(request, WebSearchRequest):
            return TaskType.WEB_SEARCH
        elif isinstance(request, SMTRequest):
            return TaskType.SMT_VERIFICATION
        else:
            return TaskType.COMPILE  # 默认
    
    def _requires_gpu(self, request: KernelRequest) -> bool:
        """判断任务是否需要GPU"""
        return isinstance(request, (TestRequest, PerfRequest))
    
    async def _worker(self, worker_name: str):
        """工作线程"""
        self.logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # 获取任务
                try:
                    priority, task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 更新统计
                with self._stats_lock:
                    self.stats['queue_size'] = self.task_queue.qsize()
                    self.stats['active_workers'] += 1
                
                # 处理任务
                await self._process_task(worker_name, task)
                
                # 更新统计
                with self._stats_lock:
                    self.stats['active_workers'] = max(0, self.stats['active_workers'] - 1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                with self._stats_lock:
                    self.stats['active_workers'] = max(0, self.stats['active_workers'] - 1)
        
        self.logger.debug(f"Worker {worker_name} stopped")
    
    async def _process_task(self, worker_name: str, task: QueueTask):
        """处理单个任务"""
        start_time = time.time()
        task_type = self._get_task_type(task.request)
        allocated_gpus = []
        
        # 创建任务指标
        metrics = TaskMetrics(
            task_id=task.task_id,
            request_id=task.request.request_id,
            conversation_id=task.request.conversation_id,
            task_type=task_type,
            status=TaskStatus.RUNNING,
            start_time=start_time
        )
        
        self.active_tasks[task.task_id] = task
        self.task_metrics.append(metrics)
        
        try:
            self.logger.info(f"Worker {worker_name} processing task {task.task_id} ({task_type.value})")
            
            # GPU资源分配
            if task.requires_gpu:
                allocated_gpus = await self._allocate_gpu_resources(task)
                if not allocated_gpus:
                    raise Exception("Failed to allocate GPU resources")
                metrics.gpu_ids = allocated_gpus
            
            # 获取处理器
            if task_type not in self.processors:
                raise Exception(f"No processor registered for task type: {task_type.value}")
            
            processor = self.processors[task_type]
            
            # 设置超时
            try:
                response = await asyncio.wait_for(
                    processor(task.request, allocated_gpus, self._get_conversation_context(task.request.conversation_id)),
                    timeout=self.task_timeout
                )
            except asyncio.TimeoutError:
                raise Exception(f"Task timed out after {self.task_timeout} seconds")
            
            # 更新对话上下文
            self._update_conversation_context(task.request.conversation_id, task_type, response)
            
            # 完成任务
            metrics.status = TaskStatus.COMPLETED
            metrics.end_time = time.time()
            metrics.processing_time = metrics.end_time - start_time
            
            if not task.future.done():
                task.future.set_result(response)
            
            # 更新统计
            with self._stats_lock:
                self.stats['completed_tasks'] += 1
                self._update_task_type_stats(task_type, metrics.processing_time, True)
            
            self.logger.info(f"Task {task.task_id} completed in {metrics.processing_time:.2f}s")
            
        except Exception as e:
            # 任务失败
            metrics.status = TaskStatus.FAILED
            metrics.end_time = time.time()
            metrics.processing_time = metrics.end_time - start_time
            metrics.error_info = str(e)
            
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            # 重试逻辑
            if task.retry_count < task.max_retries and self._should_retry(e):
                task.retry_count += 1
                metrics.retry_count = task.retry_count
                
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                
                # 重新加入队列
                await asyncio.sleep(2 ** task.retry_count)  # 指数退避
                await self.task_queue.put((task.priority + 1, task))  # 降低优先级
                
                with self._stats_lock:
                    self.stats['retried_tasks'] += 1
                
                return
            
            # 创建错误响应
            error_response = ErrorResponse(
                request_id=task.request.request_id,
                conversation_id=task.request.conversation_id,
                task_type=task_type,
                error=str(e),
                error_type=type(e).__name__,
                gpu_info=self.gpu_manager.get_gpu_info(allocated_gpus) if allocated_gpus else None
            )
            
            if not task.future.done():
                task.future.set_result(error_response)
            
            # 更新统计
            with self._stats_lock:
                self.stats['failed_tasks'] += 1
                self._update_task_type_stats(task_type, metrics.processing_time, False)
        
        finally:
            # 清理资源
            if allocated_gpus:
                await self.gpu_manager.release_gpu(task.task_id)
            
            # 从活跃任务中移除
            self.active_tasks.pop(task.task_id, None)
    
    async def _allocate_gpu_resources(self, task: QueueTask) -> List[int]:
        """分配GPU资源"""
        gpu_count = 1
        wait_for_idle = False
        
        if isinstance(task.request, PerfRequest):
            gpu_count = len(task.request.reference_codes) + 1  # 主要内核 + 参考内核数量
            wait_for_idle = task.request.wait_for_idle
        
        return await self.gpu_manager.allocate_gpu(
            task_id=task.task_id,
            conversation_id=task.request.conversation_id,
            stage=task.stage,
            task_type=self._get_task_type(task.request).value,
            gpu_count=gpu_count,
            specific_gpus=task.gpu_requirements if task.gpu_requirements else None,
            expected_duration=task.estimated_duration,
            wait_for_idle=wait_for_idle
        )
    
    def _should_retry(self, error: Exception) -> bool:
        """判断是否应该重试"""
        # GPU资源相关错误可以重试
        if "gpu" in str(error).lower() or "cuda" in str(error).lower():
            return True
        
        # 编译错误通常不需要重试
        if "compile" in str(error).lower() or "build" in str(error).lower():
            return False
        
        # 网络相关错误可以重试
        if "timeout" in str(error).lower() or "connection" in str(error).lower():
            return True
        
        return False
    
    def _get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        with self._context_lock:
            return self.conversation_contexts[conversation_id].copy()
    
    def _update_conversation_context(self, conversation_id: str, task_type: TaskType, response):
        """更新对话上下文"""
        with self._context_lock:
            context = self.conversation_contexts[conversation_id]
            
            # 记录任务历史
            if 'task_history' not in context:
                context['task_history'] = []
            
            context['task_history'].append({
                'task_type': task_type.value,
                'timestamp': time.time(),
                'success': not isinstance(response, ErrorResponse)
            })
            
            # 保持历史记录大小
            if len(context['task_history']) > 100:
                context['task_history'] = context['task_history'][-50:]
            
            # 根据任务类型更新特定上下文
            if task_type == TaskType.COMPILE and not isinstance(response, ErrorResponse):
                context['last_compile_success'] = True
                context['last_compile_time'] = time.time()
            elif task_type == TaskType.LLM_GENERATION and not isinstance(response, ErrorResponse):
                if hasattr(response.result, 'generated_code'):
                    context['last_generated_code'] = response.result.generated_code
                    context['generation_count'] = context.get('generation_count', 0) + 1
    
    def _update_task_type_stats(self, task_type: TaskType, processing_time: float, success: bool):
        """更新任务类型统计"""
        stats = self.stats['task_type_stats'][task_type.value]
        stats['total'] += 1
        
        if success:
            stats['completed'] += 1
        else:
            stats['failed'] += 1
        
        # 更新平均处理时间
        if stats['completed'] > 0:
            current_avg = stats['avg_time']
            stats['avg_time'] = (current_avg * (stats['completed'] - 1) + processing_time) / stats['completed']
    
    async def _cleanup_task(self):
        """清理任务"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                # 清理过期的任务指标
                if len(self.task_metrics) > self.max_history_size:
                    self.task_metrics = self.task_metrics[-self.max_history_size//2:]
                
                # 清理空的对话上下文
                with self._context_lock:
                    empty_contexts = [
                        conv_id for conv_id, ctx in self.conversation_contexts.items()
                        if not ctx or (time.time() - ctx.get('last_activity', 0) > 3600)  # 1小时无活动
                    ]
                    for conv_id in empty_contexts:
                        del self.conversation_contexts[conv_id]
                
                self.logger.debug(f"Cleanup completed. Metrics: {len(self.task_metrics)}, Contexts: {len(self.conversation_contexts)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self.stats.copy()
        
        # 添加实时信息
        stats['queue_size'] = self.task_queue.qsize()
        stats['active_tasks'] = len(self.active_tasks)
        stats['conversation_contexts'] = len(self.conversation_contexts)
        
        # 计算成功率
        if stats['total_tasks'] > 0:
            stats['success_rate'] = stats['completed_tasks'] / stats['total_tasks']
        else:
            stats['success_rate'] = 0.0
        
        # 活跃任务详情
        stats['active_task_details'] = {
            task_id: {
                'request_id': task.request.request_id,
                'task_type': self._get_task_type(task.request).value,
                'start_time': next((m.start_time for m in self.task_metrics if m.task_id == task_id), None),
                'gpu_ids': getattr(task, 'allocated_gpus', [])
            } for task_id, task in self.active_tasks.items()
        }
        
        return stats
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话统计信息"""
        context = self._get_conversation_context(conversation_id)
        
        return {
            'conversation_id': conversation_id,
            'task_count': len(context.get('task_history', [])),
            'generation_count': context.get('generation_count', 0),
            'last_compile_success': context.get('last_compile_success', False),
            'last_activity': context.get('last_activity', 0),
            'task_history': context.get('task_history', [])[-10:]  # 最近10个任务
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.future.done():
                task.future.cancel()
            
            # 释放GPU资源
            await self.gpu_manager.release_gpu(task_id)
            
            # 更新任务状态
            for metrics in self.task_metrics:
                if metrics.task_id == task_id:
                    metrics.status = TaskStatus.CANCELLED
                    metrics.end_time = time.time()
                    break
            
            self.active_tasks.pop(task_id, None)
            self.logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'queue_manager_running': self.running,
            'active_workers': len([w for w in self.workers if not w.done()]),
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'total_processors': len(self.processors),
            'gpu_manager_status': await self.gpu_manager.health_check(),
            'stats': self.get_stats()
        } 