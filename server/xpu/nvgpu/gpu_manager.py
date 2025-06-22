"""
GPU Manager Module

Coordinates GPU resource allocation and task scheduling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
import os

from .gpu_info import GPUInfo, GPUType, query_gpu_info, set_visible_gpus
from .task_queue import TaskQueue, TaskType, Task

logger = logging.getLogger(__name__)


class GPUManager:
    """GPU resource manager and task scheduler"""
    
    def __init__(
        self,
        refresh_interval_seconds: int = 10,
        max_task_wait_minutes: int = 30,
        log_dir: Optional[str] = None
    ):
        self.refresh_interval = refresh_interval_seconds
        self.max_task_wait_minutes = max_task_wait_minutes
        self.log_dir = log_dir or "server/logs/xpu/nvgpu"
        
        # Task queue
        self.task_queue = TaskQueue(max_wait_time_minutes=max_task_wait_minutes)
        
        # GPU state tracking
        self.gpu_infos: Dict[int, GPUInfo] = {}
        self.last_gpu_refresh: Optional[datetime] = None
        
        # Scheduler state
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"GPUManager initialized with refresh interval {refresh_interval_seconds}s")
    
    async def start(self) -> None:
        """Start the GPU manager and scheduler"""
        if self._running:
            logger.warning("GPUManager is already running")
            return
        
        self._running = True
        
        # Initial GPU discovery
        await self.refresh_gpu_info()
        
        # Start scheduler task
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("GPUManager started")
    
    async def stop(self) -> None:
        """Stop the GPU manager and scheduler"""
        if not self._running:
            return
        
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("GPUManager stopped")
    
    async def refresh_gpu_info(self) -> None:
        """Refresh GPU information from system"""
        try:
            gpu_infos = query_gpu_info()
            self.gpu_infos = {gpu.device_id: gpu for gpu in gpu_infos}
            self.last_gpu_refresh = datetime.now()
            
            logger.debug(f"Refreshed info for {len(self.gpu_infos)} GPUs")
            
        except Exception as e:
            logger.error(f"Failed to refresh GPU info: {e}")
    
    async def submit_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        execute_func: Callable[..., Awaitable[Any]],
        args: tuple = (),
        kwargs: dict = None,
        estimated_memory_mb: int = 0,
        max_wait_time_minutes: Optional[int] = None,
        preferred_gpu_types: Optional[List[GPUType]] = None
    ) -> str:
        """Submit a task for execution"""
        task_id = await self.task_queue.submit_task(
            task_type=task_type,
            name=name,
            description=description,
            execute_func=execute_func,
            args=args,
            kwargs=kwargs,
            estimated_memory_mb=estimated_memory_mb,
            max_wait_time_minutes=max_wait_time_minutes
        )
        
        # Store preferred GPU types in task metadata (if needed)
        # This could be extended to support GPU type preferences
        
        logger.info(f"Submitted task {task_id}: {name} ({task_type.value})")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return await self.task_queue.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        return await self.task_queue.cancel_task(task_id)
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status"""
        await self.refresh_gpu_info()
        
        gpu_status = {}
        for gpu_id, gpu_info in self.gpu_infos.items():
            gpu_status[str(gpu_id)] = {
                'device_id': gpu_info.device_id,
                'name': gpu_info.name,
                'gpu_type': gpu_info.gpu_type.value,
                'memory_total_mb': gpu_info.memory_total_mb,
                'memory_used_mb': gpu_info.memory_used_mb,
                'memory_usage_ratio': gpu_info.memory_usage_ratio,
                'utilization_percent': gpu_info.utilization_percent,
                'temperature_c': gpu_info.temperature_c,
                'power_draw_w': gpu_info.power_draw_w,
                'available_for_functional': gpu_info.is_available_for_functional_task,
                'available_for_performance': gpu_info.is_available_for_performance_task,
                'spec': {
                    'memory_gb': gpu_info.spec.memory_gb,
                    'compute_capability': gpu_info.spec.compute_capability,
                    'tensor_cores': gpu_info.spec.tensor_cores,
                    'max_power_w': gpu_info.spec.max_power_w,
                    'fp32_tflops': gpu_info.spec.fp32_tflops
                }
            }
        
        return gpu_status
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get task queue status"""
        return await self.task_queue.get_queue_status()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        gpu_status = await self.get_gpu_status()
        queue_status = await self.get_queue_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'gpu_manager_running': self._running,
            'last_gpu_refresh': self.last_gpu_refresh.isoformat() if self.last_gpu_refresh else None,
            'gpus': gpu_status,
            'queue': queue_status
        }
    
    def set_visible_gpus(self, gpu_ids: List[int]) -> None:
        """Set visible GPUs for the current process"""
        set_visible_gpus(gpu_ids)
        logger.info(f"Set visible GPUs to: {gpu_ids}")
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self._running:
            try:
                # Refresh GPU information
                await self.refresh_gpu_info()
                
                # Clean up expired tasks
                expired_count = await self.task_queue.cleanup_expired_tasks()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired tasks")
                
                # Schedule tasks on available GPUs
                await self._schedule_tasks()
                
                # Wait before next iteration
                await asyncio.sleep(self.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(self.refresh_interval)
        
        logger.info("Scheduler loop stopped")
    
    async def _schedule_tasks(self) -> None:
        """Schedule tasks on available GPUs"""
        for gpu_id, gpu_info in self.gpu_infos.items():
            try:
                # 检查GPU是否可用于任务调度
                if not gpu_info.is_available_for_functional_task and not gpu_info.is_available_for_performance_task:
                    logger.debug(f"GPU {gpu_id} not available for any tasks (memory usage: {gpu_info.memory_usage_ratio:.2%})")
                    continue
                
                # Calculate available memory (in MB, accounting for current usage)
                available_memory_mb = gpu_info.memory_total_mb - gpu_info.memory_used_mb
                
                # 记录GPU状态
                logger.debug(f"GPU {gpu_id} status: memory {gpu_info.memory_used_mb}/{gpu_info.memory_total_mb}MB ({gpu_info.memory_usage_ratio:.2%}), available_memory: {available_memory_mb}MB")
                
                # 持续为这个GPU调度任务，直到没有合适的任务为止
                tasks_scheduled = 0
                while tasks_scheduled < 10:  # 防止无限循环，每次最多调度10个任务
                    # Get next suitable task for this GPU
                    task = await self.task_queue.get_next_task_for_gpu(gpu_id, available_memory_mb)
                    
                    if not task:
                        logger.debug(f"No suitable tasks for GPU {gpu_id}")
                        break  # 没有合适的任务了
                    
                    # 再次检查GPU是否可用（状态可能已变化）
                    if task.task_type == TaskType.FUNCTIONAL and not gpu_info.is_available_for_functional_task:
                        logger.warning(f"GPU {gpu_id} no longer available for functional tasks, putting task back to queue")
                        # 将任务放回队列
                        await self.task_queue._put_task_back(task)
                        break
                    elif task.task_type == TaskType.PERFORMANCE and not gpu_info.is_available_for_performance_task:
                        logger.warning(f"GPU {gpu_id} no longer available for performance tasks, putting task back to queue")
                        # 将任务放回队列
                        await self.task_queue._put_task_back(task)
                        break
                    
                    # Start the task
                    await self.task_queue.start_task(task, gpu_id)
                    
                    # Execute the task asynchronously
                    asyncio.create_task(self._execute_task(task))
                    
                    # 更新可用内存（如果是FUNCTIONAL任务的话）
                    if task.task_type == TaskType.FUNCTIONAL:
                        available_memory_mb -= task.estimated_memory_mb
                        tasks_scheduled += 1
                        logger.info(f"Scheduled functional task {task.task_id} on GPU {gpu_id}, estimated remaining memory: {available_memory_mb} MB")
                    else:
                        # PERFORMANCE任务独占GPU，停止为这个GPU调度更多任务
                        logger.info(f"Scheduled performance task {task.task_id} on GPU {gpu_id} (exclusive)")
                        break
                    
            except Exception as e:
                logger.error(f"Error scheduling tasks for GPU {gpu_id}: {e}")
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a task"""
        try:
            # Set CUDA_VISIBLE_DEVICES for this task
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(task.gpu_id)
            
            logger.info(f"Executing task {task.task_id} on GPU {task.gpu_id}")
            
            # Execute the task function
            result = await task.execute_func(*task.args, **task.kwargs)
            
            # Mark task as completed
            await self.task_queue.complete_task(task.task_id, result=result)
            
        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(f"Task {task.task_id} failed: {error_msg}")
            
            # Mark task as failed
            await self.task_queue.complete_task(task.task_id, error=error_msg)
            
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if 'original_cuda_visible' in locals():
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# # Convenience function for creating and managing GPU manager
# async def create_gpu_manager(**kwargs) -> GPUManager:
#     """Create and start a GPU manager"""
#     manager = GPUManager(**kwargs)
#     await manager.start()
#     return manager 