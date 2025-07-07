"""
GPU Manager Module

Manages a single GPU with its task queue and status monitoring.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, Callable, Awaitable, List
from datetime import datetime, timedelta

from ..base.gpu_info import GPUInfo, query_gpu_info
from .task_queue import GPUTaskQueue
from .task import Task, TaskType
from ..base.health_check import GPUHealthChecker


class GPUManager:
    """
    Manager for a single GPU
    
    Each GPUManager instance is responsible for:
    1. One specific GPU (identified by CUDA device ID and nvidia-smi ID)
    2. GPU status monitoring and information updates
    3. Task queue management for that GPU
    4. CUDA_VISIBLE_DEVICES setting for tasks
    5. CUDA error cooldown protection
    6. Error-driven GPU health checking
    """
    
    def __init__(self,
                 gpu_info: GPUInfo,
                 max_parallel_task_num: int = 4,
                 status_refresh_interval: int = 30,
                 error_cooldown_seconds: int = 30,
                 logger: Optional[logging.Logger] = None):
        
        self.gpu_info = gpu_info
        self.cuda_device_id = gpu_info.device_id
        self.nvidia_smi_id = gpu_info.nvidia_smi_id
        self.status_refresh_interval = status_refresh_interval
        self.error_cooldown_seconds = error_cooldown_seconds
        self.logger = logger or logging.getLogger(__name__)
        
        # Task queue for this GPU
        self.task_queue = GPUTaskQueue(
            gpu_id=self.cuda_device_id,
            max_parallel_task_num=max_parallel_task_num,
            logger=self.logger
        )
        
        # Status monitoring
        self.last_status_update: Optional[datetime] = None
        self._status_task: Optional[asyncio.Task] = None
        self._running = False
        
        # CUDA error cooldown mechanism
        self.last_cuda_error: Optional[datetime] = None
        self.cuda_error_count: int = 0
        self.is_in_cooldown: bool = False
        self.cooldown_reason: str = ""
        
        # GPU health checking (error-driven, not periodic)
        self.health_checker = GPUHealthChecker(
            gpu_id=self.cuda_device_id,
            logger=self.logger
        )
        self.last_health_check: Optional[datetime] = None
        self.last_health_check_passed: bool = True
        self.last_health_error: Optional[str] = None
        
        # Task requeue management
        self.killed_tasks_for_requeue: List[Task] = []
        
        self.logger.info(f"[GPU-{self.cuda_device_id}] Initialized manager for "
                        f"{gpu_info.name} (nvidia-smi ID: {self.nvidia_smi_id})")
    
    async def start(self) -> None:
        """Start the GPU manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start task queue
        await self.task_queue.start()
        
        # Start status monitoring
        self._status_task = asyncio.create_task(self._status_monitor_loop())
        
        self.logger.info(f"[GPU-{self.cuda_device_id}] Manager started")
    
    async def stop(self) -> None:
        """Stop the GPU manager"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop status monitoring
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
        
        # Stop task queue
        await self.task_queue.stop()
        
        self.logger.info(f"[GPU-{self.cuda_device_id}] Manager stopped")
    
    async def submit_task(self,
                         task_type: TaskType,
                         name: str,
                         description: str,
                         execute_func: Callable[..., Awaitable[Any]],
                         args: tuple = (),
                         kwargs: dict = {},
                         max_wait_time_minutes: int = 15) -> str:
        """Submit a task to this GPU"""
        
        # Check if GPU is available (not in cooldown and healthy)
        if not self.is_available:
            reason = []
            if self.is_in_cooldown:
                reason.append(f"in cooldown until {self._get_cooldown_expiry()}")
            if not self.last_health_check_passed:
                reason.append("health check failed")
            
            raise RuntimeError(f"GPU-{self.cuda_device_id} is not available: {', '.join(reason)}")
        
        # Create task with GPU-specific execution wrapper
        task = Task(
            name=name,
            description=description,
            task_type=task_type,
            execute_func=self._wrap_execution_func(execute_func),
            args=args,
            kwargs=kwargs,
            max_wait_time_minutes=max_wait_time_minutes,
            gpu_id=self.cuda_device_id
        )
        
        return await self.task_queue.submit_task(task)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        return await self.task_queue.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        return await self.task_queue.cancel_task(task_id)
    
    async def kill_task(self, task_id: str) -> bool:
        """Kill a running task"""
        return await self.task_queue.kill_task(task_id)
    
    async def perform_health_check(self) -> bool:
        """
        Perform health check on demand (e.g., via API)
        Only allowed when GPU has no running tasks
        """
        # Check if GPU has running tasks
        if self.task_queue.has_running_tasks():
            raise RuntimeError(f"GPU-{self.cuda_device_id} has running tasks, cannot perform health check")
        
        return await self._perform_health_check()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU status"""
        # Refresh GPU info if needed
        await self._refresh_gpu_info()
        
        # Get queue status
        queue_status = await self.task_queue.get_queue_status()
        
        return {
            # Hardware info
            'cuda_device_id': self.cuda_device_id,
            'nvidia_smi_id': self.nvidia_smi_id,
            'name': self.gpu_info.name,
            'gpu_type': self.gpu_info.gpu_type.value,
            
            # Current status
            'memory_total_mb': self.gpu_info.memory_total_mb,
            'memory_used_mb': self.gpu_info.memory_used_mb,
            'memory_free_mb': self.gpu_info.memory_free_mb,
            'memory_usage_ratio': self.gpu_info.memory_usage_ratio,
            'utilization_percent': self.gpu_info.utilization_percent,
            'temperature_c': self.gpu_info.temperature_c,
            'power_draw_w': self.gpu_info.power_draw_w,
            
            # Availability
            'is_available': self.is_available,
            'is_in_cooldown': self.is_in_cooldown,
            'cooldown_reason': self.cooldown_reason,
            
            # CUDA error tracking
            'cuda_error_count': self.cuda_error_count,
            'last_cuda_error': self.last_cuda_error.isoformat() if self.last_cuda_error else None,
            'cooldown_expires_at': self._get_cooldown_expiry().isoformat() if self.is_in_cooldown else None,
            
            # Health check status
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'last_health_check_passed': self.last_health_check_passed,
            'last_health_error': self.last_health_error,
            
            # Specifications
            'spec': {
                'memory_gb': self.gpu_info.spec.memory_gb,
                'compute_capability': self.gpu_info.spec.compute_capability,
                'max_power_w': self.gpu_info.spec.max_power_w,
            },
            
            # Task queue status
            'queue': queue_status,
            
            # Status metadata
            'last_status_update': self.last_status_update.isoformat() if self.last_status_update else None,
            'manager_running': self._running
        }
    
    def _wrap_execution_func(self, original_func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Wrap execution function to set CUDA_VISIBLE_DEVICES and handle CUDA errors"""
        async def wrapped_func(*args, **kwargs):
            # Set CUDA_VISIBLE_DEVICES to only this GPU
            original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device_id)
            
            try:
                self.logger.debug(f"[GPU-{self.cuda_device_id}] Set CUDA_VISIBLE_DEVICES={self.cuda_device_id}")
                result = await original_func(*args, **kwargs)
                return result
            except Exception as e:
                # Check if this is a CUDA error
                if self._is_cuda_error(e):
                    await self._handle_cuda_error(e)
                raise
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_devices is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
                else:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                
                self.logger.debug(f"[GPU-{self.cuda_device_id}] Restored CUDA_VISIBLE_DEVICES")
        
        return wrapped_func
    
    def _is_cuda_error(self, error: Exception) -> bool:
        """Check if an error is a CUDA-related error"""
        error_str = str(error).lower()
        cuda_keywords = [
            'cuda', 'gpu', 'device', 'out of memory', 'nvml',
            'runtime error', 'driver error', 'kernel launch',
            'device assert', 'illegal memory access'
        ]
        return any(keyword in error_str for keyword in cuda_keywords)
    
    async def _handle_cuda_error(self, error: Exception) -> None:
        """Handle CUDA error by killing all running tasks and entering cooldown mode"""
        self.last_cuda_error = datetime.now()
        self.cuda_error_count += 1
        self.is_in_cooldown = True
        self.cooldown_reason = f"CUDA error: {str(error)[:100]}..."
        
        self.logger.error(f"[GPU-{self.cuda_device_id}] CUDA error occurred (count: {self.cuda_error_count}): {error}")
        
        # Kill all running tasks immediately
        killed_tasks = await self.task_queue.kill_all_running_tasks()
        
        if killed_tasks:
            self.logger.warning(f"[GPU-{self.cuda_device_id}] Killed {len(killed_tasks)} running tasks due to CUDA error")
            
            # Store killed tasks for re-queuing after recovery
            self.killed_tasks_for_requeue = killed_tasks
        
        self.logger.warning(f"[GPU-{self.cuda_device_id}] Entering cooldown mode for {self.error_cooldown_seconds} seconds")
    
    def _get_cooldown_expiry(self) -> Optional[datetime]:
        """Get when the cooldown period expires"""
        if not self.last_cuda_error:
            return None
        return self.last_cuda_error + timedelta(seconds=self.error_cooldown_seconds)
    
    def _is_cooldown_expired(self) -> bool:
        """Check if cooldown period has expired"""
        if not self.last_cuda_error:
            return True
        return datetime.now() >= self._get_cooldown_expiry()
    
    async def _perform_health_check(self) -> bool:
        """Perform GPU health check"""
        try:
            self.logger.debug(f"[GPU-{self.cuda_device_id}] Starting health check")
            
            # Perform isolated health check
            is_healthy = await self.health_checker.check_gpu_health()
            
            self.last_health_check = datetime.now()
            self.last_health_check_passed = is_healthy
            self.last_health_error = None if is_healthy else "Health check failed"
            
            if is_healthy:
                self.logger.debug(f"[GPU-{self.cuda_device_id}] Health check passed")
            else:
                self.logger.warning(f"[GPU-{self.cuda_device_id}] Health check failed")
                
            return is_healthy
                
        except Exception as e:
            self.last_health_check = datetime.now()
            self.last_health_check_passed = False
            self.last_health_error = str(e)
            self.logger.error(f"[GPU-{self.cuda_device_id}] Health check error: {e}")
            return False
    
    async def _try_recovery_from_cooldown(self) -> None:
        """Try to recover from cooldown by performing health check"""
        if not self.is_in_cooldown:
            return
        
        # Only try recovery if cooldown has expired
        if not self._is_cooldown_expired():
            return
        
        # Only perform health check if GPU has no running tasks
        if self.task_queue.has_running_tasks():
            self.logger.debug(f"[GPU-{self.cuda_device_id}] Cannot perform health check during cooldown recovery: GPU has running tasks")
            return
        
        self.logger.info(f"[GPU-{self.cuda_device_id}] Cooldown expired, performing health check for recovery")
        
        # Perform health check
        is_healthy = await self._perform_health_check()
        
        if is_healthy:
            # Recovery successful
            self.is_in_cooldown = False
            self.cooldown_reason = ""
            
            # Re-queue killed tasks if any
            if hasattr(self, 'killed_tasks_for_requeue') and self.killed_tasks_for_requeue:
                await self.task_queue.requeue_tasks(self.killed_tasks_for_requeue)
                self.killed_tasks_for_requeue = []
            
            self.logger.info(f"[GPU-{self.cuda_device_id}] Recovery successful, GPU available again")
        else:
            # Health check failed, extend cooldown
            self.last_cuda_error = datetime.now()  # Reset cooldown timer
            self.cooldown_reason = f"Health check failed: {self.last_health_error}"
            self.logger.warning(f"[GPU-{self.cuda_device_id}] Health check failed, extending cooldown for another {self.error_cooldown_seconds} seconds")
    
    async def _status_monitor_loop(self) -> None:
        """Monitor GPU status periodically"""
        while self._running:
            try:
                await self._refresh_gpu_info()
                
                # Try recovery from cooldown if needed
                await self._try_recovery_from_cooldown()
                
                await asyncio.sleep(self.status_refresh_interval)
                
            except Exception as e:
                self.logger.error(f"[GPU-{self.cuda_device_id}] Status monitor error: {e}")
                await asyncio.sleep(self.status_refresh_interval)
    
    async def _refresh_gpu_info(self) -> None:
        """Refresh GPU information from system"""
        try:
            # Query all GPUs and find ours
            all_gpu_infos = query_gpu_info(available_gpu_ids=[])  # Get all for monitoring
            
            # Find our GPU by nvidia-smi ID
            for gpu_info in all_gpu_infos:
                if gpu_info.nvidia_smi_id == self.nvidia_smi_id:
                    # Update our GPU info while preserving the original CUDA device ID
                    original_cuda_id = self.gpu_info.device_id
                    self.gpu_info = gpu_info
                    self.gpu_info.device_id = original_cuda_id  # Keep original CUDA device ID mapping
                    self.last_status_update = datetime.now()
                    break
            else:
                self.logger.warning(f"[GPU-{self.cuda_device_id}] Could not find nvidia-smi ID {self.nvidia_smi_id}")
                
        except Exception as e:
            self.logger.error(f"[GPU-{self.cuda_device_id}] Failed to refresh GPU info: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if GPU is available for tasks"""
        # Check basic availability
        basic_available = (self.gpu_info.memory_usage_ratio < 0.8 and 
                          self.gpu_info.temperature_c < 85)
        
        # Check cooldown and health status
        return (basic_available and 
                not self.is_in_cooldown and 
                self.last_health_check_passed)
    
    @property
    def load_score(self) -> float:
        """
        Calculate a load score for load balancing
        Lower score = less loaded
        """
        # If not available, return very high score
        if not self.is_available:
            return float('inf')
        
        # Base score from memory usage and utilization
        memory_score = self.gpu_info.memory_usage_ratio * 50
        utilization_score = self.gpu_info.utilization_percent * 0.3
        temperature_score = max(0, (self.gpu_info.temperature_c - 60) * 0.2)
        
        # Add task queue load
        exclusive_tasks = len(self.task_queue.exclusive_queue)
        shared_tasks = len(self.task_queue.shared_queue)
        running_exclusive = 1 if self.task_queue.running_exclusive_task else 0
        running_shared = len(self.task_queue.running_shared_tasks)
        
        # Exclusive tasks have higher weight
        queue_score = (exclusive_tasks * 20 + shared_tasks * 5 + 
                      running_exclusive * 50 + running_shared * 10)
        
        # Add penalty for recent errors
        error_penalty = self.cuda_error_count * 10
        
        return memory_score + utilization_score + temperature_score + queue_score + error_penalty 