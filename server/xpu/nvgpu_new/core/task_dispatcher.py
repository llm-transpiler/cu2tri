"""
Task Dispatcher Module

Manages multiple GPUs and dispatches tasks across them with load balancing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable

from .gpu_manager import GPUManager
from ..base.gpu_info import get_available_gpus
from .task import TaskType


class TaskDispatcher:
    """
    Task dispatcher that manages multiple GPUs and distributes tasks.
    
    Handles:
    - GPU discovery and management
    - Load balancing across GPUs
    - Task routing to optimal GPUs
    - System-wide status monitoring
    """
    
    def __init__(self,
                 available_gpu_ids: Optional[List[int]] = None,
                 max_parallel_task_num: int = 4,
                 gpu_manager_refresh_interval: int = 30,
                 error_cooldown_seconds: int = 30,
                 logger: Optional[logging.Logger] = None):
        
        self.available_gpu_ids = available_gpu_ids or []
        self.max_parallel_task_num = max_parallel_task_num
        self.gpu_manager_refresh_interval = gpu_manager_refresh_interval
        self.error_cooldown_seconds = error_cooldown_seconds
        self.logger = logger or logging.getLogger(__name__)
        
        # GPU managers
        self.gpu_managers: Dict[int, GPUManager] = {}
        
        # Task registry for tracking tasks across GPUs
        self.global_task_registry: Dict[str, int] = {}  # task_id -> gpu_id
        
        # Control flags
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        
        if self.available_gpu_ids:
            self.logger.info(f"[TaskDispatcher] Initialized with available GPUs: {self.available_gpu_ids}")
        else:
            self.logger.info(f"[TaskDispatcher] Initialized with all available GPUs")
        self.logger.info(f"[TaskDispatcher] CUDA error cooldown: {self.error_cooldown_seconds} seconds")
    
    async def start(self) -> None:
        """Start the task dispatcher"""
        if self._running:
            return
        
        self._running = True
        
        # Initialize GPU managers
        await self._initialize_gpu_managers()
        
        # Start all GPU managers
        for gpu_manager in self.gpu_managers.values():
            await gpu_manager.start()
        
        # Start GPU discovery refresh task
        self._refresh_task = asyncio.create_task(self._gpu_refresh_loop())
        
        self.logger.info(f"[TaskDispatcher] Started with {len(self.gpu_managers)} GPUs")
    
    async def stop(self) -> None:
        """Stop the task dispatcher"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop GPU refresh task
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        
        # Stop all GPU managers
        for gpu_manager in self.gpu_managers.values():
            await gpu_manager.stop()
        
        self.logger.info("[TaskDispatcher] Stopped")
    
    async def submit_task(self,
                         task_type: TaskType,
                         name: str,
                         description: str,
                         execute_func: Callable[..., Awaitable[Any]],
                         args: tuple = (),
                         kwargs: dict = {},
                         preferred_gpu_id: Optional[int] = None,
                         max_wait_time_minutes: int = 15) -> str:
        """
        Submit a task to the best available GPU.
        
        Args:
            task_type: Type of task (exclusive or shared)
            name: Task name
            description: Task description
            execute_func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            preferred_gpu_id: Preferred GPU ID (optional)
            max_wait_time_minutes: Maximum wait time in queue
            
        Returns:
            str: Task ID
            
        Raises:
            RuntimeError: If no GPUs are available
        """
        # Select GPU
        gpu_id = await self._select_gpu(preferred_gpu_id)

        if gpu_id is None:
            available_gpus = [gpu_id for gpu_id, manager in self.gpu_managers.items() if manager.is_available]
            cooldown_gpus = [gpu_id for gpu_id, manager in self.gpu_managers.items() if manager.is_in_cooldown]
            unhealthy_gpus = [gpu_id for gpu_id, manager in self.gpu_managers.items() if not manager.last_health_check_passed]
            
            raise RuntimeError(f"No available GPUs for task '{name}'. "
                             f"Available: {available_gpus}, "
                             f"In cooldown: {cooldown_gpus}, "
                             f"Unhealthy: {unhealthy_gpus}")
        
        # Submit task to selected GPU
        gpu_manager = self.gpu_managers[gpu_id]
        task_id = await gpu_manager.submit_task(
            task_type=task_type,
            name=name,
            description=description,
            execute_func=execute_func,
            args=args,
            kwargs=kwargs,
            max_wait_time_minutes=max_wait_time_minutes
        )
        
        # Register task globally
        self.global_task_registry[task_id] = gpu_id
        
        self.logger.info(f"[TaskDispatcher] Submitted {task_type.value} task '{name}' ({task_id}) to GPU-{gpu_id}")
        return task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task by task ID"""
        if task_id not in self.global_task_registry:
            self.logger.warning(f"[TaskDispatcher] Task {task_id} not found in global registry")
            return False
        
        gpu_id = self.global_task_registry[task_id]
        
        if gpu_id not in self.gpu_managers:
            self.logger.error(f"[TaskDispatcher] GPU {gpu_id} not found for task {task_id}")
            return False
        
        # Delegate to the appropriate GPU manager
        success = await self.gpu_managers[gpu_id].cancel_task(task_id)
        
        if success:
            # Remove from global registry
            del self.global_task_registry[task_id]
            self.logger.info(f"[TaskDispatcher] Task {task_id} cancelled on GPU-{gpu_id}")
        
        return success
    
    async def kill_task(self, task_id: str) -> bool:
        """Kill a running task by task ID"""
        if task_id not in self.global_task_registry:
            self.logger.warning(f"[TaskDispatcher] Task {task_id} not found in global registry")
            return False
        
        gpu_id = self.global_task_registry[task_id]
        
        if gpu_id not in self.gpu_managers:
            self.logger.error(f"[TaskDispatcher] GPU {gpu_id} not found for task {task_id}")
            return False
        
        # Delegate to the appropriate GPU manager
        success = await self.gpu_managers[gpu_id].kill_task(task_id)
        
        if success:
            # Remove from global registry
            del self.global_task_registry[task_id]
            self.logger.info(f"[TaskDispatcher] Task {task_id} killed on GPU-{gpu_id}")
        
        return success
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by task ID"""
        if task_id not in self.global_task_registry:
            return None
        
        gpu_id = self.global_task_registry[task_id]
        
        if gpu_id not in self.gpu_managers:
            return None
        
        return await self.gpu_managers[gpu_id].get_task_status(task_id)
    
    def get_available_gpu_ids(self) -> List[int]:
        """Get list of available GPU IDs"""
        return [gpu_id for gpu_id, manager in self.gpu_managers.items() if manager.is_available]
    
    def get_gpu_count(self) -> int:
        """Get total number of managed GPUs"""
        return len(self.gpu_managers)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get status from all GPU managers
        gpu_statuses = {}
        for gpu_id, manager in self.gpu_managers.items():
            try:
                gpu_statuses[str(gpu_id)] = await manager.get_status()
            except Exception as e:
                self.logger.error(f"[TaskDispatcher] Failed to get status for GPU {gpu_id}: {e}")
                gpu_statuses[str(gpu_id)] = {"error": str(e)}
        
        # Calculate system statistics
        available_gpus = sum(
            1 if status.get('is_available', False) else 0
            for status in gpu_statuses.values()
            if 'error' not in status
        )
        
        cooldown_gpus = sum(
            1 if status.get('is_in_cooldown', False) else 0
            for status in gpu_statuses.values()
            if 'error' not in status
        )
        
        unhealthy_gpus = sum(
            1 if not status.get('last_health_check_passed', True) else 0
            for status in gpu_statuses.values()
            if 'error' not in status
        )
        
        # Queue statistics
        queue_stats = {}
        for gpu_id, manager in self.gpu_managers.items():
            try:
                queue_status = await manager.task_queue.get_queue_status()
                queue_stats[str(gpu_id)] = queue_status
            except Exception as e:
                self.logger.error(f"[TaskDispatcher] Failed to get queue status for GPU {gpu_id}: {e}")
        
        return {
            'dispatcher_running': self._running,
            'total_gpus': len(self.gpu_managers),
            'available_gpus': available_gpus,
            'cooldown_gpus': cooldown_gpus,
            'unhealthy_gpus': unhealthy_gpus,
            'available_gpu_ids': self.available_gpu_ids,
            'global_task_count': len(self.global_task_registry),
            'queue_statistics': queue_stats,
            'configuration': {
                'max_parallel_task_num': self.max_parallel_task_num,
                'error_cooldown_seconds': self.error_cooldown_seconds,
                'refresh_interval': self.gpu_manager_refresh_interval,
            },
            'gpus': gpu_statuses
        }
    
    async def update_available_gpus(self, new_available_gpu_ids: List[int]) -> Dict[str, Any]:
        """Update the list of available GPUs dynamically"""
        self.logger.info(f"[TaskDispatcher] Updating available GPUs to: {new_available_gpu_ids}")
        
        # Update the available GPU list
        old_available_gpu_ids = self.available_gpu_ids.copy()
        self.available_gpu_ids = new_available_gpu_ids
        
        # Get current and new GPU info
        current_gpu_infos = get_available_gpus(available_gpu_ids=self.available_gpu_ids)
        current_gpu_ids = set(current_gpu_infos.keys())
        managed_gpu_ids = set(self.gpu_managers.keys())
        
        # Track changes
        added_gpus = []
        removed_gpus = []
        
        # Add new GPUs
        new_gpu_ids = current_gpu_ids - managed_gpu_ids
        for gpu_id in new_gpu_ids:
            gpu_info = current_gpu_infos[gpu_id]
            
            gpu_manager = GPUManager(
                gpu_info=gpu_info,
                max_parallel_task_num=self.max_parallel_task_num,
                status_refresh_interval=self.gpu_manager_refresh_interval,
                error_cooldown_seconds=self.error_cooldown_seconds,
                logger=self.logger
            )
            
            await gpu_manager.start()
            self.gpu_managers[gpu_id] = gpu_manager
            added_gpus.append(gpu_id)
            self.logger.info(f"[TaskDispatcher] Added GPU-{gpu_id}: {gpu_info.name}")
        
        # Remove old GPUs
        removed_gpu_ids = managed_gpu_ids - current_gpu_ids
        for gpu_id in removed_gpu_ids:
            gpu_manager = self.gpu_managers.pop(gpu_id)
            await gpu_manager.stop()
            removed_gpus.append(gpu_id)
            self.logger.info(f"[TaskDispatcher] Removed GPU-{gpu_id}")
        
        result = {
            "old_available_gpus": old_available_gpu_ids,
            "new_available_gpus": self.available_gpu_ids,
            "added_gpus": added_gpus,
            "removed_gpus": removed_gpus,
            "current_gpu_managers": list(self.gpu_managers.keys())
        }
        
        self.logger.info(f"[TaskDispatcher] GPU update completed: {result}")
        return result
    
    async def _select_gpu(self, preferred_gpu_id: Optional[int]) -> Optional[int]:
        """Select GPU for a task"""
        
        # If specific GPU requested, check if available
        if preferred_gpu_id is not None:
            if preferred_gpu_id in self.gpu_managers:
                gpu_manager = self.gpu_managers[preferred_gpu_id]
                if gpu_manager.is_available:
                    return preferred_gpu_id
                else:
                    self.logger.warning(f"[TaskDispatcher] Requested GPU-{preferred_gpu_id} is not available "
                                      f"(cooldown: {gpu_manager.is_in_cooldown}, healthy: {gpu_manager.last_health_check_passed})")
            else:
                self.logger.warning(f"[TaskDispatcher] Requested GPU-{preferred_gpu_id} does not exist")
        
        # Otherwise, select GPU with lowest load
        available_gpus = []
        
        for gpu_id, gpu_manager in self.gpu_managers.items():
            if gpu_manager.is_available:
                available_gpus.append(gpu_id)
        
        if not available_gpus:
            return None
        
        # Select GPU with lowest load score
        best_gpu = available_gpus[0]
        best_score = float('inf')
        
        for gpu_id in available_gpus:
            gpu_manager = self.gpu_managers[gpu_id]
            score = gpu_manager.load_score
            
            if score < best_score:
                best_score = score
                best_gpu = gpu_id
        
        return best_gpu
    
    async def _initialize_gpu_managers(self) -> None:
        """Initialize GPU managers for all available GPUs"""
        try:
            gpu_infos = get_available_gpus(available_gpu_ids=self.available_gpu_ids)
            
            for cuda_device_id, gpu_info in gpu_infos.items():
                gpu_manager = GPUManager(
                    gpu_info=gpu_info,
                    max_parallel_task_num=self.max_parallel_task_num,
                    status_refresh_interval=self.gpu_manager_refresh_interval,
                    error_cooldown_seconds=self.error_cooldown_seconds,
                    logger=self.logger
                )
                
                self.gpu_managers[cuda_device_id] = gpu_manager
                self.logger.info(f"[TaskDispatcher] Initialized GPU-{cuda_device_id}: {gpu_info.name}")
            
            if not self.gpu_managers:
                self.logger.warning("[TaskDispatcher] No GPUs available after filtering")
            
        except Exception as e:
            self.logger.error(f"[TaskDispatcher] Failed to initialize GPU managers: {e}")
    
    async def _gpu_refresh_loop(self) -> None:
        """Periodically refresh GPU availability"""
        while self._running:
            try:
                await asyncio.sleep(self.gpu_manager_refresh_interval * 2)  # Less frequent than individual GPU refresh
                
                # Check for new GPUs or removed GPUs
                current_gpu_infos = get_available_gpus(available_gpu_ids=self.available_gpu_ids)
                current_gpu_ids = set(current_gpu_infos.keys())
                managed_gpu_ids = set(self.gpu_managers.keys())
                
                # Handle new GPUs
                new_gpu_ids = current_gpu_ids - managed_gpu_ids
                for gpu_id in new_gpu_ids:
                    gpu_info = current_gpu_infos[gpu_id]
                    
                    gpu_manager = GPUManager(
                        gpu_info=gpu_info,
                        max_parallel_task_num=self.max_parallel_task_num,
                        status_refresh_interval=self.gpu_manager_refresh_interval,
                        error_cooldown_seconds=self.error_cooldown_seconds,
                        logger=self.logger
                    )
                    await gpu_manager.start()
                    self.gpu_managers[gpu_id] = gpu_manager
                    self.logger.info(f"[TaskDispatcher] Added new GPU-{gpu_id}: {gpu_info.name}")
                
                # Handle removed GPUs
                removed_gpu_ids = managed_gpu_ids - current_gpu_ids
                for gpu_id in removed_gpu_ids:
                    gpu_manager = self.gpu_managers.pop(gpu_id)
                    await gpu_manager.stop()
                    self.logger.warning(f"[TaskDispatcher] Removed GPU-{gpu_id}")
                
            except Exception as e:
                self.logger.error(f"[TaskDispatcher] GPU refresh error: {e}") 