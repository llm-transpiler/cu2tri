"""
GPU Task Queue Module

Manages task queues for a single GPU with exclusive and shared task scheduling.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional

from .task import Task, TaskType, TaskStatus


class GPUTaskQueue:
    """
    Per-GPU task queue with dynamic scheduling based on task types
    
    Scheduling Rules:
    1. EXCLUSIVE tasks require exclusive GPU access
    2. SHARED tasks can run in parallel with other SHARED tasks
    3. Maximum parallel SHARED tasks is configurable
    """
    
    def __init__(self, 
                 gpu_id: int,
                 max_parallel_task_num: int = 4,
                 logger: Optional[logging.Logger] = None):
        
        self.gpu_id = gpu_id
        self.max_parallel_task_num = max_parallel_task_num
        self.logger = logger or logging.getLogger(__name__)
        
        # Task queues by type
        self.exclusive_queue: deque[Task] = deque()
        self.shared_queue: deque[Task] = deque()
        
        # Currently running tasks
        self.running_exclusive_task: Optional[Task] = None
        self.running_shared_tasks: Dict[str, Task] = {}  # task_id -> Task
        
        # Completed task history (limited size)
        self.completed_tasks: Dict[str, Task] = {}
        self.max_completed_history = 3000
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger.info(f"[GPU-{self.gpu_id}] Initialized task queue with "
                        f"max_parallel_task_num={self.max_parallel_task_num}")
    
    async def start(self) -> None:
        """Start the task queue scheduler"""
        if self._running:
            return
            
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info(f"[GPU-{self.gpu_id}] Task queue scheduler started")
    
    async def stop(self) -> None:
        """Stop the task queue scheduler"""
        if not self._running:
            return
            
        self._running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all pending tasks
        async with self._lock:
            for task in list(self.exclusive_queue) + list(self.shared_queue):
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                task.error = "System shutdown"
        
        self.logger.info(f"[GPU-{self.gpu_id}] Task queue scheduler stopped")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the queue"""
        async with self._lock:
            task.gpu_id = self.gpu_id
            
            if task.task_type == TaskType.EXCLUSIVE:
                self.exclusive_queue.append(task)
                self.logger.info(f"[GPU-{self.gpu_id}] Queued EXCLUSIVE task: {task.task_id} ({task.name})")
            elif task.task_type == TaskType.SHARED:  # SHARED
                self.shared_queue.append(task)
                self.logger.info(f"[GPU-{self.gpu_id}] Queued SHARED task: {task.task_id} ({task.name})")
            else:
                self.logger.error(f"[GPU-{self.gpu_id}] Unknown task type for task {task.task_id}")
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        async with self._lock:
            # Check if task is running
            if self.running_exclusive_task and self.running_exclusive_task.task_id == task_id:
                self.logger.warning(f"[GPU-{self.gpu_id}] Cannot cancel running exclusive task {task_id}")
                return False
            
            for task in self.running_shared_tasks:
                if task.task_id == task_id:
                    self.logger.warning(f"[GPU-{self.gpu_id}] Cannot cancel running shared task {task_id}")
                    return False
            
            # Check exclusive queue
            for task in self.exclusive_queue:
                if task.task_id == task_id:
                    self.exclusive_queue.remove(task)
                    await task.cancel()
                    self.logger.info(f"[GPU-{self.gpu_id}] Cancelled exclusive task {task_id}")
                    return True
            
            # Check shared queue
            for task in self.shared_queue:
                if task.task_id == task_id:
                    self.shared_queue.remove(task)
                    await task.cancel()
                    self.logger.info(f"[GPU-{self.gpu_id}] Cancelled shared task {task_id}")
                    return True
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    self.logger.info(f"[GPU-{self.gpu_id}] Task {task_id} is already completed")
                    return False
            
            self.logger.warning(f"[GPU-{self.gpu_id}] Task {task_id} not found")
            return False
    
    async def kill_task(self, task_id: str) -> bool:
        """Kill a running task"""
        async with self._lock:
            # Check if task is running exclusive
            if self.running_exclusive_task and self.running_exclusive_task.task_id == task_id:
                task = self.running_exclusive_task
                success = await task.kill()
                if success:
                    self.running_exclusive_task = None
                    self.logger.info(f"[GPU-{self.gpu_id}] Killed exclusive task {task_id}")
                return success
            
            # Check if task is running shared
            for task in self.running_shared_tasks[:]:  # Use slice to avoid modification during iteration
                if task.task_id == task_id:
                    success = await task.kill()
                    if success:
                        self.running_shared_tasks.remove(task)
                        self.logger.info(f"[GPU-{self.gpu_id}] Killed shared task {task_id}")
                    return success
            
            # Try to cancel if not running
            return await self.cancel_task(task_id)
    
    async def kill_all_running_tasks(self) -> List[Task]:
        """
        Kill all running tasks and return them for potential re-queuing.
        Used when CUDA error occurs.
        
        Returns:
            List of killed tasks that can be re-queued
        """
        killed_tasks = []
        
        async with self._lock:
            # Kill exclusive task
            if self.running_exclusive_task:
                task = self.running_exclusive_task
                success = await task.kill()
                if success:
                    killed_tasks.append(task)
                    self.logger.info(f"[GPU-{self.gpu_id}] Killed exclusive task {task.task_id} due to CUDA error")
                self.running_exclusive_task = None
            
            # Kill all shared tasks
            for task in self.running_shared_tasks[:]:
                success = await task.kill()
                if success:
                    killed_tasks.append(task)
                    self.logger.info(f"[GPU-{self.gpu_id}] Killed shared task {task.task_id} due to CUDA error")
            
            self.running_shared_tasks.clear()
            
            self.logger.warning(f"[GPU-{self.gpu_id}] Killed {len(killed_tasks)} running tasks due to CUDA error")
        
        return killed_tasks
    
    async def requeue_tasks(self, tasks: List[Task]) -> None:
        """
        Re-queue tasks that were killed due to CUDA error.
        Reset their status and add them back to appropriate queues.
        """
        async with self._lock:
            for task in tasks:
                # Reset task status
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.error = None
                task.result = None
                task.task_coroutine = None
                
                # Add back to appropriate queue
                if task.task_type == TaskType.EXCLUSIVE:
                    self.exclusive_queue.append(task)
                    self.logger.info(f"[GPU-{self.gpu_id}] Re-queued exclusive task {task.task_id}")
                else:
                    self.shared_queue.append(task)
                    self.logger.info(f"[GPU-{self.gpu_id}] Re-queued shared task {task.task_id}")
            
            self.logger.info(f"[GPU-{self.gpu_id}] Re-queued {len(tasks)} tasks after CUDA error recovery")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        async with self._lock:
            # Check running tasks
            if self.running_exclusive_task and self.running_exclusive_task.task_id == task_id:
                return self.running_exclusive_task.to_dict()
            
            for task in self.running_shared_tasks.values():
                if task.task_id == task_id:
                    return task.to_dict()
            
            # Check queued tasks
            for task in self.exclusive_queue + self.shared_queue:
                if task.task_id == task_id:
                    return task.to_dict()
            
            # Check completed tasks
            for task in self.completed_tasks.values():
                if task.task_id == task_id:
                    return task.to_dict()
            
            return None
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information"""
        async with self._lock:
            return {
                'gpu_id': self.gpu_id,
                'exclusive_queue_size': len(self.exclusive_queue),
                'shared_queue_size': len(self.shared_queue),
                'running_exclusive': self.running_exclusive_task is not None,
                'running_shared_count': len(self.running_shared_tasks),
                'max_parallel_task_num': self.max_parallel_task_num,
                'completed_tasks_count': len(self.completed_tasks),
                'total_tasks_processed': len(self.completed_tasks)
            }
    
    def has_running_tasks(self) -> bool:
        """Check if there are currently running tasks (without async lock for quick check)"""
        return (self.running_exclusive_task is not None or 
                len(self.running_shared_tasks) > 0)
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self._running:
            try:
                await self._schedule_tasks()
                await self._cleanup_expired_tasks()
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"[GPU-{self.gpu_id}] Scheduler error: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _schedule_tasks(self) -> None:
        """Main task scheduling logic"""
        async with self._lock:
            # Priority 1: If an exclusive task is running, wait for it to complete
            if self.running_exclusive_task:
                return
            
            # Priority 2: If there's an exclusive task waiting and no shared tasks running
            if self.exclusive_queue and not self.running_shared_tasks:
                await self._start_exclusive_task()
                return
            
            # Priority 3: Start shared tasks if possible
            if self.shared_queue and len(self.running_shared_tasks) < self.max_parallel_task_num:
                await self._start_shared_tasks()
    
    async def _start_exclusive_task(self) -> None:
        """Start an exclusive task"""
        if not self.exclusive_queue:
            return
        
        task = self.exclusive_queue.popleft()
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_exclusive_task = task
        
        self.logger.info(f"[GPU-{self.gpu_id}] Starting EXCLUSIVE task: {task.task_id} ({task.name})")
        
        # Execute task in background
        asyncio.create_task(self._execute_task(task))
    
    async def _start_shared_tasks(self) -> None:
        """Start shared tasks up to the parallel limit"""
        available_slots = self.max_parallel_task_num - len(self.running_shared_tasks)
        
        for _ in range(min(available_slots, len(self.shared_queue))):
            task = self.shared_queue.popleft()
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.running_shared_tasks[task.task_id] = task
            
            self.logger.info(f"[GPU-{self.gpu_id}] Starting SHARED task: {task.task_id} ({task.name})")
            
            # Execute task in background
            asyncio.create_task(self._execute_task(task))
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a single task"""
        try:
            # Execute the task function
            if task.execute_func:
                result = await task.execute_func(*task.args, **task.kwargs)
                await self._complete_task(task, result=result)
            else:
                await self._complete_task(task, error="No execution function provided")
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[GPU-{self.gpu_id}] Task {task.task_id} failed: {error_msg}")
            await self._complete_task(task, error=error_msg)
    
    async def _complete_task(self, task: Task, result: Any = None, error: str = None) -> None:
        """Complete a task and update state"""
        async with self._lock:
            task.completed_at = datetime.now()
            task.result = result
            task.error = error
            task.status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED
            
            # Remove from running tasks
            if task.task_type == TaskType.EXCLUSIVE:
                self.running_exclusive_task = None
            else:
                self.running_shared_tasks.pop(task.task_id, None)
            
            # Add to completed history
            self._add_to_completed(task)
            
            execution_time = task.execution_time_seconds
            self.logger.info(f"[GPU-{self.gpu_id}] Completed {task.task_type.value} task: {task.task_id} "
                           f"({task.name}) in {execution_time:.2f}s")
    
    async def _cleanup_expired_tasks(self) -> None:
        """Clean up expired pending tasks"""
        async with self._lock:
            current_time = datetime.now()
            
            # Clean exclusive queue
            expired_exclusive = []
            remaining_exclusive = deque()
            
            while self.exclusive_queue:
                task = self.exclusive_queue.popleft()
                if task.is_expired:
                    expired_exclusive.append(task)
                else:
                    remaining_exclusive.append(task)
            
            self.exclusive_queue = remaining_exclusive
            
            # Clean shared queue
            expired_shared = []
            remaining_shared = deque()
            
            while self.shared_queue:
                task = self.shared_queue.popleft()
                if task.is_expired:
                    expired_shared.append(task)
                else:
                    remaining_shared.append(task)
            
            self.shared_queue = remaining_shared
            
            # Mark expired tasks as failed
            for task in expired_exclusive + expired_shared:
                task.status = TaskStatus.FAILED
                task.completed_at = current_time
                task.error = f"Task expired after {task.max_wait_time_minutes} minutes"
                self._add_to_completed(task)
                
                self.logger.warning(f"[GPU-{self.gpu_id}] Task expired: {task.task_id} ({task.name})")
    
    def _add_to_completed(self, task: Task) -> None:
        """Add task to completed history with size limit"""
        self.completed_tasks[task.task_id] = task
        
        # Limit completed history size
        if len(self.completed_tasks) > self.max_completed_history:
            # Remove oldest 10% of tasks
            oldest_tasks = sorted(self.completed_tasks.values(), 
                                key=lambda t: t.completed_at or datetime.min)
            tasks_to_remove = oldest_tasks[:self.max_completed_history // 10]
            
            for old_task in tasks_to_remove:
                self.completed_tasks.pop(old_task.task_id, None) 