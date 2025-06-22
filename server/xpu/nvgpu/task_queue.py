"""
Task Queue Management Module

Manages functional and performance tasks with proper scheduling and resource allocation.
"""

import asyncio
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Awaitable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task type enumeration"""
    FUNCTIONAL = "functional"  # Multiple tasks can run on same GPU if memory allows
    PERFORMANCE = "performance"  # Requires exclusive GPU access


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    execute_func: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Task metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    
    # Resource requirements
    estimated_memory_mb: int = 0
    max_wait_time_minutes: int = 30
    
    # Results
    result: Any = None
    error: Optional[str] = None
    gpu_id: Optional[int] = None
    
    @property
    def wait_time_seconds(self) -> float:
        """Calculate current wait time in seconds"""
        if self.started_at:
            return 0.0
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def execution_time_seconds(self) -> float:
        """Calculate execution time in seconds"""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded maximum wait time"""
        return self.wait_time_seconds > (self.max_wait_time_minutes * 60)


class TaskQueue:
    """Task queue manager for GPU resource allocation"""
    
    def __init__(self, max_wait_time_minutes: int = 30):
        self.max_wait_time_minutes = max_wait_time_minutes
        
        # Task queues
        self.functional_queue: List[Task] = []
        self.performance_queue: List[Task] = []
        
        # Running tasks tracking
        self.running_tasks: Dict[str, Task] = {}  # task_id -> Task
        self.gpu_functional_tasks: Dict[int, List[str]] = {}  # gpu_id -> [task_ids]
        self.gpu_performance_task: Dict[int, Optional[str]] = {}  # gpu_id -> task_id
        
        # Task history
        self.completed_tasks: Dict[str, Task] = {}
        
        # Synchronization
        self._lock = asyncio.Lock()
        
        logger.info("TaskQueue initialized")
    
    async def submit_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        execute_func: Callable[..., Awaitable[Any]],
        args: tuple = (),
        kwargs: dict = None,
        estimated_memory_mb: int = 0,
        max_wait_time_minutes: Optional[int] = None
    ) -> str:
        """Submit a new task to the queue"""
        if kwargs is None:
            kwargs = {}
        
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            name=name,
            description=description,
            execute_func=execute_func,
            args=args,
            kwargs=kwargs,
            estimated_memory_mb=estimated_memory_mb,
            max_wait_time_minutes=max_wait_time_minutes or self.max_wait_time_minutes
        )
        
        async with self._lock:
            if task_type == TaskType.FUNCTIONAL:
                self.functional_queue.append(task)
                logger.info(f"Submitted functional task: {task_id} ({name})")
            else:
                self.performance_queue.append(task)
                logger.info(f"Submitted performance task: {task_id} ({name})")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status information"""
        async with self._lock:
            # Check running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                return self._task_to_status_dict(task)
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return self._task_to_status_dict(task)
            
            # Check pending tasks
            for task in self.functional_queue + self.performance_queue:
                if task.task_id == task_id:
                    return self._task_to_status_dict(task)
        
        return None
    
    def _task_to_status_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task to status dictionary"""
        return {
            'task_id': task.task_id,
            'name': task.name,
            'description': task.description,
            'task_type': task.task_type.value,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'wait_time_seconds': task.wait_time_seconds,
            'execution_time_seconds': task.execution_time_seconds,
            'gpu_id': task.gpu_id,
            'error': task.error,
            'estimated_memory_mb': task.estimated_memory_mb
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        async with self._lock:
            # Remove from functional queue
            for i, task in enumerate(self.functional_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.functional_queue.pop(i)
                    self.completed_tasks[task_id] = task
                    logger.info(f"Cancelled functional task: {task_id}")
                    return True
            
            # Remove from performance queue
            for i, task in enumerate(self.performance_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.performance_queue.pop(i)
                    self.completed_tasks[task_id] = task
                    logger.info(f"Cancelled performance task: {task_id}")
                    return True
        
        return False
    
    async def get_next_task_for_gpu(self, gpu_id: int, gpu_available_memory_mb: int) -> Optional[Task]:
        """Get next suitable task for the specified GPU"""
        async with self._lock:
            # Check if GPU is currently running a performance task
            if self.gpu_performance_task.get(gpu_id):
                return None
            
            # Try to get a performance task first (higher priority)
            if not self.gpu_functional_tasks.get(gpu_id):  # No functional tasks running
                for i, task in enumerate(self.performance_queue):
                    if not task.is_expired:
                        self.performance_queue.pop(i)
                        return task
            
            # Try to get a functional task
            current_functional_tasks = self.gpu_functional_tasks.get(gpu_id, [])
            current_memory_usage = sum(
                self.running_tasks[tid].estimated_memory_mb 
                for tid in current_functional_tasks 
                if tid in self.running_tasks
            )
            
            for i, task in enumerate(self.functional_queue):
                if task.is_expired:
                    continue
                
                # Check if there's enough memory
                required_memory = task.estimated_memory_mb
                if current_memory_usage + required_memory <= gpu_available_memory_mb * 0.67:
                    self.functional_queue.pop(i)
                    return task
        
        return None
    
    async def start_task(self, task: Task, gpu_id: int) -> None:
        """Mark task as started on specified GPU"""
        async with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.gpu_id = gpu_id
            
            self.running_tasks[task.task_id] = task
            
            if task.task_type == TaskType.PERFORMANCE:
                self.gpu_performance_task[gpu_id] = task.task_id
                logger.info(f"Started performance task {task.task_id} on GPU {gpu_id}")
            else:
                if gpu_id not in self.gpu_functional_tasks:
                    self.gpu_functional_tasks[gpu_id] = []
                self.gpu_functional_tasks[gpu_id].append(task.task_id)
                logger.info(f"Started functional task {task.task_id} on GPU {gpu_id}")
    
    async def complete_task(self, task_id: str, result: Any = None, error: str = None) -> None:
        """Mark task as completed"""
        async with self._lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Attempted to complete non-running task: {task_id}")
                return
            
            task = self.running_tasks[task_id]
            task.completed_at = datetime.now()
            task.result = result
            task.error = error
            task.status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED
            
            # Remove from running tasks tracking
            del self.running_tasks[task_id]
            
            # Remove from GPU tracking
            gpu_id = task.gpu_id
            if task.task_type == TaskType.PERFORMANCE:
                if gpu_id in self.gpu_performance_task:
                    del self.gpu_performance_task[gpu_id]
            else:
                if gpu_id in self.gpu_functional_tasks:
                    try:
                        self.gpu_functional_tasks[gpu_id].remove(task_id)
                        if not self.gpu_functional_tasks[gpu_id]:
                            del self.gpu_functional_tasks[gpu_id]
                    except ValueError:
                        pass
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            
            status_str = "completed" if error is None else "failed"
            logger.info(f"Task {task_id} {status_str} on GPU {gpu_id} after {task.execution_time_seconds:.2f}s")
    
    async def cleanup_expired_tasks(self) -> int:
        """Remove expired tasks from queues"""
        expired_count = 0
        
        async with self._lock:
            # Clean functional queue
            expired_functional = [task for task in self.functional_queue if task.is_expired]
            self.functional_queue = [task for task in self.functional_queue if not task.is_expired]
            
            # Clean performance queue
            expired_performance = [task for task in self.performance_queue if task.is_expired]
            self.performance_queue = [task for task in self.performance_queue if not task.is_expired]
            
            # Mark expired tasks as failed
            for task in expired_functional + expired_performance:
                task.status = TaskStatus.FAILED
                task.error = f"Task expired after waiting {task.wait_time_seconds:.1f} seconds"
                self.completed_tasks[task.task_id] = task
                expired_count += 1
                logger.warning(f"Task {task.task_id} expired after {task.wait_time_seconds:.1f}s wait")
        
        return expired_count
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        async with self._lock:
            return {
                'functional_queue_length': len(self.functional_queue),
                'performance_queue_length': len(self.performance_queue),
                'running_tasks_count': len(self.running_tasks),
                'completed_tasks_count': len(self.completed_tasks),
                'gpu_functional_tasks': dict(self.gpu_functional_tasks),
                'gpu_performance_task': dict(self.gpu_performance_task)
            }
    
    async def _put_task_back(self, task: Task) -> None:
        """Put a task back to the appropriate queue"""
        async with self._lock:
            if task.task_type == TaskType.FUNCTIONAL:
                self.functional_queue.insert(0, task)  # 插入到队列前面，优先处理
                logger.info(f"Put functional task {task.task_id} back to queue")
            else:
                self.performance_queue.insert(0, task)  # 插入到队列前面，优先处理
                logger.info(f"Put performance task {task.task_id} back to queue") 