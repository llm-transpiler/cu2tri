"""
Task Management Module

Defines the Task class and related functionality for GPU task execution.
"""

import asyncio
import logging
import signal
import os
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable, Optional, Dict
from dataclasses import dataclass, field
import uuid


class TaskType(Enum):
    """Task execution type"""
    EXCLUSIVE = "exclusive"    # Requires exclusive GPU access
    SHARED = "shared"         # Can share GPU with other shared tasks
    
    def __str__(self):
        return self.value


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    KILLED = "killed"
    def __str__(self):
        return self.value


@dataclass
class Task:
    """
    Represents a GPU task with execution tracking and control capabilities.
    
    Each task encapsulates:
    - Task metadata (name, description, type)
    - Execution function and parameters
    - Status tracking (pending -> waiting -> running -> completed/failed/cancelled/killed)
    - Timing information
    - GPU assignment
    - Process control for real termination
    """
    
    name: str
    description: str
    task_type: TaskType
    execute_func: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    max_wait_time_minutes: int = 30
    gpu_id: Optional[int] = None
    
    # Auto-generated fields
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    
    # Process control
    process: Optional[asyncio.subprocess.Process] = None
    task_coroutine: Optional[asyncio.Task] = None
    
    # Internal state
    _logger: Optional[logging.Logger] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize task after creation"""
        if self._logger is None:
            self._logger = logging.getLogger(f"Task-{self.task_id[:8]}")
        
        self._logger.info(f"[Task] Created task '{self.name}' (type: {self.task_type.value})")
    
    @property
    def wait_time_seconds(self) -> float:
        """Calculate wait time before execution started"""
        if self.started_at is None:
            return (datetime.now() - self.created_at).total_seconds()
        return (self.started_at - self.created_at).total_seconds()
    
    @property
    def execution_time_seconds(self) -> float:
        """Calculate execution time"""
        if self.started_at is None:
            return 0.0
        
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running"""
        return self.status == TaskStatus.RUNNING
    
    @property
    def is_finished(self) -> bool:
        """Check if task has finished (completed, failed, cancelled, or killed)"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.KILLED]
    
    @property
    def can_be_cancelled(self) -> bool:
        """Check if task can be cancelled"""
        return self.status in [TaskStatus.PENDING, TaskStatus.WAITING]
    
    @property
    def can_be_killed(self) -> bool:
        """Check if task can be killed"""
        return self.status == TaskStatus.RUNNING
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired based on max_wait_time_minutes"""
        if self.status not in [TaskStatus.PENDING, TaskStatus.WAITING]:
            return False
        
        elapsed_minutes = (datetime.now() - self.created_at).total_seconds() / 60
        return elapsed_minutes > self.max_wait_time_minutes
    
    async def execute(self) -> Any:
        """
        Execute the task with proper status tracking and error handling.
        
        Returns:
            Task execution result
            
        Raises:
            Exception: If task execution fails
        """
        try:
            # Update status to running
            self.status = TaskStatus.RUNNING
            self.started_at = datetime.now()
            
            self._logger.info(f"[Task] Starting execution on GPU-{self.gpu_id}")
            
            # Create a coroutine task for execution
            self.task_coroutine = asyncio.create_task(
                self.execute_func(*self.args, **self.kwargs)
            )
            
            # Wait for execution to complete
            self.result = await self.task_coroutine
            
            # Mark as completed
            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()
            
            self._logger.info(f"[Task] Completed successfully in {self.execution_time_seconds:.2f}s")
            return self.result
            
        except asyncio.CancelledError:
            # Task was cancelled
            self.status = TaskStatus.KILLED
            self.completed_at = datetime.now()
            self.error = "Task was killed"
            self._logger.warning("[Task] " + self.error)
            raise
            
        except Exception as e:
            # Task failed
            self.status = TaskStatus.FAILED
            self.completed_at = datetime.now()
            self.error = str(e)
            
            self._logger.error(f"[Task] Failed after {self.execution_time_seconds:.2f}s: {e}")
            raise
    
    async def cancel(self) -> bool:
        """
        Cancel the task if it's still pending or waiting.
        
        Returns:
            True if task was cancelled, False otherwise
        """
        if not self.can_be_cancelled:
            self._logger.warning(f"[Task] Cannot cancel task in status: {self.status}")
            return False
        
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
        self.error = "Task was cancelled"
        
        self._logger.info("[Task] Task cancelled")
        return True
    
    async def kill(self) -> bool:
        """
        Kill the running task by cancelling its coroutine.
        
        Returns:
            True if task was killed, False otherwise
        """
        if not self.can_be_killed:
            self._logger.warning(f"[Task] Cannot kill task in status: {self.status}")
            return False
        
        try:
            # Cancel the task coroutine
            if self.task_coroutine and not self.task_coroutine.done():
                self.task_coroutine.cancel()
                
                # Wait for cancellation to complete
                try:
                    await self.task_coroutine
                except asyncio.CancelledError:
                    pass
            
            # If there's a subprocess, kill it
            if self.process and self.process.returncode is None:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if termination timeout
                    self.process.kill()
                    await self.process.wait()
            
            self.status = TaskStatus.KILLED
            self.completed_at = datetime.now()
            self.error = "Task was killed"
            
            self._logger.info("[Task] Task killed successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"[Task] Failed to kill task: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'wait_time_seconds': self.wait_time_seconds,
            'execution_time_seconds': self.execution_time_seconds,
            'gpu_id': self.gpu_id,
            'error': self.error,
            'result': self.result
        } 