"""Task queue management for NVGPU server."""
import threading
from collections import deque

from server.nvgpu.models import Task, TaskStatus
from server.nvgpu.logger import setup_logger
from server.common.timezone import now_timestamp
from server.common.task_refs import format_task_ref

logger = setup_logger("task_queue")


class TaskQueue:
    """Thread-safe task queue with global and per-GPU queues."""
    
    def __init__(self):
        self.lock = threading.RLock()
        
        # Global task queue (pending tasks)
        self.global_queue: deque[Task] = deque()
        
        # Per-GPU task queues (queued for specific GPU)
        self.gpu_queues: dict[int, deque[Task]] = {}
        
        # All tasks by ID
        self.tasks: dict[str, Task] = {}
    
    def submit_task(self, task: Task) -> str:
        """Submit a new task."""
        with self.lock:
            task.status = TaskStatus.PENDING
            self.tasks[task.task_id] = task
            self.global_queue.append(task)
            # Start lifecycle timers
            task.timer.start("total")
            task.timer.start("waiting")
            task.timer.start("pending")
            
        # Log with type and label if set
        type_info = f", type={task.task_type.value}" if task.task_type else ""
        label_info = f", label={task.task_label}" if task.task_label else ""
        logger.info(f"TASK {format_task_ref(task)} submitted: mode={task.task_mode.value}{type_info}{label_info}, "
                   f"script={task.script_path}, gpu={task.gpu_id}")
        return task.task_id
    
    def push_front(self, task: Task):
        """Push a task to the front of global queue (for requeue after error).
        
        Args:
            task: Task to requeue at front
        """
        with self.lock:
            task.status = TaskStatus.PENDING
            task.assigned_gpu = None  # Clear assignment
            task.queued_timestamp = None
            task.start_timestamp = None
            task.end_timestamp = None
            # The same Task instance is reused so task.task_id remains stable.
            self.global_queue.appendleft(task)
            task.timer.start("pending")
            logger.info(f"TASK {format_task_ref(task)} requeued at front (priority)")
    
    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        """List all tasks, optionally filtered by status."""
        with self.lock:
            if status:
                return [t for t in self.tasks.values() if t.status == status]
            return list(self.tasks.values())
    
    def pop_pending_task(self) -> Task | None:
        """Pop a pending task from global queue."""
        with self.lock:
            if not self.global_queue:
                return None
            return self.global_queue.popleft()
    
    def queue_task_for_gpu(self, task: Task, gpu_id: int):
        """Queue a task for a specific GPU."""
        with self.lock:
            if gpu_id not in self.gpu_queues:
                self.gpu_queues[gpu_id] = deque()
            
            task.status = TaskStatus.QUEUED
            task.assigned_gpu = gpu_id
            task.queued_timestamp = now_timestamp()  # Record when task was assigned to GPU queue
            
            # Stop pending timer, start queue timer
            pending_duration = task.timer.stop("pending")
            if pending_duration is not None:
                task.phase_duration_ms["pending"] = pending_duration
            task.timer.start("queue")
            self.gpu_queues[gpu_id].append(task)
            
            logger.info(f"TASK {format_task_ref(task)} queued for GPU {gpu_id}")
    
    def pop_gpu_task(self, gpu_id: int) -> Task | None:
        """Pop a task from GPU-specific queue."""
        with self.lock:
            if gpu_id not in self.gpu_queues or not self.gpu_queues[gpu_id]:
                return None
            return self.gpu_queues[gpu_id].popleft()
    
    def get_queue_size(self, gpu_id: int | None = None) -> int:
        """Get queue size for global or specific GPU queue."""
        with self.lock:
            if gpu_id is None:
                return len(self.global_queue)
            return len(self.gpu_queues.get(gpu_id, []))
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or queued task."""
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            
            if task.status not in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                logger.error(f"Cannot cancel TASK {format_task_ref(task)} with status {task.status.value}")
                return False
            
            self._finalize_timing_on_cancel(task)
            
            # Remove from global queue
            try:
                self.global_queue.remove(task)
            except ValueError:
                pass
            
            # Remove from GPU queue
            if task.assigned_gpu is not None:
                gpu_id = task.assigned_gpu
                if gpu_id in self.gpu_queues:
                    try:
                        self.gpu_queues[gpu_id].remove(task)
                    except ValueError:
                        pass
            
            task.status = TaskStatus.CANCELLED
            logger.info(f"TASK {format_task_ref(task)} cancelled successfully")
            return True
    
    def force_cancel_task(self, task_id: str, task_runner) -> bool:
        """Force cancel a task, including running tasks.
        
        Args:
            task_id: Task ID to cancel
            task_runner: TaskRunner instance to kill running processes
            
        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            if task_id not in self.tasks:
                logger.error(f"Cannot force cancel TASK {task_id}: not found")
                return False
            
            task = self.tasks[task_id]
            task_ref_str = format_task_ref(task)
            # For pending or queued tasks, use regular cancel
            if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                return self.cancel_task(task_id)
            
            # For running tasks, kill the process
            if task.status == TaskStatus.RUNNING:
                logger.info(f"Force cancelling running TASK {task_ref_str}")
                if task_runner.kill_task(task):
                    task.status = TaskStatus.CANCELLED
                    task.error_message = "Cancelled by user (force)"
                    task.end_timestamp = now_timestamp()
                    logger.info(f"TASK {task_ref_str} force cancelled")
                    return True
                else:
                    logger.error(f"Failed to kill running TASK {task_ref_str}")
                    return False
            
            # Already finished tasks cannot be cancelled
            logger.warning(f"Cannot cancel TASK {task_ref_str} with status {task.status.value}")
            return False
    
    def get_statistics(self) -> dict:
        """Get queue statistics."""
        with self.lock:
            stats = {
                "global_queue_size": len(self.global_queue),
                "total_tasks": len(self.tasks),
                "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
                "queued": sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED),
                "running": sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
                "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
                "gpu_queues": {gpu_id: len(queue) for gpu_id, queue in self.gpu_queues.items()}
            }
            return stats

    def _finalize_timing_on_cancel(self, task: Task) -> None:
        """Stop relevant timers when a task is cancelled."""
        if task.status == TaskStatus.PENDING:
            pending_duration = task.timer.stop("pending")
            if pending_duration is not None:
                task.phase_duration_ms["pending"] = pending_duration
        if task.status == TaskStatus.QUEUED:
            queue_duration = task.timer.stop("queue")
            if queue_duration is not None:
                task.phase_duration_ms["queue"] = queue_duration
        waiting_duration = task.timer.stop("waiting")
        if waiting_duration is not None:
            task.phase_duration_ms["waiting"] = waiting_duration
        total_duration = task.timer.stop("total")
        if total_duration is not None:
            task.phase_duration_ms["total"] = total_duration
