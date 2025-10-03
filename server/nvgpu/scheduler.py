"""Task scheduler for NVGPU server."""
import threading
import time
from typing import Optional

from gpu_manager import GPUManager
from task_queue import TaskQueue
from task_runner import TaskRunner
from models import TaskStatus
from config import config
from logger import setup_logger

logger = setup_logger("scheduler")


class Scheduler:
    """Schedules and dispatches tasks to available GPUs."""
    
    def __init__(self, gpu_manager: GPUManager, task_queue: TaskQueue, task_runner: TaskRunner):
        self.gpu_manager = gpu_manager
        self.task_queue = task_queue
        self.task_runner = task_runner
        
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
    
    def _schedule_round(self):
        """One round of scheduling: assign pending tasks to GPUs and execute queued tasks."""
        # Phase 1: Assign pending tasks to GPU queues
        while True:
            task = self.task_queue.pop_pending_task()
            if not task:
                break
            
            # Find available GPU
            gpu_id = self.gpu_manager.find_available_gpu(task.gpu_id)
            if gpu_id is None:
                # No GPU available, put back to queue
                self.task_queue.global_queue.appendleft(task)
                break
            
            # Queue task for specific GPU
            self.task_queue.queue_task_for_gpu(task, gpu_id)
        
        # Phase 2: Execute queued tasks on available GPUs
        for gpu in self.gpu_manager.list_gpus():
            gpu_id = gpu.gpu_id
            
            # Check if GPU can accept new task
            if not gpu.can_accept_task():
                continue
            
            # Get next task from GPU queue
            task = self.task_queue.pop_gpu_task(gpu_id)
            if not task:
                continue
            
            # Execute task in a separate thread
            thread = threading.Thread(
                target=self._execute_task,
                args=(task, gpu_id),
                daemon=True
            )
            thread.start()
    
    def _execute_task(self, task, gpu_id: int):
        """Execute a single task on GPU."""
        # Mark task as running on GPU
        self.gpu_manager.mark_task_running(gpu_id, task.task_id)
        
        try:
            # Run the task
            success = self.task_runner.run_task(task, gpu_id)
            
            # Check for GPU errors based on task result
            if not success or task.exit_code != 0:
                # Check stderr for GPU-related errors (if file is small enough)
                if task.log_file and task.stderr_size > 0 and task.stderr_size < 1024 * 1024:  # < 1MB
                    try:
                        from pathlib import Path
                        stderr_path = Path(task.log_file).parent / f"{task.task_id}.stderr"
                        if stderr_path.exists():
                            stderr_content = stderr_path.read_text(encoding='utf-8', errors='replace')
                            if any(err in stderr_content.lower() 
                                  for err in ["cuda error", "gpu error", "out of memory"]):
                                logger.warning(f"GPU error detected in task {task.task_id}")
                                # Could trigger severe error here if needed
                                # self.gpu_manager.trigger_severe_error(gpu_id, "GPU error in task")
                    except Exception as e:
                        logger.debug(f"Could not check stderr for GPU errors: {e}")
        finally:
            # Mark task as completed on GPU
            self.gpu_manager.mark_task_completed(gpu_id, task.task_id)
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Skip scheduling if severe error is active
                if self.gpu_manager.severe_error_active:
                    logger.debug("Severe error active, skipping scheduling")
                    time.sleep(config.scheduler_interval)
                    continue
                
                # Run one scheduling round
                self._schedule_round()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
            
            time.sleep(config.scheduler_interval)
    
    def start(self):
        """Start the scheduler."""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")

