"""GPU manager for monitoring and managing GPU resources."""
import threading
import time
from datetime import datetime, timedelta

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from models import GPU, GPUStatus, GPUMode
from config import config
from logger import setup_logger

logger = setup_logger("gpu_manager")

# Global GPU config loader (set by main)
gpu_config_loader = None


class GPUManager:
    """Manages GPU resources, monitoring, and status."""
    
    def __init__(self):
        self.gpus: dict[int, GPU] = {}
        self.lock = threading.RLock()
        self.monitor_thread: threading.Thread | None = None
        self.running = False
        self.nvml_initialized = False
        
        # Severe error handling
        self.severe_error_active = False
        self.severe_error_time: datetime | None = None
        
        # Dependencies (set after initialization)
        self.task_queue = None
        self.task_runner = None
        
        self._initialize_nvml()
    
    def set_dependencies(self, task_queue, task_runner):
        """Set dependencies for advanced error handling.
        
        Args:
            task_queue: TaskQueue instance
            task_runner: TaskRunner instance
        """
        self.task_queue = task_queue
        self.task_runner = task_runner
        logger.debug("GPU manager dependencies set")
    
    def _initialize_nvml(self):
        """Initialize NVML for GPU monitoring."""
        if not NVML_AVAILABLE:
            logger.warning("pynvml not available, GPU monitoring disabled")
            return
        
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized, found {device_count} GPUs")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.nvml_initialized = False
    
    def register_gpu(self, gpu_id: int, mode: GPUMode = None, memory_threshold: float = None, 
                    max_concurrent_tasks: int = None) -> bool:
        """Register a GPU for use."""
        with self.lock:
            if gpu_id in self.gpus:
                logger.warning(f"GPU {gpu_id} already registered")
                return False
            
            gpu = GPU(
                gpu_id=gpu_id,
                mode=mode or config.default_gpu_mode,
                memory_threshold=memory_threshold or config.default_memory_threshold,
                max_concurrent_tasks=max_concurrent_tasks or config.default_max_concurrent_tasks,
                status=GPUStatus.ONLINE
            )
            self.gpus[gpu_id] = gpu
            logger.info(f"Registered GPU {gpu_id} with mode={gpu.mode}, "
                       f"threshold={gpu.memory_threshold}, max_tasks={gpu.max_concurrent_tasks}")
            return True
    
    def unregister_gpu(self, gpu_id: int) -> bool:
        """Unregister a GPU."""
        with self.lock:
            if gpu_id not in self.gpus:
                logger.warning(f"GPU {gpu_id} not registered")
                return False
            
            gpu = self.gpus[gpu_id]
            if gpu.running_tasks:
                logger.error(f"Cannot unregister GPU {gpu_id}: has running tasks")
                return False
            
            del self.gpus[gpu_id]
            logger.info(f"Unregistered GPU {gpu_id}")
            return True
    
    def set_gpu_status(self, gpu_id: int, status: GPUStatus) -> bool:
        """Set GPU status (online/offline/maintenance)."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            self.gpus[gpu_id].status = status
            logger.info(f"GPU {gpu_id} status changed to {status.value}")
            return True
    
    def set_gpu_mode(self, gpu_id: int, mode: GPUMode, manual: bool = True) -> bool:
        """Set GPU execution mode.
        
        Args:
            gpu_id: GPU ID
            mode: GPU mode to set
            manual: If True, sets as manual override. If False, just changes current mode.
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            gpu = self.gpus[gpu_id]
            gpu.mode = mode
            
            if manual:
                gpu.manual_mode = mode
                logger.info(f"GPU {gpu_id} manual mode set to {mode.value}")
            else:
                logger.debug(f"GPU {gpu_id} mode changed to {mode.value} (task-driven)")
            
            return True
    
    def clear_manual_mode(self, gpu_id: int) -> bool:
        """Clear manual mode override for a GPU.
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            self.gpus[gpu_id].manual_mode = None
            logger.info(f"GPU {gpu_id} manual mode cleared")
            return True
    
    def set_gpu_mode_for_task(self, gpu_id: int, task_id: str, task_mode) -> bool:
        """Set GPU mode based on task requirements.
        
        This is called when a task starts running.
        
        Args:
            gpu_id: GPU ID
            task_id: Task ID
            task_mode: Task mode (exclusive/shared)
            
        Returns:
            True if successful, False otherwise
        """
        from config import TaskMode, GPUMode
        
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            gpu = self.gpus[gpu_id]
            
            # If manual mode is set, respect it (don't change)
            if gpu.manual_mode:
                logger.debug(f"GPU {gpu_id} has manual mode {gpu.manual_mode.value}, "
                           f"not changing for task {task_id[:8]}")
                return True
            
            # Set mode based on task requirement
            if task_mode == TaskMode.EXCLUSIVE:
                gpu.mode = GPUMode.EXCLUSIVE
                gpu.mode_locked_by = task_id
                logger.info(f"GPU {gpu_id} mode set to EXCLUSIVE for task {task_id[:8]}")
            else:  # SHARED
                # Only change to shared if not locked by another task
                if not gpu.mode_locked_by:
                    gpu.mode = GPUMode.SHARED
                    logger.debug(f"GPU {gpu_id} mode set to SHARED for task {task_id[:8]}")
            
            return True
    
    def restore_gpu_mode_after_task(self, gpu_id: int, task_id: str) -> bool:
        """Restore GPU mode after a task completes.
        
        This is called when a task finishes running.
        
        Args:
            gpu_id: GPU ID
            task_id: Task ID that just finished
            
        Returns:
            True if successful, False otherwise
        """
        from config import GPUMode
        
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            gpu = self.gpus[gpu_id]
            
            # If this task locked the mode, unlock it
            if gpu.mode_locked_by == task_id:
                gpu.mode_locked_by = None
                logger.debug(f"GPU {gpu_id} mode unlocked by task {task_id[:8]}")
                
                # Restore to manual mode if set, otherwise default to shared
                if gpu.manual_mode:
                    gpu.mode = gpu.manual_mode
                    logger.debug(f"GPU {gpu_id} mode restored to manual mode {gpu.manual_mode.value}")
                else:
                    # Default to shared if no manual mode and no running tasks
                    if len(gpu.running_tasks) == 0:
                        gpu.mode = GPUMode.SHARED
                        logger.debug(f"GPU {gpu_id} mode restored to SHARED (default)")
            
            return True
    
    def set_gpu_memory_threshold(self, gpu_id: int, threshold: float) -> bool:
        """Set GPU memory threshold."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            if not 0 < threshold <= 1.0:
                logger.error(f"Invalid threshold {threshold}, must be between 0 and 1")
                return False
            
            self.gpus[gpu_id].memory_threshold = threshold
            logger.info(f"GPU {gpu_id} memory threshold set to {threshold}")
            return True
    
    def set_gpu_max_concurrent_tasks(self, gpu_id: int, max_tasks: int) -> bool:
        """Set GPU maximum concurrent tasks."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            if max_tasks < 1:
                logger.error(f"Invalid max_tasks {max_tasks}, must be >= 1")
                return False
            
            self.gpus[gpu_id].max_concurrent_tasks = max_tasks
            logger.info(f"GPU {gpu_id} max concurrent tasks set to {max_tasks}")
            return True
    
    def get_gpu(self, gpu_id: int) -> GPU | None:
        """Get GPU info."""
        with self.lock:
            return self.gpus.get(gpu_id)
    
    def list_gpus(self) -> list[GPU]:
        """List all registered GPUs."""
        with self.lock:
            return list(self.gpus.values())
    
    def find_available_gpu(self, preferred_gpu: int | None = None, task_mode=None) -> int | None:
        """Find an available GPU for task assignment.
        
        Updates GPU memory usage before checking availability to ensure
        real-time memory information is used for scheduling decisions.
        
        Args:
            preferred_gpu: Preferred GPU ID, or None for any GPU
            task_mode: Task mode (exclusive/shared) to check compatibility
        
        Returns:
            GPU ID if available, None otherwise
        """
        # Check if in severe error state (outside lock to avoid deadlock with _update_gpu_memory)
        if self.severe_error_active:
            return None
        
        # Try preferred GPU first
        if preferred_gpu is not None and preferred_gpu in self.gpus:
            # Update memory usage before checking (important for tasks with delayed GPU memory allocation)
            self._update_gpu_memory(preferred_gpu)
            
            with self.lock:
                gpu = self.gpus.get(preferred_gpu)
                if gpu and gpu.can_accept_task(task_mode):
                    return preferred_gpu
            return None
        
        # Find any available GPU
        # First update all GPU memory usages to get real-time data
        with self.lock:
            gpu_ids = list(self.gpus.keys())
        
        for gpu_id in gpu_ids:
            self._update_gpu_memory(gpu_id)
        
        # Now check which GPUs can accept tasks with this mode
        with self.lock:
            for gpu_id, gpu in self.gpus.items():
                if gpu.can_accept_task(task_mode):
                    return gpu_id
        
        return None
    
    def mark_task_running(self, gpu_id: int, task_id: str) -> bool:
        """Mark a task as running on a GPU."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            self.gpus[gpu_id].running_tasks.append(task_id)
            logger.debug(f"Task {task_id} marked running on GPU {gpu_id}")
            return True
    
    def mark_task_completed(self, gpu_id: int, task_id: str) -> bool:
        """Mark a task as completed on a GPU."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            gpu = self.gpus[gpu_id]
            if task_id in gpu.running_tasks:
                gpu.running_tasks.remove(task_id)
                logger.debug(f"Task {task_id} completed on GPU {gpu_id}")
            return True
    
    def trigger_severe_error(self, gpu_id: int, error_msg: str):
        """Trigger severe error state, kill all running tasks and requeue them.
        
        Args:
            gpu_id: GPU that encountered the error
            error_msg: Error message
        """
        with self.lock:
            self.severe_error_active = True
            self.severe_error_time = datetime.now()
            
            if gpu_id in self.gpus:
                self.gpus[gpu_id].status = GPUStatus.ERROR
                self.gpus[gpu_id].error_message = error_msg
                self.gpus[gpu_id].last_error_time = self.severe_error_time
                
                # Get all running tasks on this GPU
                running_task_ids = list(self.gpus[gpu_id].running_tasks)
            else:
                running_task_ids = []
            
            logger.error(f"SEVERE ERROR on GPU {gpu_id}: {error_msg}. "
                        f"Killing {len(running_task_ids)} running tasks.")
        
        # Kill and requeue tasks (outside lock to avoid deadlock)
        if self.task_queue and self.task_runner and running_task_ids:
            killed_tasks = []
            for task_id in running_task_ids:
                task = self.task_queue.get_task(task_id)
                if task:
                    # Kill the task process
                    if self.task_runner.kill_task(task_id):
                        logger.info(f"Killed task {task_id[:8]} due to severe error")
                        killed_tasks.append(task)
                    else:
                        logger.warning(f"Failed to kill task {task_id[:8]}")
            
            # Requeue killed tasks at front (reverse order to maintain original order)
            for task in reversed(killed_tasks):
                # Reset task state for requeue
                task.start_time = None
                task.end_time = None
                task.exit_code = None
                task.error_message = f"Requeued due to GPU {gpu_id} severe error"
                self.task_queue.push_front(task)
                logger.info(f"Requeued task {task.task_id[:8]} at front of queue")
            
            # Clear running tasks from GPU
            with self.lock:
                if gpu_id in self.gpus:
                    self.gpus[gpu_id].running_tasks.clear()
                    logger.info(f"Cleared {len(killed_tasks)} tasks from GPU {gpu_id}")
        
        logger.error(f"SEVERE ERROR handling complete. System paused for {config.error_pause_duration}s.")
    
    def clear_severe_error(self):
        """Clear severe error state and resume operations."""
        with self.lock:
            self.severe_error_active = False
            self.severe_error_time = None
            logger.info("Severe error state cleared, resuming operations")
    
    def _update_gpu_memory(self, gpu_id: int):
        """Update GPU memory usage."""
        if not self.nvml_initialized:
            return
        
        try:
            # Get nvidia-smi ID from config if available
            nvidia_smi_id = gpu_id
            if gpu_config_loader:
                mapped_id = gpu_config_loader.get_nvidia_smi_id(gpu_id)
                if mapped_id is not None:
                    nvidia_smi_id = mapped_id
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvidia_smi_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage = mem_info.used / mem_info.total
            
            with self.lock:
                if gpu_id in self.gpus:
                    self.gpus[gpu_id].current_memory_usage = usage
        except Exception as e:
            logger.debug(f"Failed to get memory info for GPU {gpu_id}: {e}")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Update memory for all GPUs
                with self.lock:
                    gpu_ids = list(self.gpus.keys())
                
                for gpu_id in gpu_ids:
                    self._update_gpu_memory(gpu_id)
                
                # Check if severe error timeout has passed - auto-resume after pause duration
                if self.severe_error_active and self.severe_error_time:
                    elapsed = (datetime.now() - self.severe_error_time).total_seconds()
                    if elapsed > config.error_pause_duration:
                        logger.info(f"Severe error pause duration ({config.error_pause_duration}s) elapsed, auto-resuming")
                        self.clear_severe_error()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            time.sleep(config.gpu_monitor_interval)
    
    def start_monitoring(self):
        """Start background GPU monitoring."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("GPU monitoring stopped")
    
    def shutdown(self):
        """Shutdown GPU manager."""
        self.stop_monitoring()
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

