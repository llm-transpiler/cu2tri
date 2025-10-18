"""GPU manager for monitoring and managing GPU resources."""
import threading
import time

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from models import GPU, GPUStatus, GPUMode, Task
from config import config, TaskStatus, TaskMode
from logger import setup_logger
from utils.timezone import now_timestamp
from utils.task_refs import format_task_ref
from profiler.timer import monotonic_elapsed_ms, monotonic_timestamp_ns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from task_queue import TaskQueue
    from task_runner import TaskRunner

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
        self.nvml_device_count = 0

        # Severe error handling
        self.severe_error_active = False
        self.severe_error_timestamp = None  # wall clock (datetime) for logging
        self.severe_error_monotonic_ns: int | None = None  # monotonic clock for elapsed checks
        self.error_gpus: set[int] = set()  # GPUs currently marked as error

        # Dependencies (set after initialization)
        self.task_queue : "TaskQueue" | None = None
        self.task_runner : "TaskRunner" | None = None

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
            self.nvml_device_count = device_count
            logger.info(f"NVML initialized, found {device_count} GPUs")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}", exc_info=True)
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
                logger.error(
                    f"Cannot unregister GPU {gpu_id}: has running tasks")
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
                logger.info(
                    f"GPU {gpu_id} manual mode set to {mode.value}")
            else:
                logger.debug(
                    f"GPU {gpu_id} mode changed to {mode.value} (task-driven)")

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

    def get_nvml_device_count(self) -> int:
        """Return number of devices detected via NVML."""
        return self.nvml_device_count

    def set_gpu_mode_for_task(
        self,
        gpu_id: int,
        task: Task,
        task_mode_override: GPUMode | None = None,
    ) -> bool:
        """Set GPU mode based on task requirements.

        This is called when a task starts running.

        Args:
            gpu_id: GPU ID
            task: Task instance
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
                logger.debug(
                    "GPU %s has manual mode %s, not changing for task %s",
                    gpu_id,
                    gpu.manual_mode.value,
                    format_task_ref(task),
                )
                return True

            # Set mode based on task requirement
            effective_task_mode = task_mode_override or task.task_mode
            if effective_task_mode == TaskMode.EXCLUSIVE:
                gpu.mode = GPUMode.EXCLUSIVE
                gpu.mode_locked_by = task.task_id
                logger.info(
                    "GPU %s mode set to EXCLUSIVE for task %s",
                    gpu_id,
                    format_task_ref(task),
                )
            else:  # SHARED
                # Only change to shared if not locked by another task
                if not gpu.mode_locked_by:
                    gpu.mode = GPUMode.SHARED
                    logger.debug(
                        "GPU %s mode set to SHARED for task %s",
                        gpu_id,
                        format_task_ref(task),
                    )

            return True

    def restore_gpu_mode_after_task(self, gpu_id: int, task: Task) -> bool:
        """Restore GPU mode after a task completes.

        This is called when a task finishes running.

        Args:
            gpu_id: GPU ID
            task: Task instance that just finished

        Returns:
            True if successful, False otherwise
        """
        from config import GPUMode

        with self.lock:
            if gpu_id not in self.gpus:
                return False

            gpu = self.gpus[gpu_id]

            # If this task locked the mode, unlock it
            if gpu.mode_locked_by == task.task_id:
                gpu.mode_locked_by = None
                logger.debug(
                    "GPU %s mode unlocked by TASK %s",
                    gpu_id,
                    format_task_ref(task),
                )

                # Restore to manual mode if set, otherwise default to shared
                if gpu.manual_mode:
                    gpu.mode = gpu.manual_mode
                    logger.debug(
                        f"GPU {gpu_id} mode restored to manual mode {gpu.manual_mode.value}")
                else:
                    # Default to shared if no manual mode and no running tasks
                    if len(gpu.running_tasks) == 0:
                        gpu.mode = GPUMode.SHARED
                        logger.debug(
                            f"GPU {gpu_id} mode restored to SHARED (default)")

            return True

    def set_gpu_memory_threshold(self, gpu_id: int, threshold: float) -> bool:
        """Set GPU memory threshold."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False

            if not 0 < threshold <= 1.0:
                logger.error(
                    f"Invalid threshold {threshold}, must be between 0 and 1")
                return False

            self.gpus[gpu_id].memory_threshold = threshold
            logger.info(
                f"GPU {gpu_id} memory threshold set to {threshold}")
            return True

    def set_gpu_max_concurrent_tasks(self, gpu_id: int, max_tasks: int) -> bool:
        """Set GPU maximum concurrent tasks."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False

            if max_tasks < 1:
                logger.error(
                    f"Invalid max_tasks {max_tasks}, must be >= 1")
                return False

            self.gpus[gpu_id].max_concurrent_tasks = max_tasks
            logger.info(
                f"GPU {gpu_id} max concurrent tasks set to {max_tasks}")
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
                queued_depth = 0
                if gpu and self.task_queue:
                    queued_depth = self.task_queue.get_queue_size(preferred_gpu)

                if (
                    gpu
                    and gpu.can_accept_task(task_mode)
                    and self._has_queue_capacity(gpu, queued_depth, task_mode)
                ):
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
                queued_depth = 0
                if self.task_queue:
                    queued_depth = self.task_queue.get_queue_size(gpu_id)

                if gpu.can_accept_task(task_mode) and self._has_queue_capacity(gpu, queued_depth, task_mode):
                    return gpu_id

        return None

    def _has_queue_capacity(self, gpu: GPU, queued_depth: int, task_mode: TaskMode | None) -> bool:
        """Check if GPU queue + running slots still have capacity."""
        requested_mode = task_mode or TaskMode.SHARED

        if requested_mode == TaskMode.EXCLUSIVE:
            # Exclusive tasks require GPU to be completely idle (no queued or running tasks)
            return queued_depth == 0 and len(gpu.running_tasks) == 0

        # Shared tasks respect max_concurrent_tasks across running + queued
        active_count = len(gpu.running_tasks) + queued_depth
        return active_count < gpu.max_concurrent_tasks

    def mark_task_running(self, gpu_id: int, task: Task) -> bool:
        """Mark a task as running on a GPU."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False

            self.gpus[gpu_id].running_tasks.append(task.task_id)
            logger.debug(f"Task {format_task_ref(task)} running on GPU {gpu_id}")
            return True

    def mark_task_completed(self, gpu_id: int, task: Task) -> bool:
        """Mark a task as completed on a GPU."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False

            gpu = self.gpus[gpu_id]
            if task.task_id in gpu.running_tasks:
                gpu.running_tasks.remove(task.task_id)
                logger.debug(f"TASK {format_task_ref(task)} completed on GPU {gpu_id}")
            return True

    def trigger_severe_error(self, gpu_id: int, error_msg: str, offending_task_id: str | None = None):
        """Trigger severe error state.

        The offending task is marked failed, any collateral tasks are terminated and
        requeued to run again once the GPU recovers.

        Args:
            gpu_id: GPU that encountered the error
            error_msg: Error message
        """
        with self.lock:
            self.severe_error_active = True
            self.severe_error_timestamp = now_timestamp()
            self.severe_error_monotonic_ns = monotonic_timestamp_ns()

            if gpu_id in self.gpus:
                self.gpus[gpu_id].status = GPUStatus.ERROR
                self.gpus[gpu_id].error_message = error_msg
                self.gpus[gpu_id].last_error_timestamp = self.severe_error_timestamp
                self.error_gpus.add(gpu_id)

                # Get all running tasks on this GPU
                running_task_ids = list(self.gpus[gpu_id].running_tasks)
            else:
                running_task_ids = []

            logger.error(f"SEVERE ERROR on GPU {gpu_id}: {error_msg}. "
                         f"Killing {len(running_task_ids)} running tasks.")

        # Kill running tasks and mark them failed or requeue collateral tasks (outside lock to avoid deadlock)
        if self.task_queue and self.task_runner and running_task_ids:
            damage_message = f"GPU damage bug: {error_msg}"
            trigger_note = (
                f"triggered by task {offending_task_id}"
                if offending_task_id
                else "triggered by severe error"
            )

            def _mark_task_failed(task: Task):
                """Mark task as failed without retry and tag damage bug message."""
                if task.status != TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED

                if task.error_message:
                    if damage_message not in task.error_message:
                        task.error_message = f"{task.error_message} | {damage_message}"
                else:
                    task.error_message = damage_message

                if task.end_timestamp is None:
                    task.end_timestamp = now_timestamp()

                logger.info(f"Marked TASK {format_task_ref(task)} as failed due to severe error (no retry)")

            collateral_tasks: list[Task] = []

            for task_id in running_task_ids:
                task = self.task_queue.get_task(task_id)
                if not task:
                    continue

                # Kill the task process if possible
                kill_success = False
                if self.task_runner:
                    try:
                        kill_success = self.task_runner.kill_task(task.task_id)
                    except Exception as e:
                        logger.error(f"Error while killing TASK {format_task_ref(task)}: {e}", exc_info=True)

                if kill_success:
                    logger.info(f"Killed TASK {format_task_ref(task)} due to severe error")
                else:
                    logger.warning(f"Could not terminate TASK {format_task_ref(task)} (maybe already exited)")

                if offending_task_id and task.task_id != offending_task_id:
                    # Collateral task: requeue instead of failing
                    task.error_message = (
                        f"Killed due to GPU {gpu_id} severe error {trigger_note}"
                    )
                    collateral_tasks.append(task)
                else:
                    _mark_task_failed(task)

            # Requeue collateral tasks at front in reverse order to maintain order
            for task in reversed(collateral_tasks):
                task.start_timestamp = None
                task.end_timestamp = None
                task.exit_code = None
                task.stdout_size = 0
                task.stderr_size = 0
                task.error_message = task.error_message or (
                    f"Requeued after GPU {gpu_id} severe error {trigger_note}"
                )
                self.task_queue.push_front(task)
                logger.info(f"Requeued collateral TASK {format_task_ref(task)} after severe error on GPU {gpu_id}")

            # Clear running tasks from GPU
            with self.lock:
                if gpu_id in self.gpus:
                    cleared_count = len(self.gpus[gpu_id].running_tasks)
                    self.gpus[gpu_id].running_tasks.clear()
                    logger.info(
                        f"Cleared {cleared_count} tasks from GPU {gpu_id}")

        logger.error(
            f"SEVERE ERROR handling complete. System paused for {config.error_pause_duration}s.")

    def clear_severe_error(self):
        """Clear severe error state and resume operations."""
        with self.lock:
            self.severe_error_active = False
            self.severe_error_timestamp = None
            self.severe_error_monotonic_ns = None
            recovered_gpus = list(self.error_gpus)
            for error_gpu_id in recovered_gpus:
                gpu = self.gpus.get(error_gpu_id)
                if gpu:
                    gpu.status = GPUStatus.ONLINE
                    gpu.error_message = None
                    logger.info(f"GPU {error_gpu_id} recovered from severe error and is back online")
            self.error_gpus.clear()
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
            logger.debug(
                f"Failed to get memory info for GPU {gpu_id}: {e}", exc_info=True)

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
                if self.severe_error_active and self.severe_error_monotonic_ns is not None:
                    elapsed_ms = monotonic_elapsed_ms(self.severe_error_monotonic_ns)
                    if elapsed_ms > config.error_pause_duration * 1000:
                        logger.info(
                            f"Severe error pause duration ({config.error_pause_duration}s) elapsed, auto-resuming")
                        self.clear_severe_error()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

            time.sleep(config.gpu_monitor_interval)

    def start_monitoring(self):
        """Start background GPU monitoring."""
        if self.running:
            logger.warning("Monitoring already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
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
