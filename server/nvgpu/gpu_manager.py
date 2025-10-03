"""GPU manager for monitoring and managing GPU resources."""
import threading
import time
from typing import Dict, Optional, List
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
        self.gpus: Dict[int, GPU] = {}
        self.lock = threading.RLock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.nvml_initialized = False
        
        # Severe error handling
        self.severe_error_active = False
        self.severe_error_time: Optional[datetime] = None
        
        self._initialize_nvml()
    
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
    
    def set_gpu_mode(self, gpu_id: int, mode: GPUMode) -> bool:
        """Set GPU execution mode."""
        with self.lock:
            if gpu_id not in self.gpus:
                return False
            
            self.gpus[gpu_id].mode = mode
            logger.info(f"GPU {gpu_id} mode changed to {mode.value}")
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
    
    def get_gpu(self, gpu_id: int) -> Optional[GPU]:
        """Get GPU info."""
        with self.lock:
            return self.gpus.get(gpu_id)
    
    def list_gpus(self) -> List[GPU]:
        """List all registered GPUs."""
        with self.lock:
            return list(self.gpus.values())
    
    def find_available_gpu(self, preferred_gpu: Optional[int] = None) -> Optional[int]:
        """Find an available GPU for task assignment.
        
        Updates GPU memory usage before checking availability to ensure
        real-time memory information is used for scheduling decisions.
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
                if gpu and gpu.can_accept_task():
                    return preferred_gpu
            return None
        
        # Find any available GPU
        # First update all GPU memory usages to get real-time data
        with self.lock:
            gpu_ids = list(self.gpus.keys())
        
        for gpu_id in gpu_ids:
            self._update_gpu_memory(gpu_id)
        
        # Now check which GPUs can accept tasks
        with self.lock:
            for gpu_id, gpu in self.gpus.items():
                if gpu.can_accept_task():
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
        """Trigger severe error state, pausing all tasks."""
        with self.lock:
            self.severe_error_active = True
            self.severe_error_time = datetime.now()
            
            if gpu_id in self.gpus:
                self.gpus[gpu_id].status = GPUStatus.ERROR
                self.gpus[gpu_id].error_message = error_msg
                self.gpus[gpu_id].last_error_time = self.severe_error_time
            
            logger.error(f"SEVERE ERROR on GPU {gpu_id}: {error_msg}. All tasks paused.")
    
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

