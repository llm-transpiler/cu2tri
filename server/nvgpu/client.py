"""Python client for NVGPU Server."""
import requests
import time
from typing import Any
from dataclasses import dataclass

from profiler.timer import monotonic_elapsed_ms, monotonic_timestamp_ns


@dataclass
class TaskResult:
    """Task result."""
    task_id: str
    status: str
    task_mode: str | None = None  # Task mode (exclusive/shared)
    task_type: str | None = None  # Task type (functional/performance/both)
    task_label: str | None = None  # Task label
    exit_code: int | None = None
    log_file: str | None = None
    error_message: str | None = None
    submit_timestamp: str | None = None
    queued_timestamp: str | None = None  # Time when assigned to GPU queue
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    stdout_size: int = 0
    stderr_size: int = 0
    gpu_id: int | None = None
    # Computed timing fields (in milliseconds, 2 decimal places)
    pending_duration_ms: float | None = None
    queue_duration_ms: float | None = None
    waiting_duration_ms: float | None = None
    running_duration_ms: float | None = None
    total_duration_ms: float | None = None
    execution_duration_ms: dict[str, float] | None = None

class NVGPUClient:
    """Client for interacting with NVGPU Server."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """Initialize client.
        
        Args:
            base_url: Base URL of the NVGPU server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def submit_task(
        self,
        script_path: str,
        task_mode: str | None = None,
        task_type: str | None = None,
        task_label: str | None = None,
        work_dir: str = ".",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        gpu_id: int | None = None
    ) -> str:
        """Submit a task to the server.
        
        Smart defaults:
        - task_type="functional" -> task_mode="shared" (if not specified)
        - task_type="performance" -> task_mode="exclusive" (if not specified)
        - task_type="both" -> task_mode="exclusive" (if not specified)
        - No task_type/mode -> task_mode="shared" (safe default)
        
        Args:
            script_path: Path to the Python script to run
            task_mode: Task mode ("exclusive" or "shared"), controls GPU behavior.
                      Optional - smart defaults applied based on task_type.
            task_type: Business categorization ("functional", "performance", "both").
                      Optional - for statistics and filtering.
            task_label: Specific identification tag (e.g., "xpiler_cuda/add_3_3_256/cuda_vs_triton").
                       Optional - for precise identification.
            work_dir: Working directory for the script
            args: Command line arguments for the script
            env: Environment variables
            gpu_id: Specific GPU ID, or None for auto-assignment
            
        Returns:
            Task ID
            
        Raises:
            RuntimeError: If submission fails
            
        Examples:
            # Simple usage (smart defaults)
            submit_task("test.py")  # shared mode
            submit_task("test.py", task_type="functional")  # shared mode
            submit_task("test.py", task_type="performance")  # exclusive mode
            
            # With specific label
            submit_task("test.py", 
                       task_type="functional",
                       task_label="xpiler_cuda/add_3_3_256/cuda_vs_triton")
            
            # Explicit control
            submit_task("test.py", task_mode="exclusive", task_type="functional")
        """
        payload = {
            "script_path": script_path,
            "work_dir": work_dir,
            "args": args or [],
        }
        
        # Add optional parameters only if provided
        if task_mode is not None:
            payload["task_mode"] = task_mode
        if task_type is not None:
            payload["task_type"] = task_type
        if task_label is not None:
            payload["task_label"] = task_label
        if env:
            payload["env"] = env
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        
        response = self.session.post(
            f"{self.base_url}/tasks",
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to submit task: {response.text}")
        
        result = response.json()
        return result["task_id"]
    
    def submit_task_in_script_dir(
        self,
        script_path: str,
        task_mode: str | None = None,
        task_type: str | None = None,
        task_label: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        gpu_id: int | None = None
    ) -> str:
        """Submit task with work_dir automatically set to script's directory.
        
        This is a convenience method that automatically uses the script's
        parent directory as the working directory. Useful when the script
        needs to access files in its own directory.
        
        Args:
            script_path: Path to the Python script to run
            task_mode: Task mode ("exclusive" or "shared"), controls GPU behavior
            task_type: Business categorization ("functional", "performance", "both")
            task_label: Specific identification tag
            args: Command line arguments for the script
            env: Environment variables
            gpu_id: Specific GPU ID, or None for auto-assignment
            
        Returns:
            Task ID
            
        Example:
            >>> # These are equivalent:
            >>> client.submit_task(
            ...     script_path="/workspace/test/script.py",
            ...     work_dir="/workspace/test"
            ... )
            >>> client.submit_task_in_script_dir(
            ...     script_path="/workspace/test/script.py"
            ... )
        """
        from pathlib import Path
        work_dir = str(Path(script_path).parent.absolute())
        
        return self.submit_task(
            script_path=script_path,
            task_mode=task_mode,
            task_type=task_type,
            task_label=task_label,
            work_dir=work_dir,
            args=args,
            env=env,
            gpu_id=gpu_id
        )
    
    def get_task(self, task_id: str) -> TaskResult:
        """Get task status and results.
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskResult object
            
        Raises:
            RuntimeError: If task not found
        """
        response = self.session.get(f"{self.base_url}/tasks/{task_id}", timeout=10)
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get task: {response.text}")
        
        data = response.json()
        return TaskResult(
            task_id=data["task_id"],
            status=data["status"],
            task_mode=data.get("task_mode"),
            task_type=data.get("task_type"),
            task_label=data.get("task_label"),
            exit_code=data.get("exit_code"),
            log_file=data.get("log_file"),
            error_message=data.get("error_message"),
            submit_timestamp=data.get("submit_timestamp"),
            queued_timestamp=data.get("queued_timestamp"),
            start_timestamp=data.get("start_timestamp"),
            end_timestamp=data.get("end_timestamp"),
            stdout_size=data.get("stdout_size", 0),
            stderr_size=data.get("stderr_size", 0),
            gpu_id=data.get("assigned_gpu"),
            pending_duration_ms=data.get("pending_duration_ms"),
            queue_duration_ms=data.get("queue_duration_ms"),
            waiting_duration_ms=data.get("waiting_duration_ms"),
            running_duration_ms=data.get("running_duration_ms"),
            total_duration_ms=data.get("total_duration_ms"),
            execution_duration_ms=data.get("execution_duration_ms"),
        )
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
        poll_interval: float = 2.0
    ) -> TaskResult:
        """Wait for task to complete.
        
        Args:
            task_id: Task ID
            timeout: Maximum time to wait in seconds, None for no timeout
            poll_interval: Polling interval in seconds
            
        Returns:
            TaskResult object
            
        Raises:
            TimeoutError: If timeout is reached
            RuntimeError: If task fails
        """
        start_ns = monotonic_timestamp_ns()
        
        while True:
            result = self.get_task(task_id)
            
            if result.status in ["completed", "failed", "cancelled"]:
                return result
            
            if timeout and monotonic_elapsed_ms(start_ns) > timeout * 1000:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def cancel_task(self, task_id: str, force: bool = False) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            force: If True, will kill running tasks. If False, only cancels pending/queued tasks.
            
        Returns:
            True if cancelled successfully
            
        Raises:
            RuntimeError: If cancellation fails (e.g., task not found, task in wrong state)
        """
        response = self.session.post(
            f"{self.base_url}/tasks/{task_id}/cancel",
            params={"force": force},
            timeout=10
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to cancel task: {response.text}")
        
        return True
    
    def list_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        """List all tasks.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of task dictionaries
        """
        url = f"{self.base_url}/tasks"
        if status:
            url += f"?status={status}"
        
        response = self.session.get(url, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list tasks: {response.text}")
        
        return response.json()["tasks"]
    
    def list_gpus(self) -> list[dict[str, Any]]:
        """List all GPUs.
        
        Returns:
            List of GPU dictionaries
        """
        response = self.session.get(f"{self.base_url}/gpus", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list GPUs: {response.text}")
        
        return response.json()["gpus"]
    
    def get_gpu(self, gpu_id: int) -> dict[str, Any]:
        """Get GPU information.
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            GPU dictionary
        """
        response = self.session.get(f"{self.base_url}/gpus/{gpu_id}", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get GPU: {response.text}")
        
        return response.json()
    
    def set_gpu_status(self, gpu_id: int, status: str) -> bool:
        """Set GPU status.
        
        Args:
            gpu_id: GPU ID
            status: Status ("online", "offline", "maintenance")
            
        Returns:
            True if successful
        """
        response = self.session.put(
            f"{self.base_url}/gpus/{gpu_id}/status",
            json={"status": status},
            timeout=10
        )
        return response.status_code == 200
    
    def set_gpu_mode(self, gpu_id: int, mode: str, manual: bool = True) -> bool:
        """Set GPU mode.
        
        Args:
            gpu_id: GPU ID
            mode: Mode ("exclusive" or "shared")
            manual: If True, sets as manual override that persists.
                   If False, just changes current mode (task-driven).
            
        Returns:
            True if successful
        """
        response = self.session.put(
            f"{self.base_url}/gpus/{gpu_id}/mode",
            json={"mode": mode, "manual": manual},
            timeout=10
        )
        return response.status_code == 200
    
    def clear_gpu_manual_mode(self, gpu_id: int) -> bool:
        """Clear manual mode override for a GPU.
        
        This allows the GPU to use task-driven mode switching.
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            True if successful
        """
        response = self.session.delete(
            f"{self.base_url}/gpus/{gpu_id}/mode",
            timeout=10
        )
        return response.status_code == 200
    
    def set_gpu_memory_threshold(self, gpu_id: int, threshold: float) -> bool:
        """Set GPU memory threshold.
        
        Args:
            gpu_id: GPU ID
            threshold: Memory threshold (0-1)
            
        Returns:
            True if successful
        """
        response = self.session.put(
            f"{self.base_url}/gpus/{gpu_id}/memory_threshold",
            json={"threshold": threshold},
            timeout=10
        )
        return response.status_code == 200
    
    def set_gpu_max_concurrent_tasks(self, gpu_id: int, max_tasks: int) -> bool:
        """Set GPU maximum concurrent tasks.
        
        Args:
            gpu_id: GPU ID
            max_tasks: Maximum concurrent tasks
            
        Returns:
            True if successful
        """
        response = self.session.put(
            f"{self.base_url}/gpus/{gpu_id}/max_concurrent_tasks",
            json={"max_tasks": max_tasks},
            timeout=10
        )
        return response.status_code == 200
    
    def get_stats(self) -> dict[str, Any]:
        """Get server statistics.
        
        Returns:
            Statistics dictionary
        """
        response = self.session.get(f"{self.base_url}/stats", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get stats: {response.text}")
        
        return response.json()
    
    def register_gpu(self, gpu_id: int, mode: str = "shared", memory_threshold: float = 0.75) -> bool:
        """Register a new GPU.
        
        Args:
            gpu_id: GPU ID
            mode: GPU mode
            memory_threshold: Memory threshold
            
        Returns:
            True if successful
        """
        response = self.session.post(
            f"{self.base_url}/gpus/register",
            json={
                "gpu_id": gpu_id,
                "mode": mode,
                "memory_threshold": memory_threshold
            },
            timeout=10
        )
        return response.status_code == 200
    
    def unregister_gpu(self, gpu_id: int) -> bool:
        """Unregister a GPU.
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            True if successful
        """
        response = self.session.post(
            f"{self.base_url}/gpus/{gpu_id}/unregister",
            timeout=10
        )
        return response.status_code == 200
    
    def clear_severe_error(self) -> bool:
        """Clear severe error state.
        
        Returns:
            True if successful
        """
        response = self.session.post(
            f"{self.base_url}/gpus/clear_error",
            timeout=10
        )
        return response.status_code == 200
    
    def get_task_log(
        self,
        task_id: str,
        log_type: str = "summary",
        offset: int = 0,
        limit: int = 102400
    ) -> dict[str, Any]:
        """Get task log content.
        
        Args:
            task_id: Task ID
            log_type: Log type (summary, stdout, stderr)
            offset: Byte offset to start reading from
            limit: Maximum bytes to read (default 100KB)
            
        Returns:
            Dictionary with log content and metadata:
            - task_id: Task ID
            - log_type: Type of log
            - log_path: Absolute path to log file
            - content: Log content (string)
            - total_size: Total size of log file in bytes
            - offset: Starting offset of returned content
            - size: Size of returned content in bytes
            - truncated: Whether content was truncated
            - has_more: Whether there is more content to read
            
        Raises:
            RuntimeError: If request fails
        """
        response = self.session.get(
            f"{self.base_url}/tasks/{task_id}/log",
            params={
                "log_type": log_type,
                "offset": offset,
                "limit": limit
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get task log: {response.text}")
        
        return response.json()
    
    def get_full_task_log(
        self,
        task_id: str,
        log_type: str = "summary",
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> str:
        """Get full task log content, fetching in chunks if necessary.
        
        Args:
            task_id: Task ID
            log_type: Log type (summary, stdout, stderr)
            chunk_size: Size of chunks to fetch (default 1MB)
            
        Returns:
            Full log content as string
            
        Raises:
            RuntimeError: If request fails
        """
        content_parts = []
        offset = 0
        
        while True:
            result = self.get_task_log(task_id, log_type, offset, chunk_size)
            content_parts.append(result["content"])
            
            if not result.get("has_more", False):
                break
            
            offset += result["size"]
        
        return "".join(content_parts)
