"""Python client for NVGPU Server."""
import requests
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Task result."""
    task_id: str
    status: str
    exit_code: Optional[int] = None
    log_file: Optional[str] = None
    error_message: Optional[str] = None
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    stdout_size: int = 0
    stderr_size: int = 0
    gpu_id: Optional[int] = None


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
        task_type: str = "functional",
        work_dir: str = ".",
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        gpu_id: Optional[int] = None
    ) -> str:
        """Submit a task to the server.
        
        Args:
            script_path: Path to the Python script to run
            task_type: Task type ("functional" or "performance")
            work_dir: Working directory for the script
            args: Command line arguments for the script
            env: Environment variables
            gpu_id: Specific GPU ID, or None for auto-assignment
            
        Returns:
            Task ID
            
        Raises:
            RuntimeError: If submission fails
        """
        payload = {
            "script_path": script_path,
            "task_type": task_type,
            "work_dir": work_dir,
            "args": args or [],
        }
        
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
            exit_code=data.get("exit_code"),
            log_file=data.get("log_file"),
            error_message=data.get("error_message"),
            submit_time=data.get("submit_time"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            stdout_size=data.get("stdout_size", 0),
            stderr_size=data.get("stderr_size", 0),
            gpu_id=data.get("assigned_gpu"),
        )
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
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
        start_time = time.time()
        
        while True:
            result = self.get_task(task_id)
            
            if result.status in ["completed", "failed", "cancelled"]:
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def cancel_task(self, task_id: str, force: bool = False) -> bool:
        """Cancel a task.
        
        Args:
            task_id: Task ID
            force: If True, will kill running tasks. If False, only cancels pending/queued tasks.
            
        Returns:
            True if cancelled successfully
        """
        response = self.session.post(
            f"{self.base_url}/tasks/{task_id}/cancel",
            params={"force": force},
            timeout=10
        )
        return response.status_code == 200
    
    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
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
    
    def list_gpus(self) -> List[Dict[str, Any]]:
        """List all GPUs.
        
        Returns:
            List of GPU dictionaries
        """
        response = self.session.get(f"{self.base_url}/gpus", timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to list GPUs: {response.text}")
        
        return response.json()["gpus"]
    
    def get_gpu(self, gpu_id: int) -> Dict[str, Any]:
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
    
    def set_gpu_mode(self, gpu_id: int, mode: str) -> bool:
        """Set GPU mode.
        
        Args:
            gpu_id: GPU ID
            mode: Mode ("exclusive" or "shared")
            
        Returns:
            True if successful
        """
        response = self.session.put(
            f"{self.base_url}/gpus/{gpu_id}/mode",
            json={"mode": mode},
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
    
    def get_stats(self) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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

