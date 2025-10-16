"""Task runner for executing Python scripts on GPU."""
import os
import subprocess
import sys
import signal
import threading
from pathlib import Path

from models import Task, TaskStatus
from config import config
from logger import setup_logger
from profiler.timer import HostTimer, TimerSample, create_host_timer
from utils.timezone import now

logger = setup_logger("task_runner")

# Define NVGPU root directory (task_runner.py's parent directory)
NVGPU_ROOT = Path(__file__).parent.resolve()

# Global GPU config loader (set by main)
gpu_config_loader = None


def _build_task_host_timer(label: str, task_id: str | None = None) -> tuple[HostTimer, list[TimerSample]]:
    """Create a host-side timer for task runner operations."""

    captured: list[TimerSample] = []

    def _report(sample: TimerSample) -> None:
        captured.append(sample)
        status = "error" if sample.error else "ok"
        prefix = f"task={task_id[:8]} " if task_id else ""
        logger.debug(
            "%slabel=%s duration_ms=%.3f status=%s",
            prefix,
            sample.label,
            sample.duration_ms,
            status,
        )

    return create_host_timer(reporter=_report), captured


class TaskRunner:
    """Executes tasks as Python subprocess."""
    
    def __init__(self):
        self.running_processes: dict[str, subprocess.Popen] = {}  # task_id -> subprocess.Popen
        self.lock = threading.RLock()
    
    def run_task(self, task: Task, gpu_id: int) -> bool:
        """
        Execute a task on specified GPU.
        Returns True if task started successfully, False otherwise.
        """
        stdout_file = None
        stderr_file = None
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.assigned_gpu = gpu_id
            task.start_time = now()
            
            # Get absolute path of script
            script_abs_path = os.path.abspath(task.script_path)
            
            logger.info(f"Starting task {task.task_id} on GPU {gpu_id}")
            logger.info(f"  Script: {script_abs_path}")
            
            # Prepare environment
            env = os.environ.copy()
            
            # Get CUDA visible ID from config if available
            cuda_id = gpu_id
            if gpu_config_loader:
                cuda_visible_id = gpu_config_loader.get_cuda_visible_id(gpu_id)
                if cuda_visible_id is not None:
                    cuda_id = cuda_visible_id
                    logger.debug(f"GPU {gpu_id} mapped to CUDA_VISIBLE_DEVICES={cuda_id}")
            
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            
            # Merge custom environment variables
            if task.env:
                env.update(task.env)
            
            # Prepare log directory and files (use absolute path)
            log_dir = NVGPU_ROOT / "logs" / "tasks"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            task.log_file = str(log_dir / f"{task.task_id}.log")
            stdout_path = log_dir / f"{task.task_id}.stdout"
            stderr_path = log_dir / f"{task.task_id}.stderr"
            
            # Prepare command
            cmd = [sys.executable, script_abs_path] + task.args
            
            logger.debug(f"Task {task.task_id} command: {' '.join(cmd)}")
            logger.debug(f"Task {task.task_id} work_dir: {task.work_dir}")
            logger.debug(f"Task {task.task_id} CUDA_VISIBLE_DEVICES: {cuda_id}")
            
            # Open files for stdout and stderr
            stdout_file = open(stdout_path, 'w', buffering=1)  # Line buffered
            stderr_file = open(stderr_path, 'w', buffering=1)  # Line buffered
            
            # Execute subprocess with direct file output
            process = subprocess.Popen(
                cmd,
                cwd=task.work_dir,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file
            )
            
            # Register running process
            with self.lock:
                self.running_processes[task.task_id] = process
            
            timer, samples = _build_task_host_timer("task_runner.wait", task.task_id)
            try:
                # Wait for completion with timeout
                with timer.time("task_runner.wait"):
                    exit_code = process.wait(timeout=config.task_timeout)
            except subprocess.TimeoutExpired:
                # Kill the process on timeout
                logger.warning(f"Task {task.task_id} timed out, terminating...")
                process.kill()
                process.wait()  # Wait for process to be killed
                raise
            finally:
                if samples:
                    task.host_timing_ms["process_wait_ms"] = samples[-1].duration_ms
                # Unregister process
                with self.lock:
                    self.running_processes.pop(task.task_id, None)
            
            # Close files and get sizes
            stdout_file.close()
            stderr_file.close()
            stdout_file = None
            stderr_file = None
            
            task.stdout_size = stdout_path.stat().st_size
            task.stderr_size = stderr_path.stat().st_size
            
            # Save results
            task.exit_code = exit_code
            task.end_time = now()
            
            # Determine success and set final status BEFORE writing log
            if exit_code == 0:
                task.status = TaskStatus.COMPLETED
                logger.info(f"Task {task.task_id} completed successfully (exit_code=0)")
            elif exit_code < 0:
                # Negative exit code indicates signal termination (e.g., SIGSEGV = -11)
                signal_name = self._get_signal_name(abs(exit_code))
                # Don't override CANCELLED status (set by force_cancel_task)
                if task.status != TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Process terminated by signal {signal_name} ({exit_code})"
                logger.error(f"Task {task.task_id} terminated by signal {signal_name} ({exit_code})")
            else:
                # Don't override CANCELLED status
                if task.status != TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Exit code {exit_code}"
                logger.warning(f"Task {task.task_id} failed with exit code {exit_code}")
            
            # Write summary log file AFTER status is finalized
            self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path)
            
            return True
            
        except subprocess.TimeoutExpired as e:
            task.status = TaskStatus.FAILED
            task.error_message = f"Timeout after {config.task_timeout} seconds"
            task.end_time = now()
            logger.error(f"Task {task.task_id} timed out after {config.task_timeout}s")
            
            # Close files if still open
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()
            
            # Get sizes if files exist
            try:
                if stdout_path.exists():
                    task.stdout_size = stdout_path.stat().st_size
                if stderr_path.exists():
                    task.stderr_size = stderr_path.stat().st_size
            except:
                pass
            
            self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path, timeout=True)
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = now()
            logger.error(f"Task {task.task_id} failed with exception: {e}")
            
            # Close files if still open
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()
            
            # Get sizes if files exist
            try:
                if stdout_path and stdout_path.exists():
                    task.stdout_size = stdout_path.stat().st_size
                if stderr_path and stderr_path.exists():
                    task.stderr_size = stderr_path.stat().st_size
            except:
                pass
            
            self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path, exception=str(e))
            return False
    
    def _get_signal_name(self, signum: int) -> str:
        """Get signal name from signal number."""
        signal_names = {
            1: "SIGHUP",
            2: "SIGINT",
            3: "SIGQUIT",
            4: "SIGILL",
            6: "SIGABRT",
            7: "SIGBUS",
            8: "SIGFPE",
            9: "SIGKILL",
            11: "SIGSEGV",  # Segmentation fault
            13: "SIGPIPE",
            14: "SIGALRM",
            15: "SIGTERM",
        }
        return signal_names.get(signum, f"SIG{signum}")
    
    def _write_log_file(self, task: Task, script_abs_path: str, cmd: list, 
                       stdout_path: Path, stderr_path: Path,
                       timeout: bool = False, exception: str | None = None):
        """Write task execution summary log to file."""
        if not task.log_file:
            return
        
        try:
            with open(task.log_file, 'w') as f:
                f.write(f"=== Task {task.task_id} ===\n")
                f.write(f"Mode: {task.task_mode.value}\n")
                if task.task_type:
                    f.write(f"Type: {task.task_type.value}\n")
                if task.task_label:
                    f.write(f"Label: {task.task_label}\n")
                f.write(f"Script: {script_abs_path}\n")
                f.write(f"Work Dir: {os.path.abspath(task.work_dir)}\n")
                f.write(f"GPU: {task.assigned_gpu}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"\n=== Timing ===\n")
                f.write(f"Submit Time:  {task.submit_time}\n")
                if task.queued_time:
                    f.write(f"Queued Time:  {task.queued_time}\n")
                if task.start_time:
                    f.write(f"Start Time:   {task.start_time}\n")
                if task.end_time:
                    f.write(f"End Time:     {task.end_time}\n")
                
                f.write(f"\n=== Timing Breakdown (milliseconds) ===\n")
                if task.pending_time_ms is not None:
                    f.write(f"Pending Time:     {task.pending_time_ms:>10.2f} ms  (submit → GPU assignment)\n")
                if task.queue_time_ms is not None:
                    f.write(f"Queue Time:       {task.queue_time_ms:>10.2f} ms  (GPU assignment → execution start)\n")
                if task.waiting_time_ms is not None:
                    f.write(f"Total Waiting:    {task.waiting_time_ms:>10.2f} ms  (submit → execution start)\n")
                if task.execution_time_ms is not None:
                    f.write(f"Execution Time:   {task.execution_time_ms:>10.2f} ms  (execution start → end)\n")
                if task.total_time_ms is not None:
                    f.write(f"Total Time:       {task.total_time_ms:>10.2f} ms  (submit → end)\n")

                if task.host_timing_ms:
                    f.write(f"\n=== Host Timer (milliseconds) ===\n")
                    for key, value in task.host_timing_ms.items():
                        f.write(f"{key}: {value:>10.2f} ms\n")

                f.write(f"\n=== Result ===\n")
                if timeout:
                    f.write(f"Status: TIMEOUT\n")
                    f.write(f"ERROR: Task timed out after {config.task_timeout} seconds\n")
                elif exception:
                    f.write(f"Status: EXCEPTION\n")
                    f.write(f"ERROR: Exception occurred: {exception}\n")
                else:
                    f.write(f"Status: {task.status.value}\n")
                    f.write(f"Exit Code: {task.exit_code}\n")
                    
                    if task.exit_code and task.exit_code < 0:
                        signal_name = self._get_signal_name(abs(task.exit_code))
                        f.write(f"Signal: {signal_name}\n")
                        f.write(f"Note: Process was terminated by signal (possible segfault or core dump)\n")
                
                f.write(f"\n=== Output Files ===\n")
                f.write(f"STDOUT: {stdout_path} ({self._format_size(task.stdout_size)})\n")
                f.write(f"STDERR: {stderr_path} ({self._format_size(task.stderr_size)})\n")
                
                # Include small outputs inline, reference large ones
                if task.stdout_size > 0:
                    f.write(f"\n=== STDOUT Preview ===\n")
                    if task.stdout_size < 10240:  # < 10KB, include all
                        with open(stdout_path, 'r') as stdout_f:
                            f.write(stdout_f.read())
                    else:
                        # Include first 5KB
                        with open(stdout_path, 'r') as stdout_f:
                            preview = stdout_f.read(5120)
                            f.write(preview)
                            f.write(f"\n\n... (truncated, {self._format_size(task.stdout_size - 5120)} remaining)\n")
                            f.write(f"See full output in: {stdout_path}\n")
                else:
                    f.write(f"\n=== STDOUT ===\n(empty)\n")
                
                if task.stderr_size > 0:
                    f.write(f"\n=== STDERR Preview ===\n")
                    if task.stderr_size < 10240:  # < 10KB, include all
                        with open(stderr_path, 'r') as stderr_f:
                            f.write(stderr_f.read())
                    else:
                        # Include first 5KB
                        with open(stderr_path, 'r') as stderr_f:
                            preview = stderr_f.read(5120)
                            f.write(preview)
                            f.write(f"\n\n... (truncated, {self._format_size(task.stderr_size - 5120)} remaining)\n")
                            f.write(f"See full output in: {stderr_path}\n")
                else:
                    f.write(f"\n=== STDERR ===\n(empty)\n")
                
                f.write("\n=== END OF LOG ===\n")
            if task.task_label:
                logger.debug(f"Task {task.task_label}::{task.task_id} log written to {task.log_file}")
            else:
                logger.debug(f"Task {task.task_id} log written to {task.log_file}")
        except Exception as e:
            logger.error(f"Failed to write log file for task {task.task_id}: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
    
    def kill_task(self, task_id: str) -> bool:
        """Kill a running task.
        
        Args:
            task_id: Task ID to kill
            
        Returns:
            True if task was killed, False if not found or already finished
        """
        with self.lock:
            process = self.running_processes.get(task_id)
            if not process:
                logger.warning(f"Cannot kill task {task_id}: not running or already finished")
                return False
            
            try:
                # Try graceful termination first
                logger.info(f"Terminating task {task_id}...")
                process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"Task {task_id} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if termination didn't work
                    logger.warning(f"Task {task_id} did not terminate, sending SIGKILL...")
                    process.kill()
                    process.wait()
                    logger.info(f"Task {task_id} killed forcefully")
                
                return True
            except Exception as e:
                logger.error(f"Failed to kill task {task_id}: {e}")
                return False
