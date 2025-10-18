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
from utils.timezone import now_timestamp
from utils.task_refs import format_task_ref

logger = setup_logger("task_runner")

# Define NVGPU root directory (task_runner.py's parent directory)
NVGPU_ROOT = Path(__file__).parent.resolve()

# Global GPU config loader (set by main)
gpu_config_loader = None


def _create_task_timer(task: Task) -> tuple[HostTimer, list[TimerSample]]:
    """Create a task-scoped timer for task runner operations."""

    captured: list[TimerSample] = []

    def _report(sample: TimerSample) -> None:
        captured.append(sample)
        status = "err" if sample.error else "ok"
        prefix = ""
        if task:
            prefix = f"task={format_task_ref(task)} "
        logger.debug(
            "%stimer_label=%s, duration_ms=%12.3fms, status=%s",
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
        
        running_timer_started = False

        def append_error_message(message: str):
            """Attach an error message without overwriting existing context."""
            if not message:
                return
            if task.error_message:
                if message not in task.error_message:
                    task.error_message = f"{task.error_message} | {message}"
            else:
                task.error_message = message
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.assigned_gpu = gpu_id
            task.start_timestamp = now_timestamp()
            task.error_message = None
            task_ref_str = format_task_ref(task)
            
            # Stop queue/wait timers and record durations
            queue_duration = task.timer.stop("queue")
            if queue_duration is not None:
                task.phase_duration_ms["queue"] = queue_duration
            waiting_duration = task.timer.stop("waiting")
            if waiting_duration is not None:
                task.phase_duration_ms["waiting"] = waiting_duration
            
            # Get absolute path of script
            script_abs_path = os.path.abspath(task.script_path)
            
            logger.info(f"Starting task {task_ref_str} on GPU {gpu_id}")
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
            
            logger.debug(f"TASK {task_ref_str} command: {' '.join(cmd)}")
            task_work_dir = os.path.abspath(task.work_dir)
            logger.debug(f"TASK {task_ref_str} work_dir: {task_work_dir}")
            logger.debug(f"TASK {task_ref_str} CUDA_VISIBLE_DEVICES: {cuda_id}")
            
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
            
            # Start execution timer once subprocess is spawned
            task.timer.start("running")
            running_timer_started = True
            
            # Register running process
            with self.lock:
                self.running_processes[task.task_id] = process
            
            timer, samples = _create_task_timer(task)
            try:
                # Wait for completion with timeout
                with timer.time("task_runner.process_run"):
                    exit_code = process.wait(timeout=config.task_timeout)
            except subprocess.TimeoutExpired:
                # Kill the process on timeout
                logger.warning(f"TASK {task_ref_str} timed out, terminating...", exc_info=True)
                process.kill()
                process.wait()  # Wait for process to be killed
                raise
            finally:
                if samples:
                    task.execution_duration_ms = samples[-1].duration_ms
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
            task.end_timestamp = now_timestamp()
            
            # Determine success and set final status BEFORE writing log
            if exit_code == 0:
                task.status = TaskStatus.COMPLETED
                logger.info(f"TASK {task_ref_str} completed successfully (exit_code=0)")
            elif exit_code < 0:
                # Negative exit code indicates signal termination (e.g., SIGSEGV = -11)
                signal_name = self._get_signal_name(abs(exit_code))
                # Don't override CANCELLED status (set by force_cancel_task)
                if task.status != TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED
                    append_error_message(f"Process terminated by signal {signal_name} ({exit_code})")
                logger.error(f"TASK {task_ref_str} terminated by signal {signal_name} ({exit_code})")
            else:
                # Don't override CANCELLED status
                if task.status != TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED
                    append_error_message(f"Exit code {exit_code}")
                logger.warning(f"TASK {task_ref_str} failed with exit code {exit_code}")
            
            # Write summary log file AFTER status is finalized
            self._write_log_file(task, script_abs_path, cmd, stdout_path, stderr_path)
            
            return True
            
        except subprocess.TimeoutExpired as e:
            task.status = TaskStatus.FAILED
            append_error_message(f"Timeout after {config.task_timeout} seconds")
            task.end_timestamp = now_timestamp()
            logger.error(f"TASK {task_ref_str} timed out after {config.task_timeout}s", exc_info=True)
            
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
            append_error_message(str(e))
            task.end_timestamp = now_timestamp()
            logger.error(f"TASK {task_ref_str} failed with exception: {e}", exc_info=True)
            
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
        finally:
            if running_timer_started:
                running_duration = task.timer.stop("running")
                if running_duration is not None:
                    task.phase_duration_ms["running"] = running_duration
            total_duration = task.timer.stop("total")
            if total_duration is not None:
                task.phase_duration_ms["total"] = total_duration
    
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
            task_ref = format_task_ref(task)
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
                f.write(f"Submit Timestamp:  {task.submit_timestamp}\n")
                if task.queued_timestamp:
                    f.write(f"Queued Timestamp:  {task.queued_timestamp}\n")
                if task.start_timestamp:
                    f.write(f"Start Timestamp:   {task.start_timestamp}\n")
                if task.end_timestamp:
                    f.write(f"End Timestamp:     {task.end_timestamp}\n")
                
                f.write(f"\n=== Timing Breakdown (ms) ===\n")
                if task.pending_duration_ms is not None:
                    f.write(f"Pending Duration:     {task.pending_duration_ms:>10.2f} ms  ( submit          -> gpu assignment  )\n")
                if task.queue_duration_ms is not None:
                    f.write(f"Queue Duration:       {task.queue_duration_ms:>10.2f} ms  ( gpu assignment  -> execution start )\n")
                if task.waiting_duration_ms is not None:
                    f.write(f"Waiting Duration:     {task.waiting_duration_ms:>10.2f} ms  ( submit          -> execution start )\n")
                if task.running_duration_ms is not None:
                    f.write(f"Running Duration:     {task.running_duration_ms:>10.2f} ms  ( execution start -> end             )\n")
                if task.total_duration_ms is not None:
                    f.write(f"Total Duration:       {task.total_duration_ms:>10.2f} ms  ( submit          -> end             )\n")

                if task.execution_duration_ms:
                    f.write(f"\n=== Execution Duration (ms) ===\n")
                    f.write(f"Process Running Duration: {task.execution_duration_ms:>10.2f} ms\n")

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
            logger.debug(f"TASK {task_ref} log written to {task.log_file}")
        except Exception as e:
            logger.error(f"Failed to write log file for task {task_ref}: {e}", exc_info=True)
    
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
                    logger.info(f"TASK {task_id} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if termination didn't work
                    logger.warning(f"TASK {task_id} did not terminate, sending SIGKILL...", exc_info=True)
                    process.kill()
                    process.wait()
                    logger.info(f"TASK {task_id} killed forcefully", exc_info=True)
                
                return True
            except Exception as e:
                logger.error(f"Failed to kill task {task_id}: {e}", exc_info=True)
                return False
