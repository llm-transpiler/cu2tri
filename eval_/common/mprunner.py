from typing import Callable, List, Any, Dict
import traceback
import signal
import multiprocessing
from pydantic import BaseModel
import tempfile
import sys
import os
import atexit
import threading
import time
import tempfile
import signal
from typing import Optional, Dict, Set

# 设置多进程启动方法为 spawn 以支持 CUDA
multiprocessing.set_start_method('spawn', force=True)

# 全局注册表，用于跟踪活动的OutputCapture实例
_active_captures: Dict[int, 'OutputCapture'] = {}
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _emergency_cleanup():
    """紧急清理函数，在进程异常终止时调用"""
    global _active_captures
    with _cleanup_lock:
        for capture_id, capture in list(_active_captures.items()):
            try:
                capture._emergency_cleanup()
            except Exception as e:
                print(f"Emergency cleanup failed for capture {capture_id}: {e}", file=sys.__stderr__)
        _active_captures.clear()


def _signal_handler(signum, frame):
    """信号处理器，捕获进程终止信号"""
    print(f"Received signal {signum}, performing emergency cleanup...", file=sys.__stderr__)
    _emergency_cleanup()
    # 恢复默认信号处理器并重新发送信号
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _register_cleanup():
    """注册全局清理机制"""
    global _cleanup_registered
    if not _cleanup_registered:
        # 注册atexit清理函数
        atexit.register(_emergency_cleanup)
        
        # 注册信号处理器
        for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            try:
                signal.signal(sig, _signal_handler)
            except (OSError, ValueError):
                # 某些信号可能无法注册（如在线程中）
                pass
        
        _cleanup_registered = True


class SubProcResult(BaseModel):
    subproc_success: bool = False
    result: Any = None
    error: str = ""
    traceback: str = ""
    output_capture: str = ""


class OutputCapture:
    """用于捕获所有输出（包括子进程）的上下文管理器"""
    def __init__(self, output_file_path: str = None, safe_mode: bool = False, 
                 real_time_flush: bool = True, flush_interval: float = 0.1):
        self.captured_output = ""
        self.old_stdout = None
        self.old_stderr = None
        self.output_file_path = output_file_path
        self.file_temp = False
        self.safe_mode = safe_mode
        self.real_time_flush = real_time_flush  # 实时刷新模式
        self.flush_interval = flush_interval    # 刷新间隔（秒）
        self._temp_stdout_file = None
        self._temp_stderr_file = None
        self._capture_id = id(self)
        self._cleanup_done = False
        self._flush_thread = None
        self._stop_flush = threading.Event()
        
        # 注册全局清理机制
        _register_cleanup()

    def __enter__(self):
        global _active_captures
        with _cleanup_lock:
            _active_captures[self._capture_id] = self
            
        if self.safe_mode and self.output_file_path is not None:
            # 安全模式：使用临时文件避免与logger冲突
            return self._enter_safe_mode()
        else:
            # 原始模式：直接重定向到指定文件
            return self._enter_direct_mode()

    def _enter_safe_mode(self):
        """安全模式：使用临时文件，稍后合并到目标文件"""
        import tempfile
        
        # 为stdout和stderr分别创建临时文件
        self._temp_stdout_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_stdout.tmp')
        self._temp_stderr_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_stderr.tmp')
        self._temp_stdout_file.close()
        self._temp_stderr_file.close()
        
        # 保存原始的文件描述符
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)
        
        # 重定向到临时文件
        stdout_fd = os.open(self._temp_stdout_file.name, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        stderr_fd = os.open(self._temp_stderr_file.name, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        
        os.close(stdout_fd)
        os.close(stderr_fd)
        
        # 如果启用实时刷新，启动后台线程
        if self.real_time_flush:
            self._start_flush_thread()
        
        return self

    def _enter_direct_mode(self):
        """直接模式：原始实现"""
        # 创建临时文件用于捕获输出
        if self.output_file_path is None:
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            temp_file.close()
            self.output_file_path = temp_file.name
            self.file_temp = True
        else:
            self.file_temp = False 

        # 保存原始的文件描述符
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)

        # 将stdout和stderr重定向到临时文件
        temp_fd = os.open(self.output_file_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        os.dup2(temp_fd, 1)  # stdout
        os.dup2(temp_fd, 2)  # stderr
        os.close(temp_fd)

        return self

    def _start_flush_thread(self):
        """启动实时刷新线程"""
        def flush_worker():
            while not self._stop_flush.wait(self.flush_interval):
                try:
                    self._flush_to_target()
                except Exception as e:
                    print(f"Flush error: {e}", file=sys.__stderr__)
        
        self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self._flush_thread.start()

    def _flush_to_target(self):
        """将临时文件内容刷新到目标文件"""
        if not self.safe_mode or self.output_file_path is None:
            return
            
        try:
            stdout_content = ""
            stderr_content = ""
            
            # 读取临时文件的增量内容
            if self._temp_stdout_file and os.path.exists(self._temp_stdout_file.name):
                try:
                    with open(self._temp_stdout_file.name, 'r', encoding='utf-8', errors='replace') as f:
                        stdout_content = f.read()
                except Exception:
                    pass
                    
            if self._temp_stderr_file and os.path.exists(self._temp_stderr_file.name):
                try:
                    with open(self._temp_stderr_file.name, 'r', encoding='utf-8', errors='replace') as f:
                        stderr_content = f.read()
                except Exception:
                    pass
            
            # 如果有新内容，写入目标文件
            if stdout_content.strip() or stderr_content.strip():
                self._write_to_target_file(stdout_content, stderr_content, is_partial=True)
                
                # 清空临时文件（保留文件但清空内容）
                try:
                    if stdout_content.strip():
                        open(self._temp_stdout_file.name, 'w').close()
                    if stderr_content.strip():
                        open(self._temp_stderr_file.name, 'w').close()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Flush to target failed: {e}", file=sys.__stderr__)

    def _write_to_target_file(self, stdout_content: str, stderr_content: str, is_partial: bool = False):
        """写入内容到目标文件"""
        import fcntl
        
        # 合并内容，添加标识符以区分输出源
        combined_content = ""
        if stdout_content.strip():
            combined_content += f"=== CAPTURED STDOUT {'(PARTIAL)' if is_partial else ''} ===\n{stdout_content}\n"
        if stderr_content.strip():
            combined_content += f"=== CAPTURED STDERR {'(PARTIAL)' if is_partial else ''} ===\n{stderr_content}\n"
            
        if not combined_content.strip():
            return
            
        # 使用文件锁安全地写入目标文件
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with open(self.output_file_path, 'a', encoding='utf-8') as f:
                    # 尝试获取文件锁
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (OSError, IOError):
                        # 如果无法获取锁，等待后重试
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (2 ** attempt))
                            continue
                        else:
                            # 最后一次尝试，即使无法获取锁也要写入
                            pass
                    
                    # 写入内容，添加时间戳
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    marker = "OutputCapture (Real-time)" if is_partial else "OutputCapture"
                    f.write(f"\n=== {marker} @ {timestamp} ===\n")
                    f.write(combined_content)
                    f.write(f"=== End {marker} @ {timestamp} ===\n\n")
                    f.flush()
                    
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (OSError, IOError):
                        pass
                    
                    break
                
                self.captured_output += combined_content
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Warning: Failed to write captured output to {self.output_file_path}: {e}", file=sys.__stderr__)
                else:
                    time.sleep(retry_delay * (2 ** attempt))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 停止刷新线程
        if self._flush_thread:
            self._stop_flush.set()
            try:
                self._flush_thread.join(timeout=2.0)
            except:
                pass
        
        # 刷新缓冲区
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except:
            pass

        # 恢复原始的文件描述符
        try:
            if self.old_stdout is not None:
                os.dup2(self.old_stdout, 1)
                os.close(self.old_stdout)
            if self.old_stderr is not None:
                os.dup2(self.old_stderr, 2)
                os.close(self.old_stderr)
        except:
            pass

        if self.safe_mode and self.output_file_path is not None:
            self._exit_safe_mode()
        else:
            self._exit_direct_mode()
            
        # 从全局注册表中移除
        global _active_captures
        with _cleanup_lock:
            _active_captures.pop(self._capture_id, None)
        
        self._cleanup_done = True

    def _exit_safe_mode(self):
        """安全模式退出：合并临时文件到目标文件"""
        stdout_content = ""
        stderr_content = ""
        
        # 读取临时文件内容
        try:
            if self._temp_stdout_file and os.path.exists(self._temp_stdout_file.name):
                with open(self._temp_stdout_file.name, 'r', encoding='utf-8', errors='replace') as f:
                    stdout_content = f.read()
        except Exception:
            pass
            
        try:
            if self._temp_stderr_file and os.path.exists(self._temp_stderr_file.name):
                with open(self._temp_stderr_file.name, 'r', encoding='utf-8', errors='replace') as f:
                    stderr_content = f.read()
        except Exception:
            pass
        
        # 合并内容到captured_output
        combined_content = ""
        if stdout_content.strip():
            combined_content += f"=== CAPTURED STDOUT ===\n{stdout_content}\n"
        if stderr_content.strip():
            combined_content += f"=== CAPTURED STDERR ===\n{stderr_content}\n"
            
        self.captured_output = combined_content
        
        # 如果不是实时刷新模式，现在写入目标文件
        if not self.real_time_flush and combined_content.strip():
            self._write_to_target_file(stdout_content, stderr_content, is_partial=False)
        
        # 清理临时文件
        self._cleanup_temp_files()

    def _exit_direct_mode(self):
        """直接模式退出：原始实现"""
        # 读取捕获的输出，使用错误处理来处理二进制数据
        try:
            with open(self.output_file_path, 'r', encoding='utf-8') as f:
                self.captured_output = f.read()
        except UnicodeDecodeError:
            # 如果有二进制数据，使用错误替换模式
            with open(self.output_file_path, 'r', encoding='utf-8', errors='replace') as f:
                self.captured_output = f.read()

        # 清理临时文件
        if self.file_temp:
            try:
                os.unlink(self.output_file_path)
            except:
                pass

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if self._temp_stdout_file and os.path.exists(self._temp_stdout_file.name):
                os.unlink(self._temp_stdout_file.name)
        except Exception:
            pass
            
        try:
            if self._temp_stderr_file and os.path.exists(self._temp_stderr_file.name):
                os.unlink(self._temp_stderr_file.name)
        except Exception:
            pass

    def _emergency_cleanup(self):
        """紧急清理：在进程异常终止时调用"""
        if self._cleanup_done:
            return
            
        try:
            print(f"Emergency cleanup for OutputCapture {self._capture_id}", file=sys.__stderr__)
            
            # 停止刷新线程
            if self._flush_thread:
                self._stop_flush.set()
            
            # 尝试恢复文件描述符
            try:
                if self.old_stdout is not None:
                    os.dup2(self.old_stdout, 1)
                    os.close(self.old_stdout)
                if self.old_stderr is not None:
                    os.dup2(self.old_stderr, 2)
                    os.close(self.old_stderr)
            except:
                pass
            
            # 在安全模式下，尝试保存临时文件内容
            if self.safe_mode and self.output_file_path is not None:
                try:
                    stdout_content = ""
                    stderr_content = ""
                    
                    if self._temp_stdout_file and os.path.exists(self._temp_stdout_file.name):
                        try:
                            with open(self._temp_stdout_file.name, 'r', encoding='utf-8', errors='replace') as f:
                                stdout_content = f.read()
                        except:
                            pass
                            
                    if self._temp_stderr_file and os.path.exists(self._temp_stderr_file.name):
                        try:
                            with open(self._temp_stderr_file.name, 'r', encoding='utf-8', errors='replace') as f:
                                stderr_content = f.read()
                        except:
                            pass
                    
                    # 如果有内容，尝试写入目标文件
                    if stdout_content.strip() or stderr_content.strip():
                        emergency_content = f"\n=== EMERGENCY SAVE @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                        emergency_content += "=== PROCESS TERMINATED UNEXPECTEDLY ===\n"
                        if stdout_content.strip():
                            emergency_content += f"=== CAPTURED STDOUT ===\n{stdout_content}\n"
                        if stderr_content.strip():
                            emergency_content += f"=== CAPTURED STDERR ===\n{stderr_content}\n"
                        emergency_content += "=== END EMERGENCY SAVE ===\n\n"
                        
                        try:
                            with open(self.output_file_path, 'a', encoding='utf-8') as f:
                                f.write(emergency_content)
                                f.flush()
                        except Exception as e:
                            print(f"Failed to save emergency content: {e}", file=sys.__stderr__)
                            
                            # 如果无法写入目标文件，至少保存到一个紧急文件
                            try:
                                emergency_file = f"{self.output_file_path}.emergency_{os.getpid()}"
                                with open(emergency_file, 'w', encoding='utf-8') as f:
                                    f.write(emergency_content)
                                print(f"Emergency content saved to: {emergency_file}", file=sys.__stderr__)
                            except:
                                pass
                                
                except Exception as e:
                    print(f"Emergency save failed: {e}", file=sys.__stderr__)
            
            # 清理临时文件
            self._cleanup_temp_files()
            
        except Exception as e:
            print(f"Emergency cleanup failed: {e}", file=sys.__stderr__)
        
        self._cleanup_done = True

    def get_output(self):
        return self.captured_output

def mp_run(
    worker_func: Callable,
    args: list | tuple = (),
    kwargs: dict = {},
    timeout: int = 300,
    SpecSubProcResult: BaseModel = SubProcResult,
) -> SubProcResult:
    """
    在子进程中运行worker函数，确保进程正确清理
    
    Args:
        worker_func: 要在子进程中执行的函数
        args: 传递给worker_func的参数
        kwargs: 传递给worker_func的关键字参数
        timeout: 超时时间（秒）
        SpecSubProcResult: 结果模型类
    
    Returns:
        SubProcResult: 包含执行结果的对象
    """
    result_queue = None
    process = None
    
    try:
        # 创建队列和进程
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=worker_func,
            args=(result_queue, *args),
            kwargs=kwargs
        )
        process.start()
        process.join(timeout=timeout)
        
        # 处理超时或异常退出
        if process.is_alive():
            print(f"Process {process.pid} timeout, terminating...")
            process.terminate()
            process.join(timeout=5)  # 给更多时间让进程正常退出
            
            if process.is_alive():
                print(f"Process {process.pid} still alive after terminate, killing...")
                process.kill()
                process.join(timeout=2)
                
                if process.is_alive():
                    print(f"Warning: Process {process.pid} still alive after kill")
            
            return SpecSubProcResult(subproc_success=False, result={}, error="Test timeout, process terminated")
        
        # 检查进程退出码
        if process.exitcode is None:
            return SpecSubProcResult(subproc_success=False, result={}, error="Process exit code is None")
        elif process.exitcode < 0:
            try:
                signal_name = signal.Signals(-process.exitcode).name
            except (ValueError, AttributeError):
                signal_name = f"Signal[{process.exitcode}]"
            return SpecSubProcResult(subproc_success=False, result={}, error=f"Process terminated by signal: {signal_name}")
        elif process.exitcode > 0:
            return SpecSubProcResult(subproc_success=False, result={}, error=f"Process exited with code: {process.exitcode}")
        
        # 获取结果
        if not result_queue.empty():
            try:
                result = result_queue.get(timeout=5)  # 给队列操作设置超时
                return SpecSubProcResult(subproc_success=True, result=result, error="")
            except Exception as queue_error:
                return SpecSubProcResult(subproc_success=False, result={}, error=f"Failed to get result from queue: {queue_error}")
        else:
            return SpecSubProcResult(subproc_success=False, result={}, error="Process exited normally but no result found")
            
    except Exception as e:
        error_msg = f"Process execution failed: {e}"
        print(f"mp_run exception: {error_msg}")
        return SpecSubProcResult(subproc_success=False, result={}, error=f"{error_msg}", traceback=traceback.format_exc())
    
    finally:
        # 确保进程和队列被正确清理
        if process is not None:
            try:
                if process.is_alive():
                    print(f"Cleaning up alive process {process.pid}")
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                # 确保进程对象被正确关闭
                process.close()
            except Exception as cleanup_error:
                print(f"Error during process cleanup: {cleanup_error}")
        
        if result_queue is not None:
            try:
                # 清空队列
                while not result_queue.empty():
                    result_queue.get_nowait()
                result_queue.close()
                result_queue.join_thread()
            except Exception as queue_cleanup_error:
                print(f"Error during queue cleanup: {queue_cleanup_error}")