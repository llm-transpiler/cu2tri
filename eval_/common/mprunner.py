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
    
    # 使用非阻塞锁避免死锁
    cleanup_timeout = 5.0  # 5秒超时
    lock_acquired = False
    
    try:
        # 尝试获取锁，如果失败则继续
        lock_acquired = _cleanup_lock.acquire(timeout=cleanup_timeout)
        
        captures_to_clean = list(_active_captures.items()) if lock_acquired else []
        
        # 如果无法获取锁，至少尝试清理已知的捕获实例
        if not lock_acquired:
            print("Emergency cleanup: Unable to acquire lock, attempting direct cleanup", file=sys.__stderr__)
            # 直接尝试清理，不依赖全局注册表
            try:
                # 尝试恢复标准输出
                if hasattr(sys, '_original_stdout'):
                    sys.stdout = sys._original_stdout
                if hasattr(sys, '_original_stderr'):
                    sys.stderr = sys._original_stderr
            except:
                pass
            return
        
        print(f"Emergency cleanup: Found {len(captures_to_clean)} active captures", file=sys.__stderr__)
        
        for capture_id, capture in captures_to_clean:
            try:
                # 设置超时防止单个清理操作阻塞太久
                capture._emergency_cleanup()
            except Exception as e:
                print(f"Emergency cleanup failed for capture {capture_id}: {e}", file=sys.__stderr__)
        
        _active_captures.clear()
        
    except Exception as e:
        print(f"Emergency cleanup encountered error: {e}", file=sys.__stderr__)
    finally:
        if lock_acquired:
            try:
                _cleanup_lock.release()
            except:
                pass


def _signal_handler(signum, frame):
    """信号处理器，捕获进程终止信号"""
    print(f"Received signal {signum}, performing emergency cleanup...", file=sys.__stderr__)
    
    try:
        # 执行紧急清理，设置更短的超时
        cleanup_start = time.time()
        _emergency_cleanup()
        cleanup_duration = time.time() - cleanup_start
        print(f"Emergency cleanup completed in {cleanup_duration:.2f}s", file=sys.__stderr__)
        
    except Exception as e:
        print(f"Emergency cleanup failed: {e}", file=sys.__stderr__)
    
    # 对于子进程，快速退出而不重新发送信号
    # 避免与主进程的信号处理器冲突
    print(f"Exiting subprocess due to signal {signum}", file=sys.__stderr__)
    
    # 给一点时间让日志输出完成
    try:
        sys.stderr.flush()
        time.sleep(0.05)  # 减少等待时间
    except:
        pass
    
    # 使用对应的退出码
    if signum == signal.SIGTERM:
        os._exit(128 + signal.SIGTERM)  # 143
    elif signum == signal.SIGINT:
        os._exit(128 + signal.SIGINT)   # 130
    else:
        os._exit(128 + signum)


def _register_cleanup():
    """注册全局清理机制"""
    global _cleanup_registered
    if not _cleanup_registered:
        # 注册atexit清理函数
        atexit.register(_emergency_cleanup)
        
        # 注册信号处理器 - 包括更多可能导致异常退出的信号
        signals_to_handle = [
            signal.SIGTERM,  # 终止信号
            signal.SIGINT,   # 中断信号 (Ctrl+C)
            signal.SIGHUP,   # 挂起信号
            signal.SIGUSR1,  # 用户定义信号1
            signal.SIGUSR2,  # 用户定义信号2
        ]
        
        # 某些严重错误信号通常不应该被捕获，但我们可以尝试处理
        # 注意：SIGSEGV, SIGABRT等通常不应该被捕获，因为它们表示程序严重错误
        critical_signals = [
            signal.SIGSEGV,  # 段错误
            signal.SIGABRT,  # 异常终止
            signal.SIGFPE,   # 浮点异常
            signal.SIGILL,   # 非法指令
            signal.SIGBUS,   # 总线错误
        ]
        
        for sig in signals_to_handle:
            try:
                signal.signal(sig, _signal_handler)
                print(f"Registered signal handler for {sig.name}", file=sys.__stderr__)
            except (OSError, ValueError, AttributeError) as e:
                print(f"Failed to register handler for signal {sig}: {e}", file=sys.__stderr__)
        
        # 对于严重错误信号，我们尝试注册但更谨慎
        for sig in critical_signals:
            try:
                # 只在Linux/Unix系统上尝试注册这些信号
                if hasattr(os, 'name') and os.name == 'posix':
                    signal.signal(sig, _critical_signal_handler)
                    print(f"Registered critical signal handler for {sig.name}", file=sys.__stderr__)
            except (OSError, ValueError, AttributeError) as e:
                # 这些信号注册失败是正常的，不输出错误
                pass
        
        _cleanup_registered = True


def _critical_signal_handler(signum, frame):
    """处理严重错误信号的处理器"""
    try:
        signal_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        signal_name = f"Signal[{signum}]"
    
    print(f"CRITICAL: Received {signal_name} ({signum}) - attempting fast cleanup", file=sys.__stderr__)
    
    # 对于严重错误，我们需要更快速的清理
    try:
        # 快速恢复标准输出
        try:
            if hasattr(sys, '_original_stdout'):
                sys.stdout = sys._original_stdout
            if hasattr(sys, '_original_stderr'):
                sys.stderr = sys._original_stderr
        except:
            pass
        
        # 尝试快速清理，但不要花太多时间（减少超时时间）
        cleanup_start = time.time()
        _emergency_cleanup()
        cleanup_duration = time.time() - cleanup_start
        print(f"Critical cleanup completed in {cleanup_duration:.2f}s", file=sys.__stderr__)
        
    except Exception as e:
        print(f"Critical cleanup failed: {e}", file=sys.__stderr__)
    
    # 对于严重错误信号，快速退出
    print(f"Fast exit due to critical signal {signal_name}", file=sys.__stderr__)
    try:
        sys.stderr.flush()
    except:
        pass
    
    # 使用_exit快速退出，不执行清理
    os._exit(128 + signum)


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
        """直接模式：直接重定向stdout/stderr到文件"""
        if self.output_file_path is None:
            # 如果没有指定输出文件，创建临时文件
            import tempfile
            self._temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
            self.output_file_path = self._temp_file.name
            self.file_temp = True
        else:
            self.file_temp = False 

        # 确保输出文件的父目录存在
        import os
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)

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
            
        cleanup_start = time.time()
        cleanup_timeout = 3.0  # 3秒超时
        
        try:
            print(f"Emergency cleanup for OutputCapture {self._capture_id}", file=sys.__stderr__)
            
            # 停止刷新线程（快速处理）
            if self._flush_thread:
                self._stop_flush.set()
                # 不等待线程结束，因为在紧急情况下可能会阻塞
            
            # 尝试恢复文件描述符（高优先级操作）
            try:
                if self.old_stdout is not None:
                    os.dup2(self.old_stdout, 1)
                    os.close(self.old_stdout)
                    self.old_stdout = None
                if self.old_stderr is not None:
                    os.dup2(self.old_stderr, 2)
                    os.close(self.old_stderr)
                    self.old_stderr = None
            except Exception as fd_error:
                print(f"FD restore failed: {fd_error}", file=sys.__stderr__)
            
            # 检查是否超时
            if time.time() - cleanup_start > cleanup_timeout:
                print(f"Emergency cleanup timeout after FD restore", file=sys.__stderr__)
                self._cleanup_done = True
                return
            
            # 在安全模式下，尝试保存临时文件内容（较低优先级）
            if self.safe_mode and self.output_file_path is not None:
                try:
                    self._emergency_save_content(cleanup_start, cleanup_timeout)
                except Exception as save_error:
                    print(f"Emergency save failed: {save_error}", file=sys.__stderr__)
            
            # 清理临时文件（如果还有时间）
            if time.time() - cleanup_start < cleanup_timeout:
                try:
                    self._cleanup_temp_files()
                except Exception as cleanup_error:
                    print(f"Temp file cleanup failed: {cleanup_error}", file=sys.__stderr__)
            
        except Exception as e:
            print(f"Emergency cleanup failed: {e}", file=sys.__stderr__)
        finally:
            self._cleanup_done = True
            cleanup_duration = time.time() - cleanup_start
            print(f"Emergency cleanup for {self._capture_id} completed in {cleanup_duration:.2f}s", file=sys.__stderr__)
    
    def _emergency_save_content(self, cleanup_start, cleanup_timeout):
        """紧急保存内容的辅助方法"""
        stdout_content = ""
        stderr_content = ""
        
        # 快速读取临时文件内容
        if self._temp_stdout_file and os.path.exists(self._temp_stdout_file.name):
            try:
                # 限制读取大小避免阻塞
                with open(self._temp_stdout_file.name, 'r', encoding='utf-8', errors='replace') as f:
                    stdout_content = f.read(1024 * 1024)  # 最多读取1MB
            except:
                pass
                
        if time.time() - cleanup_start > cleanup_timeout:
            return
                
        if self._temp_stderr_file and os.path.exists(self._temp_stderr_file.name):
            try:
                with open(self._temp_stderr_file.name, 'r', encoding='utf-8', errors='replace') as f:
                    stderr_content = f.read(1024 * 1024)  # 最多读取1MB
            except:
                pass
        
        # 如果有内容且还有时间，尝试写入目标文件
        if (stdout_content.strip() or stderr_content.strip()) and time.time() - cleanup_start < cleanup_timeout:
            emergency_content = f"\n=== EMERGENCY SAVE @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            emergency_content += f"=== PROCESS TERMINATED UNEXPECTEDLY (PID: {os.getpid()}) ===\n"
            if stdout_content.strip():
                emergency_content += f"=== CAPTURED STDOUT (truncated) ===\n{stdout_content[:10000]}\n"
            if stderr_content.strip():
                emergency_content += f"=== CAPTURED STDERR (truncated) ===\n{stderr_content[:10000]}\n"
            emergency_content += "=== END EMERGENCY SAVE ===\n\n"
            
            try:
                # 使用非阻塞写入
                with open(self.output_file_path, 'a', encoding='utf-8') as f:
                    f.write(emergency_content)
                    f.flush()
            except Exception as e:
                # 如果无法写入目标文件，保存到紧急文件
                try:
                    emergency_file = f"{self.output_file_path}.emergency_{os.getpid()}_{int(time.time())}"
                    with open(emergency_file, 'w', encoding='utf-8') as f:
                        f.write(emergency_content)
                    print(f"Emergency content saved to: {emergency_file}", file=sys.__stderr__)
                except:
                    pass

    def get_output(self):
        return self.captured_output


# class OutputCapture:
#     """用于捕获所有输出（包括子进程）的上下文管理器"""
#     def __init__(self, output_file_path: str = None):
#         self.captured_output = ""
#         self.old_stdout = None
#         self.old_stderr = None
#         self.temp_file = None
#         self.output_file_path = output_file_path

#     def __enter__(self):
#         # 创建临时文件用于捕获输出
#         if self.output_file_path is None:
#             self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
#             self.temp_file.close()
#         else:
#             self.temp_file = open(self.output_file_path, 'w')

#         # 保存原始的文件描述符
#         self.old_stdout = os.dup(1)
#         self.old_stderr = os.dup(2)

#         # 将stdout和stderr重定向到临时文件
#         temp_fd = os.open(self.temp_file.name, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
#         os.dup2(temp_fd, 1)  # stdout
#         os.dup2(temp_fd, 2)  # stderr
#         os.close(temp_fd)

#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # 刷新缓冲区
#         sys.stdout.flush()
#         sys.stderr.flush()

#         # 恢复原始的文件描述符
#         os.dup2(self.old_stdout, 1)
#         os.dup2(self.old_stderr, 2)
#         os.close(self.old_stdout)
#         os.close(self.old_stderr)

#         # 读取捕获的输出，使用错误处理来处理二进制数据
#         try:
#             with open(self.temp_file.name, 'r', encoding='utf-8') as f:
#                 self.captured_output = f.read()
#         except UnicodeDecodeError:
#             # 如果有二进制数据，使用错误替换模式
#             with open(self.temp_file.name, 'r', encoding='utf-8', errors='replace') as f:
#                 self.captured_output = f.read()
        
#         self.temp_file.close()

#     def get_output(self):
#         return self.captured_output

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
        print(f"mp_run: {worker_func.__name__}")
        # 创建队列和进程
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=worker_func,
            args=(result_queue, *args),
            kwargs=kwargs
        )
        process.start()
        print(f"mp_run process start: {process.pid}")
        process.join(timeout=timeout)
        print(f"mp_run process end: {process.pid}")
        print(f"mp_run end: {worker_func.__name__}")
        
        # 处理超时或异常退出
        if process.is_alive():
            print(f"Process {process.pid} timeout after {timeout}s, terminating gracefully...")
            
            # 第一步：发送SIGTERM信号
            try:
                process.terminate()
                print(f"Sent SIGTERM to process {process.pid}")
            except Exception as e:
                print(f"Failed to terminate process {process.pid}: {e}")
            
            # 等待进程优雅退出
            process.join(timeout=10)  # 给更多时间让紧急清理完成
            
            if process.is_alive():
                print(f"Process {process.pid} still alive after SIGTERM, sending SIGKILL...")
                try:
                    process.kill()
                    print(f"Sent SIGKILL to process {process.pid}")
                except Exception as e:
                    print(f"Failed to kill process {process.pid}: {e}")
                
                # 最后等待
                process.join(timeout=5)
                
                if process.is_alive():
                    print(f"Warning: Process {process.pid} still alive after SIGKILL")
            else:
                print(f"Process {process.pid} terminated gracefully")
            
            return SpecSubProcResult(subproc_success=False, result={}, error=f"Process timeout after {timeout}s, terminated")
        
        # 检查进程退出码并详细分析异常情况
        if process.exitcode is None:
            return SpecSubProcResult(subproc_success=False, result={}, error="Process exit code is None (process may still be running)")
        elif process.exitcode < 0:
            # 负数退出码表示被信号终止
            signal_num = -process.exitcode
            try:
                signal_name = signal.Signals(signal_num).name
            except (ValueError, AttributeError):
                signal_name = f"Signal[{signal_num}]"
            
            # 分析具体的信号类型
            error_msg = f"Process terminated by signal: {signal_name} ({signal_num})"
            
            # 检查是否是严重错误信号
            critical_signals = {
                signal.SIGSEGV: "Segmentation fault (core dump)",
                signal.SIGABRT: "Process aborted (core dump)", 
                signal.SIGFPE: "Floating point exception",
                signal.SIGILL: "Illegal instruction",
                signal.SIGBUS: "Bus error (bad memory access)",
                signal.SIGSYS: "Bad system call",
                signal.SIGTRAP: "Trace/breakpoint trap"
            }
            
            if signal_num in critical_signals:
                error_msg += f" - {critical_signals[signal_num]}"
                print(f"WARNING: Process {process.pid} crashed with {error_msg}", file=sys.stderr)
            elif signal_num == signal.SIGKILL:
                error_msg += " - Force killed (likely due to timeout or resource issues)"
            elif signal_num == signal.SIGTERM:
                error_msg += " - Graceful termination requested"
            
            return SpecSubProcResult(subproc_success=False, result={}, error=error_msg)
        elif process.exitcode > 0:
            error_msg = f"Process exited with code: {process.exitcode}"
            
            # 分析常见的退出码
            if process.exitcode == 1:
                error_msg += " (General error)"
            elif process.exitcode == 2:
                error_msg += " (Misuse of shell command)"
            elif process.exitcode == 126:
                error_msg += " (Command invoked cannot execute)"
            elif process.exitcode == 127:
                error_msg += " (Command not found)"
            elif process.exitcode == 128:
                error_msg += " (Invalid argument to exit)"
            elif process.exitcode > 128:
                # 128 + signal_num 通常表示被信号终止
                potential_signal = process.exitcode - 128
                error_msg += f" (Possibly terminated by signal {potential_signal})"
            
            return SpecSubProcResult(subproc_success=False, result={}, error=error_msg)
        
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