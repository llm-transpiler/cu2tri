from typing import Callable, List, Any, Dict
import traceback
import signal
import multiprocessing
from pydantic import BaseModel, Field

# 设置多进程启动方法为 spawn 以支持 CUDA
multiprocessing.set_start_method('spawn', force=True)

class SubProcResult(BaseModel):
    subproc_success: bool = False
    result: Any = None
    error: str = ""

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
        return SpecSubProcResult(subproc_success=False, result={}, error=f"{error_msg}\ntraceback: {traceback.format_exc()}")
    
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