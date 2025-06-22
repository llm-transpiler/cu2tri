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
    # 创建队列和进程
    try:
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
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join()
            return SpecSubProcResult(subproc_success=False, result={}, error="Test timeout, no result found")
        
        # 检查进程退出码
        if process.exitcode < 0:
            try:
                signal_name = signal.Signals(-process.exitcode).name
            except (ValueError, AttributeError):
                signal_name = f"Signal[{process.exitcode}]"
            return SpecSubProcResult(subproc_success=False, result={}, error=f"Process terminated by signal: {signal_name}")
        
        # 获取结果
        if not result_queue.empty():
            return SpecSubProcResult(subproc_success=True, result=result_queue.get(), error="")
        else:
            return SpecSubProcResult(subproc_success=False, result={}, error="Process exited abnormally, no result found")
    except Exception as e:
        return SpecSubProcResult(subproc_success=False, result={}, error=f"Process exited abnormally, no result found: {e} \ntraceback: {traceback.format_exc()}")