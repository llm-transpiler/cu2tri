"""
基于multiprocessing的子进程运行器模块

使用multiprocessing在子进程中运行Triton测试，防止core dump影响主进程
相比subprocess方式，避免了临时文件生成和字符串脚本的开销
"""

import os
import multiprocessing
import traceback

# 设置多进程启动方法为 spawn 以支持 CUDA
multiprocessing.set_start_method('spawn', force=True)

from ..common.config import EvalConfig, DEFAULT_CONFIG, DEFAULT_RANDOM_SEED
from ..common.verifier import CompareResult, _compare_tensor_results, PerformanceResult
from ..common.loader import _load_pyfile_module
from ..common.loader import _load_pyfile_module_attr, load_cuda_extension_from_cufile
from ..common.mprunner import mp_run
from ..common.benchmark import benchmark_kernel
from ..common.verifier import _normalize_input
from ..common.mprunner import SubProcResult
from ..common.mprunner import OutputCapture


def _compare_triton_torch_executor(result_queue, triton_file, torch_ref_file, random_seed, config, log_file_path):
    """Worker function for correctness testing"""
    with OutputCapture(log_file_path) as capture:
        try:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            import torch
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                
                torch_ref = _load_torch_reference(torch_ref_file)
                torch_ref_model = torch_ref.Model(*torch_ref.get_init_inputs())
                torch_ref_model.eval()
                torch_ref_model.to(torch.device("cuda"))
                
                test_inputs = torch_ref.get_inputs()
                
                triton_inputs = _normalize_input(test_inputs)
                
                triton_func = _load_triton_kernel(triton_file)
                triton_outputs = torch_ref_model.forward(*triton_inputs, fn=triton_func)
                torch.cuda.synchronize()
                
                torch_inputs = _normalize_input(triton_inputs)
                torch_outputs = torch_ref_model.forward(*torch_inputs, fn=torch_ref.module_fn)
                torch.cuda.synchronize()
                
                match_result = _compare_tensor_results(triton_outputs, torch_outputs, config.rtol, config.atol)
                result_queue.put(match_result)
                
        except Exception as e:
            result_queue.put(CompareResult(comp_exec_success=False, error=str(e), traceback=traceback.format_exc(), output_capture=capture.get_output()))


def triton_compare_torch_worker(
    triton_file: str,
    torch_ref_file: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    config: EvalConfig = DEFAULT_CONFIG,
    log_file_path: str = None,
    timeout: int = DEFAULT_CONFIG.subproc_timeout
) -> SubProcResult:
    return mp_run(_compare_triton_torch_executor, (triton_file, torch_ref_file, random_seed, config, log_file_path), timeout=timeout)


def _compare_triton_cuda_executor(result_queue, triton_file, cuda_ref_file, torch_ref_file, random_seed, config, log_file_path):
    """Worker function for triton vs cuda correctness testing"""
    with OutputCapture(log_file_path) as capture:
        try:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            import torch
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                cuda_fn = _load_cuda_kernel(cuda_ref_file, config)
                
                torch_ref = _load_torch_reference(torch_ref_file)
                torch_ref_model = torch_ref.Model(*torch_ref.get_init_inputs())
                torch_ref_model.eval()
                torch_ref_model.to(torch.device("cuda"))
                
                test_inputs = torch_ref.get_inputs()
                inputs = _normalize_input(test_inputs)
                
                triton_fn = _load_triton_kernel(triton_file)
                triton_outputs = torch_ref_model.forward(*inputs, fn=triton_fn)
                torch.cuda.synchronize()
                
                cuda_inputs_ref = _normalize_input(test_inputs)
                cuda_result = torch_ref_model.forward(*cuda_inputs_ref, fn=cuda_fn)
                torch.cuda.synchronize()
                
                match_result = _compare_tensor_results(triton_outputs, cuda_result, config.rtol, config.atol)
                result_queue.put(match_result)
                
        except Exception as e:
            result_queue.put(CompareResult(comp_exec_success=False, error=str(e), traceback=traceback.format_exc(), output_capture=capture.get_output()))

def triton_compare_cuda_worker(
    triton_file: str,
    cuda_ref_file: str,
    torch_ref_file: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    config: EvalConfig = DEFAULT_CONFIG,
    log_file_path: str = None,
    timeout: int = DEFAULT_CONFIG.subproc_timeout
) -> SubProcResult:
    return mp_run(_compare_triton_cuda_executor, (triton_file, cuda_ref_file, torch_ref_file, random_seed, config, log_file_path), timeout=timeout)


def _compare_cuda_torch_executor(result_queue, cuda_file, torch_ref_file, random_seed, config, log_file_path):
    """Worker function for cuda vs torch correctness testing"""
    with OutputCapture(log_file_path) as capture:
        try:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            import torch
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                
                cuda_fn = _load_cuda_kernel(cuda_file, config)
                # 重新获取inputs用于torch计算
                torch_ref = _load_torch_reference(torch_ref_file)
                
                torch_ref_model = torch_ref.Model(*torch_ref.get_init_inputs())
                torch_ref_model.eval()
                torch_ref_model.to(torch.device("cuda"))
                torch_inputs = torch_ref.get_inputs()
                cuda_inputs = _normalize_input(torch_inputs)
                
                cuda_result = torch_ref_model.forward(*cuda_inputs, fn=cuda_fn)
                torch.cuda.synchronize()
                
                torch_inputs = _normalize_input(cuda_inputs)
                torch_outputs = torch_ref_model.forward(*torch_inputs, fn=torch_ref.module_fn)
                torch.cuda.synchronize()

                match_result = _compare_tensor_results(cuda_result, torch_outputs, config.rtol, config.atol)
                result_queue.put(match_result)
                
        except Exception as e:
            result_queue.put(CompareResult(comp_exec_success=False, error=str(e), traceback=traceback.format_exc(), output_capture=capture.get_output()))


def cuda_compare_torch_worker(
    cuda_file: str,
    torch_ref_file: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    config: EvalConfig = DEFAULT_CONFIG,
    log_file_path: str = None,
    timeout: int = DEFAULT_CONFIG.subproc_timeout
) -> SubProcResult:
    return mp_run(_compare_cuda_torch_executor, (cuda_file, torch_ref_file, random_seed, config, log_file_path), timeout=timeout)


def _perf_triton_executor(result_queue, triton_file, torch_ref_file, random_seed, warmup_runs, test_runs, log_file_path):
    """Worker function for performance testing"""
    with OutputCapture(log_file_path) as capture:
        try:
            import torch # must be here
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                
                torch_ref = _load_torch_reference(torch_ref_file)
                torch_ref_model = torch_ref.Model(*torch_ref.get_init_inputs())
                torch_ref_model.eval()
                torch_ref_model.to(torch.device("cuda"))
                
                test_inputs = torch_ref.get_inputs()
                
                # 加载Triton内核
                triton_fn = _load_triton_kernel(triton_file)
                triton_inputs = _normalize_input(test_inputs)
                
                # 性能测试
                def triton_test_func():
                    return torch_ref_model.forward(*triton_inputs, fn=triton_fn)
                    # return triton_fn(*triton_inputs)
                triton_time = benchmark_kernel(triton_test_func, [], warmup=warmup_runs, iterations=test_runs)
                
                result_queue.put(PerformanceResult(perf_exec_success=True, perf_time_ms=triton_time, output_capture=capture.get_output()))
                
        except Exception as e:
            result_queue.put(PerformanceResult(perf_exec_success=False, error=str(e), traceback=traceback.format_exc(), output_capture=capture.get_output()))


def triton_perf_worker(
    triton_file: str,
    torch_ref_file: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    log_file_path: str = None,
    config: EvalConfig = DEFAULT_CONFIG,
) -> SubProcResult:
    return mp_run(_perf_triton_executor, (triton_file, torch_ref_file, random_seed, config.warmup_runs, config.test_runs, log_file_path), timeout=config.subproc_timeout)

def _perf_cuda_executor(result_queue, cuda_file, torch_ref_file, random_seed, config, log_file_path):
    """Worker function for CUDA performance testing"""
    with OutputCapture(log_file_path) as capture:
        try:
            import torch
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                
                cuda_fn = _load_cuda_kernel(cuda_file, config)
                torch_ref = _load_torch_reference(torch_ref_file)
                torch_ref_model = torch_ref.Model(*torch_ref.get_init_inputs())
                torch_ref_model.eval()
                torch_ref_model.to(torch.device("cuda"))
                
                test_inputs = torch_ref.get_inputs()
                cuda_inputs = _normalize_input(test_inputs)                
                
                # 性能测试
                def cuda_test_func():
                    return torch_ref_model.forward(*cuda_inputs, fn=cuda_fn)
                cuda_time = benchmark_kernel(cuda_test_func, [], warmup=config.warmup_runs, iterations=config.test_runs)
                
                result_queue.put(PerformanceResult(perf_exec_success=True, perf_time_ms=cuda_time, output_capture=capture.get_output()))
                
        except Exception as e:
            result_queue.put(PerformanceResult(perf_exec_success=False, error=str(e), traceback=traceback.format_exc(), output_capture=capture.get_output()))

def cuda_perf_worker(
    cuda_file: str,
    torch_ref_file: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    log_file_path: str = None,
    config: EvalConfig = DEFAULT_CONFIG,
) -> SubProcResult:
    return mp_run(_perf_cuda_executor, (cuda_file, torch_ref_file, random_seed, config, log_file_path), timeout=config.subproc_timeout)

def _perf_torch_executor(result_queue, torch_file, random_seed, config, log_file_path):
    """Worker function for Torch performance testing"""
    with OutputCapture(log_file_path) as capture:
        try:
            import torch
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                
                torch_ref = _load_torch_reference(torch_file)
                test_inputs = torch_ref.get_inputs()
                torch_inputs = _normalize_input(test_inputs)
                
                # 创建torch模型实例
                torch_ref_model = torch_ref.Model(*torch_ref.get_init_inputs())
                torch_ref_model.eval()
                torch_ref_model.to(torch.device("cuda"))
                
                # 性能测试
                def torch_test_func():
                    return torch_ref_model.forward(*torch_inputs, fn=torch_ref.module_fn)
                torch_time = benchmark_kernel(torch_test_func, [], warmup=config.warmup_runs, iterations=config.test_runs)
                
                result_queue.put(PerformanceResult(perf_exec_success=True, perf_time_ms=torch_time, output_capture=capture.get_output()))
                
        except Exception as e:
            result_queue.put(PerformanceResult(perf_exec_success=False, error=str(e), traceback=traceback.format_exc(), output_capture=capture.get_output()))

def torch_perf_worker(
    torch_file: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    log_file_path: str = None,
    config: EvalConfig = DEFAULT_CONFIG,
) -> SubProcResult:
    return mp_run(_perf_torch_executor, (torch_file, random_seed, config, log_file_path), timeout=config.subproc_timeout)

def _load_torch_reference(torch_ref_file: str):
    return _load_pyfile_module(torch_ref_file)

def _load_triton_kernel(triton_file: str):
    return _load_pyfile_module_attr(triton_file, "forward")

def _load_cuda_kernel(cuda_ref_file: str, config: EvalConfig):
    extension = load_cuda_extension_from_cufile(cuda_ref_file, config)
    return getattr(extension, "forward")